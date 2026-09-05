"""Generic duplex channel to an isolated engine subprocess.

This is the transport layer shared by every out-of-process engine. It exists for
three reasons, in descending order of importance:

1. **Reclamation.** Killing a process is the only way to return a CUDA/ROCm/OpenVINO
   context's device memory to the OS. In-process teardown (``gc.collect()`` plus
   ``ctranslate2.clear_caches()``) demonstrably reclaims none of it: an idle purge on
   this codebase logs ``CUDA VRAM=193 MB -> CUDA VRAM=193 MB (Delta: +0 MB)``.
2. **Crash containment.** A native segfault in CTranslate2, ROCm or the OpenVINO GPU
   plugin currently kills the API process. Out-of-process it kills one worker, which
   is respawned on the next call.
3. **Mutually exclusive runtimes.** One interpreter cannot host a CUDA context and an
   OpenVINO GPU context (see ``modules/core/config.py``), nor a CUDA torch build and a
   ROCm torch build. Separate processes are the only way to run them together.

**Memory:** a channel owns exactly one worker process, and callers are expected to
create one channel *per engine type* rather than per hardware unit. The worker keeps
the same ``unit_id -> model`` pool the parent used to hold, so isolation relocates
resident models rather than duplicating them. Audio crosses the boundary as a path,
never as a decoded array, and transcription results stream segment-by-segment instead
of accumulating a full list in the child.

**Duplex.** ``multiprocessing.Pipe`` is bidirectional. :meth:`stream` reads worker
events while allowing the *same* thread to send control messages (``cancel``) between
events, which is what lets cooperative preemption survive the process boundary. Only
one thread ever writes each direction, so no additional synchronisation is required
beyond the channel lock.

The concurrency invariants here (generation stamping, shutdown races, lock-free
teardown) are ported deliberately from ``whisperx_worker_client``; see that module for
the original rationale. Phase 5 of the isolation work retires the duplicate.
"""

import logging
import multiprocessing
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from itertools import count
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import Any, Optional

logger = logging.getLogger(__name__)

_CTX = multiprocessing.get_context("spawn")

#: Poll chunk used when no fixed RPC deadline is configured, so an unbounded wait still
#: re-checks the shutdown flag periodically.
_SHUTDOWN_POLL_INTERVAL_SEC = 2.0

#: Budget for draining a cancelled stream back to a clean pipe before giving up and
#: respawning. Generous enough for a worker to finish the chunk it is mid-way through,
#: short enough that a wedged worker is replaced rather than waited on.
_ABANDON_DRAIN_POLL_SEC = 0.5
_ABANDON_DRAIN_TIMEOUT_SEC = 30.0

#: Events that end a stream normally. "cancelled" is the worker acknowledging a cancel it
#: acted on; it terminates the stream exactly as "done" does, and is separate so the two
#: outcomes stay distinguishable rather than a partial run reading as a complete one.
_TERMINAL_EVENTS = ("done", "cancelled")

#: What _drain_to_terminal_event accepts as "the pipe is clean again" -- the above plus the
#: error terminator, which also ends the worker's side of the request.
_DRAIN_TERMINAL_EVENTS = (*_TERMINAL_EVENTS, "error")


#: Sentinel distinguishing "the pipe died" from "nothing arrived this poll". A plain False
#: would conflate the two, and only one of them means the drain can never succeed.
_PIPE_BROKEN = object()


def _poll_once(conn: Connection, deadline: float):
    """Wait briefly for one message, returning (received, message).

    ``received`` is :data:`_PIPE_BROKEN` when the connection is gone, False when the poll
    simply timed out with the worker still alive, and True when ``message`` is real.
    """
    try:
        remaining = min(_ABANDON_DRAIN_POLL_SEC, max(0.0, deadline - time.monotonic()))
        if not conn.poll(remaining):
            return False, None
        return True, conn.recv()
    except (EOFError, OSError, BrokenPipeError):
        return _PIPE_BROKEN, None


def _is_terminal(message: Any) -> bool:
    """Whether a drained message ends the worker's side of the request."""
    return isinstance(message, dict) and message.get("event") in _DRAIN_TERMINAL_EVENTS


def _drain_outcome(received: Any, message: Any) -> Optional[bool]:
    """Whether this poll settled the drain, and how. None means "keep going".

    A dead pipe settles it as failure and a terminal event as success; anything else --
    a timed-out poll, or a progress event from a worker still talking -- leaves the caller
    to keep draining until its own clock runs out.
    """
    if received is _PIPE_BROKEN:
        return False
    if received and _is_terminal(message):
        return True
    return None


class WorkerError(RuntimeError):
    """Raised when an isolated worker reports a failure, dies, or times out."""


class WorkerReportedError(WorkerError):
    """Raised when the worker itself reported a handler error, rather than dying.

    Separated from its base so a caller can retry a *dead* worker -- respawning is often
    enough -- without also retrying a failure the worker was healthy enough to describe.
    Retrying the latter just runs the same failing work twice.
    """


def _reported_error_for(error_cls: type[WorkerError]) -> type[WorkerError]:
    """Return the reported-error subclass matching ``error_cls``.

    A caller that passes its own error type still gets a distinguishable reported variant,
    so ``except <their type>`` keeps working while ``except WorkerReportedError`` narrows.
    """
    if error_cls is WorkerError:
        return WorkerReportedError
    if issubclass(error_cls, WorkerReportedError):
        return error_cls
    return type(f"{error_cls.__name__}Reported", (WorkerReportedError, error_cls), {})


class WorkerChannel:
    """Owns one lazily-started, long-lived worker process and serializes calls to it.

    ``worker_main`` must be a module-level function (spawn pickles it by reference)
    taking a single :class:`Connection`.
    """

    def __init__(
        self,
        worker_main: Callable[[Connection], None],
        *,
        name: str,
        log_tag: str,
        error_cls: type[WorkerError] = WorkerError,
        call_timeout_sec: float = 0.0,
        lock_warn_sec: float = 5.0,
    ) -> None:
        self._worker_main = worker_main
        self._name = name
        self._log_tag = log_tag
        self._error_cls = error_cls
        # A worker that reports a failure is alive and said what went wrong; a worker that
        # dies says nothing. Callers distinguish them to decide whether a retry is useful.
        self._reported_error_cls = _reported_error_for(error_cls)
        #: 0 disables the RPC deadline so genuinely long CPU jobs are not killed
        #: mid-flight. Set > 0 for a hung-worker ceiling in tests/ops.
        self._call_timeout_sec = call_timeout_sec
        self._lock_warn_sec = lock_warn_sec

        self._lock = threading.Lock()
        self._request_ids = count()
        # Bumped in _teardown_worker, the single funnel every destroy path goes through,
        # so a handle cached against a prior generation is reliably invalidated no
        # matter *why* the worker went away.
        self._state: dict[str, Any] = {"process": None, "conn": None, "generation": 0}

        # Set while shutdown() runs. With an unbounded deadline an in-flight call on a
        # hung worker would otherwise hold the lock forever and shutdown could never
        # get in to kill it; the waiter polls this flag and gives up on its own.
        self._shutdown_requested = threading.Event()
        self._shutdown_state_lock = threading.Lock()
        self._shutdown_state = {"in_progress": 0}

    # --- locking -----------------------------------------------------------------

    @contextmanager
    def _locked(self, operation: str, **context: Any) -> Iterator[None]:
        """Hold the channel lock, warning when a caller blocks on contention."""
        if not self._lock.acquire(timeout=self._lock_warn_sec):
            ctx = " ".join(f"{key}={value}" for key, value in sorted(context.items()) if value is not None)
            suffix = f" ({ctx})" if ctx else ""
            logger.warning(
                "[%s] Blocked on lock for >%.1fs during %s%s; waiting for serialized access",
                self._log_tag,
                self._lock_warn_sec,
                operation,
                suffix,
            )
            self._lock.acquire()
        try:
            yield
        finally:
            self._lock.release()

    # --- lifecycle ---------------------------------------------------------------

    @property
    def lock(self):
        """The call lock, held for the whole of a call or a stream.

        Public because callers outside this module need to *observe* it: telemetry decides
        whether an accelerator is busy from whether its preprocessor's lock is held, and an
        isolated preprocessor's work happens in the worker, so this is the only place that
        state exists. Exposed as the lock object, not a boolean, so it presents the same
        `.locked()` interface as the in-process manager's own lock.

        For observation only -- acquire it through the channel's own methods, never here.
        """
        return self._lock

    def generation(self) -> int:
        """Return the current worker generation, reaping a since-died process first.

        Callers use this as a pre-flight cache-validity check before sending a cached
        handle. Without the liveness reap they would observe a stale (pre-death)
        generation and treat an already-invalid handle as good.
        """
        with self._locked("generation"):
            process = self._state["process"]
            if process is not None and not process.is_alive():
                self._teardown_worker()
            return self._state["generation"]

    def is_running(self) -> bool:
        """Whether a live worker process currently exists (no spawn side effect)."""
        process = self._state["process"]
        return process is not None and process.is_alive()

    def _ensure_worker(self) -> None:
        process = self._state["process"]
        if process is not None and process.is_alive():
            return
        if process is not None:
            # Reap the dead worker through the normal funnel so its pipe handle is
            # closed and the generation invalidated, rather than leaking both.
            self._teardown_worker()

        parent_conn, child_conn = _CTX.Pipe()
        process = _CTX.Process(target=self._worker_main, args=(child_conn,), daemon=True, name=self._name)
        process.start()
        child_conn.close()
        self._state["process"] = process
        self._state["conn"] = parent_conn
        logger.info(
            "[%s] Started isolated worker process (pid=%s, generation=%s)",
            self._log_tag,
            process.pid,
            self._state["generation"],
        )

    def _close_connection(self, conn: Optional[Connection]) -> None:
        if conn is None:
            return
        try:
            conn.close()
        except OSError:
            pass

    def _terminate_process(self, process: Optional[BaseProcess]) -> None:
        if process is None or not process.is_alive():
            return
        process.terminate()
        process.join(timeout=10)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)

    def _teardown_worker(self) -> None:
        """Lock-free teardown. Callers hold the lock already, or don't need it.

        This is the single place the generation is bumped, so every destroy path
        (shutdown, RPC timeout/error, dead-process reap) invalidates cached handles.
        """
        self._close_connection(self._state["conn"])
        self._terminate_process(self._state["process"])
        self._state["process"] = None
        self._state["conn"] = None
        self._state["generation"] += 1

    # --- request/response --------------------------------------------------------

    def _poll_timeout(self) -> float:
        if self._call_timeout_sec > 0:
            return self._call_timeout_sec
        return _SHUTDOWN_POLL_INTERVAL_SEC

    def _wait_for_message(self, conn: Connection) -> bool:
        """Block for one inbound message.

        Returns False when the fixed deadline elapses, or -- with no deadline -- when a
        concurrent shutdown is requested, so a hung worker cannot pin the lock forever.
        """
        deadline_enabled = self._call_timeout_sec > 0
        while True:
            if conn.poll(self._poll_timeout()):
                return True
            if deadline_enabled:
                return False
            if self._shutdown_requested.is_set():
                return False

    def _reject_if_shutdown_requested(self, cmd: str) -> None:
        """Reject a call queued behind shutdown rather than spawning a worker that is
        about to be torn down mid-request."""
        if self._shutdown_requested.is_set():
            raise self._error_cls(f"{self._log_tag} worker shut down during '{cmd}'")

    def _send_request(self, cmd: str, args: dict[str, Any], *, stream: bool) -> int:
        """Send one request. Must run with the lock held.

        A pipe that dies on *send* is as fatal as one that dies on recv, and has to be
        reported the same way -- callers catch the channel's error type, not raw
        BrokenPipeError -- so the teardown lives here too rather than only around recv.
        """
        request_id = next(self._request_ids)
        conn = self._state["conn"]
        try:
            conn.send({"id": request_id, "cmd": cmd, "args": args, "stream": stream})
        except (EOFError, OSError, BrokenPipeError) as exc:
            # Read the exit status before tearing down -- teardown clears the process
            # handle, and with it the only evidence of how the worker died.
            detail = self._death_detail()
            # The lock is held, so tear down directly; shutdown() would deadlock here.
            self._teardown_worker()
            raise self._error_cls(f"{self._log_tag} died during '{cmd}' ({detail}): {exc or 'pipe closed'}") from exc
        return request_id

    def _death_detail(self) -> str:
        """Describe how the worker died, for an error a reader can act on.

        A native crash closes the pipe and surfaces as a bare EOFError, so without the
        exit status the message is just "worker died" with an empty reason. A negative
        code is the fatal signal -- -11 (SIGSEGV) points at the vendor runtime rather
        than at anything in Python.
        """
        process = self._state["process"]
        # A child that has just died is not reaped until it is waited on, and until then
        # exitcode is None -- which would report "no exit status" for every crash.
        try:
            process.join(timeout=2)
        except (AttributeError, AssertionError, ValueError):
            pass
        try:
            code = int(getattr(process, "exitcode", None))
        except (TypeError, ValueError):
            # Still running, already reaped, or not a real process (tests).
            return "no exit status"
        if code < 0:
            signal_note = " (SIGSEGV -- native crash in the vendor runtime)" if code == -11 else ""
            return f"killed by signal {-code}{signal_note}"
        return f"exit code {code}"

    def _receive(self, cmd: str) -> dict[str, Any]:
        """Await one message. Must run with the lock held."""
        conn = self._state["conn"]
        try:
            if not self._wait_for_message(conn):
                # The lock is held here, so tear down directly -- shutdown() takes the
                # same non-reentrant lock and would deadlock this thread against itself.
                self._teardown_worker()
                self._reject_if_shutdown_requested(cmd)
                raise self._error_cls(f"{self._log_tag} worker timed out after {self._call_timeout_sec}s during '{cmd}'")
            return conn.recv()
        except (EOFError, OSError, BrokenPipeError) as exc:
            detail = self._death_detail()
            self._teardown_worker()
            raise self._error_cls(f"{self._log_tag} died during '{cmd}' ({detail}): {exc or 'pipe closed'}") from exc

    def call_with_generation(self, cmd: str, **args: Any) -> tuple[Any, int]:
        """Send one command and return ``(result, generation)`` under a single lock hold.

        Load/cache callers must use this so the stamped generation cannot observe a
        stale value from a separate :meth:`generation` call that released the lock
        before the creating RPC completed.
        """
        with self._locked("call", cmd=cmd):
            self._reject_if_shutdown_requested(cmd)
            self._ensure_worker()
            self._send_request(cmd, args, stream=False)
            response = self._receive(cmd)
            if not response.get("ok"):
                raise self._reported_error_cls(response.get("error", f"Unknown {self._log_tag} worker error"))
            return response.get("result"), self._state["generation"]

    def call(self, cmd: str, **args: Any) -> Any:
        """Send a command and block for its result."""
        result, _generation = self.call_with_generation(cmd, **args)
        return result

    # --- streaming ---------------------------------------------------------------

    def stream(self, cmd: str, **args: Any) -> Iterator[dict[str, Any]]:
        """Run a streaming command, yielding each worker event until ``done``.

        The channel lock is held for the whole stream, so the caller **must** consume or
        close the generator; abandoning it without closing would strand the lock. Both
        normal exhaustion and ``GeneratorExit`` (an early ``break``) release it, and an
        early exit also cancels the in-flight worker request so the child does not keep
        decoding into a pipe nobody reads.

        Call :meth:`cancel_in_stream` from the consuming thread between yields to
        request cooperative preemption; the worker checks for it between chunks.
        """
        with self._locked("stream", cmd=cmd):
            self._reject_if_shutdown_requested(cmd)
            self._ensure_worker()
            self._send_request(cmd, args, stream=True)
            completed = False
            try:
                while True:
                    message = self._receive(cmd)
                    event = message.get("event")
                    if event in _TERMINAL_EVENTS:
                        # Both end the stream and both leave the pipe clean, so neither needs
                        # the abandon path. They are distinguished only so a cancelled stream
                        # is not mistaken for an exhausted one by anything reading the wire.
                        completed = True
                        return
                    if event == "error":
                        # Terminal too: the worker has finished this request and left the
                        # pipe clean. Raising with completed still False sent a healthy
                        # worker down the abandon path, where it drained fruitlessly for 30
                        # seconds and was then killed -- so one handler error cost a full
                        # model reload on top of the error itself.
                        completed = True
                        raise self._reported_error_cls(message.get("error", f"Unknown {self._log_tag} worker error"))
                    yield message
            finally:
                if not completed:
                    self._abandon_stream()

    def cancel_in_stream(self) -> None:
        """Ask the worker to stop the in-flight streaming command at its next check.

        Safe to call from the thread consuming :meth:`stream` (it already holds the
        lock, and the parent is the only writer in this direction). A dead worker is not
        an error here -- the stream will surface that on its next read.
        """
        conn = self._state["conn"]
        if conn is None:
            return
        try:
            conn.send({"control": "cancel"})
        except (OSError, BrokenPipeError):
            pass

    def _abandon_stream(self) -> None:
        """Drop a stream the caller stopped consuming, preserving the worker if possible.

        The worker may still be mid-command with unread events queued; a later RPC would
        otherwise read *those* instead of its own reply. Cancelling and draining to the
        terminal event restores a clean pipe while keeping the loaded model, which
        matters because cooperative preemption abandons streams routinely -- tearing
        down every time would force a full model reload on each preemption.

        Teardown remains the fallback: if the worker does not reach a terminal event
        promptly it is not trustworthy, and a respawn beats mismatched responses.
        """
        self.cancel_in_stream()
        if not self._drain_to_terminal_event():
            self._teardown_worker()

    def _drain_to_terminal_event(self) -> bool:
        """Read until this request's ``done``/``error``. False if the pipe can't be cleaned.

        Bounded by elapsed time rather than by a count of idle polls. Counting only the
        polls that timed out meant a worker still emitting progress events reset nothing
        and consumed no budget, so a chatty worker could hold the channel lock -- and with
        it every dashboard poll -- indefinitely. The clock runs whether the worker is
        talking or silent.
        """
        conn = self._state["conn"]
        if conn is None:
            return False
        deadline = time.monotonic() + _ABANDON_DRAIN_TIMEOUT_SEC
        while time.monotonic() < deadline:
            received, message = _poll_once(conn, deadline)
            outcome = _drain_outcome(received, message)
            if outcome is not None:
                return outcome
        return False

    # --- shutdown ----------------------------------------------------------------

    def _begin_shutdown(self) -> None:
        with self._shutdown_state_lock:
            self._shutdown_state["in_progress"] += 1
            self._shutdown_requested.set()

    def _end_shutdown(self) -> None:
        with self._shutdown_state_lock:
            self._shutdown_state["in_progress"] = max(0, self._shutdown_state["in_progress"] - 1)
            if self._shutdown_state["in_progress"] == 0:
                self._shutdown_requested.clear()

    def shutdown(self) -> None:
        """Terminate the worker, releasing every byte of device and host memory it held.

        The flag is set before acquiring the lock so an in-flight call blocked on a hung
        worker notices and releases the lock instead of pinning it forever. Concurrent
        shutdowns are counted so the first to finish cannot clear the flag out from
        under a sibling still in flight.
        """
        self._begin_shutdown()
        try:
            with self._locked("shutdown"):
                self._teardown_worker()
        finally:
            self._end_shutdown()
