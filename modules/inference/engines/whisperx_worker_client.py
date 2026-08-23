"""Parent-side client for the isolated WhisperX subprocess.

See ``whisperx_worker`` for why this runs out-of-process. This module owns
a single lazily-started, long-lived worker and serializes calls to it with
a lock (Flask may call in from multiple request threads). The worker is
disposable: :func:`shutdown` terminates it outright, which is also how we
guarantee its VRAM/RAM is fully reclaimed during the app's periodic
"purge everything" idle cleanup — a fresh worker is spawned on next use.

The worker is started with ``multiprocessing`` ``spawn`` against
``worker_main``. Production entry (``whisper_pro_asr.py``) and package
``__init__`` modules are spawn-safe so the child does not import the main
torch stack before ``_activate_isolated_lib_path()`` runs.
"""

import logging
import multiprocessing
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from itertools import count
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import Any

from modules.inference.engines import whisperx_worker

logger = logging.getLogger(__name__)

_CTX = multiprocessing.get_context("spawn")
_LOCK = threading.Lock()
# "generation" is bumped every time a *new* worker process is actually spawned
# (not on reuse of a still-alive one). Callers that cache handles into the
# worker's `objects` pool (diarization.py's ALIGN_POOL/DIARIZE_POOL,
# WhisperXEngine's model_handle) stamp the generation returned by
# call_with_generation() under the same lock as the creating RPC, and
# compare against generation() before trusting a cached handle -- a handle
# from a prior generation refers to an `objects` dict that no longer exists
# (the new worker process starts with an empty one), so it must be treated
# as a miss rather than sent to the worker and fail there.
_STATE: dict[str, Any] = {"process": None, "conn": None, "generation": 0}
_REQUEST_IDS = count()
#: Default 0 disables the RPC deadline so long CPU WhisperX jobs are not
#: killed mid-flight. Set WHISPERX_WORKER_CALL_TIMEOUT_SEC > 0 to enable a
#: hung-worker ceiling (seconds) for tests/ops.
_CALL_TIMEOUT_SEC = float(os.environ.get("WHISPERX_WORKER_CALL_TIMEOUT_SEC", "0"))
#: Operational warn threshold when a caller blocks waiting for serialized worker
#: access. Does not change lock semantics — only surfaces contention. Override
#: via WHISPERX_WORKER_LOCK_WARN_SEC for tests/tuning.
_LOCK_WARN_SEC = float(os.environ.get("WHISPERX_WORKER_LOCK_WARN_SEC", "5"))
#: Set while shutdown() is in progress. A threading.Lock has no way to interrupt
#: a thread that already holds it, so with WHISPERX_WORKER_CALL_TIMEOUT_SEC<=0
#: (deliberately unbounded, for legitimate long CPU jobs) an in-flight
#: call_with_generation() waiting on a hung worker would otherwise hold _LOCK
#: forever, and shutdown()/idle-cleanup -- which also needs _LOCK to terminate
#: that same worker -- could never make progress. The in-flight call's bounded
#: poll loop (see _wait_for_response) checks this flag each chunk and gives up
#: on its own instead, releasing _LOCK so shutdown() can proceed.
_SHUTDOWN_REQUESTED = threading.Event()
#: Guards _SHUTDOWN_IN_PROGRESS so concurrent shutdown() callers don't clear the
#: flag out from under each other -- with a plain Event, the first shutdown() to
#: finish would clear _SHUTDOWN_REQUESTED in its `finally` while a second, still
#: in-flight shutdown() (or an in-flight call_with_generation() relying on the
#: flag to notice a *different* shutdown() is underway) is still relying on it
#: being set. Tracked as a count of active shutdown() calls instead: the flag
#: stays set as long as at least one is in progress, and is only cleared once
#: the last one finishes.
_SHUTDOWN_STATE_LOCK = threading.Lock()
_SHUTDOWN_STATE: dict[str, int] = {"in_progress": 0}
#: Poll chunk used while WHISPERX_WORKER_CALL_TIMEOUT_SEC<=0, so a wait with no
#: fixed deadline still periodically re-checks _SHUTDOWN_REQUESTED.
_SHUTDOWN_POLL_INTERVAL_SEC = 2.0


@contextmanager
def _locked(operation: str, **context: Any) -> Iterator[None]:
    """Hold ``_LOCK`` for the duration of the ``with`` block, with contention warnings."""
    if not _LOCK.acquire(timeout=_LOCK_WARN_SEC):
        ctx = " ".join(f"{key}={value}" for key, value in sorted(context.items()) if value is not None)
        suffix = f" ({ctx})" if ctx else ""
        logger.warning(
            "[WhisperXWorker] Blocked on lock for >%.1fs during %s%s; waiting for serialized access",
            _LOCK_WARN_SEC,
            operation,
            suffix,
        )
        _LOCK.acquire()
    try:
        yield
    finally:
        _LOCK.release()


def generation() -> int:
    """Return the current worker generation. See _STATE["generation"] for why this matters.

    Also detects a stored worker process that has died since the last call() (e.g.
    externally killed/OOM-killed, with no call() attempted since) and reaps it here
    via _teardown_worker() (which bumps the generation itself). Callers that use
    generation() as a pre-flight cache-validity check (diarization.py's
    _cached_handle_for_current_generation, WhisperXEngine._ensure_current_model_handle)
    call it *before* ever sending a handle to the worker -- without this, they'd
    observe a stale (pre-death) generation number and treat an already-invalid
    cached handle as still valid, sending it to a not-yet-respawned worker only to
    have that call fail.
    """
    with _locked("generation"):
        process = _STATE["process"]
        if process is not None and not process.is_alive():
            _teardown_worker()
        return _STATE["generation"]


def _ensure_worker() -> None:
    process = _STATE["process"]
    if process is not None and process.is_alive():
        return
    if process is not None:
        # Stored process exists but has died -- close its stale connection, reap it,
        # and invalidate the generation (via _teardown_worker) before replacing
        # _STATE, instead of just overwriting process/conn and leaking the dead
        # worker's pipe handle.
        _teardown_worker()

    parent_conn, child_conn = _CTX.Pipe()
    process = _CTX.Process(
        target=whisperx_worker.worker_main,
        args=(child_conn,),
        daemon=True,
        name="whisperx-worker",
    )
    process.start()
    child_conn.close()
    _STATE["process"] = process
    _STATE["conn"] = parent_conn
    logger.info("[WhisperXWorker] Started isolated worker process (pid=%s, generation=%s)", process.pid, _STATE["generation"])


class WhisperXWorkerError(RuntimeError):
    """Raised when the isolated WhisperX worker reports a failure."""


def _poll_timeout() -> float:
    """Per-iteration poll chunk: the full fixed deadline when
    WHISPERX_WORKER_CALL_TIMEOUT_SEC is enabled (a single poll suffices), else
    the short _SHUTDOWN_POLL_INTERVAL_SEC chunk used by _wait_for_response's
    loop for the deliberately-unbounded default."""
    if _CALL_TIMEOUT_SEC > 0:
        return _CALL_TIMEOUT_SEC
    return _SHUTDOWN_POLL_INTERVAL_SEC


def _wait_for_response(conn: Connection) -> bool:
    """Wait for a worker response. Returns False if the fixed
    WHISPERX_WORKER_CALL_TIMEOUT_SEC deadline elapses, or -- when no such
    deadline is configured -- if shutdown() is requested concurrently while
    still waiting, so a hung worker can't block a caller (and _LOCK) forever."""
    deadline_enabled = _CALL_TIMEOUT_SEC > 0
    while True:
        if conn.poll(_poll_timeout()):
            return True
        if deadline_enabled:
            return False
        if _SHUTDOWN_REQUESTED.is_set():
            return False


def _reject_if_shutdown_requested(cmd: str) -> None:
    """Reject a call queued behind shutdown()'s lock acquisition instead of spinning
    up a fresh worker only to have it torn down mid-request once shutdown() gets the
    lock -- same treatment as a call against an already-dead worker."""
    if _SHUTDOWN_REQUESTED.is_set():
        raise WhisperXWorkerError(f"WhisperX worker shut down during '{cmd}'")


def _send_and_await_response(cmd: str, args: dict[str, Any]) -> dict[str, Any]:
    """Send one RPC and block for its response. Must run with ``_LOCK`` held."""
    conn = _STATE["conn"]
    request_id = next(_REQUEST_IDS)
    try:
        conn.send({"id": request_id, "cmd": cmd, "args": args})
        if not _wait_for_response(conn):
            # _LOCK is already held here, so call the lock-free teardown directly --
            # shutdown() acquires the same (non-reentrant) _LOCK and would deadlock
            # this thread against itself. A hung/deadlocked worker is treated the
            # same as a dead one: tear it down so the next call spawns a fresh
            # process instead of hanging forever.
            _teardown_worker()
            _reject_if_shutdown_requested(cmd)
            raise WhisperXWorkerError(f"WhisperX worker timed out after {_CALL_TIMEOUT_SEC}s during '{cmd}'")
        return conn.recv()
    except (EOFError, OSError, BrokenPipeError) as exc:
        # _LOCK is already held here, so call the lock-free teardown directly --
        # shutdown() acquires the same (non-reentrant) _LOCK and would deadlock this
        # thread against itself.
        _teardown_worker()
        raise WhisperXWorkerError(f"WhisperX worker died during '{cmd}': {exc}") from exc


def call_with_generation(cmd: str, **args: Any) -> tuple[Any, int]:
    """Send a command and return ``(result, generation)`` under one ``_LOCK`` hold.

    Load/cache callers must use this so the stamped generation cannot observe a
    stale value from a separate ``generation()`` call that released the lock
    before the creating RPC completed.
    """
    with _locked("call", cmd=cmd):
        _reject_if_shutdown_requested(cmd)
        _ensure_worker()
        response = _send_and_await_response(cmd, args)
        if not response.get("ok"):
            raise WhisperXWorkerError(response.get("error", "Unknown WhisperX worker error"))
        return response.get("result"), _STATE["generation"]


def call(cmd: str, **args: Any) -> Any:
    """Send a command to the isolated WhisperX worker and block for the result."""
    result, _generation = call_with_generation(cmd, **args)
    return result


def _close_connection(conn: Connection | None) -> None:
    if conn is None:
        return
    try:
        conn.close()
    except OSError:
        pass


def _terminate_process(process: BaseProcess | None) -> None:
    if process is None or not process.is_alive():
        return
    process.terminate()
    process.join(timeout=10)
    if process.is_alive():
        process.kill()
        process.join(timeout=5)


def _teardown_worker() -> None:
    """Lock-free worker teardown. Callers must hold _LOCK already (call()'s
    exception path) or not need it (shutdown(), which acquires it itself).

    Bumps _STATE["generation"] here -- the single place every path that can
    destroy the worker funnels through (shutdown(), call()'s timeout/error
    paths, _ensure_worker()'s dead-process reap, generation()'s own dead-process
    detection) -- so any handle cached against the prior generation is reliably
    invalidated regardless of *why* the worker went away."""
    _close_connection(_STATE["conn"])
    _terminate_process(_STATE["process"])
    _STATE["process"] = None
    _STATE["conn"] = None
    _STATE["generation"] += 1


def _begin_shutdown() -> None:
    with _SHUTDOWN_STATE_LOCK:
        _SHUTDOWN_STATE["in_progress"] += 1
        _SHUTDOWN_REQUESTED.set()


def _end_shutdown() -> None:
    with _SHUTDOWN_STATE_LOCK:
        _SHUTDOWN_STATE["in_progress"] = max(0, _SHUTDOWN_STATE["in_progress"] - 1)
        if _SHUTDOWN_STATE["in_progress"] == 0:
            _SHUTDOWN_REQUESTED.clear()


def shutdown() -> None:
    """Terminate the worker process, releasing all VRAM/RAM it holds.

    Sets _SHUTDOWN_REQUESTED before acquiring _LOCK: with the default
    unbounded RPC deadline, an in-flight call_with_generation() blocked on a
    hung worker would otherwise hold _LOCK forever, and this call -- needing
    the same lock -- could never get in to terminate it. The flag lets that
    in-flight call notice within _SHUTDOWN_POLL_INTERVAL_SEC and tear the
    worker down itself, releasing _LOCK so this acquire can proceed.

    Tracks concurrent shutdown() callers via _SHUTDOWN_IN_PROGRESS so the flag
    stays set as long as any of them is still in flight, and is only cleared
    once the last one finishes -- otherwise the first caller to finish would
    clear it out from under a still-running sibling call.
    """
    _begin_shutdown()
    try:
        with _locked("shutdown"):
            _teardown_worker()
    finally:
        _end_shutdown()
