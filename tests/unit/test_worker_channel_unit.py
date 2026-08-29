"""Unit tests for the shared isolated-worker channel.

The channel really spawns an OS subprocess, which is not appropriate for a fast unit
test; these mock the process/connection objects to exercise the channel's own logic
(worker lifecycle, error handling, shutdown sequencing) in isolation. The end-to-end
behaviour against a real subprocess is covered in
tests/inference/engines/test_worker_channel.py.

These were originally written against whisperx_worker_client, which owned a private copy
of this machinery. The client now delegates here, so the invariants are tested once.
"""

# The unit under test is the module's internals; reaching them by name is the point.

import threading
from collections.abc import Generator
from unittest import mock

import pytest

from modules.inference.engines import worker_channel
from modules.inference.engines.worker_channel import WorkerChannel, WorkerError


def _worker_main(conn):
    raise AssertionError("mocked channel must not start a real worker")


@pytest.fixture(name="channel")
def _channel() -> Generator[WorkerChannel, None, None]:
    yield WorkerChannel(_worker_main, name="test-worker", log_tag="TestWorker")


def _mock_process(alive: bool = True) -> mock.MagicMock:
    process = mock.MagicMock()
    process.is_alive.return_value = alive
    process.pid = 4242
    return process


# --- worker lifecycle -------------------------------------------------------------


def test_ensure_worker_spawns_when_none_running(channel):
    """A first-ever spawn must not bump the generation.

    The counter only advances when a worker is actually torn down, which cannot have
    happened yet if nothing was ever cached against a prior generation.
    """
    parent_conn, child_conn = mock.MagicMock(), mock.MagicMock()
    new_process = _mock_process(alive=True)
    with mock.patch.object(worker_channel, "_CTX") as mock_ctx:
        mock_ctx.Pipe.return_value = (parent_conn, child_conn)
        mock_ctx.Process.return_value = new_process
        channel._ensure_worker()

    mock_ctx.Process.assert_called_once_with(target=mock.ANY, args=(child_conn,), daemon=True, name="test-worker")
    new_process.start.assert_called_once()
    child_conn.close.assert_called_once()
    assert channel._state["process"] is new_process
    assert channel._state["conn"] is parent_conn
    assert channel.generation() == 0


def test_ensure_worker_reuses_alive_process(channel):
    """A live worker must be reused, leaving the generation untouched."""
    existing_process = _mock_process(alive=True)
    channel._state.update({"process": existing_process, "conn": mock.MagicMock(), "generation": 3})

    with mock.patch.object(worker_channel, "_CTX") as mock_ctx:
        channel._ensure_worker()

    mock_ctx.Process.assert_not_called()
    assert channel._state["process"] is existing_process
    assert channel.generation() == 3


def test_ensure_worker_respawns_when_dead(channel):
    """A died worker is reaped through teardown, bumping the generation."""
    channel._state.update({"process": _mock_process(alive=False), "conn": mock.MagicMock(), "generation": 1})

    parent_conn, child_conn = mock.MagicMock(), mock.MagicMock()
    new_process = _mock_process(alive=True)
    with mock.patch.object(worker_channel, "_CTX") as mock_ctx:
        mock_ctx.Pipe.return_value = (parent_conn, child_conn)
        mock_ctx.Process.return_value = new_process
        channel._ensure_worker()

    assert channel._state["process"] is new_process
    assert channel.generation() == 2


def test_generation_detects_dead_process_reaps_and_bumps(channel):
    """generation() must notice a worker that died with no intervening call.

    Pre-flight cache-validity checks rely on this: without the reap they would observe a
    stale generation and treat an already-invalid handle as good.
    """
    dead_process = _mock_process(alive=False)
    conn = mock.MagicMock()
    channel._state.update({"process": dead_process, "conn": conn, "generation": 5})

    assert channel.generation() == 6
    dead_process.terminate.assert_not_called()
    conn.close.assert_called_once()
    assert channel._state["process"] is None
    assert channel._state["conn"] is None


# --- calls ------------------------------------------------------------------------


def test_call_returns_result_on_ok_response(channel):
    """A successful response hands the caller its result."""
    conn = mock.MagicMock()
    conn.poll.return_value = True
    conn.recv.return_value = {"id": 0, "ok": True, "result": {"text": "hi"}}
    channel._state.update({"process": _mock_process(alive=True), "conn": conn})

    assert channel.call("transcribe", audio_path="clip.wav") == {"text": "hi"}
    sent = conn.send.call_args.args[0]
    assert sent["cmd"] == "transcribe"
    assert sent["args"]["audio_path"] == "clip.wav"


def test_call_with_generation_stamps_under_the_same_lock(channel):
    """The generation is stamped inside the same lock hold that produced the result."""
    conn = mock.MagicMock()
    conn.poll.return_value = True
    conn.recv.return_value = {"id": 0, "ok": True, "result": "handle-1"}
    channel._state.update({"process": _mock_process(alive=True), "conn": conn, "generation": 7})

    assert channel.call_with_generation("load_model", model_id="tiny") == ("handle-1", 7)


def test_call_raises_on_ok_false(channel):
    """A failure the worker reports becomes an error the caller can catch."""
    conn = mock.MagicMock()
    conn.poll.return_value = True
    conn.recv.return_value = {"id": 0, "ok": False, "error": "boom"}
    channel._state.update({"process": _mock_process(alive=True), "conn": conn})

    with pytest.raises(WorkerError, match="boom"):
        channel.call("ping")


def test_call_raises_and_tears_down_on_broken_pipe(channel):
    """The error path tears down directly while holding the lock.

    shutdown() takes the same non-reentrant lock and would deadlock the thread against
    itself, so this exercises the real teardown and checks state is actually cleared.
    """
    conn = mock.MagicMock()
    conn.send.side_effect = BrokenPipeError("pipe gone")
    process = _mock_process(alive=True)
    channel._state.update({"process": process, "conn": conn})

    with pytest.raises(WorkerError, match="died during 'transcribe'"):
        channel.call("transcribe")

    process.terminate.assert_called_once()
    conn.close.assert_called_once()
    assert channel._state["process"] is None
    assert channel._state["conn"] is None


@pytest.mark.parametrize("exc_type", [EOFError, OSError])
def test_call_treats_recv_failure_as_a_dead_worker(channel, exc_type: type[Exception]):
    """A broken pipe is a dead worker, and is reported as one."""
    conn = mock.MagicMock()
    conn.recv.side_effect = exc_type("gone")
    channel._state.update({"process": _mock_process(alive=True), "conn": conn})

    with pytest.raises(WorkerError):
        channel.call("ping")

    assert channel._state["process"] is None
    assert channel._state["conn"] is None


def test_call_timeout_tears_down_and_next_call_respawns(channel):
    """A worker that never answers is treated as dead, and the next call gets a new one."""
    conn = mock.MagicMock()
    conn.poll.return_value = False
    process = _mock_process(alive=True)
    channel._state.update({"process": process, "conn": conn, "generation": 1})
    channel._call_timeout_sec = 0.01

    with pytest.raises(WorkerError, match="timed out"):
        channel.call("transcribe")

    process.terminate.assert_called_once()
    assert channel._state["process"] is None

    new_conn = mock.MagicMock()
    new_conn.poll.return_value = True
    new_conn.recv.return_value = {"id": 0, "ok": True, "result": "handle-2"}
    with mock.patch.object(worker_channel, "_CTX") as mock_ctx:
        mock_ctx.Pipe.return_value = (new_conn, mock.MagicMock())
        mock_ctx.Process.return_value = _mock_process(alive=True)
        assert channel.call("transcribe") == "handle-2"

    mock_ctx.Process.assert_called_once()
    assert channel.generation() == 2


# --- teardown primitives ----------------------------------------------------------


def test_close_connection_handles_none(channel):
    """There is nothing to close before a worker has ever started."""
    channel._close_connection(None)


def test_close_connection_closes_normally(channel):
    """A live connection is closed."""
    conn = mock.MagicMock()
    channel._close_connection(conn)
    conn.close.assert_called_once()


def test_close_connection_swallows_oserror(channel):
    """Teardown cannot fail on a connection that is already gone."""
    conn = mock.MagicMock()
    conn.close.side_effect = OSError("already closed")
    channel._close_connection(conn)


def test_terminate_process_handles_none(channel):
    """There is nothing to terminate before a worker has ever started."""
    channel._terminate_process(None)


def test_terminate_process_skips_when_not_alive(channel):
    """A process that has already exited is not signalled again."""
    process = _mock_process(alive=False)
    channel._terminate_process(process)
    process.terminate.assert_not_called()


def test_terminate_process_terminates_gracefully(channel):
    """A live worker is asked to exit before anything harsher."""
    process = _mock_process(alive=True)
    process.is_alive.side_effect = [True, False]
    channel._terminate_process(process)
    process.terminate.assert_called_once()
    process.kill.assert_not_called()


def test_terminate_process_force_kills_when_still_alive(channel):
    """A worker that ignores the request is killed rather than waited on forever."""
    process = _mock_process(alive=True)
    process.is_alive.side_effect = [True, True]
    channel._terminate_process(process)
    process.terminate.assert_called_once()
    process.kill.assert_called_once()
    assert process.join.call_count == 2


# --- shutdown ---------------------------------------------------------------------


def test_shutdown_closes_terminates_and_resets(channel):
    """Shutdown releases the pipe, the process, and the cached handles together."""
    conn = mock.MagicMock()
    channel._state.update({"process": _mock_process(alive=False), "conn": conn})

    channel.shutdown()

    conn.close.assert_called_once()
    assert channel._state["process"] is None
    assert channel._state["conn"] is None


def test_shutdown_invalidates_the_generation(channel):
    """Idle cleanup uses shutdown(), and the respawned worker's pool is empty."""
    channel._state.update({"process": _mock_process(alive=True), "conn": mock.MagicMock(), "generation": 4})

    channel.shutdown()

    assert channel.generation() == 5


def test_shutdown_is_a_noop_on_empty_state(channel):
    """Shutting down before anything started is harmless."""
    channel.shutdown()
    assert channel._state["process"] is None


def test_the_poll_is_chunked_so_a_long_deadline_still_notices_shutdown(channel):
    """A configured deadline must not become the shutdown latency.

    ``isolated_engine`` sets a 30-minute call timeout. Handing that to ``poll`` as one
    timeout reintroduced the bug the deadline was added to fix, one level up: shutdown went
    unnoticed for the whole span, so a stuck worker still pinned the interpreter at exit.
    The wait is chunked at the shutdown poll interval regardless of the deadline.
    """
    conn = mock.MagicMock()
    channel._state.update({"process": _mock_process(alive=True), "conn": conn, "generation": 1})
    channel._call_timeout_sec = 1800.0
    timeouts: list[float] = []

    def _hung_poll(timeout):
        timeouts.append(timeout)
        return False

    conn.poll.side_effect = _hung_poll
    channel._shutdown_requested.set()

    assert channel._wait_for_message(conn) is False, "the shutdown flag must end the wait"
    assert timeouts == [worker_channel._SHUTDOWN_POLL_INTERVAL_SEC], "the whole deadline was handed to poll()"


def test_shutdown_interrupts_a_hung_unbounded_call(channel):
    """With no RPC deadline, a call blocked on a hung worker must not pin the lock.

    shutdown() needs the same lock, so the in-flight call has to notice the shutdown
    flag and give up rather than the two deadlocking against each other.
    """
    conn = mock.MagicMock()
    channel._state.update({"process": _mock_process(alive=True), "conn": conn, "generation": 3})
    channel._call_timeout_sec = 0.0

    poll_started = threading.Event()

    def _hung_poll(_timeout):
        poll_started.set()
        return False

    conn.poll.side_effect = _hung_poll
    call_errors: list[Exception] = []

    def _blocked_call():
        try:
            channel.call("transcribe")
        except WorkerError as exc:
            call_errors.append(exc)

    with mock.patch.object(worker_channel, "_SHUTDOWN_POLL_INTERVAL_SEC", 0.05):
        call_thread = threading.Thread(target=_blocked_call, daemon=True)
        call_thread.start()
        assert poll_started.wait(timeout=2.0)

        shutdown_thread = threading.Thread(target=channel.shutdown, daemon=True)
        shutdown_thread.start()
        shutdown_thread.join(timeout=5.0)
        assert not shutdown_thread.is_alive(), "shutdown() deadlocked behind the hung call's lock"

        call_thread.join(timeout=5.0)

    assert len(call_errors) == 1, "shutdown did not interrupt the hung unbounded call"
    assert "shut down" in str(call_errors[0])
    assert (channel._state["process"], channel._state["conn"]) == (None, None)


def test_lock_contention_warns_once_beyond_the_threshold(channel):
    """A caller blocked on the lock past the warn threshold logs once, then waits."""
    channel._lock_warn_sec = 0.01
    held = threading.Lock()
    holder_ready = threading.Event()
    release_event = threading.Event()
    warning_seen = threading.Event()
    channel._state["generation"] = 9

    # 1.0s, not 5.0: long enough that ordinary scheduling never trips it, short enough that a
    # regression which stops emitting the warning fails in about a second instead of five.
    def _hold_lock():
        with held:
            holder_ready.set()
            release_event.wait(timeout=1.0)

    holder = threading.Thread(target=_hold_lock, daemon=True)
    holder.start()
    assert holder_ready.wait(timeout=1.0)
    channel._lock = held

    def _release_after_warning():
        # finally, and no assertion. An assert in a worker thread is swallowed by the
        # runtime, so when the warning never arrived this thread died without ever setting
        # release_event -- the holder then sat on the lock for its full 5s timeout and the
        # real failure surfaced as a slow, confusing hang. A missing warning is reported by
        # mock_warning.assert_called_once() on the main thread, where it belongs.
        try:
            warning_seen.wait(timeout=1.0)
        finally:
            release_event.set()

    releaser = threading.Thread(target=_release_after_warning, daemon=True)
    releaser.start()
    try:
        with mock.patch.object(worker_channel.logger, "warning") as mock_warning:
            mock_warning.side_effect = lambda *_a, **_k: warning_seen.set()
            assert channel.generation() == 9
        mock_warning.assert_called_once()
        assert "generation" in str(mock_warning.call_args)
    finally:
        release_event.set()
        releaser.join(timeout=1.0)
        holder.join(timeout=1.0)


class TestAbandoningAStreamTheCallerStoppedConsuming:
    """Cooperative preemption abandons streams routinely, so this is a hot path.

    The worker may still be mid-command with unread events queued. Draining to the terminal
    event restores a clean pipe while keeping the loaded model; tearing the worker down
    instead would force a full model reload on every preemption.
    """

    def _channel(self, conn):
        channel = worker_channel.WorkerChannel.__new__(worker_channel.WorkerChannel)
        channel._state = {"conn": conn, "process": None}
        return channel

    def test_a_stream_that_reaches_a_terminal_event_keeps_the_worker(self):
        """Preemption abandons streams routinely; a reload each time would be ruinous."""
        conn = mock.MagicMock()
        conn.poll.return_value = True
        conn.recv.side_effect = [{"event": "progress"}, {"event": "done"}]
        channel = self._channel(conn)
        with mock.patch.object(channel, "_teardown_worker") as teardown:
            channel._abandon_stream()
        teardown.assert_not_called()

    def test_a_cancel_acknowledgement_also_ends_the_drain(self):
        """ "cancelled" terminates the stream exactly as "done" does."""
        conn = mock.MagicMock()
        conn.poll.return_value = True
        conn.recv.return_value = {"event": "cancelled"}
        channel = self._channel(conn)
        with mock.patch.object(channel, "_teardown_worker") as teardown:
            channel._abandon_stream()
        teardown.assert_not_called()

    def test_a_worker_that_never_reaches_a_terminal_event_is_torn_down(self):
        """A worker that will not settle is not trustworthy; a respawn beats mismatched replies."""
        conn = mock.MagicMock()
        conn.poll.return_value = False
        channel = self._channel(conn)
        # Zero budget rather than a fake clock: _poll_once reads the clock too, so counting
        # monotonic() calls couples the test to the drain loop's internals.
        with (
            mock.patch.object(worker_channel, "_ABANDON_DRAIN_TIMEOUT_SEC", 0.0),
            mock.patch.object(channel, "_teardown_worker") as teardown,
        ):
            channel._abandon_stream()
        teardown.assert_called_once()

    def test_a_dead_pipe_tears_the_worker_down(self):
        """A pipe that cannot be drained cannot be trusted for the next RPC."""
        conn = mock.MagicMock()
        conn.poll.side_effect = EOFError("pipe gone")
        channel = self._channel(conn)
        with mock.patch.object(channel, "_teardown_worker") as teardown:
            channel._abandon_stream()
        teardown.assert_called_once()

    def test_cancelling_on_a_dead_pipe_is_not_an_error(self):
        """The stream surfaces a dead worker on its next read; cancel must not raise here."""
        conn = mock.MagicMock()
        conn.send.side_effect = BrokenPipeError("gone")
        self._channel(conn).cancel_in_stream()

    def test_cancelling_without_a_connection_is_a_no_op(self):
        """Shutdown races call this after the connection is gone."""
        self._channel(None).cancel_in_stream()
