"""Unit tests for the shared isolated-worker channel.

The channel really spawns an OS subprocess, which is not appropriate for a fast unit
test; these mock the process/connection objects to exercise the channel's own logic
(worker lifecycle, error handling, shutdown sequencing) in isolation. The end-to-end
behaviour against a real subprocess is covered in
tests/inference/engines/test_worker_channel.py.

These were originally written against whisperx_worker_client, which owned a private copy
of this machinery. The client now delegates here, so the invariants are tested once.
"""

# pylint: disable=protected-access
# The unit under test is the module's internals. Reaching them by name is the point
# of these tests, not an accident: the public surface is a thin wrapper and testing
# only through it would leave the rules below unpinned.

import threading
from collections.abc import Generator
from unittest import mock

import pytest

from modules.inference.engines import worker_channel
from modules.inference.engines.worker_channel import WorkerChannel, WorkerError


def _worker_main(conn):  # pragma: no cover - never executed; spawn is mocked out
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

    def _hold_lock():
        with held:
            holder_ready.set()
            release_event.wait(timeout=5.0)

    holder = threading.Thread(target=_hold_lock, daemon=True)
    holder.start()
    assert holder_ready.wait(timeout=1.0)
    channel._lock = held

    def _release_after_warning():
        assert warning_seen.wait(timeout=5.0)
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
