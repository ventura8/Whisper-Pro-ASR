"""Unit tests for modules/inference/engines/whisperx_worker_client.py.

The real client spawns an actual OS subprocess (multiprocessing.get_context("spawn")),
which isn't appropriate for a fast unit test - these mock the process/connection
objects to exercise the client's own logic (worker lifecycle, error handling,
shutdown sequencing) in isolation.
"""

import threading
from collections.abc import Generator
from unittest import mock

import pytest

from modules.inference.engines import whisperx_worker_client as client
from modules.inference.engines.whisperx_worker_client import (
    _SHUTDOWN_REQUESTED,
    _SHUTDOWN_STATE,
    _STATE,
    _close_connection,
    _ensure_worker,
    _terminate_process,
)


@pytest.fixture(autouse=True)
def reset_client_state() -> Generator[None, None, None]:
    """Ensure each test starts with a clean, unstarted worker state.

    Also clears _SHUTDOWN_REQUESTED and _SHUTDOWN_STATE["in_progress"]: shutdown()
    sets the Event and increments the in-progress counter, and interrupted
    shutdown threads can otherwise leak either into a later test (making call()
    raise WhisperXWorkerError or block on a stale shutdown)."""
    _STATE["process"] = None
    _STATE["conn"] = None
    _STATE["generation"] = 0
    _SHUTDOWN_REQUESTED.clear()
    _SHUTDOWN_STATE["in_progress"] = 0
    yield
    _STATE["process"] = None
    _STATE["conn"] = None
    _STATE["generation"] = 0
    _SHUTDOWN_REQUESTED.clear()
    _SHUTDOWN_STATE["in_progress"] = 0


def _mock_process(alive: bool = True) -> mock.MagicMock:
    process = mock.MagicMock()
    process.is_alive.return_value = alive
    process.pid = 4242
    return process


def test_ensure_worker_spawns_when_none_running():
    """_ensure_worker should start a fresh process/pipe when none exists yet.

    A first-ever spawn (no prior process to tear down) does not bump the
    generation: the invalidation counter only advances when a worker is actually
    torn down (see _teardown_worker), which can't have happened yet if nothing
    was cached against a prior generation to begin with."""
    parent_conn, child_conn = mock.MagicMock(), mock.MagicMock()
    new_process = _mock_process(alive=True)
    with mock.patch.object(client, "_CTX") as mock_ctx:
        mock_ctx.Pipe.return_value = (parent_conn, child_conn)
        mock_ctx.Process.return_value = new_process
        _ensure_worker()

    mock_ctx.Process.assert_called_once_with(
        target=mock.ANY,
        args=(child_conn,),
        daemon=True,
        name="whisperx-worker",
    )
    new_process.start.assert_called_once()
    child_conn.close.assert_called_once()
    assert _STATE["process"] is new_process
    assert _STATE["conn"] is parent_conn
    assert client.generation() == 0


def test_ensure_worker_reuses_alive_process():
    """_ensure_worker should not spawn a new process if the current one is alive,
    and must not bump the generation counter (no new process, no cache invalidation)."""
    existing_process = _mock_process(alive=True)
    _STATE["process"] = existing_process
    _STATE["conn"] = mock.MagicMock()
    _STATE["generation"] = 3

    with mock.patch.object(client, "_CTX") as mock_ctx:
        _ensure_worker()

    mock_ctx.Process.assert_not_called()
    assert _STATE["process"] is existing_process
    assert client.generation() == 3


def test_ensure_worker_respawns_when_dead():
    """_ensure_worker should start a new process if the stored one has died,
    and bump the generation counter (via _teardown_worker reaping the dead
    process) so stale cached handles are invalidated."""
    dead_process = _mock_process(alive=False)
    _STATE["process"] = dead_process
    _STATE["conn"] = mock.MagicMock()
    _STATE["generation"] = 1

    parent_conn, child_conn = mock.MagicMock(), mock.MagicMock()
    new_process = _mock_process(alive=True)
    with mock.patch.object(client, "_CTX") as mock_ctx:
        mock_ctx.Pipe.return_value = (parent_conn, child_conn)
        mock_ctx.Process.return_value = new_process
        _ensure_worker()

    assert _STATE["process"] is new_process
    assert client.generation() == 2


def test_generation_detects_dead_process_reaps_and_bumps():
    """generation() must detect a stored worker that died without an intervening
    call() (e.g. externally OOM-killed), reap it via _teardown_worker(), and bump
    the generation immediately -- so a pre-flight cache-validity check elsewhere
    (diarization.py, whisperx_engine.py) observes the death right away instead of
    treating a stale handle as still valid."""
    dead_process = _mock_process(alive=False)
    conn = mock.MagicMock()
    _STATE["process"] = dead_process
    _STATE["conn"] = conn
    _STATE["generation"] = 5

    assert client.generation() == 6
    dead_process.terminate.assert_not_called()  # already not alive, nothing to terminate
    conn.close.assert_called_once()
    assert _STATE["process"] is None
    assert _STATE["conn"] is None


def test_call_returns_result_on_ok_response():
    """call() should return the worker's result payload when ok=True."""
    conn = mock.MagicMock()
    conn.poll.return_value = True
    conn.recv.return_value = {"id": 0, "ok": True, "result": {"text": "hi"}}
    _STATE["process"] = _mock_process(alive=True)
    _STATE["conn"] = conn

    assert client.call("transcribe", audio_path="clip.wav") == {"text": "hi"}
    conn.send.assert_called_once()
    sent = conn.send.call_args.args[0]
    assert sent["cmd"] == "transcribe"
    assert sent["args"]["audio_path"] == "clip.wav"


def test_call_with_generation_returns_result_and_generation():
    """call_with_generation must stamp the generation observed under the same lock."""
    conn = mock.MagicMock()
    conn.poll.return_value = True
    conn.recv.return_value = {"id": 0, "ok": True, "result": "handle-1"}
    _STATE["process"] = _mock_process(alive=True)
    _STATE["conn"] = conn
    _STATE["generation"] = 7

    result, generation = client.call_with_generation("load_model", model_id="tiny")
    assert result == "handle-1"
    assert generation == 7


def test_call_raises_worker_error_on_ok_false():
    """call() should raise WhisperXWorkerError when the worker reports failure."""
    conn = mock.MagicMock()
    conn.poll.return_value = True
    conn.recv.return_value = {"id": 0, "ok": False, "error": "boom"}
    _STATE["process"] = _mock_process(alive=True)
    _STATE["conn"] = conn

    with pytest.raises(client.WhisperXWorkerError, match="boom"):
        client.call("ping")


def test_call_raises_worker_error_on_broken_pipe():
    """call() should shut down the worker and raise WhisperXWorkerError if the pipe dies.

    Does not mock shutdown()/_teardown_worker(): call()'s exception path calls
    _teardown_worker() directly while still holding _LOCK (shutdown() itself
    acquires that same non-reentrant lock and would deadlock here), so this
    exercises the real teardown and verifies _STATE actually gets cleared."""
    conn = mock.MagicMock()
    conn.send.side_effect = BrokenPipeError("pipe gone")
    process = _mock_process(alive=True)
    _STATE["process"] = process
    _STATE["conn"] = conn

    with pytest.raises(client.WhisperXWorkerError, match="died during 'transcribe'"):
        client.call("transcribe")

    process.terminate.assert_called_once()
    conn.close.assert_called_once()
    assert _STATE["process"] is None
    assert _STATE["conn"] is None


@pytest.mark.parametrize("exc_type", [EOFError, OSError])
def test_call_raises_worker_error_on_recv_failure(exc_type: type[Exception]):
    """call() should also treat EOFError/OSError on recv as a dead worker, and
    really tear the worker down (not just raise) so the next call respawns."""
    conn = mock.MagicMock()
    conn.recv.side_effect = exc_type("gone")
    process = _mock_process(alive=True)
    _STATE["process"] = process
    _STATE["conn"] = conn

    with pytest.raises(client.WhisperXWorkerError):
        client.call("ping")

    assert _STATE["process"] is None
    assert _STATE["conn"] is None


def test_call_times_out_tears_down_and_respawns_a_new_generation_on_next_call():
    """A worker that never responds before the configured timeout must be treated
    as dead: call() raises a timeout-specific WhisperXWorkerError and clears
    _STATE, and the *next* call() must spawn a genuinely new worker process
    (bumping generation) rather than reusing the timed-out one."""
    conn = mock.MagicMock()
    conn.poll.return_value = False
    process = _mock_process(alive=True)
    _STATE["process"] = process
    _STATE["conn"] = conn
    _STATE["generation"] = 1

    with mock.patch.object(client, "_CALL_TIMEOUT_SEC", 0.01):
        with pytest.raises(client.WhisperXWorkerError, match="timed out"):
            client.call("transcribe")

    process.terminate.assert_called_once()
    conn.close.assert_called_once()
    assert _STATE["process"] is None
    assert _STATE["conn"] is None

    new_conn = mock.MagicMock()
    new_conn.poll.return_value = True
    new_conn.recv.return_value = {"id": 0, "ok": True, "result": "handle-2"}
    new_process = _mock_process(alive=True)
    with mock.patch.object(client, "_CTX") as mock_ctx:
        mock_ctx.Pipe.return_value = (new_conn, mock.MagicMock())
        mock_ctx.Process.return_value = new_process
        result = client.call("transcribe")

    assert result == "handle-2"
    mock_ctx.Process.assert_called_once()
    assert client.generation() == 2


def test_close_connection_handles_none():
    """_close_connection(None) must be a no-op."""
    _close_connection(None)


def test_close_connection_closes_normally():
    """_close_connection should close a real connection object."""
    conn = mock.MagicMock()
    _close_connection(conn)
    conn.close.assert_called_once()


def test_close_connection_swallows_oserror():
    """_close_connection must not raise if close() itself fails."""
    conn = mock.MagicMock()
    conn.close.side_effect = OSError("already closed")
    _close_connection(conn)


def test_terminate_process_handles_none():
    """_terminate_process(None) must be a no-op."""
    _terminate_process(None)


def test_terminate_process_skips_when_not_alive():
    """A not-alive process must not be terminated/killed."""
    process = _mock_process(alive=False)
    _terminate_process(process)
    process.terminate.assert_not_called()


def test_terminate_process_terminates_gracefully():
    """A process that exits promptly after terminate() should not be killed."""
    process = _mock_process(alive=True)
    process.join.side_effect = lambda timeout=None: setattr(process, "_joined", True)
    process.is_alive.side_effect = [True, False]
    _terminate_process(process)
    process.terminate.assert_called_once()
    process.kill.assert_not_called()


def test_terminate_process_force_kills_when_still_alive_after_terminate():
    """A process still alive after terminate() must be force-killed."""
    process = _mock_process(alive=True)
    process.is_alive.side_effect = [True, True]
    _terminate_process(process)
    process.terminate.assert_called_once()
    process.kill.assert_called_once()
    assert process.join.call_count == 2


def test_shutdown_closes_connection_and_terminates_process_then_resets_state():
    """shutdown() must close the connection, terminate the process, and clear state."""
    conn = mock.MagicMock()
    process = _mock_process(alive=False)
    _STATE["process"] = process
    _STATE["conn"] = conn

    client.shutdown()

    conn.close.assert_called_once()
    assert _STATE["process"] is None
    assert _STATE["conn"] is None


def test_shutdown_invalidates_the_generation():
    """shutdown() must bump the generation counter (via _teardown_worker) so any
    handle cached against the pre-shutdown generation is treated as stale --
    shutdown() is used for the app's periodic "purge everything" idle cleanup,
    and a subsequent respawned worker's objects dict is empty."""
    _STATE["process"] = _mock_process(alive=True)
    _STATE["conn"] = mock.MagicMock()
    _STATE["generation"] = 4

    client.shutdown()

    assert client.generation() == 5


def test_shutdown_is_a_noop_on_already_empty_state():
    """shutdown() with no active worker must not raise."""
    client.shutdown()
    assert _STATE["process"] is None
    assert _STATE["conn"] is None


def _hung_poll_side_effect(poll_started: threading.Event):
    """conn.poll side effect simulating a worker that never responds.

    Only reachable once call_with_generation already holds _LOCK (it's
    acquired, and conn.send() has happened, before the poll loop starts) --
    signaling here lets a caller safely start a competing shutdown() only
    after the call thread genuinely holds the lock, removing any race on
    which thread acquires it first."""

    def _side_effect(_timeout):
        poll_started.set()
        return False

    return _side_effect


def _run_blocked_call(call_errors: list[Exception]) -> None:
    try:
        client.call("transcribe")
    except client.WhisperXWorkerError as exc:
        call_errors.append(exc)


def test_shutdown_interrupts_a_hung_unbounded_call_instead_of_deadlocking():
    """With the default unbounded RPC deadline (_CALL_TIMEOUT_SEC<=0), a
    call_with_generation() blocked on a hung worker must not hold _LOCK
    forever -- shutdown() (which needs the same lock) has to be able to
    interrupt it via _SHUTDOWN_REQUESTED and actually complete, rather than
    the two deadlocking against each other."""
    conn = mock.MagicMock()
    process = _mock_process(alive=True)
    _STATE["process"] = process
    _STATE["conn"] = conn
    _STATE["generation"] = 3

    call_errors: list[Exception] = []
    poll_started = threading.Event()
    conn.poll.side_effect = _hung_poll_side_effect(poll_started)

    with mock.patch.object(client, "_SHUTDOWN_POLL_INTERVAL_SEC", 0.05), mock.patch.object(client, "_CALL_TIMEOUT_SEC", 0.0):
        call_thread = threading.Thread(target=_run_blocked_call, args=(call_errors,), daemon=True)
        call_thread.start()
        assert poll_started.wait(timeout=2.0)

        # shutdown() must complete within a bounded time, not hang behind the
        # in-flight call's lock hold.
        shutdown_thread = threading.Thread(target=client.shutdown, daemon=True)
        shutdown_thread.start()
        shutdown_thread.join(timeout=5.0)
        assert not shutdown_thread.is_alive(), "shutdown() deadlocked behind the hung call's lock"

        call_thread.join(timeout=5.0)

    assert len(call_errors) == 1, "shutdown did not interrupt hung unbounded call (call_errors empty)"
    assert "shut down" in str(call_errors[0])
    assert (_STATE["process"], _STATE["conn"]) == (None, None)


def test_acquire_lock_warns_when_blocked_beyond_threshold(monkeypatch: pytest.MonkeyPatch):
    """Callers blocked on _LOCK beyond the warn threshold should log once, then wait."""
    monkeypatch.setattr(client, "_LOCK_WARN_SEC", 0.01)
    held = threading.Lock()
    holder_ready = threading.Event()
    release_event = threading.Event()
    warning_seen = threading.Event()
    _STATE["generation"] = 9

    def _hold_lock():
        with held:
            holder_ready.set()
            release_event.wait(timeout=5.0)

    holder = threading.Thread(target=_hold_lock, daemon=True)
    holder.start()
    assert holder_ready.wait(timeout=1.0)
    monkeypatch.setattr(client, "_LOCK", held)

    def _release_after_warning():
        assert warning_seen.wait(timeout=5.0)
        release_event.set()

    releaser = threading.Thread(target=_release_after_warning, daemon=True)
    releaser.start()
    try:
        with mock.patch.object(client.logger, "warning") as mock_warning:

            def _record_warning(*_args, **_kwargs):
                warning_seen.set()

            mock_warning.side_effect = _record_warning
            assert client.generation() == 9
        mock_warning.assert_called_once()
        assert mock_warning.call_args.args[2] == "generation"
    finally:
        release_event.set()
        releaser.join(timeout=1.0)
        holder.join(timeout=1.0)
