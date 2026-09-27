"""Focused coverage tests for modules/core/process_exec.py."""

import asyncio
from types import SimpleNamespace
from unittest import mock

import pytest

from modules.core import process_exec


def _noop_line_callback(_line: str) -> None:
    """No-op callback used for stream tests."""


def test_run_coroutine_sync_uses_asyncio_run_without_running_loop():
    """No running loop should route through asyncio.run."""

    async def _sample():
        return 7

    def _run_and_close(coro):
        coro.close()
        return 7

    with (
        mock.patch("asyncio.get_running_loop", side_effect=RuntimeError),
        mock.patch("asyncio.run", side_effect=_run_and_close) as mock_run,
    ):
        result = getattr(process_exec, "_run_coroutine_sync")(_sample())
        assert result == 7
        mock_run.assert_called_once()


def test_run_coroutine_sync_uses_thread_executor_when_loop_running():
    """An active loop should run coroutine via a helper thread."""

    async def _sample():
        return 9

    executor = mock.MagicMock()
    executor.__enter__.return_value = executor
    executor.__exit__.return_value = False

    def _submit(_fn, coro):
        coro.close()
        return SimpleNamespace(result=lambda: 9)

    executor.submit.side_effect = _submit

    running_loop = mock.Mock()
    running_loop.is_running.return_value = True

    with (
        mock.patch("asyncio.get_running_loop", return_value=running_loop),
        mock.patch("modules.core.process_exec.ThreadPoolExecutor", return_value=executor),
    ):
        result = getattr(process_exec, "_run_coroutine_sync")(_sample())
        assert result == 9
        executor.submit.assert_called_once()


def test_terminate_process_noop_if_already_finished():
    """Terminate helper should return immediately for completed processes."""
    proc = SimpleNamespace(returncode=0, terminate=mock.Mock())
    asyncio.run(getattr(process_exec, "_terminate_process")(proc))
    proc.terminate.assert_not_called()


def test_terminate_process_graceful_path():
    """Terminate helper should terminate and wait without kill when process exits in time."""
    proc = SimpleNamespace(returncode=None, terminate=mock.Mock(), kill=mock.Mock())

    async def _wait():
        proc.returncode = 0
        return 0

    proc.wait = _wait

    asyncio.run(getattr(process_exec, "_terminate_process")(proc))

    proc.terminate.assert_called_once_with()
    proc.kill.assert_not_called()


def test_terminate_process_kills_after_timeout():
    """Terminate helper should kill process if graceful wait times out."""
    proc = SimpleNamespace(returncode=None, terminate=mock.Mock(), kill=mock.Mock())

    async def _wait():
        proc.returncode = -9
        return -9

    proc.wait = _wait

    async def _raise_timeout(awaitable, *_, **__):
        awaitable.close()
        raise asyncio.TimeoutError

    with mock.patch("asyncio.wait_for", side_effect=_raise_timeout):
        asyncio.run(getattr(process_exec, "_terminate_process")(proc))

    proc.terminate.assert_called_once_with()
    proc.kill.assert_called_once_with()


def test_run_capture_raises_when_check_enabled_and_nonzero_returncode():
    """run_capture(check=True) should raise CommandExecutionError on non-zero return code."""
    result = process_exec.CommandResult(returncode=3, stdout="", stderr="boom")
    with (
        mock.patch("modules.core.process_exec._run_capture_async", return_value=None),
        mock.patch("modules.core.process_exec._run_coroutine_sync", return_value=result),
    ):
        with pytest.raises(process_exec.CommandExecutionError):
            process_exec.run_capture(["cmd"], check=True)


def test_check_output_text_returns_stdout():
    """check_output_text should return captured stdout from successful command."""
    with mock.patch("modules.core.process_exec.run_capture") as mock_run:
        mock_run.return_value = process_exec.CommandResult(returncode=0, stdout="ok", stderr="")
        assert process_exec.check_output_text(["cmd"]) == "ok"


def test_run_stream_raises_for_nonzero_returncode():
    """run_stream should raise CommandExecutionError for non-zero result codes."""
    with (
        mock.patch("modules.core.process_exec._run_stream_async", return_value=None),
        mock.patch("modules.core.process_exec._run_coroutine_sync", return_value=2),
    ):
        with pytest.raises(process_exec.CommandExecutionError):
            process_exec.run_stream(["cmd"], timeout=1.0, on_line=lambda _line: None)


def test_run_capture_returns_result_when_check_disabled():
    """run_capture should return the command result when check=False."""
    result = process_exec.CommandResult(returncode=7, stdout="out", stderr="err")
    with (
        mock.patch("modules.core.process_exec._run_capture_async", return_value=None),
        mock.patch("modules.core.process_exec._run_coroutine_sync", return_value=result),
    ):
        assert process_exec.run_capture(["cmd"], check=False) == result


def test_run_capture_async_success_decodes_none_buffers():
    """_run_capture_async should normalize missing stdout/stderr buffers to empty strings."""
    proc = SimpleNamespace(returncode=0)

    async def _communicate():
        return None, None

    proc.communicate = _communicate

    with mock.patch("asyncio.create_subprocess_exec", return_value=proc):
        result = asyncio.run(getattr(process_exec, "_run_capture_async")(["cmd"], timeout=1.0))

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_run_capture_async_timeout_raises_command_timeout():
    """_run_capture_async should translate wait_for timeout into CommandTimeoutError."""
    proc = SimpleNamespace(returncode=None)

    async def _communicate():
        return b"", b""

    proc.communicate = _communicate

    with (
        mock.patch("asyncio.create_subprocess_exec", return_value=proc),
        mock.patch("asyncio.wait_for", side_effect=asyncio.TimeoutError),
        mock.patch("modules.core.process_exec._terminate_process", new_callable=mock.AsyncMock) as mock_terminate,
    ):
        capture = getattr(process_exec, "_run_capture_async")(["cmd"], timeout=0.01)
        with pytest.raises(process_exec.CommandTimeoutError):
            asyncio.run(capture)

    mock_terminate.assert_awaited_once_with(proc)


def test_run_stream_async_raises_when_stdout_unavailable():
    """_run_stream_async should fail fast when subprocess stdout stream is unavailable."""
    proc = SimpleNamespace(stdout=None)

    async def _wait():
        return 0

    proc.wait = _wait

    with mock.patch("asyncio.create_subprocess_exec", return_value=proc):
        stream = getattr(process_exec, "_run_stream_async")(["cmd"], timeout=1.0, on_line=_noop_line_callback)
        with pytest.raises(RuntimeError, match="stream is unavailable"):
            asyncio.run(stream)


def test_run_stream_async_streams_lines_and_returns_code():
    """_run_stream_async should stream decoded lines and return subprocess return code."""
    lines = [b"alpha\n", b"beta\n", b""]

    async def _readline():
        return lines.pop(0)

    stdout = SimpleNamespace(readline=_readline)
    proc = SimpleNamespace(stdout=stdout, returncode=0)

    async def _wait():
        proc.returncode = 0
        return 0

    proc.wait = _wait
    seen_lines: list[str] = []

    def _collect_line(line: str) -> None:
        seen_lines.append(line)

    with mock.patch("asyncio.create_subprocess_exec", return_value=proc):
        result = asyncio.run(getattr(process_exec, "_run_stream_async")(["cmd"], timeout=1.0, on_line=_collect_line))

    assert result == 0
    assert seen_lines == ["alpha\n", "beta\n"]


def test_run_stream_async_timeout_raises_command_timeout():
    """_run_stream_async should terminate and raise CommandTimeoutError on timeout."""
    proc = SimpleNamespace(stdout=SimpleNamespace(readline=mock.AsyncMock(return_value=b"")), returncode=None)

    async def _wait():
        return 0

    proc.wait = _wait

    with (
        mock.patch("asyncio.create_subprocess_exec", return_value=proc),
        mock.patch("asyncio.wait_for", side_effect=asyncio.TimeoutError),
        mock.patch("modules.core.process_exec._terminate_process", new_callable=mock.AsyncMock) as mock_terminate,
    ):
        stream = getattr(process_exec, "_run_stream_async")(["cmd"], timeout=0.01, on_line=_noop_line_callback)
        with pytest.raises(process_exec.CommandTimeoutError):
            asyncio.run(stream)

    mock_terminate.assert_awaited_once_with(proc)


def test_run_stream_returns_code_when_zero():
    """run_stream should return zero return code when subprocess succeeds."""
    with (
        mock.patch("modules.core.process_exec._run_stream_async", return_value=None),
        mock.patch("modules.core.process_exec._run_coroutine_sync", return_value=0),
    ):
        assert process_exec.run_stream(["cmd"], timeout=1.0, on_line=_noop_line_callback) == 0
