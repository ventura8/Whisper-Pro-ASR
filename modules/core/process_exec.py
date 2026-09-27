"""Process execution helpers without direct subprocess imports."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class CommandResult:
    """Result of a command execution."""

    returncode: int
    stdout: str
    stderr: str


class CommandTimeoutError(RuntimeError):
    """Raised when a command exceeds the configured timeout."""


class CommandExecutionError(RuntimeError):
    """Raised when a command exits with a non-zero status."""

    def __init__(self, command: list[str], returncode: int, stderr: str = ""):
        super().__init__(f"Command failed with return code {returncode}: {' '.join(command)}")
        self.command = command
        self.returncode = returncode
        self.stderr = stderr


def _run_coroutine_sync(coro):
    """Execute a coroutine from sync code, even if a loop is already running."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()

    return asyncio.run(coro)


async def _terminate_process(proc) -> None:
    """Terminate a process gracefully, then force kill if needed."""
    if proc.returncode is not None:
        return
    proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), timeout=1.0)
    except asyncio.TimeoutError:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()


async def _run_capture_async(command: list[str]) -> CommandResult:
    # No timeout parameter: the deadline is applied by the caller around this coroutine
    # (see _run_bounded). When it expires the coroutine is cancelled, and the child must
    # not outlive it, so cancellation terminates the process before propagating.
    proc = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_b, stderr_b = await proc.communicate()
    except asyncio.CancelledError:
        await _terminate_process(proc)
        raise

    return CommandResult(
        returncode=proc.returncode,
        stdout=(stdout_b or b"").decode("utf-8", errors="replace"),
        stderr=(stderr_b or b"").decode("utf-8", errors="replace"),
    )


async def _run_stream_async(command: list[str], on_line: Callable[[str], None]) -> int:
    # Deadline applied by the caller, as for _run_capture_async; cancellation kills the child.
    proc = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    async def _consume_lines() -> None:
        if proc.stdout is None:
            raise RuntimeError("Command stream is unavailable")
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            on_line(line.decode("utf-8", errors="replace"))
        await proc.wait()

    try:
        await _consume_lines()
    except asyncio.CancelledError:
        await _terminate_process(proc)
        raise

    return proc.returncode


def _run_bounded(coro, timeout: float | None):
    """Run ``coro`` to completion from sync code under an optional deadline.

    ``asyncio.wait_for`` cancels the coroutine when the deadline passes; the coroutine
    terminates its child process on that cancellation, and the resulting TimeoutError is
    reported as CommandTimeoutError. ``timeout=None`` waits indefinitely.
    """
    try:
        return _run_coroutine_sync(asyncio.wait_for(coro, timeout=timeout))
    except TimeoutError as exc:
        raise CommandTimeoutError(f"Command timed out after {timeout}s") from exc


def run_capture(command: list[str], timeout: float | None = None, *, check: bool = False) -> CommandResult:
    """Run a command and capture stdout/stderr."""
    result: CommandResult = _run_bounded(_run_capture_async(command), timeout)
    if check and result.returncode != 0:
        raise CommandExecutionError(command, result.returncode, result.stderr)
    return result


def check_output_text(command: list[str], timeout: float | None = None) -> str:
    """Run a command and return stdout text when successful."""
    result = run_capture(command, timeout=timeout, check=True)
    return result.stdout


def run_stream(command: list[str], timeout: float, on_line: Callable[[str], None]) -> int:
    """Run a command, streaming merged stdout/stderr lines to a callback."""
    returncode: int = _run_bounded(_run_stream_async(command, on_line), timeout)
    if returncode != 0:
        raise CommandExecutionError(command, returncode)
    return returncode
