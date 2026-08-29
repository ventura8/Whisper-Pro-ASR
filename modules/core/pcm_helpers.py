"""PCM duration helpers used by `modules.core.utils.get_audio_duration`.

These helpers are split out to keep `utils.py` focused and under size limits.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

logger = logging.getLogger(__name__)


def _positive_pcm_int(flags: list[str], flag: str, default: int) -> int:
    """Parse a PCM flag as a positive int, otherwise return the safe default."""
    try:
        value = int(flags[flags.index(flag) + 1])
    except (ValueError, IndexError):
        return default
    return value if value > 0 else default


def pcm_bytes_per_second(input_flags: list[str] | None) -> float:
    """Derive PCM bytes/sec from `-ar` and `-ac` flags.

    Defaults match `STANDARD_AUDIO_FLAGS` (16kHz mono s16le): sample_rate=16000,
    channels=1, bytes_per_sample=2.
    """
    flags = list(input_flags or [])
    sample_rate = _positive_pcm_int(flags, "-ar", 16000)
    channels = _positive_pcm_int(flags, "-ac", 1)
    return float(sample_rate * channels * 2)  # s16le — 16-bit signed little-endian


def calculate_pcm_fallback_duration(file_path: str, input_flags: list[str] | None) -> float:
    """Fallback duration calculation based on file size (only valid for raw PCM)."""
    try:
        if input_flags and os.path.exists(file_path):
            f_size = os.path.getsize(file_path)
            return float(f_size) / pcm_bytes_per_second(input_flags)
    except tuple([Exception]):
        pass
    return 0.0


def _probe_duration_cmd(file_path: str, flags: list[str] | None, check_output_fn: Callable[..., str]) -> float:
    cmd = ["ffprobe", "-v", "error"]
    if flags:
        cmd.extend(flags)
    cmd.extend(
        [
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            file_path,
        ]
    )
    return float(check_output_fn(cmd, timeout=10).strip())


def _probe_duration_safe(file_path: str, flags: list[str] | None, check_output_fn: Callable[..., str]) -> float | None:
    try:
        duration = _probe_duration_cmd(file_path, flags, check_output_fn)
        return duration if duration > 0 else None
    except tuple([Exception]) as exc:
        # Logged, because this is the step whose silent failure makes a duration wrong
        # rather than absent: the caller falls through to a size-based estimate that is
        # only correct for raw PCM, so on a container it silently invents a length. At
        # debug level -- a failed probe is an expected step in the precedence chain.
        logger.debug("[Duration] ffprobe failed for %s (flags=%s): %s", file_path, flags, exc)
        return None


def _probe_explicit_flags(file_path: str, input_flags: list[str], check_output_fn: Callable[..., str]) -> float:
    duration = _probe_duration_safe(file_path, input_flags, check_output_fn)
    if duration is not None:
        return duration
    return calculate_pcm_fallback_duration(file_path, input_flags)


def _probe_contextual_flags(file_path: str, thread_flags: list[str] | None, check_output_fn: Callable[..., str]) -> float:
    if thread_flags:
        duration = _probe_duration_safe(file_path, thread_flags, check_output_fn)
        if duration is not None:
            return duration
    return calculate_pcm_fallback_duration(file_path, thread_flags)


def probe_audio_duration(
    file_path: str,
    input_flags: list[str] | None,
    thread_input_flags: list[str] | None,
    check_output_fn: Callable[..., str],
) -> float:
    """Extract media duration via ffprobe with native probe precedence and raw PCM fallback."""
    if input_flags is not None:
        return _probe_explicit_flags(file_path, input_flags, check_output_fn)

    native_duration = _probe_duration_safe(file_path, None, check_output_fn)
    if native_duration is not None:
        return native_duration

    return _probe_contextual_flags(file_path, thread_input_flags, check_output_fn)


def format_duration(seconds) -> str:
    """Convert raw seconds into a human-readable HH:MM:SS format."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
