"""PCM duration helpers used by `modules.core.utils.get_audio_duration`.

These helpers are split out to keep `utils.py` focused and under size limits.
"""

from __future__ import annotations

import os


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


def format_duration(seconds) -> str:
    """Convert raw seconds into a human-readable HH:MM:SS format."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
