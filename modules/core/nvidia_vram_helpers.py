"""NVIDIA VRAM probe helpers split from `modules.core.utils`."""

from __future__ import annotations

from shutil import which

from modules.core import process_exec


def get_nvidia_vram_usage_mb() -> int | None:
    """Return total used NVIDIA VRAM in MB across visible GPUs, or None if unavailable."""
    nvidia_smi = which("nvidia-smi")
    if not nvidia_smi:
        return None

    lines = _query_nvidia_vram_lines(nvidia_smi)
    if lines is None:
        return None

    return _parse_nvidia_vram_total(lines)


def _query_nvidia_vram_lines(nvidia_smi: str) -> list[str] | None:
    """Query nvidia-smi and return raw memory-used lines."""
    try:
        output = process_exec.check_output_text(
            [nvidia_smi, "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            timeout=5.0,
        )
    except (
        process_exec.CommandExecutionError,
        process_exec.CommandTimeoutError,
        FileNotFoundError,
        OSError,
    ):
        return None
    return output.splitlines()


def _parse_nvidia_vram_total(lines: list[str]) -> int | None:
    """Parse raw nvidia-smi memory-used lines and sum valid MB values."""
    values = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            values.append(int(stripped.split(",")[0].strip()))
        except ValueError:
            continue

    if not values:
        return None
    return sum(values)
