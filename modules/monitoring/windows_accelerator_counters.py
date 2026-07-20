"""Windows accelerator counter probe helpers.

Split out from `metrics_discovery.py` to keep that module under size limits.
"""

from __future__ import annotations

import platform
from shutil import which

from modules.core import process_exec


def _gpu_counter_command() -> str:
    return (
        "(Get-Counter '\\GPU Engine(*engtype_3D)\\Utilization Percentage', "
        "'\\GPU Engine(*engtype_Video*)\\Utilization Percentage' -ErrorAction SilentlyContinue).CounterSamples | "
        "Measure-Object -Property CookedValue -Sum | Select-Object -ExpandProperty Sum"
    )


def _npu_counter_command() -> str:
    return (
        "(Get-Counter '\\GPU Engine(*engtype_Compute)\\Utilization Percentage', "
        "'\\GPU Engine(*engtype_NPU)\\Utilization Percentage' -ErrorAction SilentlyContinue).CounterSamples | "
        "Measure-Object -Property CookedValue -Sum | Select-Object -ExpandProperty Sum"
    )


def _resolve_windows_powershell() -> str | None:
    if platform.system() != "Windows":
        return None
    return which("powershell")


def _normalize_windows_counter_value(res: str) -> int | None:
    if not res:
        return None
    if float(res) < 0:
        return None
    return min(100, int(float(res)))


def _run_windows_accelerator_counter(cmd: str) -> int | None:
    powershell_path = _resolve_windows_powershell()
    if not powershell_path:
        return None
    try:
        res = process_exec.check_output_text([powershell_path, "-Command", cmd], timeout=5.0).strip()
        return _normalize_windows_counter_value(res)
    except (
        process_exec.CommandExecutionError,
        process_exec.CommandTimeoutError,
        ValueError,
    ):
        return None


def _read_first_int_value(paths: list[str]) -> int | None:
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return int(f.read().strip())
        except (IOError, ValueError):
            continue
    return None
