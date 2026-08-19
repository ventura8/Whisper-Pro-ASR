"""AMD GPU utilization probes (sysfs vendor mapping and rocm-smi/amd-smi CSV)."""

import glob
from shutil import which

from modules.core import process_exec


def _is_header_csv_line(line: str) -> bool:
    return not line or "GPU" in line or "device" in line.lower()


def _extract_last_col_int(cols: list[str]) -> int | None:
    if len(cols) > 1 and cols[-1].isdigit():
        return int(cols[-1])
    return None


def _extract_first_digit_col(cols: list[str]) -> int | None:
    for clean in cols:
        if clean.isdigit():
            return int(clean)
    return None


def _extract_int_from_csv_line(line: str) -> int | None:
    if _is_header_csv_line(line):
        return None
    cols = [p.strip().replace("%", "") for p in line.split(",") if p.strip()]
    last = _extract_last_col_int(cols)
    return last if last is not None else _extract_first_digit_col(cols)


def _parse_amd_smi_csv(csv_text: str) -> list[int]:
    loads = []
    for raw_line in csv_text.strip().split("\n"):
        val = _extract_int_from_csv_line(raw_line.strip())
        if val is not None:
            loads.append(val)
    return loads


def _resolve_amd_smi_cmd(smi_bin: str) -> list[str]:
    if "rocm-smi" in smi_bin:
        return [smi_bin, "--showuse", "--csv"]
    return [smi_bin, "metric", "--usage", "--csv"]


def _probe_amd_smi_metrics() -> list[int]:
    """Internal probe for AMD GPU utilization using rocm-smi or amd-smi CLI."""
    smi_bin = which("rocm-smi") or which("amd-smi")
    if not smi_bin:
        return []
    try:
        res = process_exec.check_output_text(_resolve_amd_smi_cmd(smi_bin), timeout=5.0)
        return _parse_amd_smi_csv(res)
    except (process_exec.CommandExecutionError, process_exec.CommandTimeoutError, FileNotFoundError, ValueError):
        return []


def _amd_vendor_busy_paths() -> list[str]:
    paths: list[str] = []
    for vendor_path in glob.glob("/sys/class/drm/card*/device/vendor"):
        try:
            with open(vendor_path, "r", encoding="utf-8") as handle:
                vendor = handle.read().strip().lower()
        except OSError:
            continue
        if vendor == "0x1002":
            paths.append(vendor_path.replace("/vendor", "/gpu_busy_percent"))
    return sorted(paths)


def _amd_sysfs_paths(idx: int) -> list[str]:
    mapped = _amd_vendor_busy_paths()
    return [mapped[idx]] if 0 <= idx < len(mapped) else []
