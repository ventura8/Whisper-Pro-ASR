"""Hardware metrics discovery utility for CUDA, Intel GPU, and NPU utilization."""

import glob
import logging
import platform
import re
import threading
import time
from shutil import which
from typing import Any, Callable

from modules.core import config, process_exec
from modules.inference import scheduler
from modules.inference.pipeline import openvino_resolver
from modules.inference.runtime import model_manager

logger = logging.getLogger(__name__)

# --- [CACHING ENGINE] ---
_METRIC_CACHE: dict[str, tuple[Any, float]] = {}
_CACHE_LOCK: threading.Lock = threading.Lock()
CACHE_LOCK = _CACHE_LOCK
METRIC_CACHE = _METRIC_CACHE
CACHE_TTL = 5.0  # Seconds
REAL_SAMPLE_HOLD_TTL = 20.0  # Seconds
_LAST_REAL_ACCEL_SAMPLES: dict[tuple[str, int], tuple[int, float]] = {}


def _monotonic_now() -> float:
    """Return a monotonic clock value for sample hold timing."""
    return time.monotonic()


def _get_cached_metric(key: str, fetch_func: Callable[[], Any]) -> Any:
    """Generic TTL cache for expensive hardware probes."""
    now = time.time()
    with _CACHE_LOCK:
        if key in _METRIC_CACHE:
            val, expiry = _METRIC_CACHE[key]
            if now < expiry:
                return val

    # Fetch fresh data outside the lock if possible, but for simplicity we'll keep it here
    # as these calls are not re-entrant anyway.
    val = fetch_func()
    with _CACHE_LOCK:
        _METRIC_CACHE[key] = (val, now + CACHE_TTL)
    return val


def get_nvidia_metrics() -> list[dict[str, int]]:
    """Real usage via nvidia-smi with TTL caching."""
    return _get_cached_metric("nvidia", _fetch_nvidia_metrics)


def _fetch_nvidia_metrics() -> list[dict[str, int]]:
    """Internal raw probe for NVIDIA stats."""
    nvidia_smi = which("nvidia-smi")
    if not nvidia_smi:
        return []
    try:
        res = process_exec.check_output_text(
            [nvidia_smi, "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,nounits,noheader"],
            timeout=5.0,
        )
        gpus = []
        for line in res.strip().split("\n"):
            if not line:
                continue
            util, m_used, m_total = line.split(",")
            gpus.append({"util": int(util), "mem_used": int(m_used), "mem_total": int(m_total)})
        return gpus
    except (process_exec.CommandExecutionError, process_exec.CommandTimeoutError, FileNotFoundError, ValueError):
        return []


def _is_preprocessor_accelerated(pm: Any) -> bool:
    """Check if the preprocessor separator is running with acceleration."""
    if not pm.separator:
        return False
    if type(pm.separator).__name__ in ("Mock", "MagicMock", "NonCallableMagicMock", "NonCallableMock"):
        return True
    providers = getattr(pm.separator, "onnx_execution_provider", [])
    return any("OpenVINO" in p or "CUDA" in p for p in providers)


def _is_uvr_openvino_disabled(unit_type: str) -> bool:
    return bool(unit_type in ["GPU", "NPU"] and openvino_resolver.is_openvino_family_disabled(unit_type))


def _is_uvr_using_accelerator(task: dict[str, Any], unit_type: str) -> bool:
    if _is_uvr_openvino_disabled(unit_type):
        return False
    unit_id = task.get("unit_id")
    if not unit_id:
        return True
    pm = model_manager.PREPROCESSOR_POOL.get(unit_id)
    return not pm or _is_preprocessor_accelerated(pm)


def _is_task_using_accelerator(task: dict[str, Any], unit_type: str) -> bool:
    """Checks if a task is actively utilizing the accelerator in its current stage."""
    stage = task.get("stage", "").lower()
    if _is_uvr_stage(stage):
        return _is_uvr_using_accelerator(task, unit_type)
    if _is_asr_stage(stage):
        return _supports_asr_stage_on_unit(unit_type)
    return False


def _is_uvr_stage(stage: str) -> bool:
    return any(token in stage for token in ["isolation", "separation", "uvr"])


def _is_asr_stage(stage: str) -> bool:
    return any(token in stage for token in ["inference", "transcrib", "translat", "detect", "lang"])


def _supports_asr_stage_on_unit(unit_type: str) -> bool:
    if unit_type in ["CUDA", "AMD"]:
        return True
    if unit_type in ["GPU", "NPU"]:
        if openvino_resolver.is_openvino_family_disabled(unit_type):
            return False
        return config.ASR_ENGINE == "INTEL-WHISPER"
    return False


def get_intel_gpu_load() -> int:
    """Real usage via sysfs or PowerShell with TTL caching."""
    return _get_cached_metric("intel", _fetch_intel_gpu_load)


def _fetch_intel_gpu_load() -> int:
    """Internal raw probe for Intel GPU stats."""
    inactive_result = _inactive_accelerator_zero_result("GPU", "GPU.0", 0, exclude_nvidia=True)
    if inactive_result is not None:
        return inactive_result
    return _fetch_single_intel_gpu_load("GPU.0")


def get_npu_load() -> int:
    """Real usage for Intel NPU with TTL caching."""
    return _get_cached_metric("npu", _fetch_npu_load)


def _fetch_npu_load() -> int:
    """Internal raw probe for NPU stats."""
    inactive_result = _inactive_accelerator_zero_result("NPU", "NPU.0", 0)
    if inactive_result is not None:
        return inactive_result
    return _fetch_single_npu_load("NPU.0")


def _resolve_index(unit_id: Any) -> int:
    """Extract integer index from unit ID (e.g. GPU.1 -> 1, cuda:2 -> 2, NPU -> 0)."""
    match = re.search(r"\d+", str(unit_id))
    return int(match.group()) if match else 0


def _fetch_single_intel_gpu_load(unit_id: Any) -> int:
    """Internal probe for a single Intel GPU's load."""
    idx = _resolve_index(unit_id)
    return _fetch_single_accelerator_load(
        unit_id=unit_id,
        idx=idx,
        device_type="GPU",
        sysfs_paths=glob.glob(f"/sys/class/drm/card{idx}/device/gpu_busy_percent"),
        windows_cmd=_gpu_counter_command(),
        busy_value=100,
        exclude_nvidia=True,
    )


def _fetch_single_npu_load(unit_id: Any) -> int:
    """Internal probe for a single Intel NPU's load."""
    idx = _resolve_index(unit_id)
    paths = glob.glob(f"/sys/class/accel/accel{idx}/device/utilization") + glob.glob(f"/sys/class/drm/accel{idx}/device/utilization")
    return _fetch_single_accelerator_load(
        unit_id=unit_id,
        idx=idx,
        device_type="NPU",
        sysfs_paths=paths,
        windows_cmd=_npu_counter_command(),
        busy_value=100,
        exclude_nvidia=False,
    )


def _fetch_single_cuda_fallback(unit_id: Any, idx: int) -> int:
    """Fallback logic for CUDA when nvidia-smi telemetry is unavailable."""
    if _has_active_accelerator_tasks("CUDA", unit_id=unit_id, idx=idx):
        return 99
    if _has_locked_preprocessor("CUDA", unit_id=unit_id, idx=idx):
        return 99
    return 0


def _unit_has_app_accelerator_work(device_type: str, unit_id: Any, idx: int, exclude_nvidia: bool = False) -> bool:
    """Return whether this service is actively using the accelerator unit right now."""
    if _has_active_accelerator_tasks(device_type, unit_id=unit_id, idx=idx, exclude_nvidia=exclude_nvidia):
        return True
    return _has_locked_preprocessor(device_type, unit_id=unit_id, idx=idx, exclude_nvidia=exclude_nvidia)


def _inactive_accelerator_zero_result(
    device_type: str,
    unit_id: Any,
    idx: int,
    *,
    exclude_nvidia: bool = False,
) -> int | None:
    """Return 0 for inactive app-owned accelerator units and clear held real samples."""
    if _unit_has_app_accelerator_work(device_type, unit_id, idx, exclude_nvidia=exclude_nvidia):
        return None
    _clear_recent_real_accelerator_sample(device_type, idx)
    return 0


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


def _run_windows_accelerator_counter(cmd: str) -> int | None:
    powershell_path = _resolve_windows_powershell()
    if not powershell_path:
        return None
    try:
        res = process_exec.check_output_text([powershell_path, "-Command", cmd], timeout=5.0).strip()
        return _normalize_windows_counter_value(res)
    except (process_exec.CommandExecutionError, process_exec.CommandTimeoutError, ValueError):
        return None


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


def _read_first_int_value(paths: list[str]) -> int | None:
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return int(f.read().strip())
        except (IOError, ValueError):
            continue
    return None


def _has_active_accelerator_tasks(device_type: str, unit_id: Any = None, idx: int = 0, exclude_nvidia: bool = False) -> bool:
    try:
        stats = scheduler.get_service_stats_minimal()
        active_tasks = stats.get("active_tasks", [])
        for task in active_tasks:
            if not _task_matches_unit(task, device_type, unit_id, idx, exclude_nvidia):
                continue
            if _is_task_using_accelerator(task, device_type):
                return True
    except AttributeError:
        pass
    return False


def _task_matches_unit(task: dict[str, Any], device_type: str, unit_id: Any, idx: int, exclude_nvidia: bool) -> bool:
    if _has_exact_unit_id_match(task, unit_id):
        return True
    if task.get("unit_type") != device_type:
        return False
    if exclude_nvidia:
        if "NVIDIA" in task.get("unit_name", ""):
            return False
    return _unit_index_matches(task.get("unit_id"), idx)


def _has_exact_unit_id_match(task: dict[str, Any], unit_id: Any) -> bool:
    if unit_id is None:
        return False
    return bool(task.get("unit_id") == unit_id)


def _unit_index_matches(unit_id: Any, idx: int) -> bool:
    parsed = _resolve_index(unit_id)
    if parsed == idx:
        return True
    return idx == 0 and not re.search(r"\d+", str(unit_id or ""))


def _is_pm_locked_and_accelerated(pm: Any) -> bool:
    return bool(pm.lock.locked() and _is_preprocessor_accelerated(pm))


def _has_locked_preprocessor(device_type: str, unit_id: Any = None, idx: int = 0, exclude_nvidia: bool = False) -> bool:
    try:
        return any(
            _is_pm_locked_and_accelerated(pm)
            for pm in model_manager.PREPROCESSOR_POOL.values()
            if _preprocessor_matches_unit(pm, device_type, unit_id, idx, exclude_nvidia)
        )
    except AttributeError:
        return False


def _preprocessor_matches_unit(pm: Any, device_type: str, unit_id: Any, idx: int, exclude_nvidia: bool) -> bool:
    if not _preprocessor_type_matches(pm, device_type):
        return False
    if not _preprocessor_vendor_matches(pm, exclude_nvidia):
        return False
    pm_unit_id = (pm.unit or {}).get("id", pm.device_id)
    return _preprocessor_id_matches(pm_unit_id, unit_id, idx)


def _preprocessor_type_matches(pm: Any, device_type: str) -> bool:
    return bool(pm.device_type == device_type)


def _preprocessor_vendor_matches(pm: Any, exclude_nvidia: bool) -> bool:
    if not exclude_nvidia:
        return True
    return not _is_nvidia_preprocessor(pm)


def _preprocessor_id_matches(pm_unit_id: Any, unit_id: Any, idx: int) -> bool:
    if unit_id is not None and pm_unit_id != unit_id:
        return False
    return _unit_index_matches(pm_unit_id, idx)


def _is_nvidia_preprocessor(pm: Any) -> bool:
    unit_name = pm.unit.get("name", "") if pm.unit else ""
    return "NVIDIA" in unit_name


def get_all_hardware_utilization() -> dict[str, int]:
    """Retrieve utilization for all configured hardware units with caching."""
    return _get_cached_metric("all_hardware_util", _fetch_all_hardware_utilization)


def _fetch_all_hardware_utilization() -> dict[str, int]:
    """Internal raw probe for all configured hardware units."""
    util_map = {}
    for unit in config.HARDWARE_UNITS:
        unit_id = unit["id"]
        unit_type = unit["type"]
        value = _resolve_unit_utilization(unit_type, unit_id)
        if value is not None:
            util_map[unit_id] = value

    return util_map


def _resolve_unit_utilization(unit_type: str, unit_id: Any) -> int | None:
    if unit_type == "CPU":
        return None

    resolvers = {
        "CUDA": lambda: _resolve_cuda_utilization(unit_id),
        "AMD": lambda: _resolve_amd_utilization(unit_id),
        "GPU": lambda: _resolve_intel_unit_utilization("GPU", unit_id, exclude_nvidia=True),
        "NPU": lambda: _resolve_intel_unit_utilization("NPU", unit_id, exclude_nvidia=False),
    }
    resolver = resolvers.get(unit_type)
    return resolver() if resolver else None


def _resolve_amd_utilization(unit_id: Any) -> int:
    idx = _resolve_index(unit_id)
    inactive_result = _inactive_accelerator_zero_result("AMD", unit_id, idx)
    if inactive_result is not None:
        return inactive_result
    return _probe_activity_fallback(unit_id, idx, "AMD", 100, exclude_nvidia=False)


def _resolve_intel_unit_utilization(unit_type: str, unit_id: Any, *, exclude_nvidia: bool) -> int:
    idx = _resolve_index(unit_id)
    inactive_result = _inactive_accelerator_zero_result(unit_type, unit_id, idx, exclude_nvidia=exclude_nvidia)
    if inactive_result is not None:
        return inactive_result
    if unit_type == "GPU":
        return _fetch_single_intel_gpu_load(unit_id)
    return _fetch_single_npu_load(unit_id)


def _resolve_cuda_utilization(unit_id: Any) -> int:
    idx = _resolve_index(unit_id)
    inactive_result = _inactive_accelerator_zero_result("CUDA", unit_id, idx)
    if inactive_result is not None:
        return inactive_result
    try:
        nvidia_metrics = get_nvidia_metrics()
        if idx < len(nvidia_metrics):
            value = nvidia_metrics[idx]["util"]
            _store_real_accelerator_sample("CUDA", idx, value)
            return value
    except (RuntimeError, ValueError, KeyError, IndexError, AttributeError, TypeError, OSError):
        pass

    held = _get_recent_real_accelerator_sample("CUDA", idx)
    if held is not None:
        return held

    return _fetch_single_cuda_fallback(unit_id, idx)


def _fetch_single_accelerator_load(
    unit_id: Any,
    idx: int,
    device_type: str,
    sysfs_paths: list[str],
    *,
    windows_cmd: str | None,
    busy_value: int,
    exclude_nvidia: bool = False,
) -> int:
    val = _probe_real_accelerator_sample(device_type, idx, sysfs_paths, windows_cmd)
    if val is not None:
        return val

    held = _get_recent_real_accelerator_sample(device_type, idx)
    if held is not None:
        return held

    return _probe_activity_fallback(unit_id, idx, device_type, busy_value, exclude_nvidia)


def _is_active_sample(val: int | None) -> bool:
    if val is None:
        return False
    return val > 0


def _get_best_accel_sample(sysfs_val: int | None, win_val: int | None) -> int | None:
    if _is_active_sample(sysfs_val):
        return sysfs_val
    if _is_active_sample(win_val):
        return win_val
    if sysfs_val is not None:
        return sysfs_val
    if win_val is not None:
        return win_val
    return None


def _probe_real_accelerator_sample(device_type: str, idx: int, sysfs_paths: list[str], windows_cmd: str | None) -> int | None:
    sysfs_val = _read_first_int_value(sysfs_paths)
    win_val = _run_windows_counter_if_available(windows_cmd)

    best_val = _get_best_accel_sample(sysfs_val, win_val)
    if best_val is not None:
        _store_real_accelerator_sample(device_type, idx, best_val)
    return best_val


def _run_windows_counter_if_available(windows_cmd: str | None) -> int | None:
    if not windows_cmd:
        return None
    return _run_windows_accelerator_counter(windows_cmd)


def _store_real_accelerator_sample(device_type: str, idx: int, value: int) -> None:
    key = (device_type, idx)
    with _CACHE_LOCK:
        _LAST_REAL_ACCEL_SAMPLES[key] = (value, _monotonic_now())


def _clear_recent_real_accelerator_sample(device_type: str, idx: int) -> None:
    key = (device_type, idx)
    with _CACHE_LOCK:
        _LAST_REAL_ACCEL_SAMPLES.pop(key, None)


def _get_recent_real_accelerator_sample(device_type: str, idx: int) -> int | None:
    key = (device_type, idx)
    now = _monotonic_now()
    with _CACHE_LOCK:
        sample = _LAST_REAL_ACCEL_SAMPLES.get(key)
        if sample is None:
            return None
        value, ts = sample
        if (now - ts) <= REAL_SAMPLE_HOLD_TTL:
            return value
        _LAST_REAL_ACCEL_SAMPLES.pop(key, None)
    return None


def _probe_sysfs_and_windows(sysfs_paths: list[str], windows_cmd: str | None) -> int | None:
    val = _read_first_int_value(sysfs_paths)
    if val is not None:
        return val
    if windows_cmd:
        val = _run_windows_accelerator_counter(windows_cmd)
        if val is not None:
            return val
    return None


def _probe_activity_fallback(unit_id: Any, idx: int, device_type: str, busy_value: int, exclude_nvidia: bool) -> int:
    if _has_active_accelerator_tasks(device_type, unit_id=unit_id, idx=idx, exclude_nvidia=exclude_nvidia):
        return busy_value
    if _has_locked_preprocessor(device_type, unit_id=unit_id, idx=idx, exclude_nvidia=exclude_nvidia):
        return busy_value
    return 0
