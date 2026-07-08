"""
Hardware Metrics Discovery Utility
Provides real-time utilization for NVIDIA, Intel GPU, and NPU.
"""

import glob
import logging
import platform
import re
import subprocess
import threading
import time

from modules.core import config
from modules.inference import model_manager, scheduler

logger = logging.getLogger(__name__)

# --- [CACHING ENGINE] ---
_METRIC_CACHE = {}
_CACHE_LOCK = threading.Lock()
CACHE_LOCK = _CACHE_LOCK
METRIC_CACHE = _METRIC_CACHE
CACHE_TTL = 5.0  # Seconds


def _get_cached_metric(key, fetch_func):
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


def get_nvidia_metrics():
    """Real usage via nvidia-smi with TTL caching."""
    return _get_cached_metric("nvidia", _fetch_nvidia_metrics)


def _fetch_nvidia_metrics():
    """Internal raw probe for NVIDIA stats."""
    try:
        res = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,nounits,noheader"],
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
            timeout=5.0,
        )
        gpus = []
        for line in res.strip().split("\n"):
            if not line:
                continue
            util, m_used, m_total = line.split(",")
            gpus.append({"util": int(util), "mem_used": int(m_used), "mem_total": int(m_total)})
        return gpus
    except (subprocess.SubprocessError, FileNotFoundError, ValueError):
        return []


def _is_task_using_accelerator(task, unit_type):
    """Checks if a task is actively utilizing the accelerator in its current stage."""
    stage = task.get("stage", "").lower()

    # If the task is performing vocal separation / UVR, it always uses the accelerator
    if any(s in stage for s in ["isolation", "separation", "uvr"]):
        return True

    # If the task is in transcription/translation/language detection:
    if any(s in stage for s in ["inference", "transcrib", "translat", "detect", "lang"]):
        if unit_type == "CUDA":
            return True
        if unit_type in ["GPU", "NPU"] and config.ASR_ENGINE == "INTEL-WHISPER":
            return True
        return False

    # Default to True for other/unknown stages (e.g. initializing)
    return True


def get_intel_gpu_load():
    """Real usage via sysfs or PowerShell with TTL caching."""
    return _get_cached_metric("intel", _fetch_intel_gpu_load)


def _fetch_intel_gpu_load():
    """Internal raw probe for Intel GPU stats."""
    # 1. Try Linux Sysfs (WSL2/Docker with passthrough)
    try:
        paths = glob.glob("/sys/class/drm/card*/device/gpu_busy_percent")
        if paths:
            with open(paths[0], "r", encoding="utf-8") as f:
                val = int(f.read().strip())
                if val > 0:
                    return val
    except (IOError, ValueError):
        pass

    # 2. Try Windows PowerShell
    if platform.system() == "Windows":
        try:
            cmd = (
                "(Get-Counter '\\GPU Engine(*engtype_3D)\\Utilization Percentage', "
                "'\\GPU Engine(*engtype_Video*)\\Utilization Percentage' -ErrorAction SilentlyContinue).CounterSamples | "
                "Measure-Object -Property CookedValue -Sum | Select-Object -ExpandProperty Sum"
            )
            res = subprocess.check_output(
                ["powershell", "-Command", cmd], encoding="utf-8", stderr=subprocess.DEVNULL, timeout=5.0
            ).strip()
            if res and float(res) > 0:
                return min(100, int(float(res)))
        except (subprocess.SubprocessError, ValueError):
            pass

    # 3. Hybrid Fallback (Circular Import Safe)
    try:
        stats = scheduler.get_service_stats_minimal()
        active_tasks = stats.get("active_tasks", [])
        if any(
            t.get("unit_type") == "GPU"
            and "NVIDIA" not in t.get("unit_name", "")
            and _is_task_using_accelerator(t, "GPU")
            for t in active_tasks
        ):
            return 100
    except AttributeError:
        pass

    # 4. Preprocessor Lock Fallback
    try:
        for pm in model_manager.PREPROCESSOR_POOL.values():
            unit_name = pm.unit.get("name", "") if pm.unit else ""
            if pm.device_type == "GPU" and "NVIDIA" not in unit_name and pm.lock.locked():
                return 100
    except AttributeError:
        pass

    return 0


def get_npu_load():
    """Real usage for Intel NPU with TTL caching."""
    return _get_cached_metric("npu", _fetch_npu_load)


def _fetch_npu_load():
    """Internal raw probe for NPU stats."""
    # 1. Try Linux Accel Sysfs
    try:
        paths = glob.glob("/sys/class/accel/accel*/device/utilization") + glob.glob(
            "/sys/class/drm/accel*/device/utilization"
        )
        if paths:
            with open(paths[0], "r", encoding="utf-8") as f:
                val = int(f.read().strip())
                if val > 0:
                    return val
    except (IOError, ValueError):
        pass

    # 2. Try Windows PowerShell
    if platform.system() == "Windows":
        try:
            cmd = (
                "(Get-Counter '\\GPU Engine(*engtype_Compute)\\Utilization Percentage', "
                "'\\GPU Engine(*engtype_NPU)\\Utilization Percentage' -ErrorAction SilentlyContinue).CounterSamples | "
                "Measure-Object -Property CookedValue -Sum | Select-Object -ExpandProperty Sum"
            )
            res = subprocess.check_output(
                ["powershell", "-Command", cmd], encoding="utf-8", stderr=subprocess.DEVNULL, timeout=5.0
            ).strip()
            if res and float(res) > 0:
                return min(100, int(float(res)))
        except (subprocess.SubprocessError, ValueError):
            pass

    # 3. Hybrid Fallback
    try:
        stats = scheduler.get_service_stats_minimal()
        active_tasks = stats.get("active_tasks", [])
        if any(t.get("unit_type") == "NPU" and _is_task_using_accelerator(t, "NPU") for t in active_tasks):
            return 100
    except AttributeError:
        pass

    # 4. Preprocessor Lock Fallback
    try:
        for pm in model_manager.PREPROCESSOR_POOL.values():
            if pm.device_type == "NPU" and pm.lock.locked():
                return 100
    except AttributeError:
        pass

    return 0


def _resolve_index(unit_id):
    """Extract integer index from unit ID (e.g. GPU.1 -> 1, cuda:2 -> 2, NPU -> 0)."""
    match = re.search(r"\d+", str(unit_id))
    return int(match.group()) if match else 0


def _fetch_single_intel_gpu_load(unit_id):
    """Internal probe for a single Intel GPU's load."""
    idx = _resolve_index(unit_id)

    # 1. Try Linux Sysfs
    try:
        paths = glob.glob(f"/sys/class/drm/card{idx}/device/gpu_busy_percent")
        if paths:
            with open(paths[0], "r", encoding="utf-8") as f:
                val = int(f.read().strip())
                if val > 0:
                    return val
    except (IOError, ValueError):
        pass

    # 2. Try Windows PowerShell for first GPU
    if platform.system() == "Windows" and idx == 0:
        try:
            cmd = (
                "(Get-Counter '\\GPU Engine(*engtype_3D)\\Utilization Percentage', "
                "'\\GPU Engine(*engtype_Video*)\\Utilization Percentage' -ErrorAction SilentlyContinue).CounterSamples | "
                "Measure-Object -Property CookedValue -Sum | Select-Object -ExpandProperty Sum"
            )
            res = subprocess.check_output(
                ["powershell", "-Command", cmd], encoding="utf-8", stderr=subprocess.DEVNULL, timeout=5.0
            ).strip()
            if res and float(res) > 0:
                return min(100, int(float(res)))
        except (subprocess.SubprocessError, ValueError):
            pass

    # 3. Hybrid Fallback (Circular Import Safe)
    try:
        stats = scheduler.get_service_stats_minimal()
        active_tasks = stats.get("active_tasks", [])
        if any(
            (
                t.get("unit_id") == unit_id
                or (
                    t.get("unit_type") == "GPU"
                    and "NVIDIA" not in t.get("unit_name", "")
                    and (
                        _resolve_index(t.get("unit_id")) == idx
                        or (idx == 0 and not re.search(r"\d+", str(t.get("unit_id") or "")))
                    )
                )
            )
            and _is_task_using_accelerator(t, "GPU")
            for t in active_tasks
        ):
            return 100
    except AttributeError:
        pass

    # 4. Preprocessor Lock Fallback
    try:
        for pm in model_manager.PREPROCESSOR_POOL.values():
            pm_unit_id = pm.unit["id"] if pm.unit else pm.device_id
            pm_unit_name = pm.unit.get("name", "") if pm.unit else ""
            if (
                pm.device_type == "GPU"
                and "NVIDIA" not in pm_unit_name
                and _resolve_index(pm_unit_id) == idx
                and pm.lock.locked()
            ):
                return 100
    except AttributeError:
        pass

    return 0


def _fetch_single_npu_load(unit_id):
    """Internal probe for a single Intel NPU's load."""
    idx = _resolve_index(unit_id)

    # 1. Try Linux Accel Sysfs
    try:
        paths = glob.glob(f"/sys/class/accel/accel{idx}/device/utilization") + glob.glob(
            f"/sys/class/drm/accel{idx}/device/utilization"
        )
        if paths:
            with open(paths[0], "r", encoding="utf-8") as f:
                val = int(f.read().strip())
                if val > 0:
                    return val
    except (IOError, ValueError):
        pass

    # 2. Try Windows PowerShell for first NPU
    if platform.system() == "Windows" and idx == 0:
        try:
            cmd = (
                "(Get-Counter '\\GPU Engine(*engtype_Compute)\\Utilization Percentage', "
                "'\\GPU Engine(*engtype_NPU)\\Utilization Percentage' -ErrorAction SilentlyContinue).CounterSamples | "
                "Measure-Object -Property CookedValue -Sum | Select-Object -ExpandProperty Sum"
            )
            res = subprocess.check_output(
                ["powershell", "-Command", cmd], encoding="utf-8", stderr=subprocess.DEVNULL, timeout=5.0
            ).strip()
            if res and float(res) > 0:
                return min(100, int(float(res)))
        except (subprocess.SubprocessError, ValueError):
            pass

    # 3. Hybrid Fallback
    try:
        stats = scheduler.get_service_stats_minimal()
        active_tasks = stats.get("active_tasks", [])
        if any(
            (
                t.get("unit_id") == unit_id
                or (
                    t.get("unit_type") == "NPU"
                    and (
                        _resolve_index(t.get("unit_id")) == idx
                        or (idx == 0 and not re.search(r"\d+", str(t.get("unit_id") or "")))
                    )
                )
            )
            and _is_task_using_accelerator(t, "NPU")
            for t in active_tasks
        ):
            return 100
    except AttributeError:
        pass

    # 4. Preprocessor Lock Fallback
    try:
        for pm in model_manager.PREPROCESSOR_POOL.values():
            pm_unit_id = pm.unit["id"] if pm.unit else pm.device_id
            if pm.device_type == "NPU" and _resolve_index(pm_unit_id) == idx and pm.lock.locked():
                return 100
    except AttributeError:
        pass

    return 0


def _fetch_single_cuda_fallback(unit_id, idx):
    """Fallback logic for CUDA when nvidia-smi telemetry is unavailable."""
    # 1. Hybrid Fallback (Circular Import Safe)
    try:
        stats = scheduler.get_service_stats_minimal()
        active_tasks = stats.get("active_tasks", [])
        if any(
            (
                t.get("unit_id") == unit_id
                or (
                    t.get("unit_type") == "CUDA"
                    and (
                        _resolve_index(t.get("unit_id")) == idx
                        or (idx == 0 and not re.search(r"\d+", str(t.get("unit_id") or "")))
                    )
                )
            )
            and _is_task_using_accelerator(t, "CUDA")
            for t in active_tasks
        ):
            return 99
    except AttributeError:
        pass

    # 2. Preprocessor Lock Fallback
    try:
        for pm in model_manager.PREPROCESSOR_POOL.values():
            pm_unit_id = pm.unit["id"] if pm.unit else pm.device_id
            if pm.device_type == "CUDA" and _resolve_index(pm_unit_id) == idx and pm.lock.locked():
                return 99
    except AttributeError:
        pass

    return 0


def get_all_hardware_utilization():
    """Retrieve utilization for all configured hardware units with caching."""
    return _get_cached_metric("all_hardware_util", _fetch_all_hardware_utilization)


def _fetch_all_hardware_utilization():
    """Internal raw probe for all configured hardware units."""
    util_map = {}
    for unit in config.HARDWARE_UNITS:
        unit_id = unit["id"]
        unit_type = unit["type"]

        if unit_type == "CPU":
            continue
        if unit_type == "CUDA":
            idx = _resolve_index(unit_id)
            try:
                nvidia_metrics = get_nvidia_metrics()
                if idx < len(nvidia_metrics):
                    util_map[unit_id] = nvidia_metrics[idx]["util"]
                else:
                    util_map[unit_id] = _fetch_single_cuda_fallback(unit_id, idx)
            except (RuntimeError, ValueError, KeyError, IndexError, AttributeError, TypeError, OSError):
                util_map[unit_id] = _fetch_single_cuda_fallback(unit_id, idx)
        elif unit_type == "GPU":
            util_map[unit_id] = _fetch_single_intel_gpu_load(unit_id)
        elif unit_type == "NPU":
            util_map[unit_id] = _fetch_single_npu_load(unit_id)

    return util_map
