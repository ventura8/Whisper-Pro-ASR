"""
Hardware Metrics Discovery Utility
Provides real-time utilization for NVIDIA, Intel GPU, and NPU.
"""
import subprocess
import logging
import platform
import glob
import time
import threading

logger = logging.getLogger(__name__)

# --- [CACHING ENGINE] ---
_METRIC_CACHE = {}
_CACHE_LOCK = threading.Lock()
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
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,nounits,noheader"],
            encoding='utf-8', stderr=subprocess.DEVNULL
        )
        gpus = []
        for line in res.strip().split('\n'):
            if not line:
                continue
            util, m_used, m_total = line.split(',')
            gpus.append({
                "util": int(util),
                "mem_used": int(m_used),
                "mem_total": int(m_total)
            })
        return gpus
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return []


def get_intel_gpu_load():
    """Real usage via sysfs or PowerShell with TTL caching."""
    return _get_cached_metric("intel", _fetch_intel_gpu_load)


def _fetch_intel_gpu_load():
    """Internal raw probe for Intel GPU stats."""
    # 1. Try Linux Sysfs (WSL2/Docker with passthrough)
    try:
        paths = glob.glob("/sys/class/drm/card*/device/gpu_busy_percent")
        if paths:
            with open(paths[0], 'r', encoding='utf-8') as f:
                return int(f.read().strip())
    except (IOError, ValueError):
        pass

    # 2. Try Windows PowerShell
    if platform.system() == "Windows":
        try:
            cmd = (
                "powershell -Command \"(Get-Counter '\\GPU Engine(*engtype_3D)\\Utilization Percentage', "
                "'\\GPU Engine(*engtype_Video*)\\Utilization Percentage' -ErrorAction SilentlyContinue).CounterSamples | "
                "Measure-Object -Property CookedValue -Sum | Select-Object -ExpandProperty Sum\""
            )
            res = subprocess.check_output(
                cmd, shell=True, encoding='utf-8', stderr=subprocess.DEVNULL).strip()
            if res and float(res) > 0:
                return min(100, int(float(res)))
        except (subprocess.CalledProcessError, ValueError):
            pass

    # 3. Hybrid Fallback (Circular Import Safe)
    try:
        from modules.inference import scheduler  # pylint: disable=import-outside-toplevel
        stats = scheduler.get_service_stats_minimal()
        active_tasks = stats.get('active_tasks', [])
        if any(t.get('unit_type') == 'GPU' and 'NVIDIA' not in t.get('unit_name', '') for t in active_tasks):
            return 99
    except (ImportError, AttributeError):
        pass

    return 0


def get_npu_load():
    """Real usage for Intel NPU with TTL caching."""
    return _get_cached_metric("npu", _fetch_npu_load)


def _fetch_npu_load():
    """Internal raw probe for NPU stats."""
    # 1. Try Linux Accel Sysfs
    try:
        paths = glob.glob("/sys/class/accel/accel*/device/utilization") + \
            glob.glob("/sys/class/drm/accel*/device/utilization")
        if paths:
            with open(paths[0], 'r', encoding='utf-8') as f:
                return int(f.read().strip())
    except (IOError, ValueError):
        pass

    # 2. Try Windows PowerShell
    if platform.system() == "Windows":
        try:
            cmd = (
                "powershell -Command \"(Get-Counter '\\GPU Engine(*engtype_Compute)\\Utilization Percentage', "
                "'\\GPU Engine(*engtype_NPU)\\Utilization Percentage' -ErrorAction SilentlyContinue).CounterSamples | "
                "Measure-Object -Property CookedValue -Sum | Select-Object -ExpandProperty Sum\""
            )
            res = subprocess.check_output(
                cmd, shell=True, encoding='utf-8', stderr=subprocess.DEVNULL).strip()
            if res and float(res) > 0:
                return min(100, int(float(res)))
        except (subprocess.CalledProcessError, ValueError):
            pass

    # 3. Hybrid Fallback
    try:
        from modules.inference import scheduler  # pylint: disable=import-outside-toplevel
        stats = scheduler.get_service_stats_minimal()
        active_tasks = stats.get('active_tasks', [])
        if any(t.get('unit_type') == 'NPU' for t in active_tasks):
            return 100
    except (ImportError, AttributeError):
        pass

    return 0
