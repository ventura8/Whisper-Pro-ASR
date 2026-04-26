"""
Hardware Metrics Discovery Utility
Provides real-time utilization for NVIDIA, Intel GPU, and NPU.
"""
import subprocess
import logging
import platform
import glob

logger = logging.getLogger(__name__)


def get_nvidia_metrics():
    """Real usage via nvidia-smi."""
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
    except Exception:
        return []


def get_intel_gpu_load():
    """Real usage via sysfs (Linux/WSL) or PowerShell (Windows)."""
    # 1. Try Linux Sysfs (WSL2/Docker with passthrough)
    try:
        # Search for any DRM card that has busy_percent
        paths = glob.glob("/sys/class/drm/card*/device/gpu_busy_percent")
        if paths:
            with open(paths[0], 'r') as f:
                return int(f.read().strip())
    except Exception:
        pass

    # 2. Try Windows PowerShell (if running on host)
    if platform.system() == "Windows":
        try:
            # Query multiple engine types common for GPUs (3D, Video, Compute)
            cmd = "powershell -Command \"(Get-Counter '\\GPU Engine(*engtype_3D)\\Utilization Percentage', '\\GPU Engine(*engtype_Video*)\\Utilization Percentage' -ErrorAction SilentlyContinue).CounterSamples | Measure-Object -Property CookedValue -Sum | Select-Object -ExpandProperty Sum\""
            res = subprocess.check_output(
                cmd, shell=True, encoding='utf-8', stderr=subprocess.DEVNULL).strip()
            if res and float(res) > 0:
                return min(100, int(float(res)))
        except Exception:
            pass

    # 3. Hybrid Fallback: Check if any tasks are assigned to Intel GPU in model_manager
    # Note: Circular import protection
    from . import model_manager
    stats = model_manager.get_service_stats_minimal()
    active_tasks = stats.get('active_tasks', [])
    if any(t.get('unit_type') == 'GPU' and 'NVIDIA' not in t.get('unit_name', '') for t in active_tasks):
        return 99  # High activity proxy if real sensor is blind

    return 0


def get_npu_load():
    """Real usage for Intel NPU."""
    # 1. Try Linux Accel Sysfs (Latest kernels)
    try:
        # Check both accel and drm paths (accel is the new standard)
        paths = glob.glob("/sys/class/accel/accel*/device/utilization") + \
            glob.glob("/sys/class/drm/accel*/device/utilization")
        if paths:
            with open(paths[0], 'r') as f:
                return int(f.read().strip())
    except Exception:
        pass

    # 2. Try Windows PowerShell
    if platform.system() == "Windows":
        try:
            # Intel NPU engine types vary by driver version (Compute, NPU, etc.)
            cmd = "powershell -Command \"(Get-Counter '\\GPU Engine(*engtype_Compute)\\Utilization Percentage', '\\GPU Engine(*engtype_NPU)\\Utilization Percentage' -ErrorAction SilentlyContinue).CounterSamples | Measure-Object -Property CookedValue -Sum | Select-Object -ExpandProperty Sum\""
            res = subprocess.check_output(
                cmd, shell=True, encoding='utf-8', stderr=subprocess.DEVNULL).strip()
            if res and float(res) > 0:
                return min(100, int(float(res)))
        except Exception:
            pass

    # 3. Hybrid Fallback
    from . import model_manager
    stats = model_manager.get_service_stats_minimal()
    active_tasks = stats.get('active_tasks', [])
    if any(t.get('unit_type') == 'NPU' for t in active_tasks):
        return 100

    return 0
