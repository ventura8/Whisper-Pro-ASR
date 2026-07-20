"""
Configuration Helper Utilities for Whisper Pro ASR
"""

import importlib
import logging
import os
import sys
from typing import Any

logger = logging.getLogger(__name__)


def get_unit_limit(env_var: str, default: int = 1, min_value: int = 0) -> int:
    """Helper to parse hardware unit limits (supports int, ALL, AUTO)."""
    val = os.environ.get(env_var, str(default)).upper()
    if val in ["ALL", "AUTO"]:
        return 999  # Practically unlimited
    try:
        return max(min_value, int(val))
    except (ValueError, TypeError):
        return max(min_value, int(default))


def _is_explicit_amd_target(target: str) -> bool:
    dev = target.upper()
    return dev in ("AMD", "ROCM", "DML", "DIRECTML") or dev.startswith("AMD")


def _detect_explicit_amd_first(
    max_cuda: int,
    max_amd: int,
    hardware_units: list[dict[str, str]],
    state: dict[str, str],
    *,
    is_explicit_dev: bool,
    is_explicit_prep: bool,
) -> None:
    _detect_amd_hardware(max_amd, hardware_units, state)
    amd_dev = state["device"]
    amd_prep = state["prep_device"]
    _detect_cuda_hardware(max_cuda, hardware_units, state)
    if amd_dev == "AMD" and is_explicit_dev:
        state["device"] = "AMD"
        state["compute"] = "float16"
    if amd_prep == "AMD" and is_explicit_prep:
        state["prep_device"] = "AMD"


def detect_hardware(max_cuda: int, max_gpu: int, max_npu: int, max_amd: int, hardware_units: list[dict[str, str]]) -> tuple[str, str, str]:
    """Detect acceleration hardware and returns (detected_device, detected_prep_device, detected_compute)."""
    state = {"device": "CPU", "prep_device": "CPU", "compute": "int8"}
    is_explicit_dev = _is_explicit_amd_target(os.environ.get("ASR_DEVICE", ""))
    is_explicit_prep = _is_explicit_amd_target(os.environ.get("ASR_PREPROCESS_DEVICE", ""))

    if is_explicit_dev or is_explicit_prep:
        _detect_explicit_amd_first(
            max_cuda,
            max_amd,
            hardware_units,
            state,
            is_explicit_dev=is_explicit_dev,
            is_explicit_prep=is_explicit_prep,
        )
    else:
        _detect_cuda_hardware(max_cuda, hardware_units, state)
        _detect_amd_hardware(max_amd, hardware_units, state)

    _detect_intel_hardware(max_gpu, max_npu, hardware_units, state)
    _ensure_cpu_fallback_unit(hardware_units)
    return state["device"], state["prep_device"], state["compute"]


def _detect_amd_hardware(max_amd: int, hardware_units: list[dict[str, str]], state: dict[str, str]) -> None:
    try:
        if max_amd <= 0 or not _has_amd_hardware():
            return
        logger.debug("Auto-detected AMD GPU hardware.")
        _append_amd_units(max_amd, hardware_units)
        _update_amd_state(max_amd, state)
    except (ImportError, AttributeError, ValueError, TypeError, RuntimeError, OSError) as e:
        logger.debug("AMD GPU detection skipped: %s", e)


def _check_amd_dml(available: list[str]) -> bool:
    if sys.platform != "win32" or "DmlExecutionProvider" not in available:
        return False
    dev = os.environ.get("ASR_DEVICE", "").upper()
    prep = os.environ.get("ASR_PREPROCESS_DEVICE", "").upper()
    return dev == "AMD" or prep == "AMD"


def _has_rocm_hardware(available: list[str]) -> bool:
    return any(p in available for p in ("ROCMExecutionProvider", "MIGraphXExecutionProvider"))


def _check_amd_ort_providers() -> bool:
    try:
        ort = importlib.import_module("onnxruntime")
        available = ort.get_available_providers()
        return _has_rocm_hardware(available) or _check_amd_dml(available)
    except (ImportError, AttributeError, RuntimeError, OSError):
        return False


def _has_amd_wsl_driver_folder(folder_path: str) -> bool:
    return os.path.isdir(folder_path) and os.path.exists(os.path.join(folder_path, "amdxc64.so"))


def _is_amd_wsl_driver_present() -> bool:
    """Return True if AMD Radeon WSL display driver (amdxc64.so) is mounted in /usr/lib/wsl/drivers."""
    drivers_dir = "/usr/lib/wsl/drivers"
    if not os.path.isdir(drivers_dir):
        return False
    try:
        return any(_has_amd_wsl_driver_folder(os.path.join(drivers_dir, item)) for item in os.listdir(drivers_dir))
    except OSError:
        return False


def _has_amd_wsl_hardware() -> bool:
    return os.path.exists("/dev/dxg") and os.path.exists("/opt/rocm/lib/librocdxg.so") and _is_amd_wsl_driver_present()


def _has_amd_hardware() -> bool:
    """Return True if an AMD GPU card is available on Linux (/dev/kfd or DRM) or WSL2 (/dev/dxg)."""
    if os.path.exists("/dev/kfd") or _has_amd_wsl_hardware():
        return True
    return os.path.exists("/dev/dri") and _is_amd_drm_present()


def _count_amd_drm_devices() -> int:
    drm_root = "/sys/class/drm"
    if not os.path.isdir(drm_root):
        return 0
    count = 0
    for vendor_file in _iter_drm_vendor_files(drm_root):
        if _read_vendor_id(vendor_file) == "0x1002":
            count += 1
    return count


def _append_amd_units(max_amd: int, hardware_units: list[dict[str, str]]) -> None:
    drm_count = _count_amd_drm_devices()
    detected = drm_count if drm_count > 0 else 1
    units_to_use = min(detected, max_amd)
    for i in range(units_to_use):
        hardware_units.append({"type": "AMD", "id": f"amd:{i}", "name": f"AMD GPU {i}"})


def _update_amd_state(max_amd: int, state: dict[str, str]) -> None:
    if max_amd <= 0:
        return
    # Only take the ASR device slot if nothing faster claimed it yet
    if state["device"] == "CPU":
        state["device"] = "AMD"
        state["compute"] = "float16"
    # Only take the prep_device slot if nothing else (CUDA/GPU/NPU) claimed it
    if state["prep_device"] == "CPU":
        state["prep_device"] = "AMD"


def _is_amd_drm_present() -> bool:
    drm_root = "/sys/class/drm"
    if not os.path.isdir(drm_root):
        return False
    for vendor_file in _iter_drm_vendor_files(drm_root):
        if _read_vendor_id(vendor_file) == "0x1002":
            return True
    return False


def _detect_cuda_hardware(max_cuda: int, hardware_units: list[dict[str, str]], state: dict[str, str]) -> None:
    try:
        ct2 = importlib.import_module("ctranslate2")
        cuda_count = ct2.get_cuda_device_count()
        if cuda_count <= 0:
            return
        logger.debug("Auto-detected %d NVIDIA GPU(s).", cuda_count)
        _append_cuda_units(min(cuda_count, max_cuda), hardware_units)
        if min(cuda_count, max_cuda) > 0:
            state["device"] = "CUDA"
            state["prep_device"] = "CUDA"
            state["compute"] = "float16"
    except (ImportError, AttributeError, ValueError, TypeError, RuntimeError, OSError) as e:
        logger.debug("CUDA detection skipped: %s", e)


def _append_cuda_units(cuda_to_use: int, hardware_units: list[dict[str, str]]) -> None:
    for i in range(cuda_to_use):
        hardware_units.append({"type": "CUDA", "id": f"cuda:{i}", "name": f"NVIDIA GPU {i}"})


def _detect_intel_hardware(max_gpu: int, max_npu: int, hardware_units: list[dict[str, str]], state: dict[str, str]) -> None:
    try:
        ov = importlib.import_module("openvino")
        core = ov.Core()
        devices = core.available_devices
        logger.debug("OpenVINO Available Devices: %s", devices)
        gpu_count, npu_count = _append_intel_units(core, devices, max_gpu, max_npu, hardware_units, state=state)
        if gpu_count <= 0 and npu_count <= 0:
            logger.debug("OpenVINO did not report usable Intel GPU/NPU units; trying Linux device-node fallbacks")
            _append_intel_node_fallbacks(max_gpu, max_npu, hardware_units, state=state)
    except (ImportError, AttributeError, ValueError, TypeError, RuntimeError, OSError) as e:
        logger.debug("Intel accelerator detection skipped: %s", e)
        _append_intel_node_fallbacks(max_gpu, max_npu, hardware_units, state=state)


def _append_intel_node_fallbacks(max_gpu: int, max_npu: int, hardware_units: list[dict[str, str]], *, state: dict[str, str]) -> None:
    """Fallback to Linux device-node detection when OpenVINO enumeration fails."""
    _append_npu_node_fallback(max_npu, hardware_units, state=state)
    _append_gpu_node_fallback(max_gpu, hardware_units, state=state)


def _append_npu_node_fallback(max_npu: int, hardware_units: list[dict[str, str]], *, state: dict[str, str]) -> None:
    if max_npu <= 0 or not os.path.exists("/dev/accel/accel0"):
        return
    hardware_units.append({"type": "NPU", "id": "NPU", "name": "Intel NPU"})
    if state["device"] == "CPU":
        state["device"] = "NPU"
    state["prep_device"] = "NPU"
    logger.info("Detected Intel NPU via /dev/accel fallback")


def _append_gpu_node_fallback(max_gpu: int, hardware_units: list[dict[str, str]], *, state: dict[str, str]) -> None:
    if not _can_use_gpu_node_fallback(max_gpu):
        return
    hardware_units.append({"type": "GPU", "id": "GPU", "name": "Intel GPU"})
    if state["device"] == "CPU":
        state["device"] = "GPU"
    if state["prep_device"] == "CPU":
        state["prep_device"] = "GPU"
    logger.info("Detected Intel GPU via /dev/dri fallback")


def _can_use_gpu_node_fallback(max_gpu: int) -> bool:
    """Return whether Linux GPU fallback detection should be applied."""
    if max_gpu <= 0:
        return False
    if not os.path.exists("/dev/dri"):
        return False
    return _is_intel_drm_present()


def _iter_drm_vendor_files(drm_root: str) -> list[str]:
    """Return vendor file paths for DRM render nodes."""
    try:
        entries = os.listdir(drm_root)
    except OSError:
        return []
    return [os.path.join(drm_root, entry, "device", "vendor") for entry in entries if entry.startswith("renderD")]


def _read_vendor_id(vendor_file: str) -> str | None:
    """Read a DRM vendor id file and return normalized content when accessible."""
    try:
        with open(vendor_file, "r", encoding="utf-8") as handle:
            return handle.read().strip().lower()
    except OSError:
        return None


def _is_intel_drm_present() -> bool:
    """Return True when a DRM render node reports Intel vendor id 0x8086."""
    drm_root = "/sys/class/drm"
    if not os.path.isdir(drm_root):
        return os.path.exists("/dev/dri")
    for vendor_file in _iter_drm_vendor_files(drm_root):
        if _read_vendor_id(vendor_file) == "0x8086":
            return True
    return os.path.exists("/dev/dri")


def _append_intel_units(
    core: Any,
    devices: list[str],
    max_gpu: int,
    max_npu: int,
    hardware_units: list[dict[str, str]],
    *,
    state: dict[str, str],
) -> tuple[int, int]:
    gpu_detect_count = 0
    npu_detect_count = 0
    for dev in devices:
        if "GPU" in dev:
            gpu_detect_count = _append_intel_gpu(core, dev, gpu_detect_count, max_gpu, hardware_units, state=state)
            continue
        if "NPU" in dev:
            npu_detect_count = _append_intel_npu(core, dev, npu_detect_count, max_npu, hardware_units, state=state)
    return gpu_detect_count, npu_detect_count


def _append_intel_gpu(
    core: Any,
    dev: str,
    gpu_detect_count: int,
    max_gpu: int,
    hardware_units: list[dict[str, str]],
    *,
    state: dict[str, str],
) -> int:
    if gpu_detect_count >= max_gpu:
        return gpu_detect_count
    hardware_units.append({"type": "GPU", "id": dev, "name": _get_ov_device_name(core, dev)})
    if state["device"] == "CPU":
        state["device"] = "GPU"
    if state["prep_device"] in ("CPU", "CUDA"):
        state["prep_device"] = "GPU"
    return gpu_detect_count + 1


def _append_intel_npu(
    core: Any,
    dev: str,
    npu_detect_count: int,
    max_npu: int,
    hardware_units: list[dict[str, str]],
    *,
    state: dict[str, str],
) -> int:
    if npu_detect_count >= max_npu:
        return npu_detect_count
    hardware_units.append({"type": "NPU", "id": dev, "name": _get_ov_device_name(core, dev)})
    if state["device"] == "CPU":
        state["device"] = "NPU"
    state["prep_device"] = "NPU"
    return npu_detect_count + 1


def _get_ov_device_name(core: Any, dev: str) -> str:
    try:
        return str(core.get_property(dev, "FULL_DEVICE_NAME"))
    except (ImportError, AttributeError, ValueError, TypeError, RuntimeError, OSError):
        return f"Intel {dev}"


def _ensure_cpu_fallback_unit(hardware_units: list[dict[str, str]]) -> None:
    if hardware_units:
        return
    logger.info("No accelerators detected. Using Host CPU for all tasks.")
    hardware_units.append({"type": "CPU", "id": "CPU", "name": "Host CPU"})


def resolve_thread_limits(requested_asr: int, requested_prep: int, cpu_core_limit: int, max_cpu: int, device: str) -> tuple[int, int]:
    """Resolve and enforce physical hardware thread limits with priority."""
    cores = cpu_core_limit
    if max_cpu >= 999:
        return min(requested_asr, cores), min(requested_prep, cores)

    effective_pool = max(1, cpu_core_limit // max_cpu)
    asr_threads = min(requested_asr, effective_pool)
    prep_threads = min(requested_prep, _prep_cap_for_device(device, cores, effective_pool))

    _log_thread_caps(
        requested_asr=requested_asr,
        asr_threads=asr_threads,
        cpu_core_limit=cpu_core_limit,
        max_cpu=max_cpu,
        requested_prep=requested_prep,
        prep_threads=prep_threads,
        device=device,
        cores=cores,
    )
    return asr_threads, prep_threads


def _prep_cap_for_device(device: str, cores: int, effective_pool: int) -> int:
    return cores if device != "CPU" else effective_pool


def _log_thread_caps(
    *,
    requested_asr: int,
    asr_threads: int,
    cpu_core_limit: int,
    max_cpu: int,
    requested_prep: int,
    prep_threads: int,
    device: str,
    cores: int,
) -> None:
    if asr_threads < requested_asr:
        logger.info("[Config] Capping ASR_THREADS to %d (Global Limit: %d, Units: %d)", asr_threads, cpu_core_limit, max_cpu)
    if prep_threads < requested_prep and device != "CPU":
        logger.info("[Config] Capping ASR_PREPROCESS_THREADS to %d (Hardware limit)", cores)


def calculate_cpu_parallel_limit(max_cpu: int, cpu_core_limit: int, asr_threads: int, preprocess_threads: int) -> int:
    """Calculate how many multi-threaded CPU tasks can run safely."""
    if max_cpu < 999:
        return max_cpu

    cores = cpu_core_limit
    cores_per_task = max(1, asr_threads, preprocess_threads)
    limit = max(1, cores // cores_per_task)
    logger.info("[Resource] Calculated AUTO CPU parallel limit: %d (Cores: %d, Threads/Task: %d)", limit, cores, cores_per_task)
    return limit
