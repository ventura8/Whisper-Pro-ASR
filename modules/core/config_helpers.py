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
    return dev in ("ROCM", "DML", "DIRECTML") or dev.startswith("AMD")


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


def is_worker_context() -> bool:
    """Whether this interpreter is an isolated worker rather than the API process."""
    return bool(os.environ.get("WHISPER_WORKER_CONTEXT"))


def detect_hardware(max_cuda: int, max_gpu: int, max_npu: int, max_amd: int, hardware_units: list[dict[str, str]]) -> tuple[str, str, str]:
    """Detect acceleration hardware and returns (detected_device, detected_prep_device, detected_compute)."""
    if is_worker_context():
        # A worker is told which unit to use by the parent, so probing again is pure
        # cost -- and not free: ctranslate2.get_cuda_device_count() maps the NVIDIA
        # driver libraries into the process even when CUDA_VISIBLE_DEVICES hides every
        # device, which is why an Intel-only UVR worker still showed up in nvidia-smi.
        # Skipping the probe keeps a worker's address space to the one vendor it serves.
        hardware_units.append({"type": "CPU", "id": "CPU", "name": "Host CPU"})
        return "CPU", "CPU", "int8"

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

    _detect_intel_hardware(max_gpu, max_npu, hardware_units, state, is_explicit_prep=is_explicit_prep)
    _ensure_cpu_fallback_unit(hardware_units)
    return state["device"], state["prep_device"], state["compute"]


def _detect_amd_hardware(max_amd: int, hardware_units: list[dict[str, str]], state: dict[str, str]) -> None:
    try:
        if max_amd <= 0 or not _has_amd_hardware():
            return
        logger.debug("Auto-detected AMD GPU hardware.")
        before = len(hardware_units)
        _append_amd_units(max_amd, hardware_units)
        if len(hardware_units) > before:
            _update_amd_state(max_amd, state)
    except (ImportError, AttributeError, ValueError, TypeError, RuntimeError, OSError) as e:
        logger.debug("AMD GPU detection skipped: %s", e)


def _check_amd_dml(available: list[str]) -> bool:
    if sys.platform != "win32" or "DmlExecutionProvider" not in available:
        return False
    return _is_explicit_amd_target(os.environ.get("ASR_DEVICE", "")) or _is_explicit_amd_target(os.environ.get("ASR_PREPROCESS_DEVICE", ""))


def _has_rocm_hardware(available: list[str]) -> bool:
    return any(p in available for p in ("ROCMExecutionProvider", "MIGraphXExecutionProvider"))


def _check_amd_ort_providers() -> bool:
    try:
        ort = importlib.import_module("onnxruntime")
        available = ort.get_available_providers()
        return _has_rocm_hardware(available) or _check_amd_dml(available)
    except (ImportError, AttributeError, RuntimeError, OSError):
        return False


def _has_amdxc64(folder_path: str) -> bool:
    return os.path.isfile(os.path.join(folder_path, "amdxc64.so"))


def _has_nested_amdxc64(folder_path: str) -> bool:
    try:
        children = os.listdir(folder_path)
    except OSError:
        return False
    return any(_has_amdxc64(os.path.join(folder_path, child)) for child in children)


def _has_amd_wsl_driver_folder(folder_path: str) -> bool:
    """Return True if amdxc64.so is in the folder or one nested Adrenalin subdirectory."""
    if not os.path.isdir(folder_path):
        return False
    return _has_amdxc64(folder_path) or _has_nested_amdxc64(folder_path)


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


def has_amd_wsl_hardware() -> bool:
    """Public wrapper for AMD WSL hardware detection."""
    return _has_amd_wsl_hardware()


def _has_amd_win32_hardware() -> bool:
    return sys.platform == "win32" and _check_amd_ort_providers()


def _has_amd_linux_hardware() -> bool:
    return os.path.exists("/dev/dri") and _is_amd_drm_present()


def _has_rocm_runtime() -> bool:
    """Return True if this image ships a ROCm runtime that can drive an AMD GPU."""
    return os.path.isdir("/opt/rocm") or bool(os.environ.get("ROCM_PATH"))


def _has_amd_hardware() -> bool:
    """Return True if an AMD GPU card is present (presence only; see _has_rocm_runtime)."""
    if os.path.exists("/dev/kfd") or _has_amd_wsl_hardware():
        return True
    return _has_amd_win32_hardware() or _has_amd_linux_hardware()


def _count_amd_drm_devices() -> int:
    drm_root = "/sys/class/drm"
    if not os.path.isdir(drm_root):
        return 0
    count = 0
    for vendor_file in _iter_drm_vendor_files(drm_root):
        if _read_vendor_id(vendor_file) == "0x1002":
            count += 1
    return count


def _is_amd_drm_present() -> bool:
    return _count_amd_drm_devices() > 0


def _count_schedulable_amd_units() -> int:
    """Count AMD GPUs that have a usable execution provider (not WSL dxg-only)."""
    drm_count = _count_amd_drm_devices()
    if drm_count > 0:
        return drm_count
    if os.path.exists("/dev/kfd") or _has_amd_win32_hardware():
        return 1
    return 0


def _append_amd_units(max_amd: int, hardware_units: list[dict[str, str]]) -> None:
    # A card being present is not the same as this image being able to drive it. Only the
    # AMD images ship ROCm; elsewhere an AMD GPU is still visible through /dev/kfd and DRM,
    # joins the scheduler pool, and then runs on the CPU while every log line reads
    # "AMD GPU 0". Measured on a Ryzen host with an RTX 5090 and the nvidia image: 15 of 55
    # tasks were dispatched to an AMD unit in an image containing no ROCm at all -- 27% of a
    # CUDA validation silently running on the CPU.
    # DirectML on win32 drives AMD without ROCm, so the requirement applies only where
    # ROCm is the driver -- everywhere else the check would disable a working path.
    if sys.platform != "win32" and not _has_rocm_runtime():
        logger.info("AMD GPU present but this image ships no ROCm runtime; not adding it to the pool.")
        return
    units_to_use = min(_count_schedulable_amd_units(), max_amd)
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


def _detect_intel_hardware(
    max_gpu: int,
    max_npu: int,
    hardware_units: list[dict[str, str]],
    state: dict[str, str],
    *,
    is_explicit_prep: bool = False,
) -> None:
    try:
        ov = importlib.import_module("openvino")
        core = ov.Core()
        devices = core.available_devices
        logger.debug("OpenVINO Available Devices: %s", devices)
        gpu_count, npu_count = _append_intel_units(
            core,
            devices,
            max_gpu,
            max_npu,
            hardware_units,
            state=state,
            is_explicit_prep=is_explicit_prep,
        )
        if gpu_count <= 0 and npu_count <= 0:
            logger.debug("OpenVINO did not report usable Intel GPU/NPU units; trying Linux device-node fallbacks")
            _append_intel_node_fallbacks(max_gpu, max_npu, hardware_units, state=state, is_explicit_prep=is_explicit_prep)
    except (ImportError, AttributeError, ValueError, TypeError, RuntimeError, OSError) as e:
        logger.debug("Intel accelerator detection skipped: %s", e)
        _append_intel_node_fallbacks(max_gpu, max_npu, hardware_units, state=state, is_explicit_prep=is_explicit_prep)


def _append_intel_node_fallbacks(
    max_gpu: int,
    max_npu: int,
    hardware_units: list[dict[str, str]],
    *,
    state: dict[str, str],
    is_explicit_prep: bool = False,
) -> None:
    """Fallback to Linux device-node detection when OpenVINO enumeration fails."""
    _append_npu_node_fallback(max_npu, hardware_units, state=state, is_explicit_prep=is_explicit_prep)
    _append_gpu_node_fallback(max_gpu, hardware_units, state=state)


def _update_npu_state(state: dict[str, str], is_explicit_prep: bool) -> None:
    if state["device"] == "CPU":
        state["device"] = "NPU"
    _claim_npu_prep_slot(state, is_explicit_prep)


def _claim_npu_prep_slot(state: dict[str, str], is_explicit_prep: bool) -> None:
    """Give the NPU the preprocessing slot, ahead of an iGPU that is also present.

    This is deliberate, not an accident of enumeration order. UVR runs on the NPU
    (verified: the MDX-NET ONNX session initialises and returns correct output there), and
    putting it on the NPU leaves the iGPU free for ASR, which is the entire point of a
    machine that has both. Running both on the iGPU serialises them for no gain.

    A previous edit made the NPU yield this slot to the GPU, on the theory that UVR on the
    NPU was behind seven failing accuracy tests. It was not -- those failures were the ASR
    pipeline, which the NPU genuinely cannot execute (see modules/core/device_probe.py).
    The change silently disabled the NPU's one working use.
    """
    if is_explicit_prep and state.get("prep_device") == "AMD":
        return
    state["prep_device"] = "NPU"


def _append_npu_node_fallback(
    max_npu: int,
    hardware_units: list[dict[str, str]],
    *,
    state: dict[str, str],
    is_explicit_prep: bool = False,
) -> None:
    if max_npu <= 0 or not os.path.exists("/dev/accel/accel0"):
        return
    hardware_units.append({"type": "NPU", "id": "NPU", "name": "Intel NPU"})
    _update_npu_state(state, is_explicit_prep)
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
    # If DRM vendor files are present, check that an Intel vendor is present or at least no non-Intel GPU claimed all slots
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
    """Return True only when a DRM render node reports Intel vendor id 0x8086."""
    drm_root = "/sys/class/drm"
    if not os.path.isdir(drm_root):
        return False
    vendor_files = _iter_drm_vendor_files(drm_root)
    if not vendor_files:
        return False
    for vendor_file in vendor_files:
        if _read_vendor_id(vendor_file) == "0x8086":
            return True
    return False


def _append_intel_units(
    core: Any,
    devices: list[str],
    max_gpu: int,
    max_npu: int,
    hardware_units: list[dict[str, str]],
    *,
    state: dict[str, str],
    is_explicit_prep: bool = False,
) -> tuple[int, int]:
    gpu_detect_count = 0
    npu_detect_count = 0
    for dev in devices:
        if "GPU" in dev:
            gpu_detect_count = _append_intel_gpu(
                core,
                dev,
                gpu_detect_count,
                max_gpu,
                hardware_units,
                state=state,
            )
        elif "NPU" in dev:
            npu_detect_count = _append_intel_npu(
                core,
                dev,
                npu_detect_count,
                max_npu,
                hardware_units,
                state=state,
                is_explicit_prep=is_explicit_prep,
            )
    return gpu_detect_count, npu_detect_count


#: Vendor names that rule out Intel silicon when no PCI vendor id is available.
_NON_INTEL_DEVICE_MARKERS = ("nvidia", "geforce", "quadro", "tesla", "amd", "radeon", "rx ")


def _is_intel_ov_device(core: Any, dev: str) -> bool:
    """Whether an OpenVINO device is actually Intel silicon.

    OpenVINO enumerates any GPU its installed plugins can see, not only Intel ones. On a
    host whose image also carries NVIDIA's OpenCL ICD it reports the NVIDIA card as plain
    "GPU":

        devices: ['CPU', 'GPU']
          GPU: NVIDIA GeForce RTX 5090 (dGPU)
             arch: GPU: vendor=0x10de arch=v12.0.0

    Registering that as an Intel unit is not harmless. With per-unit engines it is handed
    INTEL-WHISPER, which then cannot load ("ASR engine INTEL-WHISPER failed to load on
    NVIDIA GeForce RTX 5090 (dGPU)"), and every request routed to it returns 500 -- 31
    failures across the real-audio suite on the `full` image.

    DEVICE_ARCHITECTURE carries the PCI vendor id, which is the authoritative answer.
    The device name is only a fallback for plugins that do not expose the property.
    """
    try:
        architecture = str(core.get_property(dev, "DEVICE_ARCHITECTURE"))
        if "vendor=" in architecture:
            return "vendor=0x8086" in architecture
    except (RuntimeError, TypeError, ValueError, OSError, AttributeError) as e:
        # The property is optional: a plugin that does not implement it raises RuntimeError,
        # and a mocked or partially-initialised Core raises AttributeError or TypeError.
        # Any of those simply means "no vendor id available", which the name fallback below
        # is there to handle.
        logger.debug("Could not read DEVICE_ARCHITECTURE for %s: %s", dev, e)

    # No vendor id available. Fall back to the reported name, and only reject devices that
    # name another vendor outright -- an unreadable device is assumed Intel, which is the
    # behaviour that predates this check and is right for every plugin that does not
    # expose the property. The bug this guards against had the property and reported
    # vendor=0x10de, so the authoritative path above is the one that matters.
    name = str(_get_ov_device_name(core, dev)).lower()
    return not any(vendor in name for vendor in _NON_INTEL_DEVICE_MARKERS)


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
    if not _is_intel_ov_device(core, dev):
        _log_non_intel_device(core, dev)
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
    is_explicit_prep: bool = False,
) -> int:
    if npu_detect_count >= max_npu:
        return npu_detect_count
    if not _is_intel_ov_device(core, dev):
        _log_non_intel_device(core, dev)
        return npu_detect_count
    hardware_units.append({"type": "NPU", "id": dev, "name": _get_ov_device_name(core, dev)})
    if state["device"] == "CPU":
        state["device"] = "NPU"
    _claim_npu_prep_slot(state, is_explicit_prep)
    return npu_detect_count + 1


def _log_non_intel_device(core: Any, dev: str) -> None:
    """Say that an enumerated OpenVINO device was rejected for not being Intel silicon.

    Shared by the GPU and NPU arms, which had the same message and each read the device
    name a second time to build it -- a plugin property call per rejection, duplicated.
    """
    logger.info(
        "OpenVINO reports %s (%s), which is not Intel silicon; not adding it as an Intel unit.", dev, _get_ov_device_name(core, dev)
    )


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
