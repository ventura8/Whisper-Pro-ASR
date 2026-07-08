"""
Configuration Helper Utilities for Whisper Pro ASR
"""

import importlib
import logging
import os

logger = logging.getLogger(__name__)


def get_unit_limit(env_var, default=1, min_value=1):
    """Helper to parse hardware unit limits (supports int, ALL, AUTO)."""
    val = os.environ.get(env_var, str(default)).upper()
    if val in ["ALL", "AUTO"]:
        return 999  # Practically unlimited
    try:
        return max(min_value, int(val))
    except (ValueError, TypeError):
        return max(min_value, int(default))


def detect_hardware(max_cuda, max_gpu, max_npu, hardware_units) -> tuple[str, str, str]:
    """Detect acceleration hardware and returns (detected_device, detected_prep_device, detected_compute)."""
    detected_device = "CPU"
    detected_prep_device = "CPU"
    detected_compute = "int8"

    # 1. NVIDIA Acceleration Check
    cuda_count = 0
    try:
        _ct2 = importlib.import_module("ctranslate2")
        cuda_count = _ct2.get_cuda_device_count()
        if cuda_count > 0:
            logger.debug("Auto-detected %d NVIDIA GPU(s).", cuda_count)
            cuda_to_use = min(cuda_count, max_cuda)
            if cuda_to_use > 0:
                detected_device = "CUDA"
                detected_prep_device = "CUDA"
                detected_compute = "float16"
                for i in range(cuda_to_use):
                    hardware_units.append({"type": "CUDA", "id": f"cuda:{i}", "name": f"NVIDIA GPU {i}"})
    except (ImportError, AttributeError, ValueError, TypeError, RuntimeError, OSError) as e:
        logger.debug("CUDA detection skipped: %s", e)

    # 2. Intel Accelerator Check (OpenVINO)
    try:
        _ov = importlib.import_module("openvino")
        core = _ov.Core()
        devices = core.available_devices
        logger.debug("OpenVINO Available Devices: %s", devices)

        gpu_detect_count = 0
        npu_detect_count = 0

        for dev in devices:
            if "GPU" in dev:
                if gpu_detect_count >= max_gpu:
                    continue
                try:
                    dev_name = core.get_property(dev, "FULL_DEVICE_NAME")
                except (ImportError, AttributeError, ValueError, TypeError, RuntimeError, OSError):
                    dev_name = f"Intel {dev}"
                hardware_units.append({"type": "GPU", "id": dev, "name": dev_name})
                gpu_detect_count += 1
                if detected_device == "CPU":
                    detected_device = "GPU"
                if detected_prep_device in ("CPU", "CUDA"):
                    detected_prep_device = "GPU"
            elif "NPU" in dev:
                if npu_detect_count >= max_npu:
                    continue
                try:
                    dev_name = core.get_property(dev, "FULL_DEVICE_NAME")
                except (ImportError, AttributeError, ValueError, TypeError, RuntimeError, OSError):
                    dev_name = f"Intel {dev}"
                hardware_units.append({"type": "NPU", "id": dev, "name": dev_name})
                npu_detect_count += 1
                if detected_device == "CPU":
                    detected_device = "NPU"
                detected_prep_device = "NPU"

    except (ImportError, AttributeError, ValueError, TypeError, RuntimeError, OSError) as e:
        logger.debug("Intel accelerator detection skipped: %s", e)

    if not hardware_units:
        logger.info("No accelerators detected. Using Host CPU for all tasks.")
        hardware_units.append({"type": "CPU", "id": "CPU", "name": "Host CPU"})

    return detected_device, detected_prep_device, detected_compute


def resolve_thread_limits(requested_asr, requested_prep, cpu_core_limit, max_cpu, device):
    """Resolve and enforce physical hardware thread limits with priority."""
    cores = cpu_core_limit
    if max_cpu >= 999:
        return min(requested_asr, cores), min(requested_prep, cores)

    effective_pool = max(1, cpu_core_limit // max_cpu)
    asr_threads = min(requested_asr, effective_pool)
    prep_threads = min(requested_prep, cores if device != "CPU" else effective_pool)

    if asr_threads < requested_asr:
        logger.info(
            "[Config] Capping ASR_THREADS to %d (Global Limit: %d, Units: %d)", asr_threads, cpu_core_limit, max_cpu
        )
    if prep_threads < requested_prep and device != "CPU":
        logger.info("[Config] Capping ASR_PREPROCESS_THREADS to %d (Hardware limit)", cores)
    return asr_threads, prep_threads


def calculate_cpu_parallel_limit(max_cpu, cpu_core_limit, asr_threads, preprocess_threads):
    """Calculate how many multi-threaded CPU tasks can run safely."""
    if max_cpu < 999:
        return max_cpu

    cores = cpu_core_limit
    cores_per_task = max(1, asr_threads, preprocess_threads)
    limit = max(1, cores // cores_per_task)
    logger.info(
        "[Resource] Calculated AUTO CPU parallel limit: %d (Cores: %d, Threads/Task: %d)", limit, cores, cores_per_task
    )
    return limit
