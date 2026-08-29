"""
Bootstrap Logic for Hardware Path Patching
Ensures that the correct hardware-optimized libraries are injected into the
system path before any AI-related modules are loaded.
"""

import importlib
import logging
import os
import sys
from pathlib import Path

from modules.core import boot_diagnostics
from modules.core.config_helpers import has_amd_wsl_hardware


def initialize_hardware_path():
    """
    Core hardware detection and library path redirection.
    This MUST be called before importing any AI engines.
    """
    _ensure_wsl_library_path()
    boot_logger = _get_boot_logger()
    device = os.getenv("ASR_DEVICE", os.getenv("DEVICE", "cpu")).lower()
    preprocess_device = os.getenv("ASR_PREPROCESS_DEVICE", "auto").lower()
    is_intel_hw = _detect_intel_hardware(boot_logger)
    is_nvidia_hw = _detect_nvidia_hardware()
    is_amd_hw = _detect_amd_hardware()
    target_lib, context_reason = _resolve_target_library(device, preprocess_device, is_nvidia_hw, is_intel_hw, is_amd_hw)
    _activate_target_library(boot_logger, target_lib, context_reason)


def _ensure_wsl_library_path():
    # This only affects child-process environment inheritance. It cannot change
    # dynamic-library lookup for modules already loaded in the running Python
    # interpreter, so bootstrap still relies on early invocation and explicit
    # runtime path selection for in-process imports.
    if os.path.exists("/usr/lib/wsl/lib"):
        ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if "/usr/lib/wsl/lib" not in ld_path:
            os.environ["LD_LIBRARY_PATH"] = f"/usr/lib/wsl/lib:{ld_path}" if ld_path else "/usr/lib/wsl/lib"


def _get_boot_logger():
    boot_logger = logging.getLogger("Bootstrap")
    boot_logger.propagate = False
    if not boot_logger.handlers:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        boot_logger.addHandler(sh)
        boot_logger.setLevel(logging.INFO)
    return boot_logger


def _detect_intel_hardware(boot_logger) -> bool:
    try:
        ov = importlib.import_module("openvino")
        core = ov.Core()
        return any("GPU" in d or "NPU" in d for d in core.available_devices)
    except (ImportError, AttributeError, ValueError, OSError, KeyError, RuntimeError) as e:
        boot_logger.debug("OpenVINO hardware check failed: %s", e)
        return _detect_intel_linux_nodes()


def _detect_intel_linux_nodes() -> bool:
    if os.path.exists("/dev/accel") or os.path.exists("/dev/dxg"):
        return True
    return _has_intel_drm_vendor()


def _has_intel_drm_vendor() -> bool:
    for vendor_path in Path("/sys/class/drm").glob("card*/device/vendor"):
        try:
            if vendor_path.read_text(encoding="utf-8").strip().lower() == "0x8086":
                return True
        except OSError:
            continue
    return False


def _detect_amd_hardware() -> bool:
    if os.path.exists("/dev/kfd") or has_amd_wsl_hardware():
        return True
    return _has_amd_drm_vendor()


def _has_amd_drm_vendor() -> bool:
    for vendor_path in Path("/sys/class/drm").glob("card*/device/vendor"):
        try:
            if vendor_path.read_text(encoding="utf-8").strip().lower() == "0x1002":
                return True
        except OSError:
            continue
    return False


def _detect_nvidia_hardware() -> bool:
    if os.path.exists("/dev/nvidia0") or os.path.exists("/dev/nvidiactl") or os.path.exists("/dev/nvidia-uvm"):
        return True
    return _detect_nvidia_via_ctranslate2()


def _detect_nvidia_via_ctranslate2() -> bool:
    try:
        ct2 = importlib.import_module("ctranslate2")
        return ct2.get_cuda_device_count() > 0
    except (ImportError, AttributeError, ValueError, TypeError, RuntimeError, OSError):
        return False


def _resolve_target_library(
    device: str, preprocess_device: str, is_nvidia_hw: bool, is_intel_hw: bool, is_amd_hw: bool
) -> tuple[str | None, str]:
    # Order is the whole contract here: an explicitly named preprocess device outranks the
    # ASR device, and the dual-GPU case outranks either vendor's own check. Expressed as a
    # sequence so the precedence is one readable list rather than six paired branches.
    # An explicit Intel preprocess device is only reachable when UVR does not have to share
    # this interpreter with a CUDA ASR context -- see _intel_preprocess_is_reachable.
    effective_preprocess = preprocess_device if _intel_preprocess_is_reachable(device, preprocess_device, is_nvidia_hw) else "auto"
    candidates = (
        lambda: _check_explicit_preprocess_library(effective_preprocess, is_nvidia_hw, is_intel_hw, is_amd_hw),
        lambda: _check_dual_gpu_path(device, preprocess_device, is_nvidia_hw, is_amd_hw),
        lambda: _check_nvidia_library(device, is_nvidia_hw),
        lambda: _check_amd_library(device, preprocess_device, is_amd_hw),
        lambda: _check_intel_library(device, preprocess_device, is_intel_hw),
    )
    for check in candidates:
        resolved = check()
        if resolved:
            return resolved
    return _check_cpu_library()


def _intel_preprocess_is_reachable(device: str, preprocess_device: str, is_nvidia_hw: bool) -> bool:
    """Whether an explicitly requested Intel preprocess device can actually be served.

    It cannot when ASR runs on CUDA *in this same process*. CTranslate2 binds CUDA directly,
    and an OpenVINO GPU/NPU context alongside an active CUDA one has been observed to crash
    natively -- which is why provider.py blocks the combination. Honouring the request here
    anyway loaded the Intel ONNX Runtime, and that build has no CUDA provider for the block
    to substitute, so UVR silently landed on the CPU: 0.48x against 2.40x on CUDA, on an
    RTX 3080 + UHD Graphics laptop, with every log line still naming a GPU.

    Loading the NVIDIA runtime instead is the honest outcome for that host: the request
    cannot be served, and CUDA UVR is what it will actually get.

    ``ASR_ISOLATE_PREPROCESSING`` is deliberately NOT consulted, and that is the whole point
    of this function. Isolation would remove the conflict -- UVR in its own process holds no
    CUDA context -- but the environment variable is a *request*, and isolation_policy refuses
    it on pre-Arc Intel graphics because the worker segfaults there. Trusting the request
    loaded the Intel runtime on a host that then ran UVR in-process anyway, with no CUDA
    provider to fall back to. Measured on an RTX 3080 + UHD Graphics laptop, 2026-09-08, with
    ASR_PREPROCESS_DEVICE=GPU and isolation at its default (on): UVR 0.90x and the dashboard
    correctly reporting CPU, against 8.34x once the CUDA runtime is loaded instead -- repeated
    at 8.21x on a later run. An earlier note here read 2.6x, which came from a container with
    ASR_ISOLATE_PREPROCESSING=0 and so described a path almost nobody runs. The
    accuracy suite passed either way -- a CPU fallback transcribes perfectly -- so only the
    provider and the speed showed it.

    The cost is a CUDA+Arc host, where isolation genuinely is supported and OpenVINO UVR
    would be viable: it gets CUDA UVR instead. No such machine exists in this fleet to
    measure, and a fast CUDA path beats an unverified one that degrades to CPU when the guess
    is wrong.
    """
    if not _is_intel_preprocess_request(preprocess_device):
        return True
    return not (is_nvidia_hw and _asr_would_use_cuda(device))


def _is_intel_preprocess_request(preprocess_device: str) -> bool:
    """Whether this names an Intel device rather than AUTO, CPU or another vendor."""
    return (preprocess_device or "").strip().lower() in ("intel", "openvino", "gpu", "npu")


def _asr_would_use_cuda(device: str) -> bool:
    """Whether ASR will hold a CUDA context in this process."""
    normalized = (device or "").strip().lower()
    return normalized in ("auto", "cuda") or normalized.startswith("cuda:") or normalized.startswith("nvidia")


def _check_explicit_preprocess_library(
    preprocess_device: str, is_nvidia_hw: bool, is_intel_hw: bool, is_amd_hw: bool
) -> tuple[str, str] | None:
    """Resolve from an explicitly named preprocess device, ahead of the ASR device.

    ONNX Runtime is used by UVR, VAD and hardware detection -- not by any ASR engine.
    FASTER-WHISPER is CTranslate2, INTEL-WHISPER is OpenVINO GenAI, OPENAI-WHISPER and
    WHISPERX are torch, and each brings its own runtime. So when the operator names a
    preprocess device, that device decides which ONNX variant has to be importable.

    Without this, the NVIDIA check ran first and matched on ASR_DEVICE alone (it was the
    only vendor check not given preprocess_device at all). On a hybrid NVIDIA+Intel host
    with ASR_DEVICE=AUTO, that loaded onnxruntime-gpu, which carries no OpenVINO provider,
    so ASR_PREPROCESS_DEVICE=GPU ran UVR through CPUExecutionProvider while the logs named
    the iGPU -- 0.48x against 2.40x on CUDA, five times slower while claiming acceleration.

    Only an explicit request is honoured here. AUTO falls through to the original
    ASR-device-led order, so nothing changes for hosts that do not ask for anything.
    """
    normalized = (preprocess_device or "").strip().lower()
    if normalized in ("", "auto"):
        return None
    if normalized == "cpu":
        return "/app/libs/cpu", "CPU"
    return _explicit_vendor_library(preprocess_device, is_nvidia_hw, is_intel_hw, is_amd_hw)


def _explicit_vendor_library(preprocess_device: str, is_nvidia_hw: bool, is_intel_hw: bool, is_amd_hw: bool) -> tuple[str, str] | None:
    """The vendor runtime an explicitly named, non-CPU preprocess device asks for.

    Intel first, then AMD, then NVIDIA -- the same precedence the paired branches had, kept
    as an ordered table so the order is visible in one place rather than implied by control
    flow. None when the named device does not match any vendor present on this host.
    """
    vendors = (
        (_should_use_intel_path(preprocess_device, is_intel_hw), "/app/libs/intel", "Intel OpenVINO"),
        (_should_use_amd_path(preprocess_device, is_amd_hw), "/app/libs/amd", "AMD ROCm"),
        (_should_use_nvidia_path(preprocess_device, is_nvidia_hw), "/app/libs/nvidia", "NVIDIA CUDA"),
    )
    return next(((path, label) for matched, path, label in vendors if matched), None)


def _is_auto_device_pair(device: str, preprocess_device: str) -> bool:
    return device.lower() == "auto" and preprocess_device.lower() == "auto"


def _check_dual_gpu_path(device: str, preprocess_device: str, is_nvidia_hw: bool, is_amd_hw: bool) -> tuple[str, str] | None:
    """Load AMD ONNX libs when both NVIDIA and AMD GPUs are present.

    CTranslate2 still binds CUDA directly for ASR. This override is only for
    AUTO resolution so explicit CUDA / AMD selections still follow the normal
    provider-specific path checks below.
    """
    if not _is_auto_device_pair(device, preprocess_device):
        return None
    if not is_nvidia_hw or not is_amd_hw:
        return None
    if not os.path.exists("/app/libs/amd"):
        return None
    return "/app/libs/amd", "AMD ROCm"


def _check_nvidia_library(device: str, is_nvidia_hw: bool) -> tuple[str, str] | None:
    if _should_use_nvidia_path(device, is_nvidia_hw):
        return "/app/libs/nvidia", "NVIDIA CUDA"
    return None


def _check_amd_library(device: str, preprocess_device: str, is_amd_hw: bool) -> tuple[str, str] | None:
    if _should_use_amd_path(device, is_amd_hw) or _should_use_amd_path(preprocess_device, is_amd_hw):
        return "/app/libs/amd", "AMD ROCm"
    return None


def _check_intel_library(device: str, preprocess_device: str, is_intel_hw: bool) -> tuple[str, str] | None:
    if _should_use_intel_path(device, is_intel_hw) or _should_use_intel_path(preprocess_device, is_intel_hw):
        return "/app/libs/intel", "Intel OpenVINO"
    return None


def _check_cpu_library() -> tuple[str | None, str]:
    if os.path.exists("/app/libs/cpu"):
        return "/app/libs/cpu", "CPU Runtime"
    return None, "Default"


def _is_explicit_amd_device(normalized: str) -> bool:
    if normalized in {"amd", "rocm"} or normalized.startswith("amd"):
        return True
    return sys.platform == "win32" and normalized in {"dml", "directml"}


def _should_use_amd_path(device: str, is_amd_hw: bool) -> bool:
    if not os.path.exists("/app/libs/amd"):
        return False
    normalized = device.lower()
    return _is_explicit_amd_device(normalized) or (normalized == "auto" and is_amd_hw)


def _should_use_nvidia_path(device: str, is_nvidia_hw: bool) -> bool:
    # Per-vendor images may ship without this runtime; an explicit ASR_DEVICE=CUDA must
    # not claim a path that does not exist, or resolution returns before the CPU fallback.
    if not os.path.exists("/app/libs/nvidia"):
        return False
    normalized = device.lower()
    return _is_explicit_nvidia_device(normalized) or _can_use_auto_nvidia_path(normalized, is_nvidia_hw)


def _is_explicit_nvidia_device(normalized_device: str) -> bool:
    return normalized_device == "cuda" or normalized_device.startswith("cuda:") or normalized_device.startswith("nvidia")


def _can_use_auto_nvidia_path(normalized_device: str, is_nvidia_hw: bool) -> bool:
    return normalized_device == "auto" and is_nvidia_hw and os.path.exists("/app/libs/nvidia")


def _is_explicit_intel_device(normalized_device: str) -> bool:
    return normalized_device in {"intel", "gpu", "npu"} or normalized_device.startswith("intel")


def _should_use_intel_path(device: str, is_intel_hw: bool) -> bool:
    # Same guard as the NVIDIA path above: explicit intel/gpu/npu must not resolve to a
    # runtime the image does not carry.
    if not os.path.exists("/app/libs/intel"):
        return False
    normalized = device.lower()
    return _is_explicit_intel_device(normalized) or (normalized == "auto" and is_intel_hw)


def _is_valid_target_lib(target_lib: str | None) -> bool:
    if not target_lib:
        return False
    return os.path.exists(target_lib)


def _fallback_to_cpu_library(boot_logger, target_lib: str | None) -> tuple[str | None, str]:
    """Substitute the CPU runtime for an unavailable vendor library.

    Defense in depth: the image build uninstalls the global onnxruntime, so returning
    nothing here would leave sys.path with no ONNX Runtime at all.
    """
    fallback_lib, fallback_reason = _check_cpu_library()
    if not _is_valid_target_lib(fallback_lib):
        return None, fallback_reason
    boot_logger.warning(
        "ONNX runtime path %s is unavailable in this image; falling back to %s",
        target_lib,
        fallback_reason,
    )
    return fallback_lib, fallback_reason


def _activate_target_library(boot_logger, target_lib: str | None, context_reason: str):
    if not _is_valid_target_lib(target_lib):
        target_lib, context_reason = _fallback_to_cpu_library(boot_logger, target_lib)
        if not _is_valid_target_lib(target_lib):
            return
    _prepend_to_sys_path(boot_logger, target_lib, context_reason)
    _reimport_onnxruntime(boot_logger, target_lib)
    if context_reason == "Intel OpenVINO":
        boot_diagnostics.log_intel_runtime_diagnostics(boot_logger)


def _prepend_to_sys_path(boot_logger, target_lib: str, context_reason: str) -> None:
    """Put the chosen vendor directory ahead of site-packages, once."""
    if target_lib in sys.path:
        return
    sys.path.insert(0, target_lib)
    boot_logger.info("Context: %s -> Path: %s", context_reason, target_lib)


def _reimport_onnxruntime(boot_logger, target_lib: str) -> None:
    """Drop any already-imported onnxruntime so the next import resolves to ``target_lib``.

    Without evicting the module, an onnxruntime imported before the path was adjusted stays
    in sys.modules and keeps serving the wrong vendor's build for the life of the process --
    which is the whole failure this bootstrap exists to prevent.
    """
    importlib.invalidate_caches()
    # The whole namespace, not just the top-level name. onnxruntime.capi holds the native
    # extension, and leaving those submodules behind meant the "fresh" import rebound the
    # package while every provider still came from the previous vendor's .so -- the exact
    # cross-vendor mixture this bootstrap exists to prevent, made invisible by a top-level
    # name that looked correctly reloaded.
    for name in [m for m in sys.modules if m == "onnxruntime" or m.startswith("onnxruntime.")]:
        del sys.modules[name]
    _log_onnxruntime_load(boot_logger, target_lib)


def _log_onnxruntime_load(boot_logger, target_lib: str):
    try:
        ort = importlib.import_module("onnxruntime")
        boot_logger.info("Successfully loaded ONNX %s from %s", ort.__version__, target_lib)
    except (ImportError, AttributeError, ValueError, OSError, RuntimeError) as e:
        boot_logger.warning("Failed to verify ONNX load: %s", e)


# CRITICAL: Auto-initialize on import to satisfy PEP8/Pylint order
initialize_hardware_path()
