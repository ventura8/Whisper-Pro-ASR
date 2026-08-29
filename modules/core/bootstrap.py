"""
Bootstrap Logic for Hardware Path Patching
Ensures that the correct hardware-optimized libraries are injected into the
system path before any AI-related modules are loaded.
"""

import importlib
import logging
import os
import sys
from glob import glob
from pathlib import Path

from modules.core.config_helpers import has_amd_wsl_hardware
from modules.core.constants import INTEL_ENV_KEYS


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
    candidates = (
        lambda: _check_explicit_preprocess_library(preprocess_device, is_nvidia_hw, is_intel_hw, is_amd_hw),
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
        _log_intel_runtime_diagnostics(boot_logger)


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


def _log_intel_runtime_diagnostics(boot_logger) -> None:
    _log_onnxruntime_details(boot_logger)
    _log_openvino_details(boot_logger)
    _log_optional_openvino_target_probe(boot_logger)
    boot_logger.debug(
        "Intel device nodes: /dev/accel/accel0=%s /dev/dri=%s /opt/intel/openvino=%s",
        os.path.exists("/dev/accel/accel0"),
        os.path.exists("/dev/dri"),
        os.path.exists("/opt/intel/openvino"),
    )
    _log_process_security_context(boot_logger)
    _log_intel_runtime_env(boot_logger)
    _log_intel_node_details(boot_logger)
    _log_intel_sysfs_details(boot_logger)
    _log_intel_access_diagnostics(boot_logger)


def _log_openvino_target_probe(boot_logger) -> None:
    try:
        ov = importlib.import_module("openvino")
        core = ov.Core()
        probes: list[str] = []
        for target in ("GPU", "GPU.0", "NPU", "NPU.0"):
            try:
                full_name = core.get_property(target, "FULL_DEVICE_NAME")
                probes.append(f"{target}=ok name={full_name}")
            except (AttributeError, ValueError, TypeError, RuntimeError, OSError) as e:
                probes.append(f"{target}=unavailable error={e}")
        boot_logger.info("OpenVINO target probe: %s", " | ".join(probes))
    except (ImportError, AttributeError, ValueError, TypeError, RuntimeError, OSError) as e:
        boot_logger.warning("OpenVINO target probe failed: %s", e)


def _log_optional_openvino_target_probe(boot_logger) -> None:
    probe_flag = os.getenv("INTEL_DEEP_OV_PROBE", "false").strip().lower() in {"1", "true", "yes", "on"}
    if not probe_flag:
        boot_logger.debug("OpenVINO target probe disabled (set INTEL_DEEP_OV_PROBE=true to enable GPU/NPU FULL_DEVICE_NAME probes)")
        return
    _log_openvino_target_probe(boot_logger)


def _log_process_security_context(boot_logger) -> None:
    uid, gid, groups = _get_process_identity()
    cap_eff, seccomp_mode, seccomp_filters = _read_process_security_status()

    boot_logger.debug(
        "Intel process security: uid=%s gid=%s groups=%s CapEff=%s Seccomp=%s SeccompFilters=%s",
        uid,
        gid,
        groups,
        cap_eff,
        seccomp_mode,
        seccomp_filters,
    )


def _get_process_identity() -> tuple[int, int, list[int]]:
    try:
        return os.getuid(), os.getgid(), os.getgroups()
    except (AttributeError, OSError):
        return -1, -1, []


def _read_process_security_status() -> tuple[str, str, str]:
    cap_eff = "unknown"
    seccomp_mode = "unknown"
    seccomp_filters = "unknown"

    for key, value in _iter_proc_status_entries():
        if key == "CapEff":
            cap_eff = value
        elif key == "Seccomp":
            seccomp_mode = value
        elif key == "Seccomp_filters":
            seccomp_filters = value

    return cap_eff, seccomp_mode, seccomp_filters


def _iter_proc_status_entries() -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    try:
        with open("/proc/self/status", encoding="utf-8") as status:
            for line in status:
                key, value = _parse_proc_status_line(line)
                if key:
                    entries.append((key, value))
    except OSError:
        return []
    return entries


def _parse_proc_status_line(line: str) -> tuple[str, str]:
    if ":" not in line:
        return "", ""
    key, value = line.split(":", 1)
    return key.strip(), value.strip()


def _log_intel_runtime_env(boot_logger) -> None:
    values = [f"{k}={os.getenv(k, '')}" for k in INTEL_ENV_KEYS]
    boot_logger.debug("Intel runtime env: %s", " | ".join(values))


def _log_intel_node_details(boot_logger) -> None:
    for label, pattern in (("accel", "/dev/accel/*"), ("drm-render", "/dev/dri/renderD*")):
        matches = sorted(glob(pattern))
        details: list[str] = []
        for node in matches:
            node_path = Path(node)
            try:
                st = node_path.stat()
                details.append(f"{node_path} mode={oct(st.st_mode & 0o777)} uid={st.st_uid} gid={st.st_gid}")
            except OSError as e:
                details.append(f"{node_path} stat_error={e}")
        if details:
            boot_logger.debug("Intel node details [%s]: %s", label, " | ".join(details))
        else:
            boot_logger.debug("Intel node details [%s]: none", label)


def _read_sysfs_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return "n/a"


def _log_intel_sysfs_details(boot_logger) -> None:
    _log_sysfs_class_nodes(boot_logger, "drm-render", "/sys/class/drm/renderD*/device")
    _log_sysfs_class_nodes(boot_logger, "accel", "/sys/class/accel/accel*/device")


def _log_sysfs_class_nodes(boot_logger, label: str, pattern: str) -> None:
    details: list[str] = []
    for device_path_str in sorted(glob(pattern)):
        device_path = Path(device_path_str)
        driver_link = device_path / "driver"
        driver_name = "n/a"
        if driver_link.exists():
            try:
                driver_name = driver_link.resolve().name
            except OSError:
                driver_name = "unresolved"
        details.append(
            (
                f"{device_path} vendor={_read_sysfs_file(device_path / 'vendor')} "
                f"device={_read_sysfs_file(device_path / 'device')} driver={driver_name}"
            )
        )

    if details:
        boot_logger.debug("Intel sysfs [%s]: %s", label, " | ".join(details))
    else:
        boot_logger.debug("Intel sysfs [%s]: none", label)


def _device_open_probe(path: str) -> str:
    if not os.path.exists(path):
        return "missing"
    try:
        fd = os.open(path, os.O_RDWR | os.O_CLOEXEC)
    except OSError as e:
        return f"open_failed errno={e.errno} msg={e}"
    try:
        return "open_ok"
    finally:
        os.close(fd)


def _log_intel_access_diagnostics(boot_logger) -> None:
    drm_probe = _device_open_probe("/dev/dri/renderD128")
    accel_probe = _device_open_probe("/dev/accel/accel0")

    boot_logger.debug(
        "Intel device open probe: /dev/dri/renderD128=%s | /dev/accel/accel0=%s",
        drm_probe,
        accel_probe,
    )

    try:
        uid = os.getuid()
        groups = os.getgroups()
    except (AttributeError, OSError):
        uid = -1
        groups = []

    has_device_access = drm_probe == "open_ok" and accel_probe == "open_ok"
    if uid != 0 and not has_device_access:
        boot_logger.warning(
            "Intel runtime likely misconfigured: process uid=%s groups=%s and device access probe failed. "
            "Use docker compose with Intel override or ensure the container user is in the Intel device group for /dev/dri and /dev/accel.",
            uid,
            groups,
        )


def _log_onnxruntime_details(boot_logger) -> None:
    try:
        ort = importlib.import_module("onnxruntime")
        boot_logger.debug(
            "Intel ONNX Runtime diagnostics: file=%s providers=%s",
            getattr(ort, "__file__", "unknown"),
            _safe_ort_providers(ort),
        )
    except (ImportError, AttributeError, ValueError, OSError, RuntimeError) as e:
        boot_logger.warning("Intel ONNX Runtime diagnostics failed: %s", e)


def _safe_ort_providers(ort) -> list[str]:
    try:
        return list(ort.get_available_providers())
    except (AttributeError, TypeError, ValueError, RuntimeError, OSError):
        return []


def _log_openvino_details(boot_logger) -> None:
    try:
        ov = importlib.import_module("openvino")
        core = ov.Core()
        boot_logger.debug(
            "OpenVINO diagnostics: version=%s file=%s devices=%s INTEL_OPENVINO_DIR=%s",
            getattr(ov, "__version__", "unknown"),
            getattr(ov, "__file__", "unknown"),
            list(core.available_devices),
            os.getenv("INTEL_OPENVINO_DIR", ""),
        )
    except (ImportError, AttributeError, ValueError, TypeError, RuntimeError, OSError) as e:
        boot_logger.warning("OpenVINO diagnostics failed: %s", e)


# CRITICAL: Auto-initialize on import to satisfy PEP8/Pylint order
initialize_hardware_path()
