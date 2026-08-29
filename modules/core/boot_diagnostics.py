"""Boot-time hardware diagnostics.

Split out of bootstrap so that each module stays inside the project's module-length gate.
These functions only observe and report: they probe the Intel runtime, the process security
context, the sysfs and device nodes, and the loaded ONNX Runtime, and write what they find
to the boot log. Nothing here selects a library or changes the path -- that stays in
bootstrap, and keeping the two apart makes it obvious which half can affect a boot.
"""

import importlib
import os
from glob import glob
from pathlib import Path

from modules.core.constants import INTEL_ENV_KEYS


def log_intel_runtime_diagnostics(boot_logger) -> None:
    """Log everything known about the Intel runtime, devices and access at boot."""
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

    # Either node is enough. Requiring both meant every Intel host WITHOUT an NPU -- which
    # is most of them, and every Arc or iGPU laptop -- was told its Intel runtime was
    # "likely misconfigured" while its GPU worked perfectly: /dev/accel/accel0 simply does
    # not exist there. A machine with no NPU is not a machine with a permissions problem.
    has_device_access = _any_intel_device_opened(drm_probe, accel_probe)
    if uid != 0 and not has_device_access:
        boot_logger.warning(
            "Intel runtime likely misconfigured: process uid=%s groups=%s and device access probe failed. "
            "Use docker compose with Intel override or ensure the container user is in the Intel device group for /dev/dri and /dev/accel.",
            uid,
            groups,
        )


def _any_intel_device_opened(drm_probe: str, accel_probe: str) -> bool:
    """Whether any Intel device node this host actually has could be opened.

    Requiring BOTH nodes meant every Intel host without an NPU -- most of them, and every
    Arc or iGPU laptop -- was told its runtime was "likely misconfigured" while its GPU
    worked perfectly, because /dev/accel/accel0 simply does not exist there. A machine with
    no NPU is not a machine with a permissions problem.
    """
    if drm_probe == "open_ok":
        return True
    return os.path.exists("/dev/accel/accel0") and accel_probe == "open_ok"


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
