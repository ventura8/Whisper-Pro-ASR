"""Whether a unit's vocal separation can run in a spawned worker process.

Running UVR out-of-process is worth real acceleration -- on an Arc 140T the isolated
worker separates at 4.6-5.0x through OpenVINOExecutionProvider, against 0.25x when the
in-process manager falls back to the CPU provider. But it is not safe everywhere: on Intel
UHD Graphics the worker dies outright,

    WorkerError: UVR GPU worker died during 'separate'
                 (killed by signal 11 (SIGSEGV -- native crash in the vendor runtime))

and that crash is intermittent -- the first request succeeded and a concurrent pair killed
it, so a single green run proves nothing here.

The variable is the GPU generation, not the software: both machines run the newest
published onnxruntime-openvino (1.24.1) and openvino (2026.3.1), and there is nothing to
upgrade to. OpenVINO's own DEVICE_ARCHITECTURE separates them cleanly -- ``arch=v12.0.0``
on the UHD part that crashes, ``arch=v12.74.0`` on the Arc part that does not. Intel
numbers Alchemist/Xe-HPG from 12.55, which is the same "Arc (Alchemist) or newer" boundary
docker-compose.intel-xpu.yml already documents for the XPU torch path.
"""

from __future__ import annotations

import importlib
import logging
import re

from modules.core import config

logger = logging.getLogger(__name__)

#: Oldest Intel GPU architecture from which an OpenVINO ONNX session survives a spawn child.
MIN_ISOLATABLE_INTEL_GPU_ARCH = (12, 55)

#: The NPU keeps the in-process manager: isolated UVR has never been exercised on one, and
#: an untested guess here costs a segfault rather than a slow path.
ISOLATION_UNSUPPORTED_DEVICES = ("NPU",)


def parse_intel_gpu_arch(architecture: str) -> tuple[int, int] | None:
    """Return (major, minor) from an OpenVINO DEVICE_ARCHITECTURE string, or None."""
    match = re.search(r"arch=v(\d+)\.(\d+)", str(architecture))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


#: PCI vendor id for Intel, as it appears in DEVICE_ARCHITECTURE.
_INTEL_VENDOR_ID = "vendor=0x8086"


def intel_gpu_arch_for(device_id: str) -> tuple[int, int] | None:
    """Ask OpenVINO which Intel GPU generation ``device_id`` is. None when it cannot be read.

    The architecture is only trusted once the device is confirmed Intel. OpenVINO
    enumerates any GPU its installed plugins can see, so on a hybrid host its "GPU" may be
    the NVIDIA card -- observed on an RTX 3080 + Intel UHD laptop, where it reported
    ``vendor=0x10de arch=v8.6.0``, the NVIDIA compute capability. Reading a generation off
    that and comparing it against an Intel boundary is meaningless: it happened to land on
    the safe answer there, but nothing guaranteed it would.
    """
    try:
        ov = importlib.import_module("openvino")
        target = device_id if str(device_id).upper().startswith("GPU") else "GPU"
        architecture = str(ov.Core().get_property(target, "DEVICE_ARCHITECTURE"))
        if "vendor=" in architecture and _INTEL_VENDOR_ID not in architecture:
            logger.info(
                "[Preprocess] OpenVINO's %s is not Intel silicon (%s); its architecture says "
                "nothing about the Intel GPU, so vocal isolation stays in-process.",
                target,
                architecture,
            )
            return None
        return parse_intel_gpu_arch(architecture)
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        # Any failure here means "cannot tell", and the caller treats that as not
        # isolatable -- the safe answer, since guessing wrong costs a native crash.
        logger.debug("[Preprocess] Could not read Intel GPU architecture for %s: %s", device_id, exc)
        return None


def isolation_supported(assigned_unit) -> bool:
    """Whether this unit's preprocessing can run out-of-process.

    Intel GPUs are decided by generation rather than by device type: Arc and newer are
    stable in a worker and much faster there, older parts crash. Anything we cannot
    identify keeps the in-process manager, because the failure mode of guessing wrong is a
    native crash rather than a slow path.
    """
    device_type = str((assigned_unit or {}).get("type", config.PREPROCESS_DEVICE)).upper()
    if device_type in ISOLATION_UNSUPPORTED_DEVICES:
        return False
    if device_type != "GPU":
        return True
    return _intel_gpu_is_isolatable((assigned_unit or {}).get("id", "GPU"))


def _intel_gpu_is_isolatable(device_id: str) -> bool:
    """Whether this Intel GPU's generation survives an out-of-process worker.

    Arc (Alchemist) and newer are stable there and much faster; older parts take the worker
    down with a SIGSEGV. An unreadable generation is treated as too old, because the cost of
    guessing wrong is a native crash rather than a slow path.
    """
    arch = intel_gpu_arch_for(device_id)
    if arch is None:
        logger.info("[Preprocess] Intel GPU generation unknown; keeping vocal isolation in-process.")
        return False
    if arch < MIN_ISOLATABLE_INTEL_GPU_ARCH:
        logger.info(
            "[Preprocess] Intel GPU arch v%d.%d is below v%d.%d (Arc/Alchemist); keeping vocal "
            "isolation in-process, where the OpenVINO provider is stable.",
            arch[0],
            arch[1],
            *MIN_ISOLATABLE_INTEL_GPU_ARCH,
        )
        return False
    return True
