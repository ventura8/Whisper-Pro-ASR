"""OpenVINO provider configuration and dispatch helpers."""

from __future__ import annotations

import importlib
import logging
import os

from modules.core import config as core_config

logger = logging.getLogger(__name__)

ProviderConfig = tuple[list[str], list[dict[str, object]]]


def _normalize_openvino_device_type(device_id: str) -> str:
    """Normalize any device id to an OpenVINO-compatible target name."""
    normalized = (device_id or "").strip().lower()
    if normalized in {"", "cpu", "openvino_cpu", "openvino"}:
        return "CPU"
    for prefix, replacement in (("gpu", "GPU"), ("openvino_gpu", "GPU"), ("npu", "NPU"), ("openvino_npu", "NPU")):
        normalized_family = _normalize_openvino_family_prefix(normalized, prefix, replacement)
        if normalized_family is not None:
            return normalized_family
    return device_id.upper()


def _normalize_openvino_family_prefix(normalized: str, prefix: str, replacement: str) -> str | None:
    if not normalized.startswith(prefix):
        return None
    suffix = normalized.removeprefix(prefix)
    return f"{replacement}{suffix}" if suffix else replacement


def _normalize_openvino_provider_options(options: dict | None) -> dict:
    """Return ONNX Runtime-compatible provider options."""
    return {str(key): str(value) for key, value in dict(options or {}).items() if value is not None}


def _cpu_provider_config() -> ProviderConfig:
    return ["CPUExecutionProvider"], [{}]


def _resolve_openvino_resolver():
    return importlib.import_module("modules.inference.pipeline.openvino_resolver")


def _safe_int(val: str, default: int | str) -> int | str:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _parse_cuda_device_id(device_id: str) -> int | str:
    raw = str(device_id or "0")
    lowered = raw.lower()
    if lowered == "cuda":
        return "0"
    for prefix in ("cuda:", "nvidia:"):
        if lowered.startswith(prefix):
            return _safe_int(lowered.split(prefix, 1)[1], default="0")
    return _safe_int(raw, default="0")


def cuda_provider_config(device_id: str) -> ProviderConfig:
    """Return CUDA provider config with CPU fallback semantics."""
    parsed = _parse_cuda_device_id(device_id)
    return ["CUDAExecutionProvider", "CPUExecutionProvider"], [{"device_id": parsed}]


def openvino_provider_config(device_id: str) -> ProviderConfig:
    """Return OpenVINO provider config with CPU fallback semantics."""
    resolved = _normalize_openvino_device_type(device_id)
    provider_options = _normalize_openvino_provider_options(
        {
            "device_type": resolved,
            "cache_dir": str(core_config.OV_CACHE_DIR),
            "num_streams": str(max(1, int(core_config.PREPROCESS_THREADS))),
        }
    )
    return ["OpenVINOExecutionProvider", "CPUExecutionProvider"], [provider_options]


def _get_amd_candidates() -> tuple[str, ...]:
    if os.path.exists("/dev/kfd"):
        return ("MIGraphXExecutionProvider", "ROCMExecutionProvider", "DmlExecutionProvider")
    return ("DmlExecutionProvider",)


def _get_amd_provider_order(available: list[str]) -> tuple[str, ...]:
    """Return supported AMD provider preferences.

    onnxruntime-rocm calls hipGetDeviceProperties() which throws an uncatchable C++ terminate() crash
    (HIP failure 100: no ROCm-capable device is detected) when native Linux ROCm driver node /dev/kfd is absent.
    Therefore, ROCMExecutionProvider is strictly enabled only when /dev/kfd is present.
    """
    if not _has_amd_device_node():
        return ()

    avail_set = set(available or [])
    return tuple(p for p in _get_amd_candidates() if p in avail_set)


def _resolve_amd_provider_name(available: list[str]) -> str | None:
    providers = available or []
    for prov in _get_amd_provider_order(providers):
        if prov in providers:
            return prov
    return None


def _is_wsl_missing_rocdxg(kfd_present: bool, dxg_present: bool, rocdxg_present: bool) -> bool:
    return not kfd_present and dxg_present and not rocdxg_present


def _resolve_amd_fallback_message(kfd_present: bool, dxg_present: bool, rocdxg_present: bool) -> str:
    if _is_wsl_missing_rocdxg(kfd_present, dxg_present, rocdxg_present):
        return (
            "[UVR] AMD GPU fallback to CPU: /dev/dxg (WSL2) is present but librocdxg.so is "
            "missing from /opt/rocm/lib/. librocdxg is required to bridge ROCm to /dev/dxg "
            "on WSL2 hosts without /dev/kfd. (available ORT providers: %s)"
        )
    if not kfd_present and not dxg_present:
        return (
            "[UVR] AMD GPU fallback to CPU: Neither /dev/kfd (ROCm bare-metal) nor /dev/dxg "
            "(WSL2 DXG bridge) were found. (available ORT providers: %s)"
        )
    return (
        "[UVR] AMD GPU fallback to CPU: No supported AMD execution provider found in "
        "ONNX Runtime. Available providers: %s. "
        "Expected MIGraphXExecutionProvider, ROCMExecutionProvider, or DmlExecutionProvider."
    )


def _log_amd_cpu_fallback_reason(available: list[str]) -> None:
    """Log the specific reason no AMD GPU provider could be selected."""
    fmt = _resolve_amd_fallback_message(
        os.path.exists("/dev/kfd"),
        os.path.exists("/dev/dxg"),
        os.path.exists("/opt/rocm/lib/librocdxg.so"),
    )
    logger.warning(fmt, available)


def amd_provider_config(device_id: str, available: list[str]) -> ProviderConfig:
    """Return AMD ROCm or DirectML provider config with CPU fallback."""
    raw = str(device_id or "0")
    lowered = raw.lower()
    idx_token = lowered.split("amd:", 1)[1] if lowered.startswith("amd:") else raw
    device_index = _safe_int(idx_token, default=0)
    provider_name = _resolve_amd_provider_name(available)
    if provider_name:
        logger.info(
            "[UVR] AMD GPU provider selected: %s (device_id=%s)",
            provider_name,
            device_index,
        )
        return [provider_name, "CPUExecutionProvider"], [{"device_id": device_index}]
    _log_amd_cpu_fallback_reason(available)
    return _cpu_provider_config()


def cpu_provider_config() -> ProviderConfig:
    """Return CPU-only provider config."""
    return _cpu_provider_config()


def _has_amd_device_node() -> bool:
    """Return True only if AMD GPU device nodes are mounted in docker container."""
    return os.path.exists("/dev/kfd") or (os.path.exists("/dev/dxg") and os.path.exists("/opt/rocm/lib/librocdxg.so"))


def _has_amd_provider(providers: list[str]) -> bool:
    return len(_get_amd_provider_order(providers)) > 0


# Public aliases used by sibling modules (avoids protected-access lint warnings)
has_amd_device_node = _has_amd_device_node
has_amd_provider = _has_amd_provider


def _has_openvino_accelerator(providers: list[str]) -> bool:
    if "OpenVINOExecutionProvider" not in providers:
        return False
    return _resolve_openvino_resolver().has_openvino_accelerator_device()


def auto_provider_config(available):
    """Resolve AUTO mode provider config from available runtime providers."""
    providers = list(available or [])
    if "CUDAExecutionProvider" in providers:
        return cuda_or_cpu_provider_config("0", providers)
    if _has_amd_provider(providers):
        return amd_provider_config("0", providers)
    if _has_openvino_accelerator(providers):
        return openvino_or_cpu_provider_config("GPU", providers)
    return cpu_provider_config()


def cuda_or_cpu_provider_config(device_id: str, available):
    """Resolve CUDA provider config with deterministic CPU fallback."""
    if "CUDAExecutionProvider" not in (available or []):
        return cpu_provider_config()
    return cuda_provider_config(device_id)


def openvino_or_cpu_provider_config(device_id: str, available):
    """Resolve OpenVINO provider config with deterministic CPU fallback."""
    resolver = _resolve_openvino_resolver()
    if "OpenVINOExecutionProvider" not in (available or []):
        return cpu_provider_config()
    if resolver.is_openvino_family_disabled(device_id):
        return cpu_provider_config()
    return openvino_provider_config(device_id)


_DISPATCH_MAP = {
    "CUDA": cuda_or_cpu_provider_config,
    "AMD": amd_provider_config,
    "OPENVINO": openvino_or_cpu_provider_config,
    "GPU": openvino_or_cpu_provider_config,
    "NPU": openvino_or_cpu_provider_config,
}


def provider_config_dispatch(device_type: str):
    """Return provider config resolver for the requested execution class."""
    normalized = (device_type or "AUTO").upper()
    if normalized in _DISPATCH_MAP:
        return _DISPATCH_MAP[normalized]
    if normalized == "CPU":
        return lambda _device_id, _available: cpu_provider_config()
    return lambda _device_id, available: auto_provider_config(available)


def resolve_provider_config(
    device_type: str,
    device_id: str,
    available,
):
    """Resolve provider config for requested execution class and target device id."""
    resolver = provider_config_dispatch(device_type)
    return resolver(device_id, available)
