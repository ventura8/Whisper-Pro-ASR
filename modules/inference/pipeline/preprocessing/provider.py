"""Provider and OpenVINO resolution helpers for preprocessing."""

from modules.core import config
from modules.inference.pipeline import openvino_provider_dispatch, openvino_resolver

ProviderConfig = tuple[list[str], list[dict[str, object]]]


def _cpu_provider_config() -> ProviderConfig:
    """Return deterministic CPU fallback provider config."""
    return ["CPUExecutionProvider"], [{}]


def _has_intel_accelerator(available_openvino_devices: list[str]) -> bool:
    """Return whether OpenVINO reports an Intel NPU/GPU accelerator device."""
    return openvino_resolver.has_openvino_accelerator_device(available_openvino_devices)


def _first_visible_intel_accelerator(available_openvino_devices: list[str]) -> str:
    """Return the first Intel accelerator token in runtime discovery order."""
    for device in available_openvino_devices:
        family = openvino_resolver.openvino_device_family(device)
        if family in {"GPU", "NPU"}:
            return device
    return "GPU"


def build_openvino_retry_candidates(requested: str, available_devices: list[str]) -> list[str]:
    """Build retry candidates using runtime devices or heuristic fallback."""
    return openvino_resolver.get_openvino_retry_candidates(requested, available_devices)


def resolve_openvino_device_type_for_preprocessing(device_id: str, available_devices: list[str]) -> str:
    """Resolve OpenVINO device type for preprocessing.

    GPU requests should resolve to a concrete OpenVINO GPU device when one is visible,
    while NPU keeps the generic family token because ORT rejects dotted NPU device_type values.
    """
    requested = (device_id or "GPU").upper()
    family = openvino_resolver.openvino_device_family(requested)
    if family == "NPU" and _is_generic_family_request(requested, family):
        return family
    return _resolve_openvino_device_type_with_available_devices(requested, family, available_devices)


def _is_generic_family_request(requested: str, family: str | None) -> bool:
    return bool(family and requested == family)


def _resolve_openvino_device_type_with_available_devices(
    requested: str,
    family: str | None,
    available_devices: list[str],
) -> str:
    if available_devices:
        return openvino_resolver.find_matching_openvino_device(requested, available_devices)
    if family and "." not in requested:
        return family
    return requested


def openvino_provider_config_for_preprocessing(
    device_id: str,
    _ov_cache_dir: str,
    available_devices: list[str],
    _preprocess_threads: int,
) -> ProviderConfig:
    """Return OpenVINO provider config with runtime-resolved device type.

    ``_ov_cache_dir`` is accepted (not just for interface uniformity with sibling
    provider-config functions sharing this call shape) but deliberately unused:
    this function omits ``cache_dir`` for the accelerator families (GPU/NPU):
    a second same-process OpenVINO GPU session that reloads a previously
    compiled on-disk kernel cache has been observed to SIGSEGV on Intel iGPU
    hardware (reproduced identically on openvino 2026.2.1 and 2026.3.0, and
    regardless of whether a CUDA context is also active in the process — a
    driver-level issue, not something a Python-side fix can patch around).
    Skipping the on-disk cache means every session recompiles its kernels
    in-memory instead of risking a reload of a stale/incompatible cache
    blob, trading a few seconds of extra latency per session for not
    crashing the whole process.
    """
    resolved = resolve_openvino_device_type_for_preprocessing(device_id, available_devices)
    provider_options = openvino_resolver.normalize_openvino_provider_options(
        {
            "device_type": resolved,
            "num_streams": "1",
        }
    )

    return ["OpenVINOExecutionProvider", "CPUExecutionProvider"], [provider_options]


def _resolve_auto_amd(target_prep: str, available_providers: list[str]) -> ProviderConfig | None:
    if not openvino_provider_dispatch.has_amd_provider(available_providers):
        return None
    if target_prep == "AMD":
        return openvino_provider_dispatch.amd_provider_config("0", available_providers)
    if target_prep == "AUTO" and "CUDAExecutionProvider" not in available_providers:
        return openvino_provider_dispatch.amd_provider_config("0", available_providers)
    return None


def _resolve_auto_accelerator(
    available_providers: list[str],
    available_openvino_devices: list[str],
    ov_cache_dir: str,
    preprocess_threads: int,
) -> ProviderConfig | None:
    if "CUDAExecutionProvider" in available_providers:
        return openvino_provider_dispatch.cuda_or_cpu_provider_config("0", available_providers)

    if "OpenVINOExecutionProvider" in available_providers and _has_intel_accelerator(available_openvino_devices):
        return openvino_provider_config_for_preprocessing(
            _first_visible_intel_accelerator(available_openvino_devices),
            ov_cache_dir,
            available_openvino_devices,
            preprocess_threads,
        )
    return None


def auto_provider_config_for_preprocessing(
    available_providers: list[str],
    available_openvino_devices: list[str],
    ov_cache_dir: str,
    preprocess_threads: int,
    target_prep: str = "AUTO",
) -> ProviderConfig:
    """Resolve AUTO provider strategy for preprocessing."""
    target = (target_prep or "AUTO").upper()
    amd_cfg = _resolve_auto_amd(target, available_providers)
    if amd_cfg:
        return amd_cfg

    accel_cfg = _resolve_auto_accelerator(available_providers, available_openvino_devices, ov_cache_dir, preprocess_threads)
    return accel_cfg or _cpu_provider_config()


def _resolve_openvino_or_cpu(
    device_id: str,
    available_providers: list[str],
    available_openvino_devices: list[str],
    ov_cache_dir: str,
    preprocess_threads: int,
) -> ProviderConfig:
    """Resolve OpenVINO provider for Intel targets, otherwise return CPU fallback."""
    if "OpenVINOExecutionProvider" not in available_providers:
        return _cpu_provider_config()
    if openvino_resolver.is_openvino_family_disabled(device_id):
        return _cpu_provider_config()
    return openvino_provider_config_for_preprocessing(device_id, ov_cache_dir, available_openvino_devices, preprocess_threads)


def _resolve_auto(
    target_prep: str,
    available_providers: list[str],
    available_openvino_devices: list[str],
    ov_cache_dir: str,
    preprocess_threads: int,
) -> ProviderConfig:
    """Resolve AUTO provider selection strategy."""
    return auto_provider_config_for_preprocessing(
        available_providers,
        available_openvino_devices,
        ov_cache_dir,
        preprocess_threads,
        target_prep=target_prep,
    )


def _resolve_non_cuda_amd_preprocessing(
    normalized_type: str,
    device_id: str,
    available_providers: list[str],
    *,
    available_openvino_devices: list[str],
    ov_cache_dir: str,
    preprocess_threads: int,
) -> ProviderConfig:
    if normalized_type == "CPU":
        return _cpu_provider_config()
    if normalized_type in {"OPENVINO", "GPU", "NPU"}:
        return _resolve_openvino_or_cpu(device_id, available_providers, available_openvino_devices, ov_cache_dir, preprocess_threads)
    return _resolve_auto(
        normalized_type,
        available_providers,
        available_openvino_devices,
        ov_cache_dir,
        preprocess_threads,
    )


def resolve_provider_config_for_preprocessing(
    device_type: str,
    device_id: str,
    available_providers: list[str],
    available_openvino_devices: list[str],
    ov_cache_dir: str,
    *,
    preprocess_threads: int,
) -> ProviderConfig:
    """Resolve preprocessing providers using CUDA/AMD/OpenVINO/CPU dispatch rules."""
    normalized_type = (device_type or "AUTO").upper()

    if normalized_type == "CUDA":
        return openvino_provider_dispatch.cuda_or_cpu_provider_config(device_id, available_providers)

    if normalized_type == "AMD":
        result = openvino_provider_dispatch.amd_provider_config(device_id, available_providers)
    else:
        result = _resolve_non_cuda_amd_preprocessing(
            normalized_type,
            device_id,
            available_providers,
            available_openvino_devices=available_openvino_devices,
            ov_cache_dir=ov_cache_dir,
            preprocess_threads=preprocess_threads,
        )

    return _block_openvino_alongside_cuda(result, device_id, available_providers)


def _block_openvino_alongside_cuda(result: ProviderConfig, device_id: str, available_providers: list[str]) -> ProviderConfig:
    """Redirect to CUDA whenever ASR runs on CUDA and this resolution picked OpenVINO.

    Initializing an OpenVINO GPU/NPU (Level-Zero/OpenCL) context in the same
    process as an already-active CUDA context has been observed to crash/hang
    natively on Intel iGPU hardware. This is a last-resort safety net that
    catches every path that can select OpenVINOExecutionProvider (explicit
    GPU/NPU targeting, AUTO resolution, per-unit preprocessor pooling in
    modules/inference/runtime/model_manager.py) regardless of which one
    triggered it — do not rely on upstream device-selection alone to prevent
    this combination, it has been proven to reach here anyway. Substituting
    CUDA (same vendor as ASR) rather than CPU keeps preprocessing GPU-accelerated;
    cuda_or_cpu_provider_config already falls back to CPU on its own if CUDA
    isn't actually usable in onnxruntime.
    """
    providers, _options = result
    if config.DEVICE == "CUDA" and "OpenVINOExecutionProvider" in providers:
        return openvino_provider_dispatch.cuda_or_cpu_provider_config(device_id, available_providers)
    return result
