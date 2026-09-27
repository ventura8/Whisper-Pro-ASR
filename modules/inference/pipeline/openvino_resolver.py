"""OpenVINO Execution Provider resolution, device matching, and patching utilities."""

import importlib
import json
import logging
import os
import sys
import threading

from modules.core import utils

logger = logging.getLogger(__name__)

_OPENVINO_DISABLED_FAMILIES: set[str] = set()
_OPENVINO_DISABLED_LOCK = threading.Lock()


def purge_onnxruntime_modules() -> None:
    """Purge imported ONNX modules from sys.modules to force a clean reload."""
    keys_to_remove = [k for k in sys.modules if k.startswith("onnxruntime")]
    for k in keys_to_remove:
        del sys.modules[k]


def _prepend_intel_path(path: str) -> None:
    """Move Intel runtime path to the front of sys.path deterministically."""
    sys.path = [entry for entry in sys.path if entry != path]
    sys.path.insert(0, path)


def _reload_openvino_capable_onnxruntime() -> bool:
    """Reload onnxruntime and verify OpenVINOExecutionProvider availability."""
    purge_onnxruntime_modules()
    ort = importlib.import_module("onnxruntime")
    return "OpenVINOExecutionProvider" in ort.get_available_providers()


def reload_onnxruntime_from_intel_path() -> bool:
    """Force ONNX Runtime to load from the Intel OpenVINO package path."""
    clear_openvino_disabled_families()
    intel_path = "/app/libs/intel"
    if not os.path.exists(intel_path):
        return False
    _prepend_intel_path(intel_path)
    try:
        return _reload_openvino_capable_onnxruntime()
    except (ImportError, OSError):
        return False


def is_openvino_target(device_type: str) -> bool:
    """Check if the requested device type targets the OpenVINO EP."""
    target = (device_type or "").upper()
    return target.startswith(("NPU", "GPU"))


def has_openvino_provider(curr_ort) -> bool:
    """Check if the current ONNX runtime supports OpenVINO Execution Provider."""
    try:
        return "OpenVINOExecutionProvider" in curr_ort.get_available_providers()
    except (AttributeError, ValueError):
        return False


def ensure_openvino_onnxruntime(device_type: str) -> None:
    """Hot-reload ONNX runtime from the Intel tree if OpenVINO is requested but unavailable."""
    if not is_openvino_target(device_type):
        return
    try:
        ort = importlib.import_module("onnxruntime")
    except ImportError:
        return
    if not has_openvino_provider(ort):
        logger.info("[UVR] OpenVINO provider missing in active ORT; hot-reloading from Intel runtime path.")
        reload_onnxruntime_from_intel_path()


def openvino_device_family(device_id: str) -> str | None:
    """Return normalized OpenVINO device family for the requested token."""
    upper = (device_id or "").upper()
    if upper.startswith("NPU"):
        return "NPU"
    if upper.startswith("GPU"):
        return "GPU"
    return None


def get_available_openvino_devices() -> list[str]:
    """Return available OpenVINO hardware devices."""
    try:
        openvino = importlib.import_module("openvino")
        core = openvino.Core()
        return core.available_devices
    except (ImportError, AttributeError, RuntimeError, OSError, ValueError, TypeError):
        return []


def normalize_openvino_devices(available: list[str]) -> list[tuple[str, str]]:
    """Normalize OpenVINO device names into (raw, upper) tuples."""
    return [(dev, dev.upper()) for dev in available]


def find_exact_openvino_match(normalized: list[tuple[str, str]], requested: str) -> str | None:
    """Return exact device match when available."""
    for dev, upper in normalized:
        if upper == requested:
            return dev
    return None


def find_openvino_family_prefixed_match(normalized: list[tuple[str, str]], family: str) -> str | None:
    """Return first concrete device from the requested OpenVINO family."""
    prefix = f"{family}."
    for dev, upper in normalized:
        if upper.startswith(prefix):
            return dev
    return None


def alternate_openvino_family(family: str) -> str | None:
    """Return alternate Intel OpenVINO family for cross-family fallback."""
    if family == "NPU":
        return "GPU"
    if family == "GPU":
        return "NPU"
    return None


def _find_alternate_openvino_device(normalized: list[tuple[str, str]], family: str) -> str | None:
    """Helper to find an alternate device if the primary family is unavailable."""
    alternate = alternate_openvino_family(family)
    if not alternate:
        return None
    alt_concrete = find_openvino_family_prefixed_match(normalized, alternate)
    if alt_concrete is not None:
        return alt_concrete
    return find_exact_openvino_match(normalized, alternate)


def _find_family_device(normalized: list[tuple[str, str]], requested: str, family: str) -> str | None:
    """Helper to find the best device match within the requested family."""
    if "." in requested:
        exact_family = find_exact_openvino_match(normalized, family)
        if exact_family is not None:
            return exact_family

    concrete = find_openvino_family_prefixed_match(normalized, family)
    if concrete is not None:
        return concrete

    return _find_alternate_openvino_device(normalized, family)


def find_matching_openvino_device(requested: str, available: list[str]) -> str:
    """Find the best OpenVINO device match or fallback based on available devices."""
    if not available:
        return requested

    normalized = normalize_openvino_devices(available)
    exact = find_exact_openvino_match(normalized, requested)
    if exact is not None:
        return exact

    family = openvino_device_family(requested)
    if family:
        found = _find_family_device(normalized, requested, family)
        if found is not None:
            return found

    return requested


def _resolve_when_no_openvino_devices(requested: str, requested_family: str | None) -> str:
    """Resolve fallback when OpenVINO device enumeration is unavailable."""
    if requested_family is None:
        return requested
    logger.warning("[UVR] OpenVINO device enumeration unavailable; using generic family token %s", requested_family)
    return requested_family


def _should_return_generic_family(resolved: str, requested: str, requested_family: str | None) -> bool:
    """Return whether caller should keep a generic family token for provider options."""
    return bool(resolved and requested_family is not None and "." in resolved and "." not in requested)


def resolve_openvino_device_type(device_id: str) -> str:
    """Resolve provider device to an available OpenVINO device id when possible.
    If enumeration fails, return the generic family token without a concrete suffix.
    """
    requested = (device_id or "GPU").upper()
    requested_family = openvino_device_family(requested)
    available = get_available_openvino_devices()

    if not available:
        return _resolve_when_no_openvino_devices(requested, requested_family)

    return find_matching_openvino_device(requested, available)


def extend_openvino_retry_matches(ordered: list[str], normalized: list[tuple[str, str]], matcher) -> None:
    """Append normalized devices matching predicate while preserving order/uniqueness."""
    for dev, upper in normalized:
        if matcher(upper) and dev not in ordered:
            ordered.append(dev)


def _alternate_family_for_request(requested_upper: str) -> str | None:
    """Return alternate OpenVINO family for requested target when applicable."""
    family = openvino_device_family(requested_upper)
    if family is None:
        return None
    return alternate_openvino_family(family)


def _append_exact_openvino_match(ordered: list[str], normalized: list[tuple[str, str]], expected_upper: str) -> None:
    """Append exact OpenVINO device match when present and not already queued."""
    for dev, upper in normalized:
        if upper == expected_upper and dev not in ordered:
            ordered.append(dev)


def _append_alternate_family_retry_candidates(
    ordered: list[str],
    normalized: list[tuple[str, str]],
    requested_upper: str,
) -> None:
    """Append alternate-family retry candidates for the requested OpenVINO target."""
    alternate = _alternate_family_for_request(requested_upper)
    if not alternate:
        return

    extend_openvino_retry_matches(ordered, normalized, lambda upper: upper.startswith(f"{alternate}."))
    _append_exact_openvino_match(ordered, normalized, alternate)


def openvino_heuristic_retry_candidates(requested: str) -> list[str]:
    """Return heuristic OpenVINO retry candidates when device enumeration is unavailable."""
    requested_upper = (requested or "").upper()
    family = openvino_device_family(requested_upper)
    if family is None:
        return [requested_upper]

    candidates = [requested_upper]
    if "." not in requested_upper:
        candidates.append(f"{family}.0")
    else:
        candidates.append(family)

    alternate = alternate_openvino_family(family)
    if alternate:
        candidates.extend([f"{alternate}.0", alternate])
    return dedupe_openvino_retry_candidates(candidates)


def alternate_openvino_candidates(retries: list[str], requested: str):
    """Return retry candidates excluding the requested device target."""
    for candidate in retries:
        if candidate != requested:
            yield candidate


def get_openvino_retry_candidates_for_devices(requested: str, available_devices: list[str]) -> list[str]:
    """Build retry candidates from runtime-reported OpenVINO devices."""
    requested_upper = (requested or "").upper()
    normalized = normalize_openvino_devices(available_devices)
    ordered = [requested_upper]

    extend_openvino_retry_matches(ordered, normalized, lambda upper: upper.startswith(f"{requested_upper}."))
    family = openvino_device_family(requested_upper)
    if family:
        _append_exact_openvino_match(ordered, normalized, family)
    _append_alternate_family_retry_candidates(ordered, normalized, requested_upper)
    return dedupe_openvino_retry_candidates(ordered)


def get_openvino_retry_candidates(requested: str, available_devices: list[str] | None = None) -> list[str]:
    """Return OpenVINO retry candidates using runtime devices or heuristics."""
    devices = available_devices if available_devices is not None else get_available_openvino_devices()
    if devices:
        return get_openvino_retry_candidates_for_devices(requested, devices)
    return openvino_heuristic_retry_candidates(requested)


def has_openvino_accelerator_device_in(available_devices: list[str]) -> bool:
    """Return whether the provided OpenVINO device list includes NPU/GPU devices."""
    return any((dev or "").upper().startswith(("NPU", "GPU")) for dev in available_devices)


def has_openvino_accelerator_device(available_devices: list[str] | None = None) -> bool:
    """Return whether OpenVINO accelerator devices are available in runtime context."""
    devices = available_devices if available_devices is not None else get_available_openvino_devices()
    return has_openvino_accelerator_device_in(devices)


def is_openvino_session_fallback_error(exc: Exception) -> bool:
    """Check if the error is due to a failed OpenVINO session initialization fallback."""
    if not isinstance(exc, (ValueError, RuntimeError)):
        return False
    msg = str(exc)
    return (
        "Fallback to CPUExecutionProvider" in msg
        or "Failed to create OpenVINOExecutionProvider" in msg
        or "did not initialize" in msg
        or "fell back to providers=['CPUExecutionProvider']" in msg
    )


def is_openvino_runtime_loader_error(exc: Exception) -> bool:
    """Check if the error is a critical OpenVINO library loader failure."""
    if not isinstance(exc, (ValueError, RuntimeError, OSError)):
        return False
    msg = str(exc)
    return "INTEL_OPENVINO_DIR is set but OpenVINO library wasn't able to be loaded" in msg


def mark_openvino_family_unavailable(device_type: str) -> None:
    """Mark an OpenVINO family unavailable for the current process runtime."""
    family = openvino_device_family(device_type)
    if family is None:
        return
    with _OPENVINO_DISABLED_LOCK:
        _OPENVINO_DISABLED_FAMILIES.add(family)


def is_openvino_family_disabled(device_type: str) -> bool:
    """Return whether OpenVINO family is currently disabled by fallback circuit-breaker."""
    family = openvino_device_family(device_type)
    if family is None:
        return False
    with _OPENVINO_DISABLED_LOCK:
        return family in _OPENVINO_DISABLED_FAMILIES


def clear_openvino_disabled_families() -> None:
    """Clear OpenVINO family disable state (test helper)."""
    with _OPENVINO_DISABLED_LOCK:
        _OPENVINO_DISABLED_FAMILIES.clear()


def disable_all_openvino_families_for_runtime() -> None:
    """Globally disable all OpenVINO families after a critical runtime failure."""
    with _OPENVINO_DISABLED_LOCK:
        _OPENVINO_DISABLED_FAMILIES.update(["NPU", "GPU"])


def log_openvino_cpu_fallback(session, ctx_options) -> None:
    """Log an error if ONNX Runtime fell back to CPU while OpenVINO was requested."""
    if not (ctx_options and "device_type" in ctx_options):
        return
    try:
        active_providers = session.get_providers()
    except (AttributeError, TypeError, ValueError, RuntimeError, OSError):
        return
    if "OpenVINOExecutionProvider" in active_providers:
        return
    device_type = ctx_options["device_type"]
    logger.warning(
        "[UVR] OpenVINOExecutionProvider did not initialize for device_type=%s; ONNX Runtime fell back to providers=%s",
        device_type,
        active_providers,
    )
    mark_openvino_family_unavailable(device_type)


def force_openvino_provider_if_needed(providers, ctx_options):
    """Force OpenVINO provider into session if CPU fallback occurs."""
    if ctx_options and "device_type" in ctx_options and (not providers or providers == ["CPUExecutionProvider"]):
        logger.info("[System] Intercepted CPU fallback - Forcing OpenVINOProvider")
        return ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
    return providers


def _is_ov_mergeable(providers, ctx_options) -> bool:
    if not ctx_options:
        return False
    if not providers:
        return False
    return "OpenVINOExecutionProvider" in providers


def _align_provider_options(providers, provider_options):
    if not provider_options:
        provider_options = [{} for _ in range(len(providers))]
    while len(provider_options) < len(providers):
        provider_options.append({})
    return provider_options


def normalize_openvino_session_device_type(device_type: str) -> str:
    """Normalize device_type to values accepted by ONNX Runtime OpenVINO EP."""
    target = (device_type or "").upper()
    if target.startswith("NPU"):
        return "NPU"
    return target or "GPU"


def _openvino_device_index(device_type: str) -> str | None:
    """Return numeric OpenVINO device index suffix when present."""
    target = (device_type or "").upper()
    _, dot, suffix = target.partition(".")
    if not dot or not suffix.isdigit():
        return None
    return suffix


def _openvino_load_config(device_type: str) -> str | None:
    """Return OpenVINO load_config JSON for device-specific routing when needed."""
    target = (device_type or "").upper()
    if not target.startswith("NPU"):
        return None

    index = _openvino_device_index(target)
    if index is None:
        return None

    return json.dumps({"NPU": {"DEVICE_ID": index}}, separators=(",", ":"))


def _normalize_openvino_ctx_options(ctx_options: dict | None) -> dict:
    """Normalize OpenVINO context options for ONNX Runtime compatibility."""
    normalized = dict(ctx_options or {})
    device_type = normalized.get("device_type")
    if device_type is not None:
        target = str(device_type)
        normalized["device_type"] = normalize_openvino_session_device_type(target)
        load_config = _openvino_load_config(target)
        if load_config is not None:
            normalized["load_config"] = load_config
    return normalized


def normalize_openvino_provider_options(options: dict | None) -> dict:
    """Return a normalized copy of OpenVINO provider options."""
    return _normalize_openvino_ctx_options(options)


def dedupe_openvino_retry_candidates(candidates: list[str]) -> list[str]:
    """Deduplicate exact retry candidates while preserving order."""
    deduped: list[str] = []
    seen_targets: set[str] = set()
    for candidate in candidates:
        if candidate in seen_targets:
            continue
        deduped.append(candidate)
        seen_targets.add(candidate)
    return deduped


def merge_openvino_provider_options(providers, provider_options, ctx_options):
    """Merge custom context options into the OpenVINO execution provider options."""
    if not _is_ov_mergeable(providers, ctx_options):
        return provider_options
    normalized_ctx_options = _normalize_openvino_ctx_options(ctx_options)
    logger.info("[System] Injecting OpenVINO options into session: %s", normalized_ctx_options)

    provider_options = _align_provider_options(providers, provider_options)

    ov_idx = providers.index("OpenVINOExecutionProvider")
    if not isinstance(provider_options[ov_idx], dict):
        provider_options[ov_idx] = {}

    provider_options[ov_idx].update(normalized_ctx_options)
    return provider_options


def set_openvino_context_options(target_options) -> None:
    """Set OpenVINO options in thread context."""
    if target_options and "device_type" in target_options[0]:
        utils.THREAD_CONTEXT.ov_options = _normalize_openvino_ctx_options(target_options[0])
        return
    utils.THREAD_CONTEXT.ov_options = None


def _log_openvino_cpu_fallback(session, ctx_options) -> None:
    """Backward-compatible alias for OpenVINO fallback logging helper."""
    log_openvino_cpu_fallback(session, ctx_options)
