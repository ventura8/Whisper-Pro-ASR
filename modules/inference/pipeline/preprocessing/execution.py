"""Execution-path helpers extracted from preprocessing.py to keep module size manageable."""

import logging

from modules.core import config, utils
from modules.inference.pipeline import openvino_provider_dispatch, openvino_resolver

logger = logging.getLogger(__name__)


def create_separator(lazy_import_separator, output_dir: str):
    """Create a Separator configured with project-default UVR settings."""
    return lazy_import_separator()(
        output_dir=output_dir,
        model_file_dir=config.UVR_MODEL_DIR,
        output_format="WAV",
        normalization_threshold=0.01,
        output_single_stem="Vocals",
        chunk_duration=config.UVR_CHUNK_DURATION,
        log_level=logging.INFO,
    )


def enable_separator_acceleration_flag(separator, target_providers, unit_name: str):
    """Enable audio-separator hardware acceleration when provider set is accelerated."""
    is_accelerated = any(
        provider in ["CUDAExecutionProvider", "OpenVINOExecutionProvider", "DmlExecutionProvider", "ROCMExecutionProvider"]
        for provider in target_providers
    )
    if is_accelerated:
        logger.debug("[System] Forcing hardware_acceleration_enabled for %s", unit_name)
        separator.hardware_acceleration_enabled = True


def try_openvino_candidate_load(separator, device_id: str, candidate: str, first_error: Exception) -> tuple[bool, Exception]:
    """Try loading UVR model on an alternate OpenVINO candidate device."""
    if openvino_resolver.is_openvino_family_disabled(candidate):
        logger.warning(
            "[UVR] Skipping OpenVINO retry candidate %s because family is disabled by runtime circuit-breaker",
            candidate,
        )
        return False, first_error

    providers, options = openvino_provider_dispatch.openvino_provider_config(candidate)
    separator.onnx_execution_provider = providers
    openvino_resolver.set_openvino_context_options(options)

    try:
        logger.warning(
            "[UVR] OpenVINO init failed on %s (%s); retrying with alternate device %s",
            device_id,
            first_error,
            candidate,
        )
        separator.load_model(config.VOCAL_SEPARATION_MODEL)
        logger.info("[UVR] OpenVINO initialized on alternate device %s", candidate)
        return True, first_error
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError) as retry_error:
        if openvino_resolver.is_openvino_runtime_loader_error(retry_error):
            openvino_resolver.disable_all_openvino_families_for_runtime()
        return False, retry_error


def cleanup_extra_stems(resolve_stem_path_fn, extra_stems, effective_sep, audio_path, stem_path):
    """Delete non-primary stems while preserving the selected vocal stem output."""
    for extra_stem in extra_stems:
        extra_path = resolve_stem_path_fn(extra_stem, effective_sep, audio_path)
        utils.track_file(extra_path)
        if extra_path != stem_path:
            utils.secure_remove(extra_path)


def separate_audio(
    lock,
    sep,
    audio_path,
    *,
    use_cpu_lock,
    target_options,
    active_yield_cb,
    lazy_import_separator,
    separate_with_fallback,
):
    """Run separation with provider-context injection and ENOSPC fallback handling."""
    with lock:
        if target_options and "device_type" in target_options[0]:
            utils.THREAD_CONTEXT.ov_options = target_options[0]
        else:
            utils.THREAD_CONTEXT.ov_options = None

        def _make_separator(output_dir):
            """Build a fresh Separator pinned to a specific output directory."""
            new_sep = create_separator(lazy_import_separator, output_dir)
            new_sep.onnx_execution_provider = sep.onnx_execution_provider
            if getattr(sep, "hardware_acceleration_enabled", False):
                new_sep.hardware_acceleration_enabled = True
            new_sep.load_model(config.VOCAL_SEPARATION_MODEL)
            return new_sep

        try:
            if use_cpu_lock:
                with utils.cpu_lock_ctx():
                    return separate_with_fallback(sep, _make_separator, audio_path, yield_cb=active_yield_cb)
            return separate_with_fallback(sep, _make_separator, audio_path, yield_cb=active_yield_cb)
        finally:
            utils.THREAD_CONTEXT.ov_options = None


def resolve_isolation_output_path(resolve_stem_path_fn, stems, effective_sep, audio_path):
    """Resolve and clean isolation outputs, returning the selected stem path."""
    if not stems:
        return audio_path
    stem_path = resolve_stem_path_fn(stems[0], effective_sep, audio_path)
    utils.track_file(stem_path)
    cleanup_extra_stems(resolve_stem_path_fn, stems[1:], effective_sep, audio_path, stem_path)
    return stem_path
