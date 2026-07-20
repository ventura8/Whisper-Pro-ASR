"""Helper functions for audio preprocessing, chunk tracking, and ONNX Runtime patches."""

import errno
import importlib
import logging
import math
import os
import shutil
import sys
import tempfile
import time
import types
from pathlib import Path

from modules.core import config, utils
from modules.inference import scheduler
from modules.inference.pipeline import openvino_resolver

logger = logging.getLogger(__name__)

CACHE_DIR = Path(config.PREPROCESSING_CACHE_DIR)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def candidate_output_dirs() -> list[str]:
    """Return ordered candidate output directories for UVR stem files."""
    candidates = [str(CACHE_DIR), config.PERSISTENT_TEMP_DIR, tempfile.gettempdir()]
    seen = set()
    result = []
    for d in candidates:
        if d not in seen:
            seen.add(d)
            result.append(d)
    return result


def separate_with_fallback(sep, sep_factory, audio_path, yield_cb=None, hooks=None):
    """Run UVR separation, retrying on ENOSPC with alternative directories."""
    hooks = hooks or {}
    candidate_dirs_fn = hooks.get("candidate_dirs_fn")
    disk_usage_fn = hooks.get("disk_usage_fn")

    candidates = candidate_dirs_fn() if candidate_dirs_fn else candidate_output_dirs()
    current_sep = sep
    last_err = None
    for i, out_dir in enumerate(candidates):
        stems, current_sep, last_err = _try_separate_candidate(
            current_sep,
            sep_factory,
            audio_path,
            out_dir,
            idx=i,
            yield_cb=yield_cb,
            disk_usage_fn=disk_usage_fn,
        )
        if stems is not None:
            return stems, current_sep

    # All candidates exhausted
    raise OSError(errno.ENOSPC, "No space left on any candidate output directory") from last_err


def _try_separate_candidate(
    current_sep,
    sep_factory,
    audio_path,
    out_dir,
    *,
    idx: int,
    yield_cb=None,
    disk_usage_fn=None,
):
    try:
        current_sep = _attempt_separation_in_output_dir(current_sep, sep_factory, audio_path, out_dir, idx=idx, yield_cb=yield_cb)
        return current_sep.separate(audio_path), current_sep, None
    except OSError as exc:
        if exc.errno != errno.ENOSPC:
            raise
        _log_no_space_fallback(out_dir, disk_usage_fn=disk_usage_fn)
        return None, current_sep, exc


def _attempt_separation_in_output_dir(current_sep, sep_factory, audio_path, out_dir, *, idx: int, yield_cb=None):
    os.makedirs(out_dir, exist_ok=True)
    current_sep = _maybe_recreate_separator(current_sep, sep_factory, out_dir, idx)
    _initialize_chunk_tracking_context(current_sep, audio_path, yield_cb=yield_cb)
    _ensure_separator_permanent_patch(current_sep)
    return current_sep


def _maybe_recreate_separator(current_sep, sep_factory, out_dir, idx: int):
    if idx > 0:
        return sep_factory(out_dir)
    return current_sep


def _initialize_chunk_tracking_context(current_sep, audio_path, yield_cb=None):
    audio_dur = utils.get_audio_duration(audio_path)
    chunk_duration = getattr(current_sep, "chunk_duration", None)
    _reset_chunk_tracking_state(current_sep, yield_cb=yield_cb)
    if _is_valid_chunk_duration(audio_dur, chunk_duration):
        _set_chunk_tracking_state(current_sep, audio_dur, chunk_duration)


def _is_valid_chunk_duration(audio_dur: float, chunk_duration) -> bool:
    return (
        chunk_duration
        and isinstance(chunk_duration, (int, float))
        and not hasattr(chunk_duration, "_mock_self")
        and audio_dur > chunk_duration
    )


def _reset_chunk_tracking_state(current_sep, yield_cb=None):
    utils.THREAD_CONTEXT.uvr_chunk_paths_len = 0
    utils.THREAD_CONTEXT.uvr_chunk_index = 0
    utils.THREAD_CONTEXT.uvr_audio_dur = 0.0
    utils.THREAD_CONTEXT.uvr_chunk_duration = 0
    utils.THREAD_CONTEXT.uvr_scheduler = scheduler
    utils.THREAD_CONTEXT.uvr_yield_cb = yield_cb
    setattr(current_sep, "_chunk_paths_len", 0)
    setattr(current_sep, "_chunk_index", 0)
    setattr(current_sep, "_audio_dur", 0.0)


def _set_chunk_tracking_state(current_sep, audio_dur: float, chunk_duration: float):
    total_chunks = math.ceil(audio_dur / chunk_duration)
    utils.THREAD_CONTEXT.uvr_chunk_paths_len = total_chunks
    utils.THREAD_CONTEXT.uvr_chunk_index = 0
    utils.THREAD_CONTEXT.uvr_audio_dur = audio_dur
    utils.THREAD_CONTEXT.uvr_chunk_duration = chunk_duration
    setattr(current_sep, "_chunk_paths_len", total_chunks)
    setattr(current_sep, "_chunk_index", 0)
    setattr(current_sep, "_audio_dur", audio_dur)


def _ensure_separator_permanent_patch(current_sep):
    _ensure_orig_separate_file_attr(current_sep)
    if _separator_already_patched(current_sep):
        return
    patched = _build_permanent_patched_separate_file()
    setattr(current_sep, "_separate_file", types.MethodType(patched, current_sep))
    setattr(current_sep, "_is_permanently_patched", True)


def _ensure_orig_separate_file_attr(current_sep):
    if not hasattr(current_sep, "_orig_separate_file"):
        setattr(current_sep, "_orig_separate_file", getattr(current_sep, "_separate_file"))


def _separator_already_patched(current_sep) -> bool:
    return hasattr(current_sep, "_is_permanently_patched")


def _build_permanent_patched_separate_file():
    def permanent_patched_separate_file(self, audio_file_path, custom_output_names=None):
        if _should_delegate_outer_chunk_call():
            return _run_outer_chunk_delegate(self, audio_file_path, custom_output_names)
        _update_chunk_progress_start(self)
        _run_thread_yield_cb()
        res = getattr(self, "_orig_separate_file")(audio_file_path, custom_output_names)
        if not res:
            raise RuntimeError(f"UVR separation failed on {audio_file_path}: separator produced no output stems")
        _update_chunk_progress_end(self)
        _run_thread_yield_cb()
        return res

    return permanent_patched_separate_file


def _should_delegate_outer_chunk_call() -> bool:
    chunk_paths_len = getattr(utils.THREAD_CONTEXT, "uvr_chunk_paths_len", 0)
    is_outer = not getattr(utils.THREAD_CONTEXT, "uvr_in_chunk_processing", False)
    return is_outer and chunk_paths_len > 0


def _run_outer_chunk_delegate(sep, audio_file_path, custom_output_names=None):
    utils.THREAD_CONTEXT.uvr_in_chunk_processing = True
    try:
        return getattr(sep, "_orig_separate_file")(audio_file_path, custom_output_names)
    finally:
        utils.THREAD_CONTEXT.uvr_in_chunk_processing = False


def _update_chunk_progress_start(sep):
    _update_chunk_progress(sep, start_phase=True)


def _update_chunk_progress_end(sep):
    _update_chunk_progress(sep, start_phase=False)


def _update_chunk_progress(sep, start_phase: bool):
    context = _resolve_chunk_progress_context()
    if context is None:
        return

    chunk_paths_len, sched, chunk_idx = context
    audio_dur_val = getattr(utils.THREAD_CONTEXT, "uvr_audio_dur", 0.0)
    chunk_dur = getattr(utils.THREAD_CONTEXT, "uvr_chunk_duration", 0)
    processed_dur, pct = _compute_chunk_progress(start_phase, chunk_idx, chunk_paths_len, chunk_dur, audio_dur_val)
    if not start_phase:
        setattr(utils.THREAD_CONTEXT, "uvr_chunk_index", chunk_idx)
        setattr(sep, "_chunk_index", chunk_idx)

    sched.update_task_metadata(current_position=processed_dur)
    sched.update_task_progress(
        int(pct),
        f"Vocal Separation ({chunk_idx}/{chunk_paths_len} segments | "
        f"{utils.format_duration(processed_dur)} / {utils.format_duration(audio_dur_val)})",
    )


def _resolve_chunk_progress_context():
    chunk_paths_len = getattr(utils.THREAD_CONTEXT, "uvr_chunk_paths_len", 0)
    sched = getattr(utils.THREAD_CONTEXT, "uvr_scheduler", None)
    current_idx = getattr(utils.THREAD_CONTEXT, "uvr_chunk_index", 0)
    chunk_idx = current_idx + 1
    if not (chunk_paths_len > 0 and sched):
        return None
    if not (0 <= current_idx < chunk_paths_len and 1 <= chunk_idx <= chunk_paths_len):
        return None
    return chunk_paths_len, sched, chunk_idx


def _compute_chunk_progress(
    start_phase: bool,
    chunk_idx: int,
    chunk_paths_len: int,
    chunk_dur: float,
    audio_dur_val: float,
):
    if start_phase:
        processed_dur = min((chunk_idx - 1) * chunk_dur, audio_dur_val)
        pct = 5.0 + (float(chunk_idx - 1) / chunk_paths_len) * 5.0
        return processed_dur, pct
    processed_dur = min(chunk_idx * chunk_dur, audio_dur_val)
    pct = 5.0 + (float(chunk_idx) / chunk_paths_len) * 5.0
    return processed_dur, pct


def _run_thread_yield_cb():
    current_yield_cb = getattr(utils.THREAD_CONTEXT, "uvr_yield_cb", None)
    if current_yield_cb:
        current_yield_cb()


def _log_no_space_fallback(out_dir: str, disk_usage_fn=None):
    disk_usage_fn = disk_usage_fn or shutil.disk_usage
    try:
        free_mb = disk_usage_fn(out_dir).free // (1024 * 1024)
    except OSError:
        free_mb = 0
    logger.error(
        "[UVR] No space left on %s (%d MB free) — trying next fallback.",
        out_dir,
        free_mb,
    )


def stem_resolution_candidates(effective_sep, source_audio_path, candidate_dirs_fn):
    """Build ordered candidate directories for resolving generated stem paths."""
    candidates = []
    run_output_dir = getattr(effective_sep, "output_dir", None)
    if run_output_dir:
        candidates.append(run_output_dir)
    candidates.extend(candidate_dirs_fn())
    source_parent = os.path.dirname(source_audio_path)
    if source_parent:
        candidates.append(source_parent)
    return candidates


def existing_stem_candidates(path_value, base_dirs):
    """Yield existing absolute candidates for a relative stem path."""
    seen = set()
    for base_dir in base_dirs:
        if not base_dir or base_dir in seen:
            continue
        seen.add(base_dir)
        candidate = os.path.join(base_dir, path_value)
        if os.path.exists(candidate):
            yield candidate


def run_optional_yield(yield_cb):
    """Invoke cooperative yield callback when supplied."""
    if yield_cb:
        yield_cb()


def log_isolation_complete(unit_name: str, p_start: float, audio_dur: float):
    """Emit standardized UVR completion log with speed metrics."""
    dur = time.time() - p_start
    speed_val = audio_dur / dur if dur > 0 else 0.0
    logger.info(
        "[UVR] Isolation complete on %s (Duration: %s | Audio: %s | Speed: %.2fx)",
        unit_name,
        utils.format_duration(dur),
        utils.format_duration(audio_dur),
        speed_val,
    )


def apply_onnx_optimizations(module_obj=None):
    """
    Monkeypatch ONNX Runtime and audio-separator for hardware acceleration.
    """
    try:
        curr_ort = _ensure_onnxruntime_loaded(module_obj)
        _patch_onnx_inference_session(curr_ort)
        _patch_audio_separator_onnx_check()
    except (
        ImportError,
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        OSError,
    ) as patch_err:
        logger.warning("[System] Failed to apply ONNX optimizations: %s", patch_err)


def _ensure_onnxruntime_loaded(module_obj=None):
    if module_obj is None:
        module_obj = sys.modules[__name__]
    if getattr(module_obj, "ort", None) is None:
        setattr(module_obj, "ort", importlib.import_module("onnxruntime"))
    return getattr(module_obj, "ort")


def _patch_onnx_inference_session(curr_ort):
    if getattr(curr_ort.InferenceSession, "is_patched", False) is True:
        return
    logger.debug("Optimization: Deep-patching ONNX InferenceSession...")
    original_init = curr_ort.InferenceSession.__init__

    def patched_init(
        self,
        model_path,
        sess_options=None,
        providers=None,
        provider_options=None,
        **kwargs,
    ):
        ctx_options = getattr(utils.THREAD_CONTEXT, "ov_options", None)
        providers = openvino_resolver.force_openvino_provider_if_needed(providers, ctx_options)
        provider_options = openvino_resolver.merge_openvino_provider_options(providers, provider_options, ctx_options)
        result = original_init(self, model_path, sess_options, providers, provider_options, **kwargs)
        openvino_resolver.log_openvino_cpu_fallback(self, ctx_options)
        return result

    curr_ort.InferenceSession.__init__ = patched_init
    curr_ort.InferenceSession.is_patched = True


_DEFAULT_UVR_HASH_METADATA = {
    "compensate": 1.01,
    "mdx_dim_f_set": 3072,
    "mdx_dim_t_set": 8,
    "mdx_n_fft_scale_set": 6144,
    "primary_stem": "Instrumental",
}


def _invoke_original_hash_loader(orig_load_hash, *args, **kwargs):
    if not orig_load_hash:
        return None, None
    try:
        loaded = orig_load_hash(*args, **kwargs)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        return None, exc
    return loaded or None, None


def _is_default_uvr_hash_model(*args, **kwargs) -> bool:
    haystack = " ".join(str(part) for part in args) + " " + " ".join(str(val) for val in kwargs.values())
    return "UVR-MDX-NET-Inst_HQ_3" in haystack


def _safe_load_model_data_using_hash(orig_load_hash, *args, **kwargs):
    loaded, load_error = _invoke_original_hash_loader(orig_load_hash, *args, **kwargs)
    if loaded:
        return loaded
    if orig_load_hash is None or _is_default_uvr_hash_model(*args, **kwargs):
        return dict(_DEFAULT_UVR_HASH_METADATA)
    if load_error is not None:
        raise load_error
    return None


def _patch_audio_separator_onnx_check():
    try:
        audio_separator = importlib.import_module("audio_separator.separator")
        separator_cls = audio_separator.Separator
        if getattr(separator_cls, "is_patched", False) is not True:
            logger.debug("Optimization: Patching Separator class detection logic...")
            separator_cls.check_onnxruntime = lambda self: None

            # Offline fallback for download_model_files and model parameters to prevent GitHub 429
            orig_download = getattr(separator_cls, "download_model_files", None)

            def safe_download_model_files(self, model_filename):
                model_path = os.path.join(self.model_file_dir, f"{model_filename}")
                if os.path.exists(model_path) and model_filename == "UVR-MDX-NET-Inst_HQ_3.onnx":
                    return (
                        model_filename,
                        "MDX",
                        "UVR-MDX-NET-Inst_HQ_3",
                        model_path,
                        None,
                    )
                if orig_download:
                    return orig_download(self, model_filename)
                raise FileNotFoundError(f"Model file {model_filename} not found at {model_path}")

            orig_load_hash = getattr(separator_cls, "load_model_data_using_hash", None)

            def safe_load_model_data_using_hash(*args, **kwargs):
                return _safe_load_model_data_using_hash(orig_load_hash, *args, **kwargs)

            orig_separate = getattr(separator_cls, "separate", None)

            def safe_separate(self, audio_file_path, custom_output_names=None):
                if orig_separate:
                    out = orig_separate(self, audio_file_path, custom_output_names)
                    if not out:
                        raise RuntimeError(f"audio-separator failed to process file {audio_file_path} (returned empty stems)")
                    return out
                raise RuntimeError("audio-separator separate() original implementation is missing; cannot perform vocal separation")

            separator_cls.download_model_files = safe_download_model_files
            separator_cls.load_model_data_using_hash = safe_load_model_data_using_hash
            separator_cls.separate = safe_separate
            separator_cls.is_patched = True
    except ImportError:
        pass


def _ensure_provider_option_entry_dict(provider_options, index: int):
    """Ensure provider_options[index] is a dictionary and return normalized list."""
    while len(provider_options) <= index:
        provider_options.append({})
    if not isinstance(provider_options[index], dict):
        provider_options[index] = {}
    return provider_options


def _log_openvino_cpu_fallback(session, ctx_options) -> None:
    """Backward-compatible alias used by tests for OpenVINO CPU fallback logging."""
    openvino_resolver.log_openvino_cpu_fallback(session, ctx_options)


def set_openvino_context_options(target_options) -> None:
    """Set OpenVINO options in thread context."""
    openvino_resolver.set_openvino_context_options(target_options)
