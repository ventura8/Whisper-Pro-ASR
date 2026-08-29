"""
High-Level Model Orchestration and Hardware Lifecycle Management for Whisper Pro ASR.
"""

import importlib
import logging
import os
import sys
import threading
import time
import typing

from modules.core import config, engine_registry, logging_setup, utils
from modules.inference import scheduler
from modules.inference.engines import engine_factory
from modules.inference.pipeline import diarization, language_detection, post_processing, preprocessing, vad
from modules.inference.pipeline.language_detection_core import (
    run_batch_language_detection,
    run_batch_language_detection_direct,
    run_language_detection,
    run_language_detection_core,
)
from modules.inference.runtime import gap_filling, preprocessor_pool
from modules.inference.runtime.concurrency import _check_preemption, _get_current_task_info, model_lock_ctx
from modules.inference.runtime.model_lifecycle import (
    _clear_uvr_models,
    _clear_whisper_models,
    _clear_whisperx_models,
    _format_reclamation_delta,
    _format_reclamation_memory,
    _read_reclamation_memory_snapshot,
    _run_garbage_collection_and_reclamation,
)
from modules.inference.runtime.model_segment_processing import consume_transcription_segments

__all__ = ["run_language_detection_core", "run_language_detection", "run_batch_language_detection", "run_batch_language_detection_direct"]


def early_task_registration(task_type="ASR/LD", stage="Initializing", filename=None, is_priority=False):
    """Register a task through the scheduler compatibility surface."""
    return scheduler.early_task_registration(
        task_type=task_type,
        stage=stage,
        filename=filename,
        is_priority=is_priority,
    )


def update_task_metadata(**kwargs):
    """Forward task metadata updates to the scheduler."""
    return scheduler.update_task_metadata(**kwargs)


def record_task_failure(msg: str, code: int = 400, context: str = "Task") -> None:
    """Forward failure recording to the scheduler."""
    return scheduler.record_task_failure(msg, code, context)


def update_task_progress(progress, stage=None):
    """Forward task progress updates to the scheduler."""
    return scheduler.update_task_progress(progress, stage=stage)


def is_engine_initialized():
    """Return whether the engine is initialized."""
    return scheduler.is_engine_initialized()


def is_uvr_actually_loaded():
    """Return whether UVR is loaded under the public runtime API name."""
    return any(getattr(preprocessor, "separator", None) is not None for preprocessor in PREPROCESSOR_POOL.values())


def _post_process_results(result, audio_path=None):
    """Apply the transcription post-processing filters used by the runtime pipeline."""
    return post_processing.post_process_results(result, audio_path)


def get_status():
    """Return dashboard status payload for runtime callers."""
    return {"active_units": list(MODEL_POOL.keys()), "total_units": len(config.HARDWARE_UNITS)}


def dummy_engine(*args, **kwargs):
    """Placeholder callable so the lazy engine slots below start out type-correct."""
    return (args, kwargs)


def is_engine_actually_loaded() -> bool:
    """Return whether any primary model remains loaded."""
    return bool(MODEL_POOL)


# Lazy load containers for engines
_ENGINES: typing.Dict[str, typing.Any] = {"WhisperModel": dummy_engine, "ctranslate2": dummy_engine, "IntelWhisperEngine": dummy_engine}


def _lazy_import_engines():
    """Lazily import inference engines to save 500MB+ RAM during startup."""
    if _ENGINES["WhisperModel"] is dummy_engine:
        try:
            faster_whisper = importlib.import_module("faster_whisper")
            ctranslate2 = importlib.import_module("ctranslate2")
            _ENGINES["WhisperModel"] = faster_whisper.WhisperModel
            _ENGINES["ctranslate2"] = ctranslate2
            # Ensure VAD monkeypatching is applied after faster_whisper load
            vad.lazy_import_vad()
            if config.ASR_ENGINE == "INTEL-WHISPER":
                intel_engine = importlib.import_module("modules.inference.engines.intel_engine")
                _ENGINES["IntelWhisperEngine"] = intel_engine.IntelWhisperEngine
        except ImportError as e:
            logger.warning("[Engine] Failed to lazy load engines: %s", e)


# Ensure external AI engines log to our task-aware system
logging.getLogger("faster_whisper").setLevel(logging.INFO)
logging.getLogger("audio_separator").setLevel(logging.INFO)

TASK_LOGS = logging_setup.TASK_LOGS
logger = logging.getLogger(__name__)

MODEL_POOL = {}
PREPROCESSOR_POOL = {}
ALIGN_POOL = diarization.ALIGN_POOL
DIARIZE_POOL = diarization.DIARIZE_POOL

_LIFECYCLE_STATE = {"last_activity": time.time(), "monitor_started": False}
_MONITOR_LOCK = threading.Lock()


CLEANER_STATE = {"timer": None}
_CLEANER_TIMER_LOCK = threading.Lock()
_POOL_LOCK = threading.Lock()


def _resolve_preprocessor_for_unit(unit_id: str):
    """Pick the preprocessor for ``unit_id``; the policy lives in preprocessor_pool."""
    return preprocessor_pool.resolve_preprocessor_for_unit(PREPROCESSOR_POOL, unit_id)


def _preferred_preprocessor() -> typing.Any:
    """Return a preprocessor pinned to the configured preprocess device when available."""
    return preprocessor_pool.preferred_preprocessor(PREPROCESSOR_POOL)


def load_model():
    """Initializes hardware resource mapping without eager RAM loading."""
    for unit in config.HARDWARE_UNITS:
        # Initialize preprocessor managers (they are lazy and won't load models yet)
        PREPROCESSOR_POOL[unit["id"]] = preprocessing.create_manager(unit)

    scheduler.STATE.engine_initialized = True
    if config.ENABLE_VOCAL_SEPARATION:
        scheduler.STATE.uvr_loaded = True
    return True


# Alias for backward compatibility with tests
init_pool = load_model


def _should_isolate(engine_type: str) -> bool:
    """Whether this unit's engine should run in its own process.

    WhisperX is excluded because it already proxies to its own dedicated worker;
    wrapping it again would spawn a second process for no benefit.
    """
    if not getattr(config, "ISOLATE_ENGINES", False):
        return False
    return engine_type != engine_registry.ENGINE_WHISPERX


def _build_engine(engine_type: str, model_id: str, unit: dict):
    """Create the engine for ``unit``, in-process or behind a worker."""
    if not _should_isolate(engine_type):
        return engine_factory.create_engine(engine_type, model_id, unit)
    # Imported lazily so a deployment with isolation disabled never pays for the
    # multiprocessing machinery.
    isolated_engine = importlib.import_module("modules.inference.engines.isolated_engine")
    return isolated_engine.IsolatedEngine(engine_type, model_id, unit)


def init_unit(unit):
    """Loads model for a specific hardware unit."""
    _lazy_import_engines()
    with _POOL_LOCK:
        # Bound before the try: the handler reports it, and the first statements inside can
        # themselves fail (a malformed unit dict), which would turn the real error into a
        # NameError naming nothing useful.
        engine_type = getattr(config, "ASR_ENGINE", "unknown")
        try:
            engine_type = config.engine_for_unit(unit)
            model_id = config.model_id_for_engine(engine_type)
            logger.info("[Engine] Loading %s (%s) on %s...", model_id, engine_type, unit["name"])

            model = _build_engine(engine_type, model_id, unit)

            MODEL_POOL[unit["id"]] = model
            PREPROCESSOR_POOL[unit["id"]] = preprocessing.create_manager(unit)
            scheduler.STATE.whisper_loaded = True
            scheduler.STATE.engine_initialized = True
            LAST_INIT_ERROR.pop(unit["id"], None)
            logger.info("[Engine] %s ready.", unit["id"])
        except (ValueError, RuntimeError, ImportError, AttributeError, KeyError, OSError, TypeError) as e:
            unit_id = unit.get("id", "?") if isinstance(unit, dict) else "?"
            logger.error("[Engine] Failed to load %s: %s", unit_id, e)
            LAST_INIT_ERROR[unit_id] = _describe_init_failure(engine_type, unit, e)


#: unit_id -> why its last load attempt failed, so the scheduler can report the actual
#: cause instead of the downstream "engine pool is empty" symptom.
LAST_INIT_ERROR: dict[str, str] = {}


def _describe_init_failure(engine_type: str, unit: dict, error: Exception) -> str:
    """Turn a load failure into something a caller can act on.

    A missing engine dependency is the common case and the least obvious one: not every
    image ships every engine (WhisperX is only in the `full` target), so the honest
    message names the engine and the image rather than the empty pool it causes.
    """
    if isinstance(error, ImportError):
        return (
            f"ASR engine {engine_type} is not available in this image "
            f"(missing dependency: {error.name or error}). Use an image that ships it, or select a different ASR_ENGINE."
        )
    label = unit.get("name") or unit.get("id", "?") if isinstance(unit, dict) else "?"
    return f"ASR engine {engine_type} failed to load on {label}: {type(error).__name__}: {error}"


def run_transcription(
    audio_path,
    language,
    task,
    *,
    diarize=False,
    min_speakers=None,
    max_speakers=None,
    hf_token=None,
    initial_prompt=None,
    vad_filter=True,
    word_timestamps=False,
    **_kwargs,
):
    """Executes ASR inference with hardware locking."""
    _LIFECYCLE_STATE["last_activity"] = time.time()
    perf = {"dur_iso": 0}
    with model_lock_ctx() as (model, unit_id):
        _update_audio_duration_metadata(audio_path)

        processed_path = _isolate_vocals_if_needed(audio_path, unit_id, perf)
        was_auto_detect = not language
        language, _multilingual_suspected = _detect_language_after_isolation(language, processed_path, model)

        try:
            params = {
                "language": language,
                "task": task,
                "diarize": diarize,
                "min_speakers": min_speakers,
                "max_speakers": max_speakers,
                "hf_token": hf_token,
                "initial_prompt": initial_prompt,
                "vad_filter": vad_filter,
                "word_timestamps": word_timestamps,
                # Gap-filling (see _fill_language_gaps) needs to know whether the caller
                # forced a language: an explicit request must never be second-guessed by
                # re-detecting and re-transcribing a "gap" in a different language.
                "was_auto_detect": was_auto_detect,
            }
            return _execute_transcription_pipeline(
                model,
                processed_path,
                params=params,
                unit_id=unit_id,
                perf=perf,
            )
        finally:
            _cleanup_isolated_file_safe(audio_path, processed_path)


def _update_audio_duration_metadata(audio_path):
    try:
        audio_duration = utils.get_audio_duration(audio_path)
        scheduler.update_task_metadata(video_duration=audio_duration)
    except tuple([Exception]) as e:
        logger.warning("[Engine] Failed to get audio duration early: %s", e)


def will_isolate_vocals() -> bool:
    """Whether this request will run vocal separation inside the transcription pipeline.

    Callers use this to decide whether to detect the language up front or leave it to
    run_transcription, which detects *after* separation so detection sees the same clean
    audio the decoder does (and so the dashboard never shows separation after detection).
    """
    clean_audio_override = getattr(utils.THREAD_CONTEXT, "clean_audio", None)
    return config.ENABLE_VOCAL_SEPARATION if clean_audio_override is None else bool(clean_audio_override)


def _detect_language_after_isolation(language, processed_path, model):
    """Detect the language on already-separated audio, reusing the held unit.

    Returns (language, multilingual_suspected). An explicit request short-circuits with
    suspected=False: forcing one language throughout is exactly what the caller asked
    for, and per-segment re-detection must never override that.
    """
    if language:
        return language, False
    res = language_detection.run_voting_detection_on_isolated(processed_path, model, sys.modules[__name__])
    detected = (res or {}).get("detected_language")
    if not detected:
        logger.info("[LD] No confident consensus on separated audio; leaving detection to the engine.")
    return detected, bool((res or {}).get("multilingual_suspected"))


def _isolate_vocals_if_needed(audio_path, unit_id, perf):
    processed_path = audio_path
    should_clean_audio = will_isolate_vocals()
    if should_clean_audio:
        perf["start_iso"] = time.time()
        check_preemption()
        processed_path = run_vocal_isolation_direct(audio_path, unit_id)
        check_preemption()
        perf["dur_iso"] = time.time() - perf["start_iso"]
    return processed_path


def _execute_transcription_pipeline(
    model,
    processed_path,
    *,
    params,
    unit_id,
    perf,
) -> dict:
    language = params.get("language")
    task = params.get("task")
    diarize = params.get("diarize")
    min_speakers = params.get("min_speakers")
    max_speakers = params.get("max_speakers")
    hf_token = params.get("hf_token")
    initial_prompt = params.get("initial_prompt")
    vad_filter = params.get("vad_filter")
    word_timestamps = params.get("word_timestamps")
    was_auto_detect = params.get("was_auto_detect", not language)
    op_name = "translation" if str(task).lower() == "translate" else "transcription"

    logger.info("[ASR] Starting %s on hardware unit %s", op_name, unit_id)

    perf["start_inf"] = time.time()
    check_preemption()
    scheduler.update_task_metadata(start_inference=perf["start_inf"])
    scheduler.update_task_progress(None, "Inference")
    check_preemption()
    trans_res = model.transcribe(
        processed_path,
        language=language,
        task=task,
        beam_size=config.DEFAULT_BEAM_SIZE,
        initial_prompt=initial_prompt,
        vad_filter=vad_filter,
        word_timestamps=word_timestamps,
        vad_parameters={
            "min_silence_duration_ms": config.VAD_MIN_SILENCE_DURATION_MS,
            "threshold": config.VAD_THRESHOLD,
        },
        # `language` here is always resolved -- detection has already run -- so the engine
        # cannot tell an auto-detected language from one the caller demanded. Only the
        # former may be revised per window; an explicit request must be honoured as given.
        multilingual=was_auto_detect,
    )

    results = consume_transcription_segments(
        trans_res[0],
        trans_res[1],
        task,
        diarize=diarize,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        hf_token=hf_token,
        unit_id=unit_id,
        processed_path=processed_path,
        preemption_check=check_preemption,
    )

    segment_languages = [
        {
            "start": 0.0,
            "end": round(trans_res[1].duration, 2),
            "language": trans_res[1].language,
            "confidence": trans_res[1].language_probability,
        }
    ]
    # Diarization is excluded: it speaker-fingerprints processed_path as one whole file,
    # and a gap re-transcribed on its own slice would get its own local speaker numbering
    # with no correspondence to the rest -- reconciling that is real work of its own.
    if was_auto_detect and not diarize and config.ASR_MULTILINGUAL_SEGMENTATION:
        results, segment_languages = gap_filling.fill_language_gaps(
            model,
            processed_path,
            results,
            segment_languages,
            options={
                "task": task,
                "initial_prompt": initial_prompt,
                "vad_filter": vad_filter,
                "word_timestamps": word_timestamps,
            },
            duration_sec=trans_res[1].duration,
            unit_id=unit_id,
            preemption_check=check_preemption,
        )

    perf["dur_inf"] = time.time() - perf["start_inf"]
    perf["dur_queue"] = _get_queue_duration_from_registry()

    res = {
        "text": "",  # Placeholder
        "segments": results,
        "language": trans_res[1].language,
        "language_probability": trans_res[1].language_probability,
        "video_duration_sec": trans_res[1].duration,
        "segment_languages": segment_languages,
        "performance": {
            "queue_sec": round(perf["dur_queue"], 2),
            "isolation_sec": round(perf["dur_iso"], 2),
            "inference_sec": round(perf["dur_inf"], 2),
        },
    }

    res = post_processing.post_process_results(res)
    res["text"] = utils.generate_srt(res)
    check_preemption()
    scheduler.update_task_metadata(result=res, status="completed", progress=100)
    return res


def _cleanup_isolated_file_safe(audio_path, processed_path):
    if processed_path != audio_path and os.path.exists(processed_path):
        try:
            os.remove(processed_path)
        except (IOError, OSError):
            logger.debug("[Engine] Failed to clean up isolated file")


def _get_queue_duration_from_registry():
    """Read queue duration from a single task-registry snapshot."""
    task_id, _, _, _, _, _ = _get_current_task_info()
    snapshot_now = time.time()
    with scheduler.STATE.task_registry_lock:
        task_entry = scheduler.STATE.task_registry.get(task_id) if task_id else None
        start_active = task_entry.get("start_active", snapshot_now) if task_entry else snapshot_now
        start_time = task_entry.get("start_time", snapshot_now) if task_entry else snapshot_now
    return start_active - start_time


def run_vocal_isolation(audio_path, force=False):
    """Performs UVR vocal isolation using the appropriate hardware unit."""
    with model_lock_ctx() as (_, unit_id):
        return run_vocal_isolation_direct(audio_path, unit_id, force)


def run_vocal_isolation_direct(audio_path, unit_id, force=False, stage="Vocal Separation"):
    """Direct isolation without re-acquiring the lock."""
    preprocessor = _resolve_preprocessor_for_unit(unit_id)
    if not preprocessor:
        return audio_path

    result_path = preprocessor.preprocess_audio(audio_path, force=force, yield_cb=check_preemption, stage=stage)
    if preprocessor.separator:
        scheduler.STATE.uvr_loaded = True

    # Immediate offload to save 2-4GB during the long transcription phase
    if config.AGGRESSIVE_OFFLOAD:
        preprocessor.offload()

    return result_path


def _run_idle_cleanup(timer_handle=None):
    """Timer callback to unload models when idle."""
    logger.info("[Engine] Idle timeout reached. Purging models from memory...")
    unload_models()
    with _CLEANER_TIMER_LOCK:
        if timer_handle is None or CLEANER_STATE["timer"] is timer_handle:
            CLEANER_STATE["timer"] = None


def _schedule_idle_cleanup():
    """Schedules model unloading after idle timeout."""
    if config.MODEL_IDLE_TIMEOUT <= 0:
        return
    with _CLEANER_TIMER_LOCK:
        if CLEANER_STATE["timer"] is not None:
            CLEANER_STATE["timer"].cancel()
        timer_handle = None

        def _cleanup_callback():
            _run_idle_cleanup(timer_handle)

        timer_handle = threading.Timer(config.MODEL_IDLE_TIMEOUT, _cleanup_callback)
        CLEANER_STATE["timer"] = timer_handle
        timer_handle.daemon = True
        timer_handle.start()
        logger.info("[Engine] Scheduled memory cleanup in %ds", config.MODEL_IDLE_TIMEOUT)


def _cancel_idle_cleanup():
    """Cancels any scheduled model unloading."""
    with _CLEANER_TIMER_LOCK:
        if CLEANER_STATE["timer"] is not None:
            CLEANER_STATE["timer"].cancel()
            CLEANER_STATE["timer"] = None
            logger.info("[Engine] Cancelled scheduled memory cleanup because a new task arrived")


def _shutdown_isolated_workers() -> None:
    """Terminate every engine worker, which is the only way device memory comes back.

    In-process teardown reclaims none of a CUDA/ROCm/OpenVINO context's device memory --
    an idle purge on this codebase logged ``CUDA VRAM=193 MB -> 193 MB (Delta: +0 MB)``.
    Killing the process returns it to the OS unconditionally; the next request respawns
    a worker and reloads on demand.
    """
    try:
        isolated_engine = importlib.import_module("modules.inference.engines.isolated_engine")
        isolated_engine.shutdown_all()
    except (ImportError, RuntimeError, OSError) as exc:
        logger.warning("[Engine] Failed to shut down isolated engine workers: %s", exc)

    try:
        isolated_prep = importlib.import_module("modules.inference.pipeline.preprocessing.isolated")
        isolated_prep.shutdown_all()
    except (ImportError, RuntimeError, OSError) as exc:
        logger.warning("[Engine] Failed to shut down isolated preprocessing workers: %s", exc)


def unload_models():
    """Purge all models from RAM/VRAM with extreme prejudice."""
    with _POOL_LOCK:
        mem_before = _read_reclamation_memory_snapshot()
        logger.info(
            "[Engine] Aggressive Offload: Purging models. Current memory: %s",
            _format_reclamation_memory(mem_before),
        )
        whisper_count = _clear_whisper_models(MODEL_POOL)
        uvr_count = _clear_uvr_models(PREPROCESSOR_POOL)
        _clear_whisperx_models(DIARIZE_POOL, ALIGN_POOL)
        _run_garbage_collection_and_reclamation(_ENGINES)
        _shutdown_isolated_workers()
        scheduler.STATE.whisper_loaded = False
        scheduler.STATE.uvr_loaded = False
        time.sleep(0.2)
        mem_after = _read_reclamation_memory_snapshot()
        logger.info(
            "[Engine] Reclamation complete. Memory: %s -> %s (Delta: %s, Released: %d Whisper, %d UVR)",
            _format_reclamation_memory(mem_before),
            _format_reclamation_memory(mem_after),
            _format_reclamation_delta(mem_before, mem_after),
            whisper_count,
            uvr_count,
        )


def increment_active_session():
    """Tracks active session count."""
    _LIFECYCLE_STATE["last_activity"] = time.time()
    scheduler.increment_active_session()
    cancel_idle_cleanup()


def decrement_active_session():
    """Tracks active session count and unloads if idle."""
    _LIFECYCLE_STATE["last_activity"] = time.time()
    scheduler.decrement_active_session()
    current_active = scheduler.STATE.active_sessions
    logger.debug("[Engine] Session decrement. Active sessions remaining: %d", current_active)
    if current_active == 0:
        if config.MODEL_IDLE_TIMEOUT > 0:
            _schedule_idle_cleanup()
        elif config.AGGRESSIVE_OFFLOAD:
            unload_models()


def wait_for_priority():
    """Handles priority task synchronization."""
    scheduler.wait_for_priority()


def check_preemption():
    """Public wrapper for tests and preemption hooks."""
    return _check_preemption()


def cancel_idle_cleanup():
    """Public wrapper for tests and lifecycle hooks."""
    return _cancel_idle_cleanup()
