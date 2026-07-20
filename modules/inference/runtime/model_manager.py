"""
High-Level Model Orchestration and Hardware Lifecycle Management for Whisper Pro ASR.
"""

import importlib
import logging
import os
import threading
import time
import typing

from modules.core import config, logging_setup, utils
from modules.inference import scheduler
from modules.inference.engines import engine_factory
from modules.inference.pipeline import diarization, post_processing, preprocessing, vad
from modules.inference.pipeline.language_detection_core import (
    run_batch_language_detection,
    run_batch_language_detection_direct,
    run_language_detection,
    run_language_detection_core,
)
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

_PUBLIC_API = (
    run_language_detection_core,
    run_language_detection,
    run_batch_language_detection,
    run_batch_language_detection_direct,
    model_lock_ctx,
    _check_preemption,
)


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
    return {
        "active_units": list(MODEL_POOL.keys()),
        "total_units": len(config.HARDWARE_UNITS),
    }


def dummy_engine(*args, **kwargs):
    """Dummy engine function to satisfy pylint type checker for callable targets."""
    return (args, kwargs)


def is_engine_actually_loaded() -> bool:
    """Return whether any primary model remains loaded."""
    return bool(MODEL_POOL)


# Lazy load containers for engines
_ENGINES: typing.Dict[str, typing.Any] = {
    "WhisperModel": dummy_engine,
    "ctranslate2": dummy_engine,
    "IntelWhisperEngine": dummy_engine,
}


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


def _is_accelerated_preprocess_device() -> bool:
    return config.PREPROCESS_DEVICE in {"CUDA", "GPU", "NPU", "OPENVINO", "AMD"}


def _pool_preprocessor_by_type(preferred_type: str):
    for preprocessor in PREPROCESSOR_POOL.values():
        if getattr(preprocessor, "device_type", None) == preferred_type:
            return preprocessor
    return None


def _unit_preprocessor_by_type(preferred_type: str):
    for unit in config.HARDWARE_UNITS:
        if unit.get("type") == preferred_type:
            preprocessor = PREPROCESSOR_POOL.get(unit.get("id"))
            if preprocessor is not None:
                return preprocessor
    return None


def _shared_preprocessor_for_type(preferred_type: str):
    shared_key = f"PREPROCESS::{preferred_type}"
    preprocessor = PREPROCESSOR_POOL.get(shared_key)
    if preprocessor is None:
        matched_unit = next((u for u in config.HARDWARE_UNITS if u.get("type") == preferred_type), None)
        if matched_unit is not None:
            preprocessor = preprocessing.PreprocessingManager(matched_unit)
        else:
            preprocessor = preprocessing.PreprocessingManager()
        PREPROCESSOR_POOL[shared_key] = preprocessor
    return preprocessor


def _preferred_preprocessor() -> typing.Any:
    """Return a preprocessor pinned to the configured preprocess device when available."""
    preferred_type = config.PREPROCESS_DEVICE
    preprocessor = _pool_preprocessor_by_type(preferred_type)
    if preprocessor is not None:
        return preprocessor

    preprocessor = _unit_preprocessor_by_type(preferred_type)
    if preprocessor is not None:
        return preprocessor

    return _shared_preprocessor_for_type(preferred_type)


def _resolve_preprocessor_for_unit(unit_id: str):
    if _is_accelerated_preprocess_device():
        # Prefer the preprocessor bound to the currently assigned accelerator unit
        # so concurrent tasks can use distinct hardware (e.g., CUDA + AMD) in parallel.
        unit_preprocessor = PREPROCESSOR_POOL.get(unit_id)
        if unit_preprocessor is not None:
            unit_type = str(getattr(unit_preprocessor, "device_type", "")).upper()
            if unit_type in {"CUDA", "GPU", "NPU", "OPENVINO", "AMD"}:
                return unit_preprocessor
        return _preferred_preprocessor()
    return PREPROCESSOR_POOL.get(unit_id)


def _run_idle_cleanup():
    """Timer callback to unload models when idle."""
    logger.info("[Engine] Idle timeout reached. Purging models from memory...")
    unload_models()
    with _CLEANER_TIMER_LOCK:
        CLEANER_STATE["timer"] = None


def _schedule_idle_cleanup():
    """Schedules model unloading after idle timeout."""
    if config.MODEL_IDLE_TIMEOUT <= 0:
        return
    with _CLEANER_TIMER_LOCK:
        if CLEANER_STATE["timer"] is not None:
            CLEANER_STATE["timer"].cancel()
        CLEANER_STATE["timer"] = threading.Timer(config.MODEL_IDLE_TIMEOUT, _run_idle_cleanup)
        CLEANER_STATE["timer"].daemon = True
        CLEANER_STATE["timer"].start()
        logger.info("[Engine] Scheduled memory cleanup in %ds", config.MODEL_IDLE_TIMEOUT)


def _cancel_idle_cleanup():
    """Cancels any scheduled model unloading."""
    with _CLEANER_TIMER_LOCK:
        if CLEANER_STATE["timer"] is not None:
            CLEANER_STATE["timer"].cancel()
            CLEANER_STATE["timer"] = None
            logger.info("[Engine] Cancelled scheduled memory cleanup because a new task arrived")


# Stubs for test/code backward compatibility
def _ensure_monitor_thread():
    pass


def load_model():
    """Initializes hardware resource mapping without eager RAM loading."""
    _ensure_monitor_thread()
    for unit in config.HARDWARE_UNITS:
        # Initialize preprocessor managers (they are lazy and won't load models yet)
        PREPROCESSOR_POOL[unit["id"]] = preprocessing.PreprocessingManager(unit)

    scheduler.STATE.engine_initialized = True
    if config.ENABLE_VOCAL_SEPARATION:
        scheduler.STATE.uvr_loaded = True
    return True


# Alias for backward compatibility with tests
init_pool = load_model


def init_unit(unit):
    """Loads model for a specific hardware unit."""
    _lazy_import_engines()
    with _POOL_LOCK:
        try:
            logger.info("[Engine] Loading %s on %s...", config.MODEL_ID, unit["name"])

            model = engine_factory.create_engine(config.ASR_ENGINE, config.MODEL_ID, unit)

            MODEL_POOL[unit["id"]] = model
            PREPROCESSOR_POOL[unit["id"]] = preprocessing.PreprocessingManager(unit)
            scheduler.STATE.whisper_loaded = True
            scheduler.STATE.engine_initialized = True
            logger.info("[Engine] %s ready.", unit["id"])
        except (ValueError, RuntimeError, ImportError, AttributeError, KeyError, OSError, TypeError) as e:
            logger.error("[Engine] Failed to load %s: %s", unit["id"], e)


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


def _isolate_vocals_if_needed(audio_path, unit_id, perf):
    processed_path = audio_path
    clean_audio_override = getattr(utils.THREAD_CONTEXT, "clean_audio", None)
    should_clean_audio = config.ENABLE_VOCAL_SEPARATION if clean_audio_override is None else bool(clean_audio_override)
    if should_clean_audio:
        perf["start_iso"] = time.time()
        _check_preemption()
        processed_path = run_vocal_isolation_direct(audio_path, unit_id)
        _check_preemption()
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
    op_name = "translation" if str(task).lower() == "translate" else "transcription"

    logger.info("[ASR] Starting %s on hardware unit %s", op_name, unit_id)

    perf["start_inf"] = time.time()
    _check_preemption()
    scheduler.update_task_metadata(start_inference=perf["start_inf"])
    scheduler.update_task_progress(None, "Inference")
    _check_preemption()
    trans_res = model.transcribe(
        processed_path,
        language=language,
        task=task,
        beam_size=config.DEFAULT_BEAM_SIZE,
        initial_prompt=initial_prompt,
        vad_filter=vad_filter,
        word_timestamps=word_timestamps,
        vad_parameters={"min_silence_duration_ms": config.VAD_MIN_SILENCE_DURATION_MS},
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
        preemption_check=_check_preemption,
    )

    perf["dur_inf"] = time.time() - perf["start_inf"]
    perf["dur_queue"] = _get_queue_duration_from_registry()

    res = {
        "text": "",  # Placeholder
        "segments": results,
        "language": trans_res[1].language,
        "language_probability": trans_res[1].language_probability,
        "video_duration_sec": trans_res[1].duration,
        "performance": {
            "queue_sec": round(perf["dur_queue"], 2),
            "isolation_sec": round(perf["dur_iso"], 2),
            "inference_sec": round(perf["dur_inf"], 2),
        },
    }

    res = post_processing.post_process_results(res)
    res["text"] = utils.generate_srt(res)
    _check_preemption()
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


def run_vocal_isolation_direct(audio_path, unit_id, force=False):
    """Direct isolation without re-acquiring the lock."""
    preprocessor = _resolve_preprocessor_for_unit(unit_id)
    if not preprocessor:
        return audio_path

    result_path = preprocessor.preprocess_audio(audio_path, force=force, yield_cb=_check_preemption)
    if preprocessor.separator:
        scheduler.STATE.uvr_loaded = True

    # Immediate offload to save 2-4GB during the long transcription phase
    if config.AGGRESSIVE_OFFLOAD:
        preprocessor.offload()

    return result_path


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

        scheduler.STATE.whisper_loaded = False
        scheduler.STATE.uvr_loaded = False

        time.sleep(0.2)  # Give OS time to update page tables
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
    _cancel_idle_cleanup()


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
