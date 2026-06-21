"""
High-Level Model Orchestration and Hardware Lifecycle Management for Whisper Pro ASR.
"""

import logging
import threading
import time
import gc
import os
import ctypes
import typing
import importlib
from modules import config, logging_setup, utils
from modules.inference import preprocessing, vad, scheduler, post_processing, diarization
from modules.inference.language_detection_core import (
    run_language_detection,
    run_batch_language_detection,
    run_batch_language_detection_direct,
    run_language_detection_core
)
from modules.inference.concurrency import model_lock_ctx, _check_preemption

# Re-export for public API and compatibility
_ = (
    run_language_detection,
    run_batch_language_detection,
    run_batch_language_detection_direct,
    run_language_detection_core,
    model_lock_ctx,
    _check_preemption
)

_post_process_results = post_processing.post_process_results


def dummy_engine(*args, **kwargs):
    """Dummy engine function to satisfy pylint type checker for callable targets."""
    return (args, kwargs)


# Lazy load containers for engines
_ENGINES: typing.Dict[str, typing.Any] = {
    "WhisperModel": dummy_engine,
    "ctranslate2": dummy_engine,
    "IntelWhisperEngine": dummy_engine
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
                intel_engine = importlib.import_module("modules.inference.intel_engine")
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

_LIFECYCLE_STATE = {
    "last_activity": time.time(),
    "monitor_started": False
}
_MONITOR_LOCK = threading.Lock()


CLEANER_STATE = {
    "timer": None
}
_CLEANER_TIMER_LOCK = threading.Lock()
_POOL_LOCK = threading.Lock()


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
        PREPROCESSOR_POOL[unit['id']] = preprocessing.PreprocessingManager(unit)

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
            logger.info("[Engine] Loading %s on %s...", config.MODEL_ID, unit['name'])

            if config.ASR_ENGINE == "INTEL-WHISPER" and unit['type'] in ['GPU', 'NPU', 'CPU']:
                model = _ENGINES["IntelWhisperEngine"](config.MODEL_ID, device=unit['id'])
            else:
                # Resolve device for Faster-Whisper
                if unit['type'] == 'CUDA':
                    target_device = 'cuda'
                else:
                    # Intel NPU/GPU are NOT supported by Faster-Whisper directly.
                    # Force CPU fallback for Whisper inference while allowing
                    # Preprocessing (UVR) to still use the accelerator.
                    target_device = 'cpu'
                    if unit['type'] in ['NPU', 'GPU']:
                        logger.info(
                            "[Engine] Intel accelerator detected. Whisper will use CPU for this slot.")

                model = _ENGINES["WhisperModel"](
                    config.MODEL_ID,
                    device=target_device,
                    device_index=unit.get('index', 0),
                    compute_type=config.COMPUTE_TYPE,
                    cpu_threads=config.ASR_THREADS,
                    download_root=config.OV_CACHE_DIR
                )

            MODEL_POOL[unit['id']] = model
            PREPROCESSOR_POOL[unit['id']] = preprocessing.PreprocessingManager(unit)
            scheduler.STATE.whisper_loaded = True
            scheduler.STATE.engine_initialized = True
            logger.info("[Engine] %s ready.", unit['id'])
        except (ValueError, RuntimeError, ImportError, AttributeError, KeyError, OSError, TypeError) as e:
            logger.error("[Engine] Failed to load %s: %s", unit['id'], e)


def _consume_segments(
    segments, info, task, *, diarize, min_speakers, max_speakers, hf_token, unit_id, processed_path
):
    """Consume the segments generator and run diarization if requested."""
    if diarize:
        try:
            # 1. Consume generator to collect all raw segments first
            raw_segments = []
            live_srt_blocks = []
            for segment in segments:
                _check_preemption()
                raw_segments.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text
                })

                # Live updates during transcription
                # Append only the new segment block for O(1) performance (no O(N²) rebuilds)
                seg_idx = len(raw_segments)
                block = utils.format_single_srt_block(
                    idx=seg_idx,
                    start_ts=segment.start,
                    end_ts=segment.end,
                    text=segment.text
                )
                live_srt_blocks.append(block)
                scheduler.update_task_metadata(live_text="".join(live_srt_blocks), current_position=segment.end)

                if info.duration > 0:
                    scheduler.update_task_progress(
                        min(80, int((segment.end / info.duration) * 80)),
                        f"{'Translating' if task == 'translate' else 'Transcribing'} (Seg {len(raw_segments)} | "
                        f"{utils.format_duration(segment.end)} / {utils.format_duration(info.duration)})"
                    )

                if seg_idx % 100 == 0 or seg_idx == 1:
                    logger.info("[Engine] %s segment %d (Audio: %s / %s)",
                                "Translating" if task == "translate" else "Transcribing",
                                seg_idx, utils.format_duration(segment.end), utils.format_duration(info.duration))

            if raw_segments:
                results = diarization.run_diarization(
                    processed_path=processed_path,
                    raw_segments=raw_segments,
                    info=info,
                    language=info.language,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                    hf_token=hf_token,
                    unit_id=unit_id
                )
            else:
                results = []
        except (ValueError, TypeError, KeyError, AttributeError, OSError, RuntimeError) as diarize_err:
            logger.error("[Diarization] Diarization failed: %s. Falling back to non-diarized output.", diarize_err)
            # Fallback to formatting raw_segments without diarization
            results = []
            for s in raw_segments:
                results.append({
                    "start": round(s["start"], 2),
                    "end": round(s["end"], 2),
                    "text": s["text"].strip()
                })
    else:
        results = []
        live_srt_blocks = []
        for segment in segments:
            _check_preemption()
            results.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip()
            })

            # Append the new segment's formatted SRT block
            block = utils.format_single_srt_block(
                idx=len(results),
                start_ts=segment.start,
                end_ts=segment.end,
                text=segment.text
            )
            live_srt_blocks.append(block)
            scheduler.update_task_metadata(live_text="".join(live_srt_blocks), current_position=segment.end)

            if info.duration > 0:
                scheduler.update_task_progress(
                    min(95, int((segment.end / info.duration) * 100)),
                    f"{'Translating' if task == 'translate' else 'Transcribing'} (Seg {len(results)} | "
                    f"{utils.format_duration(segment.end)} / {utils.format_duration(info.duration)})"
                )

            seg_count = len(results)
            if seg_count % 100 == 0 or seg_count == 1:
                logger.info("[Engine] %s segment %d (Audio: %s / %s)",
                            "Translating" if task == "translate" else "Transcribing",
                            seg_count, utils.format_duration(segment.end), utils.format_duration(info.duration))
    return results


def run_transcription(
    audio_path, language, task, *, diarize=False, min_speakers=None,
    max_speakers=None, hf_token=None, initial_prompt=None,
    vad_filter=True, word_timestamps=False, **_kwargs
):
    """Executes ASR inference with hardware locking."""
    _LIFECYCLE_STATE["last_activity"] = time.time()
    perf = {'dur_iso': 0}
    with model_lock_ctx() as (model, unit_id):
        # Determine and set audio duration early for estimated speed display on dashboard
        try:
            audio_duration = utils.get_audio_duration(audio_path)
            scheduler.update_task_metadata(video_duration=audio_duration)
        except tuple([Exception]) as e:
            logger.warning("[Engine] Failed to get audio duration early: %s", e)

        processed_path = audio_path
        if config.ENABLE_VOCAL_SEPARATION:
            perf['start_iso'] = time.time()
            scheduler.update_task_progress(5, "Vocal Separation")
            _check_preemption()
            processed_path = run_vocal_isolation_direct(audio_path, unit_id)
            _check_preemption()
            perf['dur_iso'] = time.time() - perf['start_iso']

        try:
            perf['start_inf'] = time.time()
            scheduler.update_task_metadata(start_inference=perf['start_inf'])
            scheduler.update_task_progress(10, "Inference")
            trans_res = model.transcribe(
                processed_path,
                language=language,
                task=task,
                beam_size=config.DEFAULT_BEAM_SIZE,
                initial_prompt=initial_prompt,
                vad_filter=vad_filter,
                vad_parameters={"min_silence_duration_ms": config.VAD_MIN_SILENCE_DURATION_MS},
                word_timestamps=word_timestamps
            )

            results = _consume_segments(
                trans_res[0], trans_res[1], task,
                diarize=diarize,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                hf_token=hf_token,
                unit_id=unit_id,
                processed_path=processed_path
            )

            perf['dur_inf'] = time.time() - perf['start_inf']

            # Fetch timing from registry for the final performance report
            perf['dur_queue'] = 0
            with scheduler.STATE.task_registry_lock:
                perf['dur_queue'] = (
                    scheduler.STATE.task_registry.get(threading.get_ident(), {})
                ).get("start_active", time.time()) - (
                    scheduler.STATE.task_registry.get(threading.get_ident(), {})
                ).get("start_time", time.time())

            res = {
                "text": "",  # Placeholder
                "segments": results,
                "language": trans_res[1].language,
                "language_probability": trans_res[1].language_probability,
                "video_duration_sec": trans_res[1].duration,
                "performance": {
                    "queue_sec": round(perf['dur_queue'], 2),
                    "isolation_sec": round(perf['dur_iso'], 2),
                    "inference_sec": round(perf['dur_inf'], 2)
                }
            }

            # Apply hallucination filters
            res = post_processing.post_process_results(res)

            res["text"] = utils.generate_srt(res)
            scheduler.update_task_metadata(result=res, status="completed", progress=100)
            return res
        finally:
            if processed_path != audio_path and os.path.exists(processed_path):
                try:
                    os.remove(processed_path)
                except (IOError, OSError):
                    logger.debug("[Engine] Failed to clean up isolated file")


def run_vocal_isolation(audio_path, force=False):
    """Performs UVR vocal isolation using the appropriate hardware unit."""
    with model_lock_ctx() as (_, unit_id):
        return run_vocal_isolation_direct(audio_path, unit_id, force)


def run_vocal_isolation_direct(audio_path, unit_id, force=False):
    """Direct isolation without re-acquiring the lock."""
    preprocessor = PREPROCESSOR_POOL.get(unit_id)
    if not preprocessor:
        return audio_path

    result_path = preprocessor.preprocess_audio(audio_path, force=force, yield_cb=_check_preemption)
    if preprocessor.separator:
        scheduler.STATE.uvr_loaded = True

    # Immediate offload to save 2-4GB during the long transcription phase
    if config.AGGRESSIVE_OFFLOAD:
        preprocessor.offload()

    return result_path


# Bottom imports removed and moved to top level


def unload_models():
    """Purge all models from RAM/VRAM with extreme prejudice."""
    with _POOL_LOCK:
        mem_before = utils.get_system_telemetry().get("app_memory_gb", 0)
        logger.info("[Engine] Aggressive Offload: Purging models. Current RAM: %s GB", mem_before)

        # 1. Clear Whisper models
        with scheduler.STATE.model_lock:
            whisper_count = len(MODEL_POOL)
            for unit_id in list(MODEL_POOL.keys()):
                model = MODEL_POOL.pop(unit_id)
                try:
                    if hasattr(model, 'unload'):
                        model.unload()
                    elif hasattr(model, 'pipeline'):
                        model.pipeline = None
                except tuple([Exception]) as e:
                    logger.debug("[Engine] Error unloading model %s: %s", unit_id, e)
                del model
            MODEL_POOL.clear()

        # 2. Clear UVR models
        uvr_count = len(PREPROCESSOR_POOL)
        for unit_id in list(PREPROCESSOR_POOL.keys()):
            pm = PREPROCESSOR_POOL.pop(unit_id)
            try:
                pm.unload_model()
            except tuple([Exception]) as e:
                logger.debug("[Engine] Error unloading UVR %s: %s", unit_id, e)
            del pm
        PREPROCESSOR_POOL.clear()

        # 3. Clear WhisperX Diarize/Align models
        for unit_id in list(DIARIZE_POOL.keys()):
            model_d = DIARIZE_POOL.pop(unit_id)
            del model_d
        DIARIZE_POOL.clear()

        for key in list(ALIGN_POOL.keys()):
            model_a, metadata = ALIGN_POOL.pop(key)
            del model_a
            del metadata
        ALIGN_POOL.clear()

        # 3. Aggressive GC and cache flushing
        gc.collect()
        gc.collect()  # Second pass for circular references

        if _ENGINES["ctranslate2"]:
            try:
                _ENGINES["ctranslate2"].clear_caches()
            except (AttributeError, RuntimeError):
                pass

        # Release GPU memory if applicable
        utils.clear_gpu_cache()

        # Force OS memory reclamation
        try:
            # Linux/Docker optimization
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except (OSError, AttributeError):
            pass

        scheduler.STATE.whisper_loaded = False
        scheduler.STATE.uvr_loaded = False

        time.sleep(0.2)  # Give OS time to update page tables
        gc.collect()

        mem_after = utils.get_system_telemetry().get("app_memory_gb", 0)
        logger.info("[Engine] Reclamation complete. RAM: %s GB -> %s GB (Released: %d Whisper, %d UVR)",
                    mem_before, mem_after, whisper_count, uvr_count)


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


def is_engine_actually_loaded():
    """Deep check of RAM state."""
    return len(MODEL_POOL) > 0


def is_uvr_actually_loaded():
    """Check if any UVR models are in RAM."""
    return any(pm.separator is not None for pm in PREPROCESSOR_POOL.values())


# Proxy functions for backward compatibility
early_task_registration = scheduler.early_task_registration
cleanup_failed_task = scheduler.cleanup_failed_task
update_task_metadata = scheduler.update_task_metadata
update_task_progress = scheduler.update_task_progress
is_engine_initialized = scheduler.is_engine_initialized


def get_status():
    """Returns engine-specific diagnostics for the telemetry system."""
    return {
        "active_units": len(MODEL_POOL),
        "total_units": len(config.HARDWARE_UNITS),
        "whisper_loaded": scheduler.STATE.whisper_loaded,
        "uvr_loaded": scheduler.STATE.uvr_loaded,
        "engine_initialized": scheduler.STATE.engine_initialized
    }
