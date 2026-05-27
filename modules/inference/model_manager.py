"""
High-Level Model Orchestration and Hardware Lifecycle Management for Whisper Pro ASR.
"""
# pylint: disable=too-many-lines
import logging
import threading
import time
import gc
import os
import ctypes
from modules import config, logging_setup, utils
from modules.inference import preprocessing, vad, scheduler, post_processing, diarization

_post_process_results = post_processing.post_process_results


# Lazy load containers for engines
_ENGINES = {
    "WhisperModel": None,
    "ctranslate2": None,
    "IntelWhisperEngine": None
}


def _lazy_import_engines():
    """Lazily import inference engines to save 500MB+ RAM during startup."""
    if _ENGINES["WhisperModel"] is None:
        try:
            from faster_whisper import WhisperModel as wm  # pylint: disable=import-outside-toplevel
            import ctranslate2 as ct  # pylint: disable=import-outside-toplevel
            _ENGINES["WhisperModel"] = wm
            _ENGINES["ctranslate2"] = ct
            # Ensure VAD monkeypatching is applied after faster_whisper load
            vad._lazy_import_vad()  # pylint: disable=protected-access
            if config.ASR_ENGINE == "INTEL-WHISPER":
                from modules.inference.intel_engine import IntelWhisperEngine as iwe  # pylint: disable=import-outside-toplevel
                _ENGINES["IntelWhisperEngine"] = iwe
        except ImportError as e:
            logger.warning("[Engine] Failed to lazy load engines: %s", e)


# Ensure external AI engines log to our task-aware system
logging.getLogger("faster_whisper").setLevel(logging.INFO)
logging.getLogger("audio_separator").setLevel(logging.INFO)

TASK_LOGS = logging_setup.TASK_LOGS
logger = logging.getLogger(__name__)

_MODEL_POOL = {}
_PREPROCESSOR_POOL = {}
_ALIGN_POOL = diarization._ALIGN_POOL  # pylint: disable=protected-access
_DIARIZE_POOL = diarization._DIARIZE_POOL  # pylint: disable=protected-access

_LIFECYCLE_STATE = {
    "last_activity": time.time(),
    "monitor_started": False
}
_MONITOR_LOCK = threading.Lock()


def _monitor_idleness():
    while True:
        time.sleep(5)
        # Check if active sessions are zero and idle timeout has elapsed
        if scheduler.STATE.active_sessions == 0:
            elapsed = time.time() - _LIFECYCLE_STATE["last_activity"]
            if elapsed > config.MODEL_IDLE_TIMEOUT:
                logger.info("[Engine] Models idle for %.1fs. Purging from memory...", elapsed)
                unload_models()


def _ensure_monitor_thread():
    if config.MODEL_IDLE_TIMEOUT <= 0:
        return
    with _MONITOR_LOCK:
        if not _LIFECYCLE_STATE["monitor_started"]:
            t = threading.Thread(target=_monitor_idleness, daemon=True, name="ModelIdleMonitor")
            t.start()
            _LIFECYCLE_STATE["monitor_started"] = True


def load_model():
    """Initializes hardware resource mapping without eager RAM loading."""
    _ensure_monitor_thread()
    for unit in config.HARDWARE_UNITS:
        # Initialize preprocessor managers (they are lazy and won't load models yet)
        _PREPROCESSOR_POOL[unit['id']] = preprocessing.PreprocessingManager(unit)

    scheduler.STATE.engine_initialized = True
    if config.ENABLE_VOCAL_SEPARATION:
        scheduler.STATE.uvr_loaded = True
    return True


# Alias for backward compatibility with tests
init_pool = load_model


def _init_unit(unit):
    """Loads model for a specific hardware unit."""
    _lazy_import_engines()
    try:
        logger.info("[Engine] Loading %s on %s...", config.MODEL_ID, unit['name'])

        if config.ASR_ENGINE == "INTEL-WHISPER" and unit['type'] in ['GPU', 'NPU', 'CPU']:
            model = _ENGINES["IntelWhisperEngine"](config.MODEL_ID, device=unit['id'])  # pylint: disable=not-callable
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

            model = _ENGINES["WhisperModel"](  # pylint: disable=not-callable
                config.MODEL_ID,
                device=target_device,
                device_index=unit.get('index', 0),
                compute_type=config.COMPUTE_TYPE,
                cpu_threads=config.ASR_THREADS,
                download_root=config.OV_CACHE_DIR
            )

        _MODEL_POOL[unit['id']] = model
        _PREPROCESSOR_POOL[unit['id']] = preprocessing.PreprocessingManager(unit)
        scheduler.STATE.whisper_loaded = True
        scheduler.STATE.engine_initialized = True
        logger.info("[Engine] %s ready.", unit['id'])
    except (ValueError, RuntimeError, IOError) as e:
        logger.error("[Engine] Failed to load %s: %s", unit['id'], e)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("[Engine] Failed to load %s (Unexpected): %s", unit['id'], e)


# pylint: disable=too-many-locals,too-many-positional-arguments


def run_transcription(
    audio_path, language, task, diarize=False, min_speakers=None,
    max_speakers=None, hf_token=None, initial_prompt=None,
    vad_filter=True, word_timestamps=False, **_kwargs
):
    """Executes ASR inference with hardware locking."""
    _LIFECYCLE_STATE["last_activity"] = time.time()
    perf = {'dur_iso': 0}
    with model_lock_ctx() as (model, unit_id):
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
            scheduler.update_task_progress(10, "Inference")
            segments, info = model.transcribe(
                processed_path,
                language=language,
                task=task,
                beam_size=config.DEFAULT_BEAM_SIZE,
                initial_prompt=initial_prompt,
                vad_filter=vad_filter,
                vad_parameters={"min_silence_duration_ms": config.VAD_MIN_SILENCE_DURATION_MS},
                word_timestamps=word_timestamps
            )

            # Consumption of generator with progress updates
            if diarize:
                try:
                    # 1. Consume generator to collect all raw segments first
                    raw_segments = []
                    for segment in segments:
                        _check_preemption()
                        raw_segments.append({
                            "start": segment.start,
                            "end": segment.end,
                            "text": segment.text
                        })

                        # Live updates during transcription
                        live_results = [{
                            "start": round(s["start"], 2),
                            "end": round(s["end"], 2),
                            "text": s["text"].strip()
                        } for s in raw_segments]
                        live_srt = utils.generate_srt({"segments": live_results})
                        scheduler.update_task_metadata(live_text=live_srt)

                        if info.duration > 0:
                            progress = min(80, int((segment.end / info.duration) * 80))
                            time_ratio = f"{utils.format_duration(segment.end)} / {utils.format_duration(info.duration)}"
                            verb = "Translating" if task == "translate" else "Transcribing"
                            scheduler.update_task_progress(
                                progress, f"{verb} (Seg {len(raw_segments)} | {time_ratio})")

                    if raw_segments:
                        results = diarization.run_diarization(
                            processed_path=processed_path,
                            raw_segments=raw_segments,
                            info=info,
                            language=language,
                            min_speakers=min_speakers,
                            max_speakers=max_speakers,
                            hf_token=hf_token,
                            unit_id=unit_id
                        )
                    else:
                        results = []
                except Exception as diarize_err:  # pylint: disable=broad-exception-caught
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
                for segment in segments:
                    _check_preemption()
                    seg_text = segment.text.strip()
                    results.append({
                        "start": round(segment.start, 2),
                        "end": round(segment.end, 2),
                        "text": seg_text
                    })

                    # Generate live SRT content (includes timestamps and newlines)
                    live_srt = utils.generate_srt({"segments": results})
                    logger.debug("[Live] Updating SRT for %d segments", len(results))
                    scheduler.update_task_metadata(live_text=live_srt)

                    if info.duration > 0:
                        progress = min(95, int((segment.end / info.duration) * 100))
                        time_ratio = f"{utils.format_duration(segment.end)} / {utils.format_duration(info.duration)}"
                        verb = "Translating" if task == "translate" else "Transcribing"
                        scheduler.update_task_progress(
                            progress, f"{verb} (Seg {len(results)} | {time_ratio})")

            perf['dur_inf'] = time.time() - perf['start_inf']

            # Fetch timing from registry for the final performance report
            perf['dur_queue'] = 0
            with scheduler.STATE.task_registry_lock:
                t_meta = scheduler.STATE.task_registry.get(threading.get_ident(), {})
                perf['dur_queue'] = t_meta.get(
                    "start_active", time.time()) - t_meta.get("start_time", time.time())

            res = {
                "text": "",  # Placeholder
                "segments": results,
                "language": info.language,
                "language_probability": info.language_probability,
                "video_duration_sec": info.duration,
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
                except (IOError, OSError) as cleanup_err:
                    logger.debug("[Engine] Failed to clean up isolated file: %s", cleanup_err)


def run_vocal_isolation(audio_path, force=False):
    """Performs UVR vocal isolation using the appropriate hardware unit."""
    with model_lock_ctx() as (_, unit_id):
        return run_vocal_isolation_direct(audio_path, unit_id, force)


def run_vocal_isolation_direct(audio_path, unit_id, force=False):
    """Direct isolation without re-acquiring the lock."""
    preprocessor = _PREPROCESSOR_POOL.get(unit_id)
    if not preprocessor:
        return audio_path

    result_path = preprocessor.preprocess_audio(audio_path, force=force, yield_cb=_check_preemption)
    if preprocessor.separator:
        scheduler.STATE.uvr_loaded = True

    # Immediate offload to save 2-4GB during the long transcription phase
    if config.AGGRESSIVE_OFFLOAD:
        preprocessor.offload()

    return result_path


# fmt: off
from modules.inference.language_detection_core import (  # pylint: disable=wrong-import-position, unused-import
    run_language_detection,
    run_batch_language_detection,
    run_batch_language_detection_direct,
    run_language_detection_core
)
# fmt: on


def update_task_result(_audio_path, result):
    """Updates the final result in the task registry."""
    scheduler.update_task_metadata(result=result)


# fmt: off
from modules.inference.concurrency import model_lock_ctx, _check_preemption  # pylint: disable=wrong-import-position, unused-import
# fmt: on


def unload_models():
    """Purge all models from RAM/VRAM with extreme prejudice."""
    mem_before = utils.get_system_telemetry().get("app_memory_gb", 0)
    logger.info("[Engine] Aggressive Offload: Purging models. Current RAM: %s GB", mem_before)

    # 1. Clear Whisper models
    with scheduler.STATE.model_lock:
        whisper_count = len(_MODEL_POOL)
        for unit_id in list(_MODEL_POOL.keys()):
            model = _MODEL_POOL.pop(unit_id)
            try:
                if hasattr(model, 'unload'):
                    model.unload()
                elif hasattr(model, 'pipeline'):
                    model.pipeline = None
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.debug("[Engine] Error unloading model %s: %s", unit_id, e)
            del model
        _MODEL_POOL.clear()

    # 2. Clear UVR models
    uvr_count = len(_PREPROCESSOR_POOL)
    for unit_id in list(_PREPROCESSOR_POOL.keys()):
        pm = _PREPROCESSOR_POOL.pop(unit_id)
        try:
            pm.unload_model()
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.debug("[Engine] Error unloading UVR %s: %s", unit_id, e)
        del pm
    _PREPROCESSOR_POOL.clear()

    # 3. Clear WhisperX Diarize/Align models
    for unit_id in list(_DIARIZE_POOL.keys()):
        model_d = _DIARIZE_POOL.pop(unit_id)
        del model_d
    _DIARIZE_POOL.clear()

    for key in list(_ALIGN_POOL.keys()):
        model_a, metadata = _ALIGN_POOL.pop(key)
        del model_a
        del metadata
    _ALIGN_POOL.clear()

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
    _ensure_monitor_thread()


def decrement_active_session():
    """Tracks active session count and unloads if idle."""
    _LIFECYCLE_STATE["last_activity"] = time.time()
    scheduler.decrement_active_session()
    current_active = scheduler.STATE.active_sessions
    logger.debug("[Engine] Session decrement. Active sessions remaining: %d", current_active)

    if current_active == 0:
        if config.MODEL_IDLE_TIMEOUT > 0:
            _ensure_monitor_thread()
        elif config.AGGRESSIVE_OFFLOAD:
            unload_models()


def wait_for_priority():
    """Handles priority task synchronization."""
    scheduler.wait_for_priority()


def is_engine_actually_loaded():
    """Deep check of RAM state."""
    return len(_MODEL_POOL) > 0


def is_uvr_actually_loaded():
    """Check if any UVR models are in RAM."""
    return any(pm.separator is not None for pm in _PREPROCESSOR_POOL.values())


# Proxy functions for backward compatibility
early_task_registration = scheduler.early_task_registration
cleanup_failed_task = scheduler.cleanup_failed_task
update_task_metadata = scheduler.update_task_metadata
update_task_progress = scheduler.update_task_progress
is_engine_initialized = scheduler.is_engine_initialized


def get_status():
    """Returns engine-specific diagnostics for the telemetry system."""
    return {
        "active_units": len(_MODEL_POOL),
        "total_units": len(config.HARDWARE_UNITS),
        "whisper_loaded": scheduler.STATE.whisper_loaded,
        "uvr_loaded": scheduler.STATE.uvr_loaded,
        "engine_initialized": scheduler.STATE.engine_initialized
    }
