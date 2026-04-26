"""
High-Level Model Orchestration and Hardware Lifecycle Management for Whisper Pro ASR.
"""
# pylint: disable=too-many-lines
import logging
import threading
import time
import gc
import contextlib
import os
import ctypes
from modules import config, logging_setup, utils
from modules.inference import preprocessing, vad, scheduler

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


def load_model():
    """Initializes hardware resource mapping without eager RAM loading."""
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


def run_transcription(audio_path, language, task, **_kwargs):
    """Executes ASR inference with hardware locking."""
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
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": config.VAD_MIN_SILENCE_DURATION_MS}
            )

            # Consumption of generator with progress updates
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
            res = _post_process_results(res)

            res["text"] = utils.generate_srt(res)
            scheduler.update_task_metadata(result=res, status="completed", progress=100)
            return res
        finally:
            if processed_path != audio_path and os.path.exists(processed_path):
                try:
                    os.remove(processed_path)
                except (IOError, OSError) as cleanup_err:
                    logger.debug("[Engine] Failed to clean up isolated file: %s", cleanup_err)


def _post_process_results(result, _audio_path=None):
    """Applies quality filters to the raw transcription output."""
    if not result or "segments" not in result:
        return result

    segments = result["segments"]
    if not segments:
        return result

    processed_segments = []
    repetition_count = 0
    last_text = ""

    for seg in segments:
        text = seg.get("text", "").strip()
        prob = seg.get("probability", 1.0)  # probability might not be in all engines

        # 1. Silence/Low confidence filter
        if prob < config.HALLUCINATION_SILENCE_THRESHOLD:
            seg["text"] = ""
            logger.debug("[Filter] Dropped segment due to low confidence (%.2f)", prob)

        # 2. Phrase filter
        elif any(phrase.lower() in text.lower() for phrase in config.HALLUCINATION_PHRASES):
            seg["text"] = ""
            logger.debug("[Filter] Dropped segment containing hallucination phrase")

        # 3. Repetition filter
        elif text == last_text and text != "":
            repetition_count += 1
            if repetition_count >= config.HALLUCINATION_REPETITION_THRESHOLD:
                seg["text"] = ""
                logger.debug("[Filter] Dropped repetitive segment")
        else:
            repetition_count = 0
            last_text = text

        processed_segments.append(seg)

    result["segments"] = processed_segments
    return result


def run_vocal_isolation(audio_path, force=False):
    """Performs UVR vocal isolation using the appropriate hardware unit."""
    with model_lock_ctx() as (_, unit_id):
        return run_vocal_isolation_direct(audio_path, unit_id, force)


def run_vocal_isolation_direct(audio_path, unit_id, force=False):
    """Direct isolation without re-acquiring the lock."""
    preprocessor = _PREPROCESSOR_POOL.get(unit_id)
    if not preprocessor:
        return audio_path

    result_path = preprocessor.preprocess_audio(audio_path, force=force)
    if preprocessor.separator:
        scheduler.STATE.uvr_loaded = True

    # Immediate offload to save 2-4GB during the long transcription phase
    if config.AGGRESSIVE_OFFLOAD:
        preprocessor.offload()

    return result_path


def run_language_detection(audio_path):
    """Optimized language detection using the faster detect_language API."""
    start_time = time.time()
    with model_lock_ctx() as (model, _):
        scheduler.update_task_progress(5, "Detection")
        res = run_language_detection_core(model, audio_path)
        res['performance'] = {"inference_sec": round(time.time() - start_time, 2)}
        scheduler.update_task_metadata(result=res)
        return res


def run_batch_language_detection(audio_path, segment_count):
    """High-performance multi-segment identification scan."""
    with model_lock_ctx() as (model, _):
        return run_batch_language_detection_direct(model, audio_path, segment_count)


def run_batch_language_detection_direct(model, audio_path, segment_count):
    """Direct batch detection without re-acquiring the lock."""
    full_audio = None
    try:
        full_audio = vad.decode_audio(audio_path)
        # Split montage into individual 30s segments
        results = []
        segment_len = int(30 * 16000)
        for i in range(segment_count):
            start = i * segment_len
            end = min(start + segment_len, len(full_audio))
            if start >= len(full_audio):
                break

            # Use .copy() to avoid keeping the full array alive via views
            chunk = full_audio[start:end].copy()
            results.append(run_language_detection_core(model, chunk, skip_vad=True))

            # Granular progress for voting (Maps 60% -> 95%)
            progress = 60 + int(((i + 1) / segment_count) * 35)
            stage = f"Inference ({i+1}/{segment_count})"
            logger.info("[Engine] %s...", stage)
            scheduler.update_task_progress(progress, stage)
        return results
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("[Engine] Batch detection failed: %s", e)
        return []
    finally:
        if full_audio is not None:
            del full_audio
        # Ensure results don't hold any references to the original buffer
        gc.collect()


def run_language_detection_core(model, audio_input, skip_vad=False):
    """Internal core using detect_language optimization."""
    if not skip_vad:
        if isinstance(audio_input, str):
            speech_ts = vad.get_speech_timestamps_from_path(audio_input)
        else:
            speech_ts = vad.get_speech_timestamps(audio_input)

        if not speech_ts:
            return {
                "detected_language": "en",
                "language": "en",
                "confidence": 0.0,
                "all_probabilities": {"en": 0.0}
            }

    try:
        # Optimization: Use detect_language to avoid full decoding
        if isinstance(audio_input, str):
            audio_input = vad.decode_audio(audio_input)

        if hasattr(audio_input, 'astype'):
            audio_input = audio_input.astype('float32')

        lang_code, lang_prob, all_probs_list = model.detect_language(audio_input)
        logger.info("[Engine] Identified: %s (%.1f%%)", lang_code, lang_prob * 100)
        return {
            "detected_language": lang_code,
            "language": lang_code,
            "confidence": lang_prob,
            "all_probabilities": dict(all_probs_list) if all_probs_list else {lang_code: lang_prob}
        }
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.info("[Engine] detect_language fallback: %s", e)
        # Fallback to minimal transcribe
        _, info = model.transcribe(audio_input, beam_size=1, task="transcribe")
        return {
            "detected_language": info.language,
            "language": info.language,
            "confidence": info.language_probability,
            "all_probabilities": dict(info.all_language_probs) if info.all_language_probs else {}
        }


def update_task_result(_audio_path, result):
    """Updates the final result in the task registry."""
    scheduler.update_task_metadata(result=result)


@contextlib.contextmanager
def model_lock_ctx(priority=None):
    """Hardware resource acquisition context with priority borrowing support."""
    is_priority = priority if priority is not None else getattr(
        utils.THREAD_CONTEXT, "is_priority", False)
    unit = None
    borrowed = False

    # 1. Acquisition logic
    queued_added = False
    try:
        while not unit:
            if is_priority:
                borrowed_unit_id = scheduler.get_preemptible_unit()
                if borrowed_unit_id:
                    unit = next(
                        (u for u in config.HARDWARE_UNITS if u['id'] == borrowed_unit_id), None)
                    if unit:
                        logger.info("[Engine] Priority task borrowed unit %s", unit['id'])
                        borrowed = True
                        break

            if scheduler.STATE.model_lock.acquire(blocking=False):
                unit = scheduler.STATE.hw_pool.get()
                break

            if not queued_added:
                scheduler.update_task_metadata(status="queued")
                scheduler.update_task_progress(None, "Waiting for Hardware")
                scheduler.increment_queued_session()
                queued_added = True

            if not is_priority:
                # Standard task: Wait on semaphore
                scheduler.STATE.model_lock.acquire()
                unit = scheduler.STATE.hw_pool.get()
                break
            # Priority task: Loop until a unit is preempted or a slot opens
            time.sleep(0.5)
    finally:
        if queued_added:
            scheduler.decrement_queued_session()

    scheduler.update_task_metadata(status="active", start_active=time.time(), unit_id=unit['id'])
    try:
        if unit['id'] not in _MODEL_POOL:
            _init_unit(unit)

        if unit['id'] not in _PREPROCESSOR_POOL:
            _PREPROCESSOR_POOL[unit['id']] = preprocessing.PreprocessingManager(unit)

        model = _MODEL_POOL.get(unit['id'])
        if model is None:
            raise RuntimeError(f"Engine pool for {unit['id']} is empty after initialization.")

        scheduler.update_task_metadata(
            unit_id=unit['id'],
            unit_type=unit['type'],
            unit_name=unit['name'],
            status="active"
        )
        yield model, unit['id']
    finally:
        if borrowed:
            # Return unit to preemptible pool so the original task can take it back
            scheduler.mark_unit_preemptible(unit['id'])
            logger.info("[Engine] Priority task finished with borrowed unit %s", unit['id'])
        else:
            scheduler.STATE.hw_pool.put(unit)
            scheduler.STATE.model_lock.release()


def _check_preemption():
    """Yields execution if a priority task is waiting."""
    if scheduler.STATE.pause_requested.is_set():
        thread_id = threading.get_ident()
        unit_id = None
        with scheduler.STATE.task_registry_lock:
            task = scheduler.STATE.task_registry.get(thread_id)
            if task:
                unit_id = task.get('unit_id')

        if unit_id:
            logger.info("[Engine] Preempting task on %s...", unit_id)
            old_stage = task.get('stage')
            scheduler.update_task_progress(task.get('progress'), "Paused for Priority Task")
            scheduler.mark_unit_preemptible(unit_id)
            scheduler.STATE.pause_confirmed.set()
            scheduler.STATE.resume_event.wait()

            # Wait until our unit is no longer "borrowed"
            while True:
                with scheduler.STATE.task_registry_lock:
                    if unit_id in scheduler.STATE.preemptible_units:
                        # It's back in the pool, we can take it
                        scheduler.STATE.preemptible_units.remove(unit_id)
                        break
                time.sleep(0.5)

            scheduler.STATE.pause_confirmed.clear()
            scheduler.update_task_progress(task.get('progress'), old_stage)
            logger.info("[Engine] Resuming task on %s", unit_id)


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
    scheduler.increment_active_session()


def decrement_active_session():
    """Tracks active session count and unloads if idle."""
    scheduler.decrement_active_session()
    current_active = scheduler.STATE.active_sessions
    logger.debug("[Engine] Session decrement. Active sessions remaining: %d", current_active)

    if config.AGGRESSIVE_OFFLOAD and current_active == 0:
        unload_models()


def wait_for_priority():
    """Handles priority task synchronization."""
    utils.THREAD_CONTEXT.is_priority = True
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
