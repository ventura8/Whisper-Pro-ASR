"""
Model Life-Cycle and Task Scheduler

This module manages the instantiation of the Faster-Whisper engine and coordinates
task scheduling. It implements a priority locking mechanism to allow short, high-priority
tasks (like Language Detection) to pre-empt long-running transcriptions.
"""
# pylint: disable=broad-exception-caught,line-too-long
# pylint: disable=no-member,protected-access,consider-using-with,cyclic-import
import logging
import queue
import threading
import time
import contextlib
import os
import math
import gc
import uuid
import soundfile as sf_lib
try:
    import torch
except ImportError:
    torch = None

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

from . import config, utils, preprocessing, vad, history_manager, telemetry_manager, metrics_discovery
from .logging_setup import TASK_LOGS
logger = logging.getLogger(__name__)
SERVICE_START_TIME = time.time()
_TELEMETRY_STOP_EVENT = threading.Event()

# --- [GLOBAL SERVICE STATE] ---
_MODEL_POOL = {}        # Map of unit_id -> WhisperModel instance
_PREPROCESSOR_POOL = {}  # Map of unit_id -> PreprocessingManager instance

# --- [LOCKING & PRE-EMPTION SCHEDULER] ---
_HW_POOL = queue.Queue()
for unit_item in config.HARDWARE_UNITS:
    _HW_POOL.put(unit_item)

# Hardware Resource Limits (Total units detected)
_ACCEL_LIMIT = len(config.HARDWARE_UNITS)
_MODEL_LOCK = threading.Semaphore(_ACCEL_LIMIT)  # Shared hardware access limit
_PRIORITY_SEQUENTIAL_LOCK = threading.Semaphore(_ACCEL_LIMIT)  # Priority task limit

_PRIORITY_LOCK = threading.Lock()        # Lock for priority counter access
_PAUSE_REQUESTED = threading.Event()      # Signals ASR loop to yield
_PAUSE_CONFIRMED = threading.Event()      # Loop confirms yielding
_RESUME_EVENT = threading.Event()         # Signal to resume ASR loop
_RESUME_EVENT.set()                       # Default: Resume state
_ACTIVE_SESSIONS = 0  # Tracks currently executing ASR/Detection tasks
_QUEUED_SESSIONS = 0  # Tracks tasks waiting for hardware/priority locks
_PRIORITY_REQUESTS = 0

# --- [TASK TRACKING REGISTRY] ---
_TASK_REGISTRY = {}
_TASK_REGISTRY_LOCK = threading.Lock()


@contextlib.contextmanager
def early_task_registration(task_type="ASR/LD", stage="Preprocessing"):
    """Registers a task immediately so it appears on the dashboard during early preparation."""
    thread_id = threading.get_ident()
    task_id = str(uuid.uuid4())

    with _TASK_REGISTRY_LOCK:
        if thread_id not in TASK_LOGS:
            TASK_LOGS[thread_id] = []
        if thread_id not in _TASK_REGISTRY:
            _TASK_REGISTRY[thread_id] = {
                "task_id": task_id,
                "filename": getattr(utils.THREAD_CONTEXT, "filename", "Unknown"),
                "start_time": time.time(),
                "status": "queued",
                "progress": 0,
                "stage": stage,
                "type": task_type,
                "video_duration": getattr(utils.THREAD_CONTEXT, "total_duration", 0),
                "caller_info": getattr(utils.THREAD_CONTEXT, "caller_info", {}),
                "request_json": getattr(utils.THREAD_CONTEXT, "request_json", {})
            }
    try:
        yield
    finally:
        # We DO NOT deregister here. Deregistration happens in model_lock_ctx's finally block,
        # or we manually handle it if the request fails before reaching model_lock_ctx.
        pass


def cleanup_failed_task():
    """Removes a task from the registry if it fails before model_lock_ctx finishes."""
    thread_id = threading.get_ident()
    with _TASK_REGISTRY_LOCK:
        if thread_id in _TASK_REGISTRY:
            del _TASK_REGISTRY[thread_id]
        if thread_id in TASK_LOGS:
            del TASK_LOGS[thread_id]


@contextlib.contextmanager
def model_lock_ctx():
    """Context manager for hardware model access with dynamic unit allocation."""
    global _QUEUED_SESSIONS, _ACTIVE_SESSIONS  # pylint: disable=global-statement
    # Re-entrancy check: If this thread already has an assigned unit, reuse it.
    current_ctx = utils.THREAD_CONTEXT
    if hasattr(current_ctx, 'assigned_unit') and current_ctx.assigned_unit:
        yield current_ctx.assigned_unit
        return

    _QUEUED_SESSIONS += 1
    start_wait = time.time()
    unit = None
    try:
        # Register task if not already registered by early_task_registration
        thread_id = threading.get_ident()

        with _TASK_REGISTRY_LOCK:
            if thread_id not in TASK_LOGS:
                TASK_LOGS[thread_id] = []

            if thread_id not in _TASK_REGISTRY:
                _TASK_REGISTRY[thread_id] = {
                    "task_id": str(uuid.uuid4()),
                    "filename": getattr(utils.THREAD_CONTEXT, "filename", "Unknown"),
                    "start_time": time.time(),
                    "status": "queued",
                    "progress": 0,
                    "stage": "Waiting in queue",
                    "type": "ASR/LD",
                    "video_duration": getattr(utils.THREAD_CONTEXT, "total_duration", 0),
                    "start_wait": start_wait,
                    "caller_info": getattr(utils.THREAD_CONTEXT, "caller_info", {}),
                    "request_json": getattr(utils.THREAD_CONTEXT, "request_json", {})
                }
            else:
                # Update existing early registration with queue info
                _TASK_REGISTRY[thread_id]["start_wait"] = start_wait
                _TASK_REGISTRY[thread_id]["status"] = "queued"

        with _MODEL_LOCK:
            _QUEUED_SESSIONS -= 1
            increment_active_session()
            # Update to active state
            with _TASK_REGISTRY_LOCK:
                if thread_id in _TASK_REGISTRY:
                    _TASK_REGISTRY[thread_id].update({
                        "status": "active",
                        "start_active": time.time(),
                        "stage": "Initializing"
                    })

            # Claim a specific hardware unit from the pool
            unit = _HW_POOL.get()
            utils.THREAD_CONTEXT.assigned_unit = unit

            with _TASK_REGISTRY_LOCK:
                if thread_id in _TASK_REGISTRY:
                    _TASK_REGISTRY[thread_id]['unit_id'] = unit['id']
                    _TASK_REGISTRY[thread_id]['unit_type'] = unit['type']
                    _TASK_REGISTRY[thread_id]['unit_name'] = unit['name']

            wait_dur = time.time() - start_wait
            if wait_dur > 0.1:
                logger.info("[System] %s acquired after %.2fs. (Active: %d, Queued: %d)",
                            unit['name'], wait_dur, _ACTIVE_SESSIONS, _QUEUED_SESSIONS)

            yield unit
    finally:
        decrement_active_session()

        thread_id = threading.get_ident()
        with _TASK_REGISTRY_LOCK:
            if thread_id in _TASK_REGISTRY:
                task_data = _TASK_REGISTRY[thread_id].copy()
                task_data["logs"] = TASK_LOGS.get(thread_id, [])
                history_manager.log_completed_task(task_data)
                del _TASK_REGISTRY[thread_id]
            if thread_id in TASK_LOGS:
                del TASK_LOGS[thread_id]

        _check_and_offload_resources()

        if unit:
            _HW_POOL.put(unit)
            utils.THREAD_CONTEXT.assigned_unit = None


def update_task_metadata(**kwargs):
    """Updates the metadata of the current thread's task in the registry."""
    thread_id = threading.get_ident()
    with _TASK_REGISTRY_LOCK:
        if thread_id in _TASK_REGISTRY:
            for key, value in kwargs.items():
                if value is not None:
                    _TASK_REGISTRY[thread_id][key] = value


def update_task_progress(progress_pct, stage=None):
    """Updates the progress and stage of the current thread's task."""
    thread_id = threading.get_ident()
    with _TASK_REGISTRY_LOCK:
        if thread_id in _TASK_REGISTRY:
            if progress_pct is not None:
                _TASK_REGISTRY[thread_id]["progress"] = progress_pct
            if stage:
                _TASK_REGISTRY[thread_id]["stage"] = stage


def update_task_result(result):
    """Stores the final result in the registry for history logging."""
    thread_id = threading.get_ident()
    with _TASK_REGISTRY_LOCK:
        if thread_id in _TASK_REGISTRY:
            # We don't want to store huge raw objects, just key highlights
            summary = {
                "text": result.get("text", "")[:2000],  # Cap for history
                "language": result.get("language", "unknown"),
                "detected_language": result.get("detected_language", "unknown"),
                "confidence": result.get("confidence", result.get("language_probability", 0.0))
            }
            if "voting_details" in result:
                summary["voting_details"] = result["voting_details"]
            _TASK_REGISTRY[thread_id]["result"] = summary
            _TASK_REGISTRY[thread_id]["response_json"] = result


def request_priority():
    """
    Registers a high-priority request (e.g., Language Detection).

    This call blocks incoming ASR tasks and signals the active ASR loop to yield
    hardware assets. Multiple priority tasks are processed sequentially.
    """
    global _PRIORITY_REQUESTS, _QUEUED_SESSIONS  # pylint: disable=global-statement
    with _PRIORITY_LOCK:
        _PRIORITY_REQUESTS += 1
        # Only request pause if we have reached hardware saturation
        # If we have free units (NPU+GPU), we can run in parallel without pausing ASR
        if _ACTIVE_SESSIONS >= _ACCEL_LIMIT:
            _PAUSE_REQUESTED.set()
            _RESUME_EVENT.clear()
        else:
            # We have headroom, ensure resume event is set so we don't block
            _RESUME_EVENT.set()

    # Ensure priority tasks respect parallel limits
    _QUEUED_SESSIONS += 1
    try:
        _PRIORITY_SEQUENTIAL_LOCK.acquire()
        logger.debug("[System] Priority lock acquired. (Queued: %d)", _QUEUED_SESSIONS)
    finally:
        _QUEUED_SESSIONS -= 1


def release_priority():
    """Signals the completion of a high-priority task and resumes ASR queue."""
    global _PRIORITY_REQUESTS  # pylint: disable=global-statement
    try:
        with _PRIORITY_LOCK:
            _PRIORITY_REQUESTS = max(0, _PRIORITY_REQUESTS - 1)
            if _PRIORITY_REQUESTS == 0:
                _PAUSE_REQUESTED.clear()
                _RESUME_EVENT.set()
    finally:
        # Release sequential lock if owned
        try:
            _PRIORITY_SEQUENTIAL_LOCK.release()
        except (RuntimeError, ValueError):
            pass


def wait_for_priority(model_lock=None):
    """
    Check-point for the ASR loop to yield to priority tasks.

    If model_lock is provided, it is released to allow the priority task to
    utilize NPU/GPU assets, then re-acquired once finished.
    """
    global _QUEUED_SESSIONS  # pylint: disable=global-statement
    # Yield ONLY if a priority task is active AND we are at hardware capacity
    if _PAUSE_REQUESTED.is_set() and _PRIORITY_REQUESTS > 0:
        logger.info(
            "[System] Hardware saturation detected with pending priority task. Yielding...")
        _PAUSE_CONFIRMED.set()
        if model_lock:
            model_lock.release()

        _QUEUED_SESSIONS += 1
        wait_start = time.time()
        try:
            # Wait for priority finished signal
            _RESUME_EVENT.wait()
        finally:
            _QUEUED_SESSIONS -= 1
            wait_dur = time.time() - wait_start
            logger.info("[System] Resuming ASR task after %.2fs priority suspension. (Active: %d, Queued: %d)",
                        wait_dur, _ACTIVE_SESSIONS, _QUEUED_SESSIONS)
            if model_lock:
                model_lock.acquire()
            _PAUSE_CONFIRMED.clear()

        logger.info(
            "[System] Priority task complete. Resuming transcription...")


def get_current_preprocessor():
    """Retrieve the unit-pinned PreprocessingManager for the current thread."""
    unit = getattr(utils.THREAD_CONTEXT, 'assigned_unit', None)
    if unit:
        return _PREPROCESSOR_POOL.get(unit['id'])
    return None


def is_engine_initialized():
    """Verify that at least one hardware unit has its engines ready."""
    return len(_MODEL_POOL) > 0 and len(_PREPROCESSOR_POOL) > 0


def _get_or_load_whisper_model(unit):
    """Lazy loader for WhisperModel instances."""
    unit_id = unit['id']
    unit_type = unit['type']
    unit_name = unit['name']

    if unit_id in _MODEL_POOL and _MODEL_POOL[unit_id] is not None:
        return _MODEL_POOL[unit_id]

    try:
        if WhisperModel is None:
            raise ImportError("faster-whisper not installed.")
        # Faster-Whisper / CUDA / CPU Path
        fw_device = "cuda" if unit_type == "CUDA" else "cpu"
        device_index = int(unit_id.split(':')[-1]) if ':' in unit_id else 0

        # Clarify that for NPU/GPU units, Faster-Whisper will fallback to CPU
        target_display = unit_name
        if unit_type in ["NPU", "GPU"] and fw_device == "cpu":
            target_display = f"Host CPU (via {unit_name} slot)"

        logger.info("[System] Loading Whisper Engine (%s) on %s...",
                    config.MODEL_ID, target_display)
        model = WhisperModel(
            config.MODEL_ID,
            device=fw_device,
            device_index=device_index,
            compute_type=config.ASR_ENGINE_COMPUTE_TYPE,
            num_workers=1,
            download_root=config.OV_CACHE_DIR,
            cpu_threads=config.ASR_THREADS
        )
        _MODEL_POOL[unit_id] = model
        return model
    except Exception as err:
        logger.error("[System] Failed to load Whisper engine on %s: %s", unit_name, err)
        raise


def load_model():
    """Initialize and warm up engines for all detected hardware units."""
    success = True

    # Enforce thread limits for PyTorch globally before initializing models
    try:
        if torch:
            torch.set_num_threads(config.ASR_THREADS)
            torch.set_num_interop_threads(config.ASR_THREADS)
    except Exception as t_err:
        logger.debug("Torch thread adjustment skipped: %s", t_err)

    for unit in config.HARDWARE_UNITS:
        unit_id = unit['id']
        unit_name = unit['name']
        try:
            # Initialize Unit-Pinned Preprocessing (UVR)
            # PreprocessingManager is lazy-loaded, so it's safe to init here regardless of offload
            _PREPROCESSOR_POOL[unit_id] = preprocessing.PreprocessingManager(assigned_unit=unit)

            # Warm up Whisper Model ONLY if aggressive offloading is DISABLED
            if not config.AGGRESSIVE_OFFLOAD:
                _get_or_load_whisper_model(unit)
                logger.info("[System] Engine pool (ASR+UVR) ready on %s", unit_name)
            else:
                # Still register the unit in the pool so the service knows it's available
                if unit_id not in _MODEL_POOL:
                    _MODEL_POOL[unit_id] = None
                logger.info("[System] Engine pool registered for %s (Deferred Load)", unit_name)

        except Exception as err:
            logger.error("[System] Failed to warm up engines on %s: %s", unit_name, err)
            success = False

    return success


# --- [INFERENCE EXECUTION] ---
def run_transcription(audio_path, language=None, task='transcribe', _batch_size=None):
    """
    Coordinates the full transcription lifecycle.
    """
    try:
        ctx_filename = getattr(utils.THREAD_CONTEXT, "filename", "System")
        if ctx_filename == "System" or "upload_" in str(ctx_filename) or "tmp" in str(ctx_filename).lower():
            utils.THREAD_CONTEXT.filename = os.path.basename(audio_path)

        # Initialize stats before locking so they are available in the queue registration
        _init_transcription_stats(audio_path)
        inference_path = audio_path

        with model_lock_ctx() as unit:
            utils.THREAD_CONTEXT.is_transcribing = True

            def checkpoint():
                wait_for_priority(model_lock=_MODEL_LOCK)

            extra_prep_time, inference_path = _preprocess_audio(audio_path, yield_cb=checkpoint)
            _log_audio_diagnostics(inference_path)

            unit_model = _get_or_load_whisper_model(unit)
            use_cpu_lock = unit['type'] != 'CUDA'

            if use_cpu_lock:
                with utils.cpu_lock_ctx():
                    segment_results, detected_language, language_probability, total_dur = \
                        _run_faster_whisper_transcription(
                            unit_model, inference_path, language, task)
            else:
                segment_results, detected_language, language_probability, total_dur = \
                    _run_faster_whisper_transcription(unit_model, inference_path, language, task)

            result = {
                "segments": segment_results,
                "text": " ".join([s["text"] for s in segment_results]),
                "duration": total_dur,
                "video_duration_sec": total_dur,
                "extra_preprocess_duration": extra_prep_time,
                "language": detected_language,
                "detected_language": detected_language,
                "language_probability": language_probability
            }

            update_task_progress(96, "Filtering Hallucinations")
            result = _post_process_vad(result, inference_path)
            result["text"] = "".join([s.get("text", "")
                                     for s in result.get("segments", [])]).strip()
            update_task_progress(98, "Finalizing Result")
            update_task_progress(100, "Complete")
            update_task_result(result)

            # Per-unit Aggressive Offloading
            if config.AGGRESSIVE_OFFLOAD:
                _offload_unit_resources(unit['id'])

            return result

    finally:
        utils.THREAD_CONTEXT.is_transcribing = False
        if 'inference_path' in locals() and inference_path != audio_path and os.path.exists(inference_path):
            try:
                os.remove(inference_path)
            except Exception:
                pass


def run_vocal_isolation(audio_path, yield_cb=None, force=False):
    """Expose vocal separation as a high-level utility for batch tasks."""
    _, inference_path = _preprocess_audio(audio_path, yield_cb=yield_cb, force=force)
    return inference_path


def run_batch_language_detection(audio_path, segment_count):
    """
    Perform a high-performance language identification scan across a concatenated montage.
    Uses single-pass VAD and in-memory numpy processing to avoid redundant I/O.
    """
    results = []

    # 1. Load the whole montage into memory (16kHz mono)
    try:
        full_audio = vad.decode_audio(audio_path)
    except Exception as e:
        logger.error("[LD] Batch decode failed: %s", e)
        return []

    # 2. Global VAD Scan (One pass for the whole montage)
    logger.info("[LD] Running global VAD scan on montage...")
    with utils.cpu_lock_ctx():
        all_speech_ts = vad.get_speech_timestamps(full_audio)

    # Calculate and log metrics without extra local variables to satisfy linting
    logger.info("[LD] Global VAD complete. Found %.2fs of speech (Removed %.2fs of silence)",
                sum(ts['end'] - ts['start'] for ts in all_speech_ts),
                (len(full_audio) / 16000) - sum(ts['end'] - ts['start'] for ts in all_speech_ts))

    with model_lock_ctx() as unit:
        model = _get_or_load_whisper_model(unit)
        use_cpu_lock = unit['type'] != 'CUDA'

        for i in range(segment_count):
            start_sec = i * 30
            end_sec = (i + 1) * 30
            step_text = f"Voting Step {i+1}/{segment_count}"
            utils.THREAD_CONTEXT.step_info = f"[LD] {step_text}"
            # Calculate progress between 10% and 90%
            prog = 10 + int(((i + 1) / segment_count) * 80)
            update_task_progress(prog, step_text)

            # Identify if this specific 30s window contains speech
            has_speech = any(ts['start'] < end_sec and ts['end'] > start_sec
                             for ts in all_speech_ts)

            if not has_speech:
                logger.info("Silence detected, skipping.")
                results.append(None)
                continue

            # Extract 30s chunk (in-memory)
            chunk = full_audio[int(start_sec * 16000):int(end_sec * 16000)]

            try:
                # Pass the numpy array directly to the inference engine
                if use_cpu_lock:
                    with utils.cpu_lock_ctx():
                        res = _run_language_detection_core(model, chunk, skip_vad=True)
                else:
                    res = _run_language_detection_core(model, chunk, skip_vad=True)
                results.append(res)
            except Exception as e:
                logger.warning("Batch sample failed: %s", e)
                results.append(None)

        # Cleanup step info for subsequent logs
        # Explicitly clear local reference before offloading
        model = None

        # Per-unit Aggressive Offloading for Batch LD
        if config.AGGRESSIVE_OFFLOAD:
            _offload_unit_resources(unit['id'])

    return results

# --- [RESOURCE MANAGEMENT] ---


def increment_active_session():
    """Register a new active task (ASR or Detection)."""
    global _ACTIVE_SESSIONS  # pylint: disable=global-statement
    _ACTIVE_SESSIONS += 1


def decrement_active_session():
    """Unregister a task and trigger resource reclamation if idle."""
    global _ACTIVE_SESSIONS  # pylint: disable=global-statement
    _ACTIVE_SESSIONS = max(0, _ACTIVE_SESSIONS - 1)
    if _ACTIVE_SESSIONS == 0:
        _check_and_offload_resources()


def get_gpu_usage():
    """Real usage via metrics discovery (NVIDIA, Intel, NPU)."""
    return metrics_discovery.get_nvidia_metrics()


def _telemetry_loop():
    """Background thread to record system stats every 60 seconds."""
    logger.info("[System] Starting telemetry recording loop")
    while not _TELEMETRY_STOP_EVENT.is_set():
        try:
            # We use psutil directly here to avoid circular imports or redundant calls
            import psutil
            cpu_usage = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            process = psutil.Process(os.getpid())
            app_mem_rss = process.memory_info().rss
            app_cpu = process.cpu_percent(interval=None)

            # Get current stats to feed to recorder
            current_stats = get_service_stats()
            # Inject psutil data that get_service_stats doesn't have (dashboard.py usually adds this)
            current_stats['system'] = {
                "cpu_percent": cpu_usage,
                "app_cpu_percent": round(app_cpu, 1),
                "memory_percent": mem.percent,
                "memory_used_gb": round(mem.used / (1024**3), 2),
                "memory_total_gb": round(mem.total / (1024**3), 2),
                "app_memory_gb": round(app_mem_rss / (1024**3), 2)
            }

            telemetry_manager.record_snapshot(current_stats)
        except Exception as e:
            logger.error(f"[Telemetry] Loop error: {e}")

        # Wait 5 seconds unless stopping
        _TELEMETRY_STOP_EVENT.wait(5)


# Start the telemetry loop on module load
threading.Thread(target=_telemetry_loop, daemon=True, name="TelemetryRecorder").start()


def get_service_stats():
    """Returns a comprehensive snapshot of the service's internal state."""
    with _TASK_REGISTRY_LOCK:
        tasks = []
        for t_id, task in _TASK_REGISTRY.items():
            task_copy = task.copy()
            task_copy["logs"] = TASK_LOGS.get(t_id, [])
            tasks.append(task_copy)

    # Calculate load factors for Intel/NPU as a proxy for usage if native APIs aren't available

    intel_load = metrics_discovery.get_intel_gpu_load()
    npu_load = metrics_discovery.get_npu_load()

    # Check UVR status across all preprocessor instances
    uvr_loaded = any(pm.separator is not None for pm in _PREPROCESSOR_POOL.values())

    # Collect history
    history, history_stats = history_manager.get_history_stats()

    return {
        "app_name": "Whisper Pro ASR",
        "version": config.VERSION,
        "status": "loaded" if is_engine_initialized() else "failed",
        "uptime_sec": round(time.time() - SERVICE_START_TIME, 2),
        "active_sessions": len([t for t in tasks if t.get('status') == 'active']),
        "queued_sessions": len([t for t in tasks if t.get('status') == 'queued']),
        "total_video_processed_sec": history_stats["all_time"],
        "history_stats": history_stats,
        "hardware_units": config.HARDWARE_UNITS,
        "telemetry": {
            "nvidia": get_gpu_usage(),
            "intel_gpu_load": intel_load,
            "npu_load": npu_load
        },
        "tasks": tasks,
        "history": history,
        "engines": {
            "whisper": {
                "status": "busy" if any(t.get('status') == 'active' for t in tasks) else ("initialized" if is_engine_initialized() else "failed"),
                "model": utils.get_pretty_model_name(config.MODEL_ID),
                "device": config.DEVICE,
                "compute_type": config.COMPUTE_TYPE
            },
            "uvr": {
                "status": "busy" if any(t.get('status') == 'active' and t.get('stage', '').lower().startswith('isolating') for t in tasks) else ("loaded" if uvr_loaded else "idle"),
                "model": utils.get_pretty_model_name(config.UVR_ENV)
            },
            "vad": {
                "status": "ready",
                "model": "Silero VAD v4"
            }
        },
        "telemetry_history": telemetry_manager.get_telemetry_history(),
        "settings": {
            "telemetry_retention_hours": int(os.environ.get("TELEMETRY_RETENTION_HOURS", 24)),
            "log_retention_days": int(os.environ.get("LOG_RETENTION_DAYS", 7))
        }
    }


def get_service_stats_minimal():
    """Lightweight status check for circular-safe metrics discovery."""
    with _TASK_REGISTRY_LOCK:
        active = []
        for task in _TASK_REGISTRY.values():
            if task.get('status') == 'active':
                active.append({
                    "unit_type": task.get('unit_type'),
                    "unit_name": task.get('unit_name', ''),
                    "unit_id": task.get('unit_id')
                })
        return {"active_tasks": active}


def _offload_unit_resources(unit_id):
    """Evict AI models for a specific hardware unit to reclaim memory."""

    # 1. Offload Whisper
    if unit_id in _MODEL_POOL and _MODEL_POOL[unit_id] is not None:
        logger.info("[System] Offloading Whisper engine from unit %s", unit_id)
        _MODEL_POOL[unit_id] = None

    # 2. Offload Preprocessor (UVR)
    pm = _PREPROCESSOR_POOL.get(unit_id)
    if pm:
        try:
            pm.offload()
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    # 3. Global reclamation
    gc.collect()
    utils.clear_gpu_cache()


def _check_and_offload_resources():
    """Reclaim RAM/VRAM only if no active or queued sessions exist."""
    # 1. GC cleanup
    gc.collect()
    utils.clear_gpu_cache()

    # 2. Heavy engine offloading (Keep resident if tasks are queued)
    if _ACTIVE_SESSIONS == 0 and _QUEUED_SESSIONS == 0:
        if config.AGGRESSIVE_OFFLOAD:
            # Offload all Whisper Models
            for u_id in list(_MODEL_POOL.keys()):
                if _MODEL_POOL[u_id] is not None:
                    logger.info(
                        "[System] Offloading Whisper engine from unit %s (Global Cleanup)", u_id)
                    _MODEL_POOL[u_id] = None

            # Offload all Preprocessors
            for p_manager in _PREPROCESSOR_POOL.values():
                try:
                    p_manager.offload()
                except Exception:  # pylint: disable=broad-exception-caught
                    pass

            # Extra GC after global offload
            gc.collect()
            utils.clear_gpu_cache()

# --- [INTERNAL HELPER UTILITIES] ---


def _preprocess_audio(audio_path, yield_cb=None, force=False):
    """Execute UVR isolation if enabled with dual-path VAD fallback."""
    extra_prep_time = 0.0
    inference_path = audio_path
    processed_path = None

    if config.ENABLE_VOCAL_SEPARATION or force:
        try:
            p_start = time.time()
            # Safety: Ensure unit is assigned. If not (e.g., priority task bypassing lock),
            # fallback to first available or CPU.
            unit = getattr(utils.THREAD_CONTEXT, 'assigned_unit', None)
            unit_id = unit['id'] if unit else config.HARDWARE_UNITS[0]['id']

            pm = _PREPROCESSOR_POOL.get(unit_id)
            if pm:
                update_task_progress(10, "Isolating Vocals (MDX-NET)")
                processed_path = pm.process_audio_file(audio_path, yield_cb=yield_cb)
                update_task_progress(25, "Filtering Noise")
                extra_prep_time = time.time() - p_start
                update_task_progress(40, "Verifying Signal")

            # Dual-Path VAD Fallback: Check if isolated audio is silent
            inference_path = _resolve_inference_path(audio_path, processed_path)
        except Exception as prep_err:
            logger.error("Vocal separation engine failed: %s", prep_err)

    return extra_prep_time, inference_path


def _resolve_inference_path(audio_path, processed_path):
    """Verifies speech presence and signal quality in processed audio."""
    if not processed_path or processed_path == audio_path:
        return audio_path

    try:
        # Step 1: Silence Check
        with utils.cpu_lock_ctx():
            speech_ts = vad.get_speech_timestamps_from_path(processed_path)
        if not speech_ts:
            logger.info("[ASR] Isolated vocals silent. Falling back to original.")
            utils.secure_remove(processed_path)
            return audio_path

        return processed_path
    except Exception as e:
        logger.debug("[ASR] Signal verification/VAD failed: %s", e)

    return processed_path


def _run_faster_whisper_transcription(model, audio_input, language=None, task='transcribe'):
    """Executes core Faster-Whisper inference loop."""
    start_time = time.time()
    update_task_progress(45, "Loading Model Assets")
    segments_gen, info = model.transcribe(
        audio_input,
        beam_size=5,
        language=language,
        task=task,
        vad_filter=True,
    )

    segment_results = []
    for s in segments_gen:
        wait_for_priority(model_lock=_MODEL_LOCK)
        segment = {
            "text": s.text,
            "start": round(s.start, 3),
            "end": round(s.end, 3),
            "timestamp": (s.start, s.end),
            "probability": math.exp(s.avg_logprob) if s.avg_logprob else 1.0
        }
        if segment["text"]:
            segment_results.append(segment)
            _log_progress_manual(s.end, info.duration, start_time, s.text)
            # Update global task registry progress (Scale 50-95%)
            if info.duration > 0:
                prog = 50 + int((s.end / info.duration) * 45)
                update_task_progress(min(95, prog), "Transcribing")

    return segment_results, info.language, info.language_probability, info.duration


def _run_intel_whisper_transcription(model, inference_path, language, task):
    """
    Placeholder for Intel OpenVINO transcription.
    Currently disabled as it is not production-ready.
    """
    raise NotImplementedError("Intel Whisper Engine is currently disabled.")


def _log_audio_diagnostics(inference_path):
    """Log file metadata for verification."""
    try:
        f_info = sf_lib.info(inference_path)
        logger.info("[ASR] Input stats: %s (%d ch, %d Hz)",
                    utils.format_duration(f_info.duration), f_info.channels, f_info.samplerate)
    except Exception as d_err:  # pylint: disable=broad-exception-caught
        logger.debug("Diagnostic check skipped: %s", d_err)


def _init_transcription_stats(audio_path):
    """Calculate and store media metadata for the current request context."""
    utils.THREAD_CONTEXT.total_duration = 0
    utils.THREAD_CONTEXT.start_time = time.time()
    update_task_progress(5, "Analyzing Signal")
    try:
        info = sf_lib.info(audio_path)
        utils.THREAD_CONTEXT.total_duration = info.duration
    except Exception:  # pylint: disable=broad-exception-caught
        pass


def _log_progress_manual(audio_pos, total_dur, start_time, text):
    """Log manual progress for both engines."""
    elapsed = time.time() - start_time
    speed = audio_pos / elapsed if elapsed > 0 else 0
    eta = (total_dur - audio_pos) / speed if speed > 0.1 else 0
    pct = (audio_pos / total_dur * 100) if total_dur > 0 else 0

    logger.info("[ASR Progress] %5.1f%% | Audio: %s/%s | Speed: %.2fx | ETA: %s | Text: %s",
                pct, utils.format_duration(
                    audio_pos), utils.format_duration(total_dur),
                speed, utils.format_duration(eta), text)


def run_language_detection(audio_path, _batch_size=None):
    """
    Orchestrates the high-priority language detection pipeline.
    """
    try:
        ctx_filename = getattr(utils.THREAD_CONTEXT, "filename", "System")
        if ctx_filename == "System" or "upload_" in str(ctx_filename) or "tmp" in str(ctx_filename).lower():
            utils.THREAD_CONTEXT.filename = os.path.basename(audio_path)

        # Initialize stats before locking
        _init_transcription_stats(audio_path)

        with model_lock_ctx() as unit:
            inference_path = audio_path
            # Optional Vocal Separation for LD
            if config.ENABLE_LD_PREPROCESSING:
                _, inference_path = _preprocess_audio(audio_path, force=True)

            try:
                model = _get_or_load_whisper_model(unit)
                use_cpu_lock = unit['type'] != 'CUDA'

                if use_cpu_lock:
                    with utils.cpu_lock_ctx():
                        result = _run_language_detection_core(model, inference_path)
                else:
                    result = _run_language_detection_core(model, inference_path)

                # --- [STRICT SILENCE FALLBACK] ---
                # If UVR removed all speech but original had it, fall back to raw audio
                if inference_path != audio_path and result.get("confidence", 0.0) == 0.0:
                    with utils.cpu_lock_ctx():
                        if vad.get_speech_timestamps_from_path(audio_path, threshold=config.LD_VAD_THRESHOLD):
                            logger.info(
                                "[LD] UVR silent. Speech found in original. Falling back to raw audio.")
                            if use_cpu_lock:
                                with utils.cpu_lock_ctx():
                                    result = _run_language_detection_core(model, audio_path)
                            else:
                                result = _run_language_detection_core(model, audio_path)

                update_task_result(result)
                return result
            finally:
                # Explicitly clear local reference before offloading
                model = None

                # Per-unit Aggressive Offloading for Language Detection
                if config.AGGRESSIVE_OFFLOAD:
                    _offload_unit_resources(unit['id'])

                # Cleanup derived assets
                if inference_path and inference_path != audio_path and os.path.exists(inference_path):
                    utils.secure_remove(inference_path)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Language detection task failed: %s", e)
        return {
            "detected_language": "en",
            "language": "en",
            "language_code": "en",
            "confidence": 0.0,
            "all_probabilities": {"en": 1.0}
        }


def _run_language_detection_core(model, audio_input, skip_vad=False):
    """Internal core for language detection without locking."""
    # Pre-check for speech to avoid hallucinations on silence
    if not skip_vad:
        if isinstance(audio_input, str):
            speech_ts = vad.get_speech_timestamps_from_path(audio_input)
        else:
            speech_ts = vad.get_speech_timestamps(audio_input)

        if not speech_ts:
            logger.info("[ASR] No speech detected in input. Returning default 'en'.")
            return {
                "detected_language": "en",
                "language": "en",
                "language_code": "en",
                "confidence": 0.0,
                "all_probabilities": {"en": 0.0}
            }

    out = model.transcribe(
        audio_input,
        beam_size=5,
        vad_filter=not skip_vad,
        task="transcribe"
    )

    # Handle different return types from engines
    if isinstance(out, tuple):
        _, info = out
        all_probs = dict(
            info.all_language_probs) if info and info.all_language_probs else {}
        detected_lang = info.language if info and info.language else "en"
        confidence = info.language_probability if info else 0.0
    else:
        # Intel/Other engine returning dict
        all_probs = out.get("all_probabilities", {})
        detected_lang = out.get("language", "en")
        confidence = out.get("language_probability", 1.0)

    return {
        "detected_language": detected_lang,
        "language": detected_lang,
        "language_code": detected_lang,
        "confidence": confidence,
        "all_probabilities": all_probs
    }


def _post_process_vad(result, _audio_path):
    """
    Apply hallucination filtering based on configured credits/silence phrases.
    """
    try:
        segments = result.get('segments', [])
        if not segments:
            return result

        phrase_drop_count = 0
        silence_drop_count = 0
        repetition_drop_count = 0

        last_text = ""
        repetition_counter = 0

        for segment in segments:
            text = segment.get('text', '').strip()

            # 1. Silence Threshold Filter (Low Confidence)
            prob = segment.get('probability', 1.0)
            if prob < config.HALLUCINATION_SILENCE_THRESHOLD:
                segment['text'] = ""
                silence_drop_count += 1
                continue

            if not text:
                continue

            # 2. Repetition Filter
            if text == last_text:
                repetition_counter += 1
            else:
                repetition_counter = 0
                last_text = text

            if repetition_counter >= config.HALLUCINATION_REPETITION_THRESHOLD:
                segment['text'] = ""
                repetition_drop_count += 1
                continue

            # 3. Hallucination Phrase Filter
            text_clean = text.lower().strip(".,!?;: ")
            for phrase in config.HALLUCINATION_PHRASES:
                if phrase in text_clean:
                    # Drop if text is nearly identical to hallucination phrase
                    if len(text_clean) < len(phrase) + 10:
                        segment['text'] = ""
                        phrase_drop_count += 1
                        break

        total_drops = phrase_drop_count + silence_drop_count + repetition_drop_count
        if total_drops > 0:
            logger.info(
                "Post-Processor: Filtered %d segments (Phrases: %d, LowConf: %d, Repeats: %d).",
                total_drops, phrase_drop_count, silence_drop_count, repetition_drop_count)

        return result
    except Exception as e:
        logger.error("VAD Post-Processing failed: %s", e)
        return result
