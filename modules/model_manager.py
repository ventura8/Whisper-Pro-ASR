"""
Model Life-Cycle and Task Scheduler

This module manages the instantiation of the Faster-Whisper engine and coordinates
task scheduling. It implements a priority locking mechanism to allow short, high-priority
tasks (like Language Detection) to pre-empt long-running transcriptions.
"""
# pylint: disable=import-outside-toplevel,broad-exception-caught,line-too-long
# pylint: disable=no-member,protected-access,consider-using-with,cyclic-import
import os
import time
import math
import contextlib
import threading
import logging
import torch  # pylint: disable=import-error
import soundfile as sf  # pylint: disable=import-error

from faster_whisper import WhisperModel  # pylint: disable=import-error
from . import config, utils, vad, intel_engine

logger = logging.getLogger(__name__)

# --- [GLOBAL SERVICE STATE] ---
WHISPER = None
SEPARATOR = None
THREAD_DATA = threading.local()

# --- [LOCKING & PRE-EMPTION SCHEDULER] ---
_MODEL_LOCK = threading.Lock()           # Global lock for the Whisper engine
_TRANSCRIBING_LOCK = threading.Lock()    # Legacy alias for tests
_PRIORITY_LOCK = threading.Lock()        # Lock for priority counter access
# Serializes multiple priority tasks
_PRIORITY_SEQUENTIAL_LOCK = threading.RLock()
_PAUSE_REQUESTED = threading.Event()      # Signals ASR loop to yield
_PAUSE_CONFIRMED = threading.Event()      # Loop confirms yielding
_RESUME_EVENT = threading.Event()         # Signal to resume ASR loop
_RESUME_EVENT.set()                       # Default: Resume state

_TRANSCRIBING_SESSIONS = 0
_PRIORITY_REQUESTS = 0


@contextlib.contextmanager
def model_lock_ctx():
    """Context manager to synchronize access to the Whisper engine."""
    with _MODEL_LOCK:
        yield


def request_priority():
    """
    Registers a high-priority request (e.g., Language Detection).

    This call blocks incoming ASR tasks and signals the active ASR loop to yield
    hardware assets. Multiple priority tasks are processed sequentially.
    """
    global _PRIORITY_REQUESTS  # pylint: disable=global-statement
    with _PRIORITY_LOCK:
        _PRIORITY_REQUESTS += 1
        _PAUSE_REQUESTED.set()
        _RESUME_EVENT.clear()

    # Ensure only one priority task runs at a time
    _PRIORITY_SEQUENTIAL_LOCK.acquire()


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
        except RuntimeError:
            pass


def wait_for_priority(model_lock=None):
    """
    Check-point for the ASR loop to yield to priority tasks.

    If model_lock is provided, it is released to allow the priority task to 
    utilize NPU/GPU assets, then re-acquired once finished.
    """
    if _PAUSE_REQUESTED.is_set():
        logger.info(
            "[System] Priority task detected. Yielding hardware resources...")
        _PAUSE_CONFIRMED.set()
        if model_lock:
            model_lock.release()

        try:
            # Wait for priority finished signal
            _RESUME_EVENT.wait()
        finally:
            if model_lock:
                # We use a non-blocking attempt first to avoid deadlocks if the lock
                # was somehow corrupted or released by something else
                model_lock.acquire()
            _PAUSE_CONFIRMED.clear()

        logger.info(
            "[System] Priority task complete. Resuming transcription...")


# --- [MODEL LOADER] ---
def load_model():
    """Initialize and warm up the selected ASR engine."""
    global WHISPER  # pylint: disable=global-statement
    engine_type = config.ASR_ENGINE
    effective_device = config.DEVICE if engine_type == "INTEL-WHISPER" else config.ASR_ENGINE_DEVICE.upper()
    logger.info("Loading ASR Engine: %s (Device: %s)...",
                engine_type, effective_device)

    try:
        if engine_type == "INTEL-WHISPER":
            WHISPER = intel_engine.IntelWhisperEngine(
                config.MODEL_ID,
                device=config.DEVICE
            )
        else:
            # Inform user about device fallback for Intel hardware in Faster-Whisper
            if config.DEVICE in ["NPU", "GPU"]:
                actual_backend = config.ASR_ENGINE_DEVICE.upper()
                logger.warning(
                    "[ASR] Intel %s detected/selected. Intel-Whisper engine is currently on-hold for quality. Falling back to %s.", config.DEVICE, actual_backend)

            # Enforce thread limits for PyTorch (Faster-Whisper backend)
            torch.set_num_threads(config.ASR_THREADS)
            torch.set_num_interop_threads(config.ASR_THREADS)

            WHISPER = WhisperModel(
                config.MODEL_ID,
                device=config.ASR_ENGINE_DEVICE,
                compute_type=config.ASR_ENGINE_COMPUTE_TYPE,
                num_workers=1,
                download_root=config.OV_CACHE_DIR,
                cpu_threads=config.ASR_THREADS
            )

        logger.info("%s loaded successfully!", engine_type)

        # Warm up preprocessing models
        if config.ENABLE_VOCAL_SEPARATION or config.ENABLE_LD_PREPROCESSING:
            try:
                from . import preprocessing
                preprocessing.get_manager().ensure_models_loaded()
            except Exception as warmup_err:
                logger.error("Preprocessing Warmup Failed: %s", warmup_err)

        return True
    except Exception as err:
        logger.error("CRITICAL ERROR LOADING ENGINE: %s", err)
        return False


# --- [INFERENCE EXECUTION] ---
def run_transcription(audio_path, language=None, task='transcribe', _batch_size=None):  # pylint: disable=too-many-locals
    """
    Coordinates the full transcription lifecycle.

    Stages: Pre-empt check -> Preprocessing (UVR) -> ASR Inference (yielding supported) -> Post-processing.
    """
    global _TRANSCRIBING_SESSIONS  # pylint: disable=global-statement
    if WHISPER is None:
        raise RuntimeError(
            "Model not loaded - check logs for initialization errors.")

    # Block if priority task is active
    wait_for_priority()

    # Track transcription state
    inference_path = audio_path
    with model_lock_ctx():
        # pylint: disable=global-statement
        global _TRANSCRIBING_SESSIONS
        _TRANSCRIBING_SESSIONS += 1
        THREAD_DATA.is_transcribing = True

        try:
            _init_transcription_stats(audio_path)

            # Stage 1: Preprocessing (Vocal Separation)
            extra_prep_time, inference_path = _preprocess_audio(audio_path)

            # Stage 2: Hardware Diagnostics
            _log_audio_diagnostics(inference_path)

            total_dur = getattr(THREAD_DATA, "total_duration", 0)
            start_time = time.time()

            if config.ASR_ENGINE == "FASTER-WHISPER":
                chunk_results, detected_language, language_probability, total_dur = \
                    _run_faster_whisper_transcription(
                        inference_path, language, task, start_time)
            else:
                chunk_results, detected_language, language_probability = \
                    _run_intel_whisper_transcription(
                        inference_path, language, task, start_time, total_dur)

            result = {
                "segments": chunk_results,
                "chunks": chunk_results,
                "text": " ".join([c["text"] for c in chunk_results]),
                "duration": total_dur,
                "video_duration_sec": total_dur,
                "extra_preprocess_duration": extra_prep_time,
                "language": detected_language,
                "detected_language": detected_language,
                "language_probability": language_probability
            }

            # Stage 4: Post-Processing (Hallucination Filtering)
            result = _post_process_vad(result, inference_path)
            result["text"] = "".join([c.get("text", "")
                                     for c in result.get("chunks", [])]).strip()

            return result

        finally:
            THREAD_DATA.is_transcribing = False
            _TRANSCRIBING_SESSIONS = max(0, _TRANSCRIBING_SESSIONS - 1)

            # Cleanup derived assets
            if inference_path and inference_path != audio_path and os.path.exists(inference_path):
                try:
                    os.remove(inference_path)
                    logger.debug(
                        "Cleaned up intermediate file: %s", inference_path)
                except Exception as cleanup_err:
                    logger.warning(
                        "Failed to cleanup intermediate file: %s", cleanup_err)


# --- [INTERNAL HELPER UTILITIES] ---

def _preprocess_audio(audio_path):
    """Execute UVR isolation if enabled."""
    extra_prep_time = 0.0
    inference_path = audio_path

    if config.ENABLE_VOCAL_SEPARATION:
        try:
            p_start = time.time()
            from . import preprocessing
            pm = preprocessing.get_manager()
            inference_path = pm.process_audio_file(audio_path, yield_cb=None)
            extra_prep_time = time.time() - p_start
            _init_transcription_stats(inference_path)
        except Exception as prep_err:
            logger.error("Vocal separation engine failed: %s", prep_err)

    return extra_prep_time, inference_path


def _run_faster_whisper_transcription(inference_path, language, task, start_time):
    """Execute transcription using the Faster-Whisper backend."""
    logger.debug("[ASR] Using native Faster-Whisper pipeline with internal VAD.")
    segments_gen, info = WHISPER.transcribe(
        inference_path,
        beam_size=config.DEFAULT_BEAM_SIZE,
        language=language,
        task=task,
        vad_filter=True,
        vad_parameters={
            "threshold": 0.35,
            "min_silence_duration_ms": config.VAD_MIN_SILENCE_DURATION_MS,
            "speech_pad_ms": config.VAD_SPEECH_PAD_MS
        },
        initial_prompt=config.INITIAL_PROMPT
    )

    chunk_results = []
    for s in segments_gen:
        wait_for_priority(model_lock=_MODEL_LOCK)
        chunk = {
            "text": s.text,
            "start": round(s.start, 3),
            "end": round(s.end, 3),
            "timestamp": (s.start, s.end),
            "probability": math.exp(s.avg_logprob) if s.avg_logprob else 1.0
        }
        if chunk["text"]:
            chunk_results.append(chunk)
            _log_progress_manual(s.end, info.duration, start_time, s.text)

    return chunk_results, info.language, info.language_probability, info.duration


def _run_intel_whisper_transcription(inference_path, language, task, start_time, total_dur):
    """Execute transcription using the Intel OpenVINO backend."""
    logger.debug("[ASR] Using Intel OpenVINO engine with native long-form handling.")
    full_raw_audio = vad.decode_audio(inference_path)
    out = WHISPER.transcribe(
        full_raw_audio,
        language=language,
        task=task,
        beam_size=config.DEFAULT_BEAM_SIZE,
        initial_prompt=config.INITIAL_PROMPT,
        vad_filter=True,
        vad_threshold=0.35,
        min_silence_duration_ms=config.VAD_MIN_SILENCE_DURATION_MS,
        speech_pad_ms=config.VAD_SPEECH_PAD_MS
    )

    chunk_results = out.get("chunks", [])
    if chunk_results:
        last_chunk = chunk_results[-1]
        _log_progress_manual(last_chunk['end'], total_dur, start_time, out.get("text", ""))

    return chunk_results, out.get("language", language), out.get("language_probability", 0.0)


def _log_audio_diagnostics(inference_path):
    """Log file metadata for verification."""
    try:
        f_info = sf.info(inference_path)
        logger.info("[ASR] Input stats: %s (%d ch, %d Hz)",
                    utils.format_duration(f_info.duration), f_info.channels, f_info.samplerate)
    except Exception as d_err:  # pylint: disable=broad-exception-caught
        logger.debug("Diagnostic check skipped: %s", d_err)


def _init_transcription_stats(audio_path):
    """Prepare session metadata for ETA calculation."""
    # Ensure attributes exist to avoid AttributeError in tests/runtime
    THREAD_DATA.total_duration = 0
    THREAD_DATA.start_time = time.time()
    try:
        info = sf.info(audio_path)
        THREAD_DATA.total_duration = info.duration
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
    Perform lightweight language identification scan.
    Returns probabilities and confidence scores.
    """
    if WHISPER is None:
        raise RuntimeError("Model not loaded.")

    try:
        with model_lock_ctx():
            out = WHISPER.transcribe(
                audio_path,
                beam_size=5,
                vad_filter=True,
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

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Language detection task failed: %s", e)
        # pylint: disable=duplicate-code
        return {
            "detected_language": "en",
            "language": "en",
            "language_code": "en",
            "confidence": 0.0,
            "all_probabilities": {"en": 1.0}
        }


def _post_process_vad(result, _audio_path):
    """
    Apply hallucination filtering based on configured credits/silence phrases.
    """
    try:
        chunks = result.get('chunks', [])
        if not chunks or not config.HALLUCINATION_PHRASES:
            return result

        phrase_drop_count = 0
        for chunk in chunks:
            text = chunk.get('text', '').strip()
            if not text:
                continue

            text_clean = text.lower().strip(".,!?;: ")
            for phrase in config.HALLUCINATION_PHRASES:
                if phrase in text_clean:
                    # Drop if text is nearly identical to hallucination phrase
                    if len(text_clean) < len(phrase) + 10:
                        chunk['text'] = ""
                        phrase_drop_count += 1
                        break

        if phrase_drop_count > 0:
            logger.info(
                "Post-Processor: Filtered %d hallucination segments.", phrase_drop_count)

        return result
    except Exception as e:
        logger.error("VAD Post-Processing failed: %s", e)
        return result
