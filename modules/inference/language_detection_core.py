"""
Core Single-Segment Language Detection Logic.
"""

import gc
import logging
import sys
import time

from modules.core import config
from modules.inference import scheduler, vad

logger = logging.getLogger(__name__)


def run_language_detection(audio_path):
    """Optimized language detection using the faster detect_language API."""
    model_manager = sys.modules["modules.inference.model_manager"]
    start_time = time.time()
    with model_manager.model_lock_ctx() as (model, _):
        scheduler.update_task_progress(5, "Detection")
        res = model_manager.run_language_detection_core(model, audio_path)
        res["performance"] = {"inference_sec": round(time.time() - start_time, 2)}
        res["segments_processed"] = 1
        scheduler.update_task_metadata(result=res)
        return res


def run_batch_language_detection(audio_path, segment_count):
    """High-performance multi-segment identification scan."""
    model_manager = sys.modules["modules.inference.model_manager"]
    with model_manager.model_lock_ctx() as (model, _):
        return model_manager.run_batch_language_detection_direct(model, audio_path, segment_count)


def run_batch_language_detection_direct(model, audio_path, segment_count):
    """Direct batch detection without re-acquiring the lock."""
    model_manager = sys.modules["modules.inference.model_manager"]
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
            results.append(model_manager.run_language_detection_core(model, chunk, skip_vad=False))

            # Granular progress for voting (Maps 60% -> 95%)
            progress = 60 + int(((i + 1) / segment_count) * 35)
            stage = f"Inference ({i + 1}/{segment_count} segments)"
            logger.info("[Engine] %s...", stage)
            scheduler.update_task_progress(progress, stage)
        return results
    except (ImportError, RuntimeError, OSError, ValueError, AttributeError, KeyError, TypeError) as e:
        logger.error("[Engine] Batch detection failed: %s", e)
        return []
    finally:
        if full_audio is not None:
            del full_audio
        # Ensure results don't hold any references to the original buffer
        gc.collect()


def run_language_detection_core(model, audio_input, skip_vad=False):
    """Internal core using detect_language optimization."""
    speech_sec = 30.0
    if not skip_vad:
        if isinstance(audio_input, str):
            speech_ts = vad.get_speech_timestamps_from_path(audio_input, threshold=config.LD_VAD_THRESHOLD)
        else:
            speech_ts = vad.get_speech_timestamps(audio_input, threshold=config.LD_VAD_THRESHOLD)

        if not speech_ts:
            return {
                "detected_language": "en",
                "language": "en",
                "confidence": 0.0,
                "all_probabilities": {"en": 0.0},
                "speech_duration": 0.0,
            }
        speech_sec = sum(ts["end"] - ts["start"] for ts in speech_ts)

    try:
        # Optimization: Use detect_language to avoid full decoding
        if isinstance(audio_input, str):
            audio_input = vad.decode_audio(audio_input)

        if hasattr(audio_input, "astype"):
            audio_input = audio_input.astype("float32")

        lang_code, lang_prob, all_probs_list = model.detect_language(audio_input)
        logger.info("[Engine] Identified: %s (%.1f%%)", lang_code, lang_prob * 100)
        all_probs = dict(all_probs_list) if all_probs_list else {lang_code: lang_prob}
        # Remove very small probabilities from the response to reduce noise
        all_probs = {k: v for k, v in all_probs.items() if v >= 0.001}
        return {
            "detected_language": lang_code,
            "language": lang_code,
            "confidence": lang_prob,
            "all_probabilities": all_probs,
            "speech_duration": round(speech_sec, 3),
        }
    except tuple([Exception]) as e:
        logger.info("[Engine] detect_language fallback: %s", e)
        # Fallback to minimal transcribe
        _, info = model.transcribe(audio_input, beam_size=1, task="transcribe")
        all_probs = dict(info.all_language_probs) if info.all_language_probs else {}
        all_probs = {k: v for k, v in all_probs.items() if v >= 0.001}
        return {
            "detected_language": info.language,
            "language": info.language,
            "confidence": info.language_probability,
            "all_probabilities": all_probs,
            "speech_duration": round(speech_sec, 3),
        }
