"""
Core Single-Segment Language Detection Logic.
"""

import gc
import logging
import sys
import time

from modules.core import config
from modules.inference import scheduler
from modules.inference.pipeline import vad

logger = logging.getLogger(__name__)


def run_language_detection(audio_path):
    """Optimized language detection using the faster detect_language API."""
    model_manager = sys.modules["modules.inference.runtime.model_manager"]
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
    model_manager = sys.modules["modules.inference.runtime.model_manager"]
    with model_manager.model_lock_ctx() as (model, _):
        return model_manager.run_batch_language_detection_direct(model, audio_path, segment_count)


def _is_isolated_engine(model) -> bool:
    """Whether ``model`` is an out-of-process engine proxy.

    Read off the model's *type*, not the instance: a MagicMock answers yes to any
    attribute asked of an instance, which would silently route every mocked engine in the
    test suite down the worker path. Its class does not, so the check stays honest.

    A class marker rather than an isinstance check because importing IsolatedEngine here
    closes a cycle -- isolated_engine imports inference_worker, which imports this module.
    """
    return getattr(type(model), "IS_ISOLATED_ENGINE", False) is True


def _detect_segments_isolated(model, audio_path, segment_count) -> list:
    """Batch-detect inside the engine's worker process.

    The worker decodes the audio itself and streams back one small result per window, so
    no audio array ever crosses the process boundary. Progress is republished here
    because the scheduler lives in this process, not the worker's.
    """
    results = []
    for event in model.detect_language_batch(audio_path, segment_count):
        results.append(event["result"])
        index = event.get("index", len(results) - 1)
        progress = 60 + int(((index + 1) / max(segment_count, 1)) * 35)
        stage = f"Inference ({index + 1}/{segment_count} segments)"
        logger.info("[Engine] %s...", stage)
        scheduler.update_task_progress(progress, stage)
    return results


def run_batch_language_detection_direct(model, audio_path, segment_count):
    """Direct batch detection without re-acquiring the lock."""
    model_manager = sys.modules["modules.inference.runtime.model_manager"]
    full_audio = None
    try:
        if _is_isolated_engine(model):
            return _detect_segments_isolated(model, audio_path, segment_count)
        full_audio = vad.decode_audio(audio_path)
        return _detect_segments(model, model_manager, full_audio, segment_count)
    except (ImportError, RuntimeError, OSError, ValueError, AttributeError, KeyError, TypeError) as e:
        logger.error("[Engine] Batch detection failed: %s", e)
        return []
    finally:
        _cleanup_batch_detection(full_audio)


def _detect_segments(model, model_manager, full_audio, segment_count) -> list:
    results = []
    segment_len = int(30 * 16000)
    for i in range(segment_count):
        start = i * segment_len
        if start >= len(full_audio):
            break
        end = min(start + segment_len, len(full_audio))
        chunk = full_audio[start:end].copy()
        results.append(model_manager.run_language_detection_core(model, chunk, skip_vad=False))

        # Granular progress for voting (Maps 60% -> 95%)
        progress = 60 + int(((i + 1) / segment_count) * 35)
        stage = f"Inference ({i + 1}/{segment_count} segments)"
        logger.info("[Engine] %s...", stage)
        scheduler.update_task_progress(progress, stage)
    return results


def _cleanup_batch_detection(full_audio):
    if full_audio is not None:
        del full_audio
    gc.collect()


def run_language_detection_core(model, audio_input, skip_vad=False):
    """Internal core using detect_language optimization."""
    speech_sec, no_speech_result = _resolve_speech_duration(audio_input, skip_vad)
    if no_speech_result is not None:
        return no_speech_result
    if _should_detect_over_the_pipe(model, audio_input):
        return _detect_language_primary(model, audio_input, speech_sec)

    audio_input = _sanitized_or_original(audio_input)
    try:
        return _detect_language_primary(model, audio_input, speech_sec)
    except tuple([Exception]) as e:
        return _detect_language_fallback(model, audio_input, speech_sec, e)


def _should_detect_over_the_pipe(model, audio_input) -> bool:
    """Whether to hand an isolated engine the path and let it decode on the far side.

    Decoding here would copy the samples across the pipe for nothing. An isolated engine
    handed raw samples instead of a path cannot take that route, which is worth saying out
    loud rather than silently falling through to the in-process path.
    """
    if not _is_isolated_engine(model):
        return False
    if isinstance(audio_input, str):
        return True
    logger.warning("[Engine] Isolated engine needs an audio path for detection; got samples.")
    return False


def _sanitized_or_original(audio_input):
    """Normalise the audio for detection, keeping the original when that is not possible."""
    try:
        return _sanitize_ld_audio_input(audio_input)
    except (ImportError, RuntimeError, OSError, ValueError, AttributeError, TypeError) as sanitize_err:
        logger.info("[Engine] Audio sanitize fallback: %s", sanitize_err)
        return audio_input


def find_uncovered_speech_gaps(segments: list[dict], speech_ts: list[dict], duration_sec: float, min_gap_sec: float = 0.5) -> list[dict]:
    """Return VAD-confirmed speech regions no transcribed segment covers.

    This is the actual, general shape of "part of the audio didn't get transcribed
    correctly": forcing one language for the whole file does not just corrupt the wrong
    portion, it frequently makes Whisper stop emitting segments there entirely rather
    than hallucinate onward -- so the failure shows up as a coverage hole, not obviously
    bad text. A 30s-chunk-grained language scan cannot see a switch that happens inside
    one chunk (a clip shorter than 30s is always exactly one chunk, so a scan at that
    granularity can never split it); a gap found after the fact needs no advance
    knowledge of where the boundary is.
    """
    covered = _covered_intervals(segments)
    gaps: list[dict] = []
    for region_start, region_end in _clamped_regions(speech_ts, duration_sec):
        gaps.extend(_gaps_within_region(region_start, region_end, covered))
    return _long_enough(gaps, min_gap_sec)


def _covered_intervals(segments: list[dict]) -> list[tuple[float, float]]:
    """Transcribed spans as sorted (start, end) pairs, dropping zero-length ones.

    Sorted because _gaps_within_region walks them in order and assumes it.
    """
    return sorted((seg["start"], seg["end"]) for seg in segments if seg["end"] > seg["start"])


def _long_enough(gaps: list[dict], min_gap_sec: float) -> list[dict]:
    """Drop gaps too short to be worth a re-detection pass."""
    return [gap for gap in gaps if gap["end"] - gap["start"] >= min_gap_sec]


def _clamped_regions(speech_ts: list[dict], duration_sec: float) -> list[tuple[float, float]]:
    """Speech regions trimmed to the file, dropping any that start past its end.

    Clamped, not discarded. A VAD region routinely overruns the decoded duration by a
    fraction of a second, and the old tail filter (`g["end"] <= duration_sec + 0.5`) threw
    the *whole* gap away when it did -- so a clip whose single speech region ran even
    slightly long got no gap-fill at all, which is exactly the untranscribed-tail case
    find_uncovered_speech_gaps exists to catch.
    """
    regions = []
    for region in speech_ts:
        end = min(float(region["end"]), duration_sec)
        if end > region["start"]:
            regions.append((float(region["start"]), end))
    return regions


def _gaps_within_region(start: float, end: float, covered: list[tuple[float, float]]) -> list[dict]:
    """Subtract every covered interval from one [start, end) speech region."""
    cursor = start
    gaps = []
    for c_start, c_end in _overlapping(covered, start, end):
        if c_start > cursor:
            gaps.append({"start": cursor, "end": min(c_start, end)})
        cursor = max(cursor, c_end)
        if cursor >= end:
            break
    if cursor < end:
        gaps.append({"start": cursor, "end": end})
    return gaps


def _overlapping(covered: list[tuple[float, float]], start: float, end: float) -> list[tuple[float, float]]:
    """The covered intervals that actually intersect [start, end)."""
    return [(c_start, c_end) for c_start, c_end in covered if c_end > start and c_start < end]


def _resolve_speech_duration(audio_input, skip_vad: bool) -> tuple[float, dict | None]:
    if skip_vad:
        return 30.0, None
    speech_ts = _get_ld_speech_ts(audio_input)
    if not speech_ts:
        return 0.0, _no_speech_detection_result()
    speech_sec = sum(ts["end"] - ts["start"] for ts in speech_ts)
    return speech_sec, None


def _no_speech_detection_result() -> dict:
    return {
        "detected_language": "en",
        "language": "en",
        "confidence": 0.0,
        "all_probabilities": {"en": 0.0},
        "speech_duration": 0.0,
    }


def _detect_language_primary(model, audio_input, speech_sec: float) -> dict:
    lang_code, lang_prob, all_probs_list = model.detect_language(audio_input)
    logger.info("[Engine] Identified: %s (%.1f%%)", lang_code, lang_prob * 100)
    all_probs = dict(all_probs_list) if all_probs_list else {lang_code: lang_prob}
    return {
        "detected_language": lang_code,
        "language": lang_code,
        "confidence": lang_prob,
        "all_probabilities": {k: v for k, v in all_probs.items() if v >= 0.001},
        "speech_duration": round(speech_sec, 3),
    }


def _get_ld_speech_ts(audio_input) -> list:
    if isinstance(audio_input, str):
        return vad.get_speech_timestamps_from_path(audio_input, threshold=config.LD_VAD_THRESHOLD)
    return vad.get_speech_timestamps(audio_input, threshold=config.LD_VAD_THRESHOLD)


def _sanitize_ld_audio_input(audio_input):
    if isinstance(audio_input, str):
        audio_input = vad.decode_audio(audio_input)
    if hasattr(audio_input, "astype"):
        audio_input = audio_input.astype("float32")
    return audio_input


def _detect_language_fallback(model, audio_input, speech_sec, e) -> dict:
    logger.info("[Engine] detect_language fallback: %s", e)
    _, info = model.transcribe(audio_input, beam_size=1, task="transcribe")
    all_probs = dict(info.all_language_probs) if info.all_language_probs else {}
    return {
        "detected_language": info.language,
        "language": info.language,
        "confidence": info.language_probability,
        "all_probabilities": {k: v for k, v in all_probs.items() if v >= 0.001},
        "speech_duration": round(speech_sec, 3),
    }
