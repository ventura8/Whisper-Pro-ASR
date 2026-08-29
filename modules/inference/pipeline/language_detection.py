"""
Language Identification Logic

This module implements a robust, multi-zone voting system for language detection.
It strategically samples audio segments, performs one-token probability scans,
and aggregates results using squared confidence weighting to ensure high accuracy.
"""

import collections
import importlib
import logging
import os
import re
import tempfile
import time
from typing import Optional

import numpy as np

from modules.core import config, process_exec, utils
from modules.inference import scheduler

# Lazy load containers
_LIBS = {"sf": None}


def _get_sf():
    """Retrieve the soundfile module dynamically."""
    if _LIBS["sf"] is None:
        _LIBS["sf"] = importlib.import_module("soundfile")
    return _LIBS["sf"]


logger = logging.getLogger(__name__)

# Pre-compiled regex for language token extraction
_LANG_PATTERN = re.compile(r"<\|([a-z]{2,3})\|>")


# --- [CORE PIPELINE] ---


# --- [VOTING & AGGREGATION ENGINE] ---


def aggregate_language_probs(segment_probs_list):
    """
    Consolidate probabilities across all segments using Squared Weighting and Speech Duration Weighting.

    Squaring the softmax probabilities punishes low-confidence noise and
    rewards clear identification 'peaks', preventing consistent false-positives
    from overriding correct detections in mixed audio.
    """
    valid = [cp for cp in segment_probs_list if cp]
    if not valid:
        return {}
    combined_scores = _combine_language_scores(valid)
    return _normalize_scores(combined_scores)


def _combine_language_scores(valid_probs: list[dict]) -> dict:
    combined_scores = {}
    for cp in valid_probs:
        weight = cp.get("_speech_duration", cp.get("speech_duration", 30.0))
        for lang, prob in cp.items():
            if _is_metadata_language_key(lang):
                continue
            combined_scores[lang] = combined_scores.get(lang, 0) + (prob**2) * weight
    return combined_scores


def _is_metadata_language_key(lang: str) -> bool:
    return lang.startswith("_") or lang in ("speech_duration", "speech_ratio")


def _normalize_scores(combined_scores: dict) -> dict:
    total_score = sum(combined_scores.values())
    if total_score <= 0:
        return {}
    return {k: v / total_score for k, v in combined_scores.items()}


# --- [INFERENCE LOGIC] ---


# --- [SMART SAMPLING ENGINE] ---


def _get_sampling_target(duration):
    """Heuristic logic to determine scan density based on file length."""
    for max_duration, target in _sampling_density_buckets():
        if duration <= max_duration:
            return target
    return 15


def _sampling_density_buckets() -> list[tuple[int, int]]:
    return [(120, 1), (600, 3), (1200, 5), (3600, 9), (10800, 13)]


# --- [VOTING CONCURRENCY] ---


def run_voting_detection(audio_path, model_manager, start_time=None):
    """
    High-level entry point for high-performance batch language voting.

    Uses a 'Montage' strategy: concatenates all sampling targets into one file,
    performs a single UVR pass, and then runs batch inference.
    """
    duration = utils.get_audio_duration(audio_path) or 300
    scans = _get_sampling_target(duration)
    logger.info(
        "[LD] Target: %s | Duration: %s | Density: %d segments",
        os.path.basename(audio_path),
        utils.format_duration(duration),
        scans,
    )
    offsets = _generate_sampling_tasks(audio_path, duration, scans)

    return _execute_batch_scan(audio_path, offsets, model_manager, scans, start_time)


def _execute_batch_scan(audio_path, offsets, model_manager, scans, start_time=None):
    """Internal orchestrator for the montage/UVR/inference pipeline."""
    perf = {"start_queue": start_time or time.time()}
    montage_path = None
    isolated_path = None
    try:
        # Montage creation is CPU-only FFmpeg work and does not need to hold a hardware unit.
        # Build it first so a paused ASR task cannot extend the priority task's unit claim.
        montage_path = _step_create_montage(audio_path, offsets, scans, perf)

        with model_manager.model_lock_ctx() as (model, unit_id):
            perf["dur_queue"] = time.time() - perf["start_queue"]

            # Phase 2: Isolation
            isolated_path = _step_isolate_vocals(montage_path, model_manager, unit_id, perf)

            # Phase 3: Inference
            res = _step_run_inference((model, model_manager), isolated_path, scans, perf)

            if res is not None:
                return res

    except (OSError, ValueError, RuntimeError, ImportError, TypeError) as e:
        logger.error("[LD] Batch consensus scan failed: %s", e)
    finally:
        _cleanup_batch_assets(montage_path, isolated_path)

    return model_manager.run_language_detection(audio_path)


def run_voting_detection_on_isolated(audio_path, model, model_manager, start_time=None):
    """Voting detection that reuses a held hardware unit and already-separated audio.

    The transcription pipeline separates vocals *before* detecting the language, so two
    things differ from :func:`run_voting_detection`: the montage needs no second UVR pass
    (the audio is already isolated), and the caller already owns the unit -- re-entering
    ``model_lock_ctx`` here would deadlock against itself.

    Returns the detection payload, or ``None`` to let the engine auto-detect.
    """
    duration = utils.get_audio_duration(audio_path) or 300
    scans = _get_sampling_target(duration)
    logger.info(
        "[LD] Target: %s | Duration: %s | Density: %d segments (on separated audio)",
        os.path.basename(audio_path),
        utils.format_duration(duration),
        scans,
    )
    offsets = _generate_sampling_tasks(audio_path, duration, scans)

    perf = {"start_queue": start_time or time.time(), "dur_queue": 0.0, "dur_iso": 0.0}
    montage_path = None
    try:
        montage_path = _step_create_montage(audio_path, offsets, scans, perf)
        return _step_run_inference((model, model_manager), montage_path or audio_path, scans, perf)
    except (OSError, ValueError, RuntimeError, ImportError, TypeError) as e:
        logger.error("[LD] Batch consensus scan on separated audio failed: %s", e)
        return None
    finally:
        _cleanup_batch_assets(montage_path, None)


def _step_create_montage(audio_path, offsets, _scans, perf):
    perf["start_montage"] = time.time()
    scheduler.update_task_progress(10, "Montage")
    path = _prepare_montage(audio_path, offsets)
    # Also register with request-level tracker so outer cleanup catches it too
    if path:
        utils.track_file(path)
    perf["dur_montage"] = time.time() - perf["start_montage"]
    return path


def _step_isolate_vocals(montage_path, model_manager, unit_id, perf):
    if not config.ENABLE_LD_PREPROCESSING:
        perf["dur_iso"] = 0.0
        return montage_path

    perf["start_iso"] = time.time()
    scheduler.update_task_progress(20, "Vocal Isolation")
    path = model_manager.run_vocal_isolation_direct(montage_path, unit_id, force=True, stage="Vocal Isolation")
    # Register with request-level tracker as a second safety net
    if path and path != montage_path:
        utils.track_file(path)
    perf["dur_iso"] = time.time() - perf["start_iso"]
    return path


def _step_run_inference(model_context, isolated_path, scans, perf):
    model, model_manager = model_context
    perf["start_inf"] = time.time()
    scheduler.update_task_progress(60, f"Inference (0/{scans} segments)")
    results = model_manager.run_batch_language_detection_direct(model, isolated_path, scans)
    perf["dur_inf"] = time.time() - perf["start_inf"]

    probs = _collect_high_confidence_votes(results)
    voting_details = aggregate_language_probs(probs)

    if not voting_details:
        logger.info("[Engine] No high-confidence voting details found, falling back to full file detection.")
        return None

    res = _format_detection_result(voting_details, scans)
    # The montage already scatters `scans` short samples across the file and detects each
    # independently before voting collapses them to one answer -- this is a free look at
    # whether those samples actually agree. Disagreement is the cheap, always-computed
    # signal that a full per-chunk scan (expensive; a full pass over the whole file) is
    # only worth paying for on genuinely multi-lingual audio.
    res["multilingual_suspected"] = _votes_disagree(results)
    res["performance"] = {
        "queue_sec": round(perf["dur_queue"], 2),
        "montage_sec": round(perf["dur_montage"], 2),
        "isolation_sec": round(perf["dur_iso"], 2),
        "inference_sec": round(perf["dur_inf"], 2),
    }
    model_manager.update_task_metadata(result=res)
    return res


#: Share of the confident votes a single-vote language must hold to count as real. Below
#: this it is treated as a lone dissenter -- noise rather than a code-switch.
_LONE_DISSENTER_SHARE = 1 / 3


def _votes_disagree(results: list[dict]) -> bool:
    """Whether the per-offset montage samples named more than one confident language.

    Every vote must clear LD_MIN_CONFIDENCE, the same bar the vote itself uses. Beyond
    that, a language backed by a single sample counts only when that sample is a
    meaningful share of the evidence: one dissenting offset out of nine is far more likely
    to be a misfire than a genuine second language, while one out of two is half of
    everything the montage saw and cannot be dismissed.

    The share test is what makes this usable on short files. Requiring two votes per
    language outright would be stricter but useless there: a clip under ten minutes is
    sampled at only one to three offsets (see _get_sampling_target), so no language could
    ever reach a second vote and code-switching would be undetectable by construction.
    """
    langs = _confident_vote_counts(results)
    total = sum(langs.values())
    if total < 2:
        return False
    confirmed = sum(1 for count in langs.values() if _vote_is_credible(count, total))
    return confirmed > 1


def _top_language_of_vote(vote: dict) -> str | None:
    """The highest-scoring real language in one offset's vote, ignoring bookkeeping keys.

    _is_metadata_language_key, not a literal "_speech_duration" comparison: the vote dict
    carries several bookkeeping keys, and any one of them other than that single hardcoded
    name could win the max() and be returned as the detected language.
    """
    return max((key for key in vote if not _is_metadata_language_key(key)), key=vote.get, default=None)


def _confident_vote_counts(results: list[dict]) -> collections.Counter:
    """Count how many montage offsets named each language, skipping low-confidence ones."""
    langs: collections.Counter = collections.Counter()
    for result in results:
        vote = _extract_segment_vote_or_none(result)
        if vote is None:
            continue
        lang = _top_language_of_vote(vote)
        if lang:
            langs[lang] += 1
    return langs


def _vote_is_credible(count: int, total: int) -> bool:
    """Whether a language's vote count is evidence rather than a single misfire."""
    return count >= 2 or count / total >= _LONE_DISSENTER_SHARE


def _collect_high_confidence_votes(results: list[dict]) -> list[dict]:
    probs = []
    for i, result in enumerate(results):
        vote = _extract_segment_vote_or_none(result)
        if vote is not None:
            probs.append(vote)
        else:
            _log_low_confidence_vote(i, result)
    return probs


def _extract_segment_vote_or_none(result: dict):
    if not (result and "all_probabilities" in result):
        return None
    conf = result.get("confidence", 0.0)
    if conf < config.LD_MIN_CONFIDENCE:
        return None
    prob_dict = result["all_probabilities"].copy()
    prob_dict["_speech_duration"] = result.get("speech_duration", 30.0)
    return prob_dict


def _log_low_confidence_vote(index: int, result: dict):
    if not result:
        return
    conf = result.get("confidence", 0.0)
    if "all_probabilities" in result and conf < config.LD_MIN_CONFIDENCE:
        logger.warning(
            "[Engine] Skipping segment %d vote due to low confidence: %s (%.1f%% < %.1f%%)",
            index + 1,
            result.get("detected_language", "unknown"),
            conf * 100,
            config.LD_MIN_CONFIDENCE * 100,
        )


def _prepare_montage(source_path, offsets):
    """Extract and concatenate audio slices into a single montage file."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=config.get_temp_dir()) as montage_tmp:
        montage_path = montage_tmp.name
    utils.track_file(montage_path)

    input_flags = getattr(utils.THREAD_CONTEXT, "input_flags", None)
    inputs, filter_parts = _build_montage_inputs_and_filters(source_path, offsets, input_flags)
    filter_complex = _build_montage_filter_complex(filter_parts)
    cmd = _build_montage_command(inputs, filter_complex, montage_path)

    logger.info("[LD] Extracting montage (%d samples)...", len(offsets))
    result = process_exec.run_capture(cmd)
    if result.returncode != 0:
        logger.error("[LD] FFmpeg montage failed (code %d): %s", result.returncode, result.stderr)
        raise RuntimeError(f"FFmpeg montage extraction failed with code {result.returncode}")

    return montage_path


def _build_montage_inputs_and_filters(source_path, offsets, input_flags):
    inputs = []
    filter_parts = []
    for i, offset in enumerate(offsets):
        segment_args = ["-ss", str(offset), "-t", "30"]
        if input_flags:
            segment_args.extend(input_flags)
        segment_args.extend(["-i", source_path])
        inputs.extend(segment_args)
        filter_parts.append(f"[{i}:a]{utils.STANDARD_NORMALIZATION_FILTERS},apad=whole_dur=30[a{i}];")
    return inputs, filter_parts


def _build_montage_filter_complex(filter_parts: list[str]) -> str:
    count = len(filter_parts)
    concat_inputs = "".join(f"[a{i}]" for i in range(count))
    return "".join(filter_parts) + f"{concat_inputs}concat=n={count}:v=0:a=1,{utils.STANDARD_NORMALIZATION_FILTERS}[out]"


def _build_montage_command(inputs: list[str], filter_complex: str, montage_path: str) -> list[str]:
    standard_audio_flags = [flag for flag in utils.STANDARD_AUDIO_FLAGS if flag != "-vn"]
    return (
        ["ffmpeg", "-y", "-threads", str(config.FFMPEG_THREADS), "-loglevel", "error"]
        + inputs
        + ["-filter_complex", filter_complex, "-map", "[out]"]
        + standard_audio_flags
        + [montage_path]
    )


def _cleanup_batch_assets(montage_path, isolated_path):
    """Ensure temporary montage files are purged, tolerating errors."""
    utils.secure_remove(montage_path)
    if isolated_path and isolated_path != montage_path:
        utils.secure_remove(isolated_path)


def _generate_sampling_tasks(audio_path, duration, scans):
    """
    Distributes sampling points across the duration.
    Uses standard 30-second samples for maximum detection context.
    """
    sample_len = 30
    zone_size = max(0, duration - sample_len) / scans
    tasks = []
    if config.SMART_SAMPLING_SEARCH:
        for i in range(scans):
            base_offset = i * zone_size
            tasks.append(_find_best_offset_in_zone(audio_path, base_offset, zone_size, duration))
    else:
        # Uniform distribution
        offsets = np.linspace(0, max(0, duration - (sample_len + 1)), scans)
        tasks = offsets.tolist()
    return tasks


def _format_detection_result(voting_details, scans):
    """Encapsulate probability aggregation into a finalized response schema."""
    if not voting_details:
        return {
            "detected_language": "en",
            "language": "en",
            "language_code": "en",
            "confidence": 0.0,
            "segments_processed": scans,
            "voting_details": {},
        }

    best_lang = max(voting_details, key=voting_details.get)
    avg_conf = voting_details[best_lang]

    # Filter out very low confidence entries (below 1% threshold)
    threshold = 0.01
    filtered_details = {k: v for k, v in voting_details.items() if v >= threshold}
    sorted_details = dict(sorted(filtered_details.items(), key=lambda item: item[1], reverse=True))

    logger.debug("[LD] Final Winner: %s (Weight: %.4f)", best_lang, avg_conf)

    return {
        "detected_language": best_lang,
        "language": best_lang,
        "language_code": best_lang,
        "confidence": avg_conf,
        "segments_processed": scans,
        "voting_details": sorted_details,
    }


def _find_best_offset_in_zone(audio_path, base_offset, zone_size, total_duration):
    """Perform localized RMS scan to find visible speech segments."""
    try:
        current_sf = _get_sf()
        info = current_sf.info(audio_path)
        file_sr = info.samplerate

        offset = _search_speech_offset(current_sf, audio_path, base_offset, file_sr, total_duration)
        if offset is not None:
            return offset

        return base_offset + (zone_size / 2)

    except tuple([Exception]):
        return base_offset + (zone_size / 2)


def _search_speech_offset(current_sf, audio_path, base_offset, file_sr, total_duration) -> Optional[float]:
    for retry in range(3):
        offset = _resolve_retry_offset(base_offset, retry, total_duration)

        # Optimization: Pick center offset if not a WAV to avoid expensive probing
        if not audio_path.lower().endswith(".wav"):
            return None

        audio = _read_audio_window(current_sf, audio_path, offset, file_sr)
        if _has_speech_energy(audio):
            return offset
    return None


def _resolve_retry_offset(base_offset: float, retry: int, total_duration: float) -> float:
    offset = base_offset + (retry * 10)
    if offset + 30 > total_duration:
        return max(0, total_duration - 30)
    return offset


def _read_audio_window(current_sf, audio_path, offset: float, file_sr: int):
    audio, _ = current_sf.read(audio_path, start=int(offset * file_sr), frames=int(30 * file_sr), dtype="float32")
    if audio.ndim == 2:
        return audio.mean(axis=1)
    return audio


def _has_speech_energy(audio) -> bool:
    return np.sqrt(np.mean(audio**2)) >= 0.005
