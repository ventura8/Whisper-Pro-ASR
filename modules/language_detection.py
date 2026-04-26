"""
Language Identification Logic

This module implements a dynamic, strategic sampling system for language detection.
It extracts multiple non-overlapping audio segments across the media to provide
high-confidence, representative language identification.
"""
# pylint: disable=no-member, protected-access, duplicate-code
import logging
import tempfile
import os
import subprocess
import concurrent.futures

from . import utils, config, preprocessing, vad

logger = logging.getLogger(__name__)


def _calculate_chunk_duration(total_duration):
    """
    Algorithm for dynamic chunk sizing based on total media duration.
    - < 1h: 5m to 8m (linear)
    - 1h to 4h: 8m to 20m (linear)
    - >= 4h: 20m (capped)
    """
    if total_duration < 3600:
        # 0s -> 300s, 3600s -> 480s
        return 300 + total_duration * (180 / 3600)
    if total_duration < 14400:
        # 3600s -> 480s, 14400s -> 1200s
        return 480 + (total_duration - 3600) * (720 / 10800)
    return 1200


def run_voting_detection(clean_wav, model_manager):
    """
    High-level entry point for language detection using Strategic Uniform Sampling.
    Calculates non-overlapping zones across the media to find representative speech.
    """
    total_dur = utils.get_audio_duration(clean_wav) or 300
    chunk_dur = min(_calculate_chunk_duration(total_dur), total_dur)

    # Strategy: Divide media into up to 5 non-overlapping zones for maximum representation.
    num_attempts = min(5, max(1, int(total_dur // chunk_dur)))
    stride = total_dur / num_attempts

    logger.info("[LD] Strategic Sampling: %d segments, stride %s, chunk %s",
                num_attempts, utils.format_duration(stride),
                utils.format_duration(chunk_dur))

    all_results = []

    # Pool Concurrency: Orchestration (FFmpeg/VAD) is light.
    # The heavy isolation logic is globally locked inside the PreprocessingManager.
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=config.get_parallel_limit("CPU")) as executor:
        prep_futures = {
            executor.submit(_prepare_detection_zone, clean_wav, i, stride, chunk_dur): i
            for i in range(num_attempts)
        }

        for i in range(num_attempts):
            last_attempt = i + 1
            try:
                future = next(f for f, idx in prep_futures.items() if idx == i)
                paths = future.result()
                result = _run_inference_on_prepared_zone(paths[0], paths[1], model_manager)
                _cleanup_files(list(paths))

                if result:
                    all_results.append(result)
                    if result.get('confidence', 0) >= config.LD_MIN_CONFIDENCE_THRESHOLD:
                        logger.info("[LD] High confidence found. Stopping early.")
                        return _format_detection_result(
                            result.get('all_probabilities', {}), last_attempt)
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error("[LD] Preparation failed for zone %d: %s", i + 1, e)

    return _format_detection_result(_aggregate_probabilities(all_results), last_attempt)


def _prepare_detection_zone(clean_wav, index, stride, chunk_dur):
    """Worker function to extract and isolate a single zone."""
    start_offset = index * stride
    if config.SMART_SAMPLING_SEARCH:
        start_offset = _find_best_speech_offset(clean_wav, start_offset, stride, chunk_dur)

    segment_path = _extract_segment(clean_wav, start_offset, chunk_dur)
    processed_path = None

    if config.ENABLE_LD_PREPROCESSING:
        try:
            processed_path = preprocessing.get_manager().process_audio_file(segment_path)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("[LD] Isolation failed for zone at %s: %s",
                         utils.format_duration(start_offset), e)

    return segment_path, processed_path


def _run_inference_on_prepared_zone(segment_path, processed_path, model_manager):
    """Execute VAD and Inference on already prepared audio files."""
    detection_path = processed_path if (
        processed_path and os.path.exists(processed_path)) else segment_path

    # Stage 1: Dual-Path VAD (Silent check)
    speech_ts = vad.get_speech_timestamps_from_path(
        detection_path, threshold=config.LD_VAD_THRESHOLD)

    if not speech_ts and processed_path and processed_path != segment_path:
        logger.info("[LD] Isolated vocals silent. Retrying VAD on original segment...")
        speech_ts = vad.get_speech_timestamps_from_path(
            segment_path, threshold=config.LD_VAD_THRESHOLD)
        if speech_ts:
            logger.info("[LD] Speech found in original segment. Using raw audio for LD.")
            detection_path = segment_path

    if not speech_ts:
        # Cleanup files if we exit early here
        _cleanup_files([segment_path, processed_path])
        return None

    # Stage 2: Dual-Path Confidence Verification
    result = model_manager.run_language_detection(detection_path)
    conf = result.get('confidence', 0) if result else 0

    if conf < 0.8 and processed_path and detection_path == processed_path:
        logger.info("[LD] Low confidence (%.2f) on isolated vocals. Verifying raw audio...", conf)
        speech_ts_raw = vad.get_speech_timestamps_from_path(
            segment_path, threshold=config.LD_VAD_THRESHOLD)

        if speech_ts_raw:
            result_raw = model_manager.run_language_detection(segment_path)
            conf_raw = result_raw.get('confidence', 0) if result_raw else 0
            if conf_raw > conf:
                logger.info("[LD] Raw audio yielded higher confidence (%.2f > %.2f). Picking raw.",
                            conf_raw, conf)
                result = result_raw

    return result


def _aggregate_probabilities(results_list):
    """Weighted voting logic: segments with higher confidence have more influence."""
    if not results_list:
        return {}

    if len(results_list) == 1:
        return results_list[0].get('all_probabilities', {})

    aggregated = {}
    total_weight = 0

    for res in results_list:
        probs = res.get('all_probabilities', {})
        # Confidence-weighted aggregation
        weight = max(res.get('confidence', 0), 0.01)
        for lang, val in probs.items():
            aggregated[lang] = aggregated.get(lang, 0) + (val * weight)
        total_weight += weight

    return {lang: val / total_weight for lang, val in aggregated.items()}


def _find_best_speech_offset(clean_wav, start_offset, stride_duration, chunk_dur):
    """
    Search within a stride to find a non-silent offset for the detection chunk.
    Uses FFmpeg's 'silencedetect' to identify speech regions efficiently.
    """
    logger.debug("[LD] Smart Search scanning stride from %s...",
                 utils.format_duration(start_offset))
    try:
        # Scan the stride for speech timestamps
        speech_ts = vad.get_speech_timestamps_from_path(
            clean_wav,
            start_offset=start_offset,
            duration=stride_duration
        )

        if speech_ts:
            # Pick the first valid speech region found in this stride
            new_offset = speech_ts[0]['start']
            # Ensure we have enough room for the chunk (at least half the duration)
            if new_offset < start_offset + stride_duration - (chunk_dur / 2):
                logger.debug("[LD] Smart Search found speech at %s",
                             utils.format_duration(new_offset))
                return new_offset

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.debug("[LD] Smart Search failed: %s", e)

    return start_offset


def _extract_segment(clean_wav, start_offset, chunk_duration):
    """Extract and normalize a segment using FFmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False,
                                     dir=config.get_temp_dir()) as segment_tmp:
        segment_path = segment_tmp.name

    cmd = [
        "ffmpeg", "-threads", "1", "-y",
        "-ss", str(start_offset), "-t", str(chunk_duration),
        "-i", clean_wav
    ] + utils.STANDARD_AUDIO_FLAGS + [
        "-af", utils.STANDARD_NORMALIZATION_FILTERS, segment_path]

    subprocess.run(cmd, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, check=True)
    return segment_path


def _cleanup_files(paths):
    """Safely remove list of files."""
    for path in paths:
        if path and os.path.exists(path):
            utils.secure_remove(path)


def _format_detection_result(voting_details, scans):
    """Encapsulate probability aggregation into a finalized response schema."""
    if not voting_details:
        return {
            "detected_language": "en",
            "language": "en",
            "language_code": "en",
            "confidence": 0.0,
            "segments_processed": scans,
            "voting_details": {}
        }

    best_lang = max(voting_details, key=voting_details.get)
    avg_conf = voting_details[best_lang]
    logger.debug("[LD] Final Winner: %s (Confidence: %.4f)", best_lang, avg_conf)

    # Stage 3: Confusion-Matrix Tie Breaking
    # If the gap between the top two results is < 5% and they are a known
    # linguistically similar pair, apply a bias to resolve the ambiguity.
    final_details = _resolve_linguistic_confusions(voting_details)
    best_lang = max(final_details, key=final_details.get)
    avg_conf = final_details[best_lang]

    # Filter out very low confidence entries (below 1% threshold)
    threshold = 0.01
    filtered_details = {
        k: v for k, v in final_details.items() if v >= threshold}
    sorted_details = dict(sorted(filtered_details.items(),
                          key=lambda item: item[1], reverse=True))

    return {
        "detected_language": best_lang,
        "language": best_lang,
        "language_code": best_lang,
        "confidence": float(avg_conf),
        "segments_processed": scans,
        "voting_details": sorted_details
    }


def _resolve_linguistic_confusions(probs):
    """
    Apply a resolution bias for common confusion pairs.
    If the gap is < 5%, we favor the more standardized or common variant.
    """
    if len(probs) < 2:
        return probs

    # Sort to find top two
    sorted_langs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    l1, p1 = sorted_langs[0]
    l2, p2 = sorted_langs[1]

    gap = p1 - p2
    if gap >= 0.05:
        return probs

    # Known pairs (alphabetic tuple for consistent lookup)
    pair = tuple(sorted([l1, l2]))
    bias = 0.03  # 3% bias shift

    # 1. Norwegian/Nynorsk: Prefer Standard Norwegian (no)
    if pair == ("nn", "no"):
        logger.info("[LD] Resolving NO/NN ambiguity (gap %.4f)", gap)
        probs["no"] = probs.get("no", 0) + bias
        probs["nn"] = max(0, probs.get("nn", 0) - bias)

    # 2. Serbian/Croatian/Bosnian: Often confused in media
    elif pair in [("hr", "sr"), ("bs", "hr"), ("bs", "sr")]:
        # We don't have a safe 'standard' here, but we can log it.
        # Users often expect 'sr' or 'hr' based on their locale.
        logger.debug("[LD] Linguistic cluster detected: %s/%s (gap %.4f)", l1, l2, gap)

    # 3. Indonesian/Malay
    elif pair == ("id", "ms"):
        logger.info("[LD] Resolving ID/MS ambiguity (gap %.4f)", gap)
        probs["id"] = probs.get("id", 0) + bias
        probs["ms"] = max(0, probs.get("ms", 0) - bias)

    return probs
