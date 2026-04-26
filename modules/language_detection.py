"""
Language Identification Logic

This module implements a dynamic, single-chunk extraction system for language detection.
It extracts a representative audio segment (minimum 5 minutes, scaling with media length)
to provide high-confidence language identification for both short clips and long movies.
"""
# pylint: disable=no-member, protected-access, duplicate-code
import logging
import tempfile
import os
import subprocess

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
    High-level entry point for language detection using iterative scanning.
    Attempts to find a representative audio chunk with active speech.
    """
    total_duration = utils.get_audio_duration(clean_wav) or 300
    chunk_dur = min(_calculate_chunk_duration(total_duration), total_duration)

    start_offset = 0
    max_attempts = 5
    attempt = 0
    max_scan_offset = total_duration * 0.5

    all_results = []

    while attempt < max_attempts and start_offset < max_scan_offset:
        attempt += 1
        logger.info("[LD] Attempt %d/%d at %s", attempt, max_attempts,
                    utils.format_duration(start_offset))

        result, next_offset = _run_detection_attempt(
            clean_wav, start_offset, chunk_dur, model_manager
        )

        if result:
            conf = result.get('confidence', 0)
            all_results.append(result)

            # Check if we hit the high-confidence threshold to stop early
            if conf >= config.LD_MIN_CONFIDENCE_THRESHOLD:
                logger.info("[LD] High confidence (%.2f) found. Stopping early.", conf)
                return _format_detection_result(result.get('all_probabilities', {}), attempt)

            logger.info("[LD] Low confidence (%.2f). Scanning next segment...", conf)
            # Force advance to next segment if we had a low confidence result
            start_offset += chunk_dur
        else:
            # If no result (e.g. no speech), advance to the next recommended offset
            start_offset = next_offset

    # If we finished the loop, determine winner using weighted voting of all results.
    final_probs = _aggregate_probabilities(all_results)
    return _format_detection_result(final_probs, attempt)


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
        # Use confidence as weight, fallback to tiny weight if 0 to avoid total_weight=0
        weight = max(res.get('confidence', 0), 0.01)
        for lang, val in probs.items():
            aggregated[lang] = aggregated.get(lang, 0) + (val * weight)
        total_weight += weight

    return {lang: val / total_weight for lang, val in aggregated.items()}


def _run_detection_attempt(clean_wav, start_offset, chunk_duration, model_manager):
    """Helper to run a single detection attempt. Returns (result, next_offset)."""
    segment_path = None
    processed_path = None
    try:
        segment_path = _extract_segment(clean_wav, start_offset, chunk_duration)
        detection_path = segment_path

        if config.ENABLE_LD_PREPROCESSING:
            pm = preprocessing.get_manager()
            processed_path = pm.process_audio_file(segment_path)
            if processed_path and os.path.exists(processed_path):
                detection_path = processed_path

        # VAD Check
        speech_ts = vad.get_speech_timestamps_from_path(
            detection_path, threshold=config.LD_VAD_THRESHOLD)

        if not speech_ts and processed_path and processed_path != segment_path:
            logger.info("[LD] Isolated vocals silent. Retrying VAD on original segment...")
            speech_ts = vad.get_speech_timestamps_from_path(
                segment_path, threshold=config.LD_VAD_THRESHOLD)
            if speech_ts:
                detection_path = segment_path

        if not speech_ts:
            logger.info("[LD] No speech at %s. Skipping...", utils.format_duration(start_offset))
            return None, start_offset + chunk_duration

        result = model_manager.run_language_detection(detection_path)
        return result, None

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("[LD] Detection attempt failed: %s", e)
        return None, start_offset + chunk_duration
    finally:
        _cleanup_files([segment_path, processed_path])


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
            try:
                os.remove(path)
            except OSError:
                pass


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

    # Filter out very low confidence entries (below 1% threshold)
    threshold = 0.01
    filtered_details = {
        k: v for k, v in voting_details.items() if v >= threshold}
    sorted_details = dict(sorted(filtered_details.items(),
                          key=lambda item: item[1], reverse=True))

    logger.debug("[LD] Final Winner: %s (Confidence: %.4f)", best_lang, avg_conf)

    return {
        "detected_language": best_lang,
        "language": best_lang,
        "language_code": best_lang,
        "confidence": float(avg_conf),
        "segments_processed": scans,
        "voting_details": sorted_details
    }
