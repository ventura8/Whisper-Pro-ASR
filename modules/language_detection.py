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

from . import utils, config, preprocessing

logger = logging.getLogger(__name__)


def run_voting_detection(clean_wav, model_manager):
    """
    High-level entry point for language detection using a single dynamic-sized chunk.
    Accounts for long media like 4h movies with a minimum 5-minute sample.
    """
    duration = utils.get_audio_duration(clean_wav) or 300

    # Dynamic size logic: 5% of duration, minimum 5 minutes (300s)
    # For a 4h movie (14400s), this is 720s (12 minutes).
    chunk_duration = max(300, duration * 0.05)

    # Cap and bound checks (don't exceed total duration)
    chunk_duration = min(chunk_duration, duration)

    # Start detection from the beginning of the media (no offset)
    start_offset = 0

    logger.info("[LD] Extracting single dynamic chunk: Offset %s, Size %s (Total: %s)",
                utils.format_duration(start_offset),
                utils.format_duration(chunk_duration),
                utils.format_duration(duration))

    processed_path = None
    segment_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False,
                                         dir=config.get_temp_dir()) as segment_tmp:
            segment_path = segment_tmp.name

        # Fast segment extraction & normalization via FFmpeg
        cmd = [
            "ffmpeg", "-threads", "1", "-y",
            "-ss", str(start_offset), "-t", str(chunk_duration),
            "-i", clean_wav
        ] + utils.STANDARD_AUDIO_FLAGS + [
            "-af", utils.STANDARD_NORMALIZATION_FILTERS, segment_path]

        subprocess.run(cmd, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=True)

        detection_path = segment_path
        if config.ENABLE_LD_PREPROCESSING:
            pm = preprocessing.get_manager()
            processed_path = pm.process_audio_file(segment_path)
            if processed_path and os.path.exists(processed_path):
                detection_path = processed_path

        # Perform single inference pass on the extracted chunk
        result = model_manager.run_language_detection(detection_path)

        # Format result to match expected schema
        # We pass 1 as 'segments_processed' to reflect the single chunk approach
        return _format_detection_result(result.get('all_probabilities', {}), 1)

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("[LD] Dynamic chunk detection failed: %s", e)
        # Fallback to direct detection on the original file if extraction fails
        return model_manager.run_language_detection(clean_wav)
    finally:
        # Cleanup temporary files
        if segment_path and os.path.exists(segment_path):
            try:
                os.remove(segment_path)
            except Exception:  # pylint: disable=broad-exception-caught
                pass
        if processed_path and processed_path != segment_path and os.path.exists(processed_path):
            try:
                os.remove(processed_path)
            except Exception:  # pylint: disable=broad-exception-caught
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
