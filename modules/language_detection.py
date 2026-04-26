"""
Language Identification Logic

This module implements a robust, multi-zone voting system for language detection.
It strategically samples audio segments, performs one-token probability scans,
and aggregates results using squared confidence weighting to ensure high accuracy.
"""
# pylint: disable=no-member, protected-access, duplicate-code
import logging
import re
import tempfile
import os
import subprocess
import soundfile as sf  # pylint: disable=import-error
import numpy as np


from . import utils, config

logger = logging.getLogger(__name__)

# Pre-compiled regex for language token extraction
_LANG_PATTERN = re.compile(r'<\|([a-z]{2,3})\|>')


# --- [CORE PIPELINE] ---


# --- [VOTING & AGGREGATION ENGINE] ---

def _aggregate_language_probs(segment_probs_list):
    """
    Consolidate probabilities across all segments using Squared Weighting.

    Squaring the softmax probabilities punishes low-confidence noise and
    rewards clear identification 'peaks', preventing consistent false-positives
    from overriding correct detections in mixed audio.
    """
    combined_scores = {}
    valid = [cp for cp in segment_probs_list if cp]
    if not valid:
        return {}

    for cp in valid:
        for lang, prob in cp.items():
            # Apply Squared Confidence Weighting
            combined_scores[lang] = combined_scores.get(lang, 0) + (prob**2)

    # Normalize scores back to probability range
    total_score = sum(combined_scores.values())
    if total_score > 0:
        return {k: v / total_score for k, v in combined_scores.items()}
    return {}


# --- [INFERENCE LOGIC] ---


# --- [SMART SAMPLING ENGINE] ---


def _get_sampling_target(duration):
    """Heuristic logic to determine scan density based on file length."""
    if duration <= 120:
        return 3
    if duration <= 600:
        return 5
    if duration <= 3600:
        return 9
    if duration <= 10800:
        return 13
    return 15


# --- [VOTING CONCURRENCY] ---

def run_voting_detection(clean_wav, model_manager):
    """
    High-level entry point for high-performance batch language voting.

    Uses a 'Montage' strategy: concatenates all sampling targets into one file,
    performs a single UVR pass, and then runs batch inference.
    """
    duration = utils.get_audio_duration(clean_wav) or 300
    scans = _get_sampling_target(duration)
    offsets = _generate_sampling_tasks(clean_wav, duration, scans)

    return _execute_batch_scan(clean_wav, offsets, model_manager, scans)


def _execute_batch_scan(clean_wav, offsets, model_manager, scans):
    """Internal orchestrator for the montage/UVR/inference pipeline."""
    montage_path = None
    isolated_path = None
    try:
        with model_manager.model_lock_ctx():
            montage_path = _prepare_montage(clean_wav, offsets)

            # Phase 2: Batch Isolation (UVR)
            # We do this ONCE for the whole montage
            logger.info("[LD] Running batch isolation on montage...")
            isolated_path = model_manager.run_vocal_isolation(montage_path, force=True)

            # Phase 3: Batch Inference
            logger.info("[LD] Running batch inference...")
            results = model_manager.run_batch_language_detection(isolated_path, scans)

            probs = [r['all_probabilities']
                     for r in results if r and 'all_probabilities' in r]
            voting_details = _aggregate_language_probs(probs)

            if not voting_details:
                # Absolute fallback to raw file detection
                return model_manager.run_language_detection(clean_wav)

            res = _format_detection_result(voting_details, scans)
            model_manager.update_task_result(res)
            return res

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("[LD] Batch consensus scan failed: %s", e)
        # Final fallback
        return model_manager.run_language_detection(clean_wav)
    finally:
        _cleanup_batch_assets(montage_path, isolated_path)


def _prepare_montage(clean_wav, offsets):
    """Extract and concatenate audio slices into a single montage file."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False,
                                     dir=config.get_temp_dir()) as montage_tmp:
        montage_path = montage_tmp.name

    # Phase 1: Montage Extraction (Concatenate all slices)
    # Using FFmpeg filter_complex for a single-pass extraction
    inputs = []
    filter_complex = ""
    for i, offset in enumerate(offsets):
        inputs.extend(["-ss", str(offset), "-t", "30", "-i", clean_wav])
        # Resample and pad each slice to ensure consistent parameters for the concat filter
        filter_complex += (f"[{i}:a]aresample=16000,aformat=sample_fmts=s16"
                           f":channel_layouts=mono,apad=whole_dur=30[a{i}];")

    for i in range(len(offsets)):
        filter_complex += f"[a{i}]"
    filter_complex += f"concat=n={len(offsets)}:v=0:a=1,{utils.STANDARD_NORMALIZATION_FILTERS}[out]"

    cmd = [
        "ffmpeg", "-y",
        "-threads", str(config.FFMPEG_THREADS),
        "-loglevel", "error"
    ] + inputs + [
        "-filter_complex", filter_complex,
        "-map", "[out]"
    ] + utils.STANDARD_AUDIO_FLAGS + [
        montage_path
    ]

    logger.info("[LD] Extracting montage (%d samples)...", len(offsets))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        logger.error("[LD] FFmpeg montage failed (code %d): %s",
                     result.returncode, result.stderr)
        raise RuntimeError(f"FFmpeg montage extraction failed with code {result.returncode}")

    return montage_path


def _cleanup_batch_assets(montage_path, isolated_path):
    """Ensure temporary montage files are purged."""
    if montage_path and os.path.exists(montage_path):
        os.remove(montage_path)
    if isolated_path and isolated_path != montage_path and os.path.exists(isolated_path):
        os.remove(isolated_path)


def _generate_sampling_tasks(clean_wav, duration, scans):
    """Calculate seek offsets for language identification probes."""
    zone_size = max(0, duration - 30) / scans
    tasks = []
    if config.SMART_SAMPLING_SEARCH:
        for i in range(scans):
            base_offset = i * zone_size
            tasks.append(_find_best_offset_in_zone(
                clean_wav, base_offset, zone_size, duration))
    else:
        offsets = np.linspace(0, max(0, duration - 31), scans)
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

    logger.debug("[LD] Final Winner: %s (Weight: %.4f)", best_lang, avg_conf)

    return {
        "detected_language": best_lang,
        "language": best_lang,
        "language_code": best_lang,
        "confidence": avg_conf,
        "segments_processed": scans,
        "voting_details": sorted_details
    }


def _find_best_offset_in_zone(audio_path, base_offset, zone_size, total_duration):
    """Perform localized RMS scan to find visible speech segments."""
    try:
        info = sf.info(audio_path)
        file_sr = info.samplerate

        for retry in range(3):
            offset = base_offset + (retry * 10)
            if offset + 30 > total_duration:
                offset = max(0, total_duration - 30)

            # Optimization: Pick center offset if not a WAV to avoid expensive probing
            # libsndfile (sf.read) doesn't support video containers.
            if not audio_path.lower().endswith('.wav'):
                return offset

            # Targeted read
            audio, _ = sf.read(audio_path, start=int(offset * file_sr),
                               frames=int(30 * file_sr), dtype='float32')

            if audio.ndim == 2:
                audio = audio.mean(axis=1)

            # Signal Strength Check
            if np.sqrt(np.mean(audio**2)) >= 0.005:
                return offset

        return base_offset + (zone_size / 2)

    except Exception:  # pylint: disable=broad-exception-caught
        return base_offset + (zone_size / 2)
