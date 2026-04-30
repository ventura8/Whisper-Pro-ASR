"""
Language Identification Logic

This module implements a robust, multi-zone voting system for language detection.
It strategically samples audio segments, performs one-token probability scans,
and aggregates results using squared confidence weighting to ensure high accuracy.
"""
import importlib
import logging
import re
import tempfile
import os
import subprocess
import time
import numpy as np
from modules.inference import scheduler
from modules import config
from modules import utils

# Lazy load containers
_LIBS = {"sf": None}


def _get_sf():
    """Retrieve the soundfile module dynamically."""
    if _LIBS["sf"] is None:
        _LIBS["sf"] = importlib.import_module("soundfile")
    return _LIBS["sf"]


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
        return 1
    if duration <= 600:
        return 3
    if duration <= 1200:
        return 5
    if duration <= 3600:
        return 9
    if duration <= 10800:
        return 13
    return 15


# --- [VOTING CONCURRENCY] ---

def run_voting_detection(audio_path, model_manager, start_time=None):
    """
    High-level entry point for high-performance batch language voting.

    Uses a 'Montage' strategy: concatenates all sampling targets into one file,
    performs a single UVR pass, and then runs batch inference.
    """
    duration = utils.get_audio_duration(audio_path) or 300
    scans = _get_sampling_target(duration)
    logger.info("[LD] Target: %s | Duration: %s | Density: %d segments",
                os.path.basename(audio_path), utils.format_duration(duration), scans)
    offsets = _generate_sampling_tasks(audio_path, duration, scans)

    return _execute_batch_scan(audio_path, offsets, model_manager, scans, start_time)


def _execute_batch_scan(audio_path, offsets, model_manager, scans, start_time=None):
    """Internal orchestrator for the montage/UVR/inference pipeline."""
    perf = {'start_queue': start_time or time.time()}
    montage_path = None
    isolated_path = None
    try:
        with model_manager.model_lock_ctx() as (model, unit_id):
            perf['dur_queue'] = time.time() - perf['start_queue']

            # Phase 1: Montage
            montage_path = _step_create_montage(audio_path, offsets, scans, perf)

            # Phase 2: Isolation
            isolated_path = _step_isolate_vocals(montage_path, model_manager, unit_id, perf)

            # Phase 3: Inference
            res = _step_run_inference((model, model_manager), isolated_path, scans,
                                      audio_path, perf)

            return res

    except (ValueError, RuntimeError, IOError) as e:
        logger.error("[LD] Batch consensus scan failed: %s", e)
        return model_manager.run_language_detection(audio_path)
    finally:
        _cleanup_batch_assets(montage_path, isolated_path)


def _step_create_montage(audio_path, offsets, _scans, perf):
    perf['start_montage'] = time.time()
    scheduler.update_task_progress(10, "Montage")
    path = _prepare_montage(audio_path, offsets)
    perf['dur_montage'] = time.time() - perf['start_montage']
    return path


def _step_isolate_vocals(montage_path, model_manager, unit_id, perf):
    if not config.ENABLE_LD_PREPROCESSING:
        perf['dur_iso'] = 0.0
        return montage_path

    perf['start_iso'] = time.time()
    scheduler.update_task_progress(20, "Vocal Isolation")
    path = model_manager.run_vocal_isolation_direct(montage_path, unit_id, force=True)
    perf['dur_iso'] = time.time() - perf['start_iso']
    return path


def _step_run_inference(model_context, isolated_path, scans, audio_path, perf):
    model, model_manager = model_context
    perf['start_inf'] = time.time()
    scheduler.update_task_progress(60, "Inference")
    results = model_manager.run_batch_language_detection_direct(model, isolated_path, scans)
    perf['dur_inf'] = time.time() - perf['start_inf']

    probs = [r['all_probabilities'] for r in results if r and 'all_probabilities' in r]
    voting_details = _aggregate_language_probs(probs)

    if not voting_details:
        return model_manager.run_language_detection(audio_path)

    res = _format_detection_result(voting_details, scans)
    res['performance'] = {
        "queue_sec": round(perf['dur_queue'], 2),
        "montage_sec": round(perf['dur_montage'], 2),
        "isolation_sec": round(perf['dur_iso'], 2),
        "inference_sec": round(perf['dur_inf'], 2)
    }
    model_manager.update_task_metadata(result=res)
    return res


def _prepare_montage(source_path, offsets):
    """Extract and concatenate audio slices into a single montage file."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False,
                                     dir=config.get_temp_dir()) as montage_tmp:
        montage_path = montage_tmp.name

    # Phase 1: Montage Extraction (Concatenate all slices)
    # Using FFmpeg filter_complex for a single-pass extraction
    inputs = []
    filter_complex = ""
    for i, offset in enumerate(offsets):
        # Extract 30s audio directly from source (video or audio)
        inputs.extend(["-ss", str(offset), "-t", "30", "-i", source_path])
        # Resample to 16kHz Stereo for optimal iGPU bandwidth and MDX-NET compatibility
        filter_complex += (f"[{i}:a]aresample=16000,aformat=sample_fmts=s16"
                           f":channel_layouts=stereo,apad=whole_dur=30[a{i}];")

    for i in range(len(offsets)):
        filter_complex += f"[a{i}]"
    filter_complex += f"concat=n={len(offsets)}:v=0:a=1,{utils.STANDARD_NORMALIZATION_FILTERS}[out]"

    cmd = [
        "ffmpeg", "-y",
        "-threads", str(config.FFMPEG_THREADS),
        "-loglevel", "error"
    ] + inputs + [
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "2",
        montage_path
    ]

    logger.info("[LD] Extracting montage (%d samples)...", len(offsets))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        logger.error("[LD] FFmpeg montage failed (code %d): %s",
                     result.returncode, result.stderr)
        raise RuntimeError(f"FFmpeg montage extraction failed with code {result.returncode}")

    return utils.track_file(montage_path)


def _cleanup_batch_assets(montage_path, isolated_path):
    """Ensure temporary montage files are purged."""
    if montage_path and os.path.exists(montage_path):
        os.remove(montage_path)
    if isolated_path and isolated_path != montage_path and os.path.exists(isolated_path):
        os.remove(isolated_path)


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
            tasks.append(_find_best_offset_in_zone(
                audio_path, base_offset, zone_size, duration))
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
        current_sf = _get_sf()
        info = current_sf.info(audio_path)
        file_sr = info.samplerate

        for retry in range(3):
            offset = base_offset + (retry * 10)
            if offset + 30 > total_duration:
                offset = max(0, total_duration - 30)

            # Optimization: Pick center offset if not a WAV to avoid expensive probing
            if not audio_path.lower().endswith('.wav'):
                return offset

            # Targeted read
            audio, _ = current_sf.read(audio_path, start=int(offset * file_sr),
                                       frames=int(30 * file_sr), dtype='float32')

            if audio.ndim == 2:
                audio = audio.mean(axis=1)

            # Signal Strength Check
            if np.sqrt(np.mean(audio**2)) >= 0.005:
                return offset

        return base_offset + (zone_size / 2)

    except Exception:  # pylint: disable=broad-exception-caught
        return base_offset + (zone_size / 2)
