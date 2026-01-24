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
from concurrent.futures import ThreadPoolExecutor
import soundfile as sf  # pylint: disable=import-error
import numpy as np
import torch  # pylint: disable=import-error

from . import utils, config, preprocessing

logger = logging.getLogger(__name__)

# Pre-compiled regex for language token extraction
_LANG_PATTERN = re.compile(r'<\|([a-z]{2,3})\|>')


# --- [CORE PIPELINE] ---

def run_detection_pipeline(model, processor, audio_path, yield_cb=None):
    """
    Execute high-fidelity language detection via zone-based sampling.

    This function analyzes multiple strategic audio zones and aggregates 
    one-token probabilities to produce a high-confidence language ID.
    """
    try:
        chunks = _collect_active_speech_chunks(audio_path, yield_cb=yield_cb)
    except Exception as err:  # pylint: disable=broad-exception-caught
        # Fallback to naive fixed-offset sampling if smart sampling fails (e.g. metadata issues)
        is_val_or_run = isinstance(err, (ValueError, RuntimeError))
        is_sf_err = type(err).__name__ == "LibsndfileError"

        if is_val_or_run or is_sf_err:
            logger.error(
                "[LD] Smart sampling failed - falling back to naive profile: %s", err)
            chunks = _sample_audio_chunks(audio_path)
        else:
            raise err

    if not chunks:
        logger.warning(
            "[LD] No valid speech identified in samples. Defaulting to English.")
        return _empty_result()

    logger.info(
        "[LD] Analyzing %d high-RMS speech segments for identification.", len(chunks))

    # Phase 1: Probability Extraction
    chunk_probs_list = _detect_languages_from_chunks(model, processor, chunks)

    # Phase 2: Aggregation (Squared Weighting)
    avg_probs = _aggregate_language_probs(chunk_probs_list)
    winner, confidence = _find_language_winner(avg_probs)

    # Logging & Diagnostics
    top_5 = sorted(avg_probs.items(),
                   key=lambda item: item[1], reverse=True)[:5]
    diag = ", ".join([f"{k}:{v*100:.1f}%" for k, v in top_5])
    logger.info("[LD] Probability Consensus: %s", diag)

    return {
        "detected_language": winner,
        "language": winner,
        "language_code": winner,
        "confidence": float(confidence),
        "all_probabilities": {k: float(v) for k, v in top_5},
        "chunks_processed": len(chunks)
    }


def _empty_result():
    """Return a default English result for cases with no valid speech."""
    return {
        "detected_language": "en",
        "language": "en",
        "language_code": "en",
        "confidence": 0.0,
        "all_probabilities": {"en": 1.0},
        "chunks_processed": 0
    }


# --- [VOTING & AGGREGATION ENGINE] ---

def _aggregate_language_probs(chunk_probs_list):
    """
    Consolidate probabilities across all chunks using Squared Weighting.

    Squaring the softmax probabilities punishes low-confidence noise and 
    rewards clear identification 'peaks', preventing consistent false-positives
    from overriding correct detections in mixed audio.
    """
    combined_scores = {}
    valid = [cp for cp in chunk_probs_list if cp]
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


def _find_language_winner(avg_probs):
    """Identify the majority candidate from the aggregated probability map."""
    if not avg_probs:
        return "en", 0.0
    winner = max(avg_probs, key=avg_probs.get)
    return winner, avg_probs[winner]


# --- [INFERENCE LOGIC] ---

def _detect_languages_from_chunks(model, processor, chunks):
    """Execute prioritized generation on a set of audio buffers."""
    all_chunk_probs = []
    lang_to_id = _get_language_token_mapping(processor)

    if not lang_to_id:
        logger.warning(
            "[LD] Language tokens missing from vocabulary. Entering slow fallback...")
        return _detect_languages_fallback(model, processor, chunks)

    id_to_lang = {v: k.strip("<|>") for k, v in lang_to_id.items()}
    lang_ids = list(lang_to_id.values())

    # Set generic prefix to ensure clean detection
    if hasattr(processor.tokenizer, 'set_prefix_tokens'):
        processor.tokenizer.set_prefix_tokens(language=None, task=None)

    total = len(chunks)
    for i, chunk in enumerate(chunks):
        pct = ((i + 1) / total) * 100
        prog_bar = '=' * int(20 * pct / 100) + '>' + '.' * \
            (20 - int(20 * pct / 100) - 1)
        logger.info("[LD Progress] %5.1f%% | Chunk %d/%d | [%s]",
                    pct, i + 1, total, prog_bar[:20])

        probs = _detect_single_chunk_probs(
            model, processor, chunk, lang_ids, id_to_lang)
        all_chunk_probs.append(probs)

    return all_chunk_probs


def _get_language_token_mapping(processor):
    """Map the tokenizer's special language tokens to their integer IDs."""
    if hasattr(processor.tokenizer, 'get_lang_to_id'):
        mapping = processor.tokenizer.get_lang_to_id()
        if mapping:
            return mapping

    # Pattern-match fallback for custom or simplified tokenizers
    mapping = {}
    for token, idx in processor.tokenizer.get_vocab().items():
        if token.startswith("<|") and token.endswith("|>"):
            inner = token[2:-2]
            if 2 <= len(inner) <= 4 and inner.isalpha() and inner.islower():
                mapping[token] = idx
    return mapping


def _detect_single_chunk_probs(model, processor, chunk, lang_ids, id_to_lang):
    """Extract softmax probabilities for the first generated token across all languages."""
    feats = processor(chunk, sampling_rate=16000,
                      return_tensors="pt").input_features
    start_token = processor.tokenizer.convert_tokens_to_ids(
        "<|startoftranscript|>")

    outputs = model.generate(
        feats,
        decoder_input_ids=torch.tensor([[start_token]], dtype=torch.long),
        max_new_tokens=1,
        num_beams=1,
        return_dict_in_generate=True,
        output_scores=True,
        forced_decoder_ids=None
    )

    logits = outputs.scores[0][0]
    vocab_size = int(logits.shape[0])
    valid_ids = [idx for idx in lang_ids if idx < vocab_size]

    if not valid_ids:
        return {"en": 0.0}

    probs = torch.softmax(logits[valid_ids], dim=-1).tolist()
    if not isinstance(probs, list):
        probs = [probs]

    return {id_to_lang[idx]: prob for idx, prob in zip(valid_ids, probs)}


def _detect_languages_fallback(model, processor, chunks):
    """Greedy decoding fallback for identification on non-standard models."""
    probs = []
    for chunk in chunks:
        ids = model.generate(
            processor(chunk, sampling_rate=16000,
                      return_tensors="pt").input_features,
            max_new_tokens=1,
            num_beams=1
        )
        decoded = processor.decode(ids[0], skip_special_tokens=False)
        match = _LANG_PATTERN.search(decoded)
        lang = match.group(1) if match else "en"
        probs.append({lang: 1.0})
    return probs


# --- [SMART SAMPLING ENGINE] ---

def _sample_audio_chunks(audio_path):
    """Fixed-offset sampling used when smart sampling fails."""
    info = sf.info(audio_path)
    duration = info.duration
    num_chunks = max(3, min(9, int(duration / 120)))
    offsets = np.linspace(0, max(0, duration - 30), num_chunks)
    return [_read_chunk(audio_path, o, 30, info.samplerate) for o in offsets]


def _get_sampling_target(duration):
    """Heuristic logic to determine scan density based on file length."""
    if duration <= 120:
        return 3
    if duration <= 600:
        return 5
    if duration <= 3600:
        return 12
    if duration <= 10800:
        return 20
    return 25


def _collect_active_speech_chunks(audio_path, yield_cb=None):
    """Collect representative audio samples using high-RMS speech targeting."""
    pm = preprocessing.get_manager() if config.ENABLE_LD_PREPROCESSING else None

    info = sf.info(audio_path)
    duration = info.duration
    target = _get_sampling_target(duration)

    logger.info("[LD] Initializing smart sampling: Targeting %d segments for %s media.",
                target, utils.format_duration(duration))

    zone_size = max(0, duration - 30) / target
    valid_chunks = []

    for i in range(target):
        start = i * zone_size
        chunk = _search_or_fallback_zone(
            audio_path, pm, yield_cb, (start, zone_size), (i + 1, target, info)
        )
        valid_chunks.append(chunk)

    return valid_chunks


def _search_or_fallback_zone(audio_path, pm, yield_cb, zone_bounds, meta):
    """Identify speech within a specific zone or fallback to the geographic center."""
    start, zone_size = zone_bounds
    curr_idx, total_target, info = meta

    if config.SMART_SAMPLING_SEARCH:
        # Search for high-RMS window
        for retry in range(3):
            offset = start + (retry * 10)
            if offset + 30 > info.duration:
                break
            raw = _read_chunk(audio_path, offset, 30, info.samplerate)

            # Diagnostic RMS check
            if np.sqrt(np.mean(raw**2)) >= 0.005:
                return pm.process_audio_chunk(raw, yield_cb=yield_cb) if pm else raw

    # Default to center of the assigned zone
    offset = start + (zone_size / 2)
    logger.info("[LD] Processing Sample %d/%d (%s -> %s)...",
                curr_idx, total_target, utils.format_duration(offset),
                utils.format_duration(min(offset + 30, info.duration)))
    raw = _read_chunk(audio_path, offset, 30, info.samplerate)
    return pm.process_audio_chunk(raw, yield_cb=yield_cb) if pm else raw


def _read_chunk(audio_path, start_sec, duration_sec, file_sr):
    """Read a specific slice of audio and normalize to 16kHz mono."""
    audio, sr = sf.read(audio_path, start=int(start_sec * file_sr),
                        frames=int(duration_sec * file_sr), dtype='float32')

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if sr != 16000:
        indices = np.linspace(
            0, len(audio) - 1, int(len(audio) * (16000 / sr))).astype(int)
        audio = audio[indices]

    # Normalize to constant 30s buffer (industry standard for Whisper encoders)
    buffer_target = 480000
    if len(audio) < buffer_target:
        padded = np.zeros(buffer_target, dtype=np.float32)
        padded[:len(audio)] = audio
        audio = padded
    return audio[:buffer_target]


# --- [VOTING CONCURRENCY] ---

def run_voting_detection(clean_wav, model_manager):
    """
    High-level entry point for parallel language voting.

    This implementation handles the orchestration of parallel segment extraction,
    optional UVR isolation, and final probability aggregation.
    """
    duration = utils.get_audio_duration(clean_wav) or 300
    scans = _get_sampling_target(duration)

    logger.info("[LD] Initializing global voting scan (%d samples)...", scans)

    # Zone calculation
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

    # Parallel segment analysis worker
    def extract_and_detect(meta):
        idx, seek_time = meta
        processed_path = None
        chunk_path = None

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as chunk_tmp:
                chunk_path = chunk_tmp.name

            logger.info("[LD] Extraction: Sample %d/%d (Seek: %s)",
                        idx + 1, scans, utils.format_duration(seek_time))

            # Fast segment extraction & normalization via FFmpeg
            # We use the original path directly to avoid full file normalization
            cmd = [
                "ffmpeg", "-threads", "1", "-y",
                "-ss", str(seek_time), "-t", "30",
                "-i", clean_wav
            ] + utils.STANDARD_AUDIO_FLAGS + [
                "-af", utils.STANDARD_NORMALIZATION_FILTERS, chunk_path]
            subprocess.run(cmd, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=True)

            detection_path = chunk_path
            if config.ENABLE_LD_PREPROCESSING:
                pm = preprocessing.get_manager()
                processed_path = pm.process_audio_file(chunk_path)
                if processed_path and os.path.exists(processed_path):
                    detection_path = processed_path

            # Sequential inference within the pool
            return model_manager.run_language_detection(detection_path)

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("[LD] Sample %d failed: %s", idx + 1, e)
            return None
        finally:
            # Persistent cleanup
            if chunk_path and os.path.exists(chunk_path):
                os.remove(chunk_path)
            if processed_path and processed_path != chunk_path and os.path.exists(processed_path):
                os.remove(processed_path)

    # Serial execution pool (respects ASR_THREADS limit by running one inference at a time)
    max_workers = 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(extract_and_detect, enumerate(tasks)))

    probs = [r['all_probabilities']
             for r in results if r and 'all_probabilities' in r]
    voting_details = _aggregate_language_probs(probs)

    if not voting_details:
        return model_manager.run_language_detection(clean_wav)

    return _format_detection_result(voting_details, scans)


def _format_detection_result(voting_details, scans):
    """Encapsulate probability aggregation into a finalized response schema."""
    if not voting_details:
        return {
            "detected_language": "en",
            "language": "en",
            "language_code": "en",
            "confidence": 0.0,
            "chunks_processed": scans,
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
        "chunks_processed": scans,
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
