"""
Speaker Diarization and Alignment Module using WhisperX.
"""
import logging
from modules import config
from modules.inference import scheduler

logger = logging.getLogger(__name__)

# Caching Pools
_ALIGN_POOL = {}
_DIARIZE_POOL = {}


def _get_whisperx_device(unit_id):
    """Resolve the WhisperX device (cuda or cpu) based on the unit ID."""
    unit = next((u for u in config.HARDWARE_UNITS if u['id'] == unit_id), None)
    unit_type = unit['type'] if unit else 'CPU'
    return 'cuda' if unit_type == 'CUDA' else 'cpu'


def _get_align_model(whisperx, lang_code, device, unit_id):
    """Load or retrieve the alignment model from the cache pool."""
    align_key = (unit_id, lang_code)
    if align_key not in _ALIGN_POOL:
        logger.info("[Diarization] Loading alignment model for language: %s on %s", lang_code, device)
        _ALIGN_POOL[align_key] = whisperx.load_align_model(
            language_code=lang_code,
            device=device
        )
    return _ALIGN_POOL[align_key]


def _get_diarize_pipeline(whisperx, token, device, unit_id):
    """Load or retrieve the diarization pipeline from the cache pool."""
    if unit_id not in _DIARIZE_POOL:
        scheduler.update_task_progress(90, "Loading Diarization Model")
        logger.info("[Diarization] Loading diarization pipeline on %s...", device)
        _DIARIZE_POOL[unit_id] = whisperx.diarization.DiarizationPipeline(  # pylint: disable=no-member
            use_auth_token=token,
            device=device
        )
    return _DIARIZE_POOL[unit_id]


def run_diarization(
    *, processed_path, raw_segments, info, language, min_speakers, max_speakers, hf_token, unit_id
):
    """Aligns segments and performs speaker diarization using whisperx."""
    # Resolve device and import whisperx
    whisperx_device = _get_whisperx_device(unit_id)
    import whisperx  # pylint: disable=import-outside-toplevel,import-error

    # 1. Get/load alignment model
    scheduler.update_task_progress(83, "Loading Alignment Model")
    model_a, metadata = _get_align_model(whisperx, info.language or language or "en", whisperx_device, unit_id)

    # 2. Align segments
    scheduler.update_task_progress(85, "Aligning Transcription")
    logger.info("[Diarization] Aligning segments...")
    audio = whisperx.load_audio(processed_path)
    alignment_result = whisperx.align(
        raw_segments,
        model_a,
        metadata,
        audio,
        device=whisperx_device,
        return_char_alignments=False
    )

    # 3. Diarization pipeline
    token = hf_token or config.HF_TOKEN
    if not token:
        raise ValueError(
            "HF_TOKEN is required for speaker diarization. "
            "Please set HF_TOKEN environment variable or pass hf_token parameter."
        )

    diarize_pipeline = _get_diarize_pipeline(whisperx, token, whisperx_device, unit_id)

    # Run diarization
    scheduler.update_task_progress(93, "Diarizing Speakers")
    logger.info("[Diarization] Running speaker diarization...")
    diarize_segments = diarize_pipeline(
        audio,
        min_speakers=min_speakers,
        max_speakers=max_speakers
    )

    # 4. Assign speakers
    scheduler.update_task_progress(97, "Assigning Speakers")
    logger.info("[Diarization] Assigning speakers to segments...")
    alignment_result = whisperx.assign_word_speakers(diarize_segments, alignment_result)

    # 5. Format back to results
    results = []
    for seg in alignment_result["segments"]:
        results.append({
            "start": round(seg.get("start", 0.0), 2),
            "end": round(seg.get("end", 0.0), 2),
            "text": seg.get("text", "").strip(),
            "speaker": seg.get("speaker")
        })
    return results
