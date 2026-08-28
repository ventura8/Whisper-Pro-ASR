"""
Speaker Diarization and Alignment Module using WhisperX.

Known Limitation — Very Long Files (15 h+)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
WhisperX's ``load_audio()``, ``align()``, and ``DiarizationPipeline()``
load the *entire* processed audio file into RAM as float32 numpy arrays.
At 16 kHz mono the memory cost is approximately:

    duration_sec × 16 000 samples/sec × 4 bytes/sample
    ≈ 3.5 GB for 15 hours
    ≈ 5.5 GB for 24 hours

On top of the raw audio buffer, the alignment model and the diarization
pipeline hold their own state, so peak process RSS during diarization of
a 15-hour file can exceed **8–10 GB**.

If your deployment target cannot accommodate this, either:
  • Disable diarization for long files on the client side (``diarize=false``),
  • Increase the container/host memory accordingly, or
  • Set ``MAX_DIARIZE_DURATION_SEC`` to restrict diarization to shorter files.
"""

import logging
import os
from typing import Any

from modules.core import config, utils
from modules.inference import scheduler
from modules.inference.engines import whisperx_worker_client as worker

logger = logging.getLogger(__name__)

# Caching Pools
ALIGN_POOL = {}
DIARIZE_POOL = {}

# Duration threshold (seconds) above which a RAM warning is emitted before
# attempting diarization.  Set to 0 to disable the warning.  Set the env var
# ``MAX_DIARIZE_DURATION_SEC`` to a positive value to *skip* diarization
# entirely for files longer than this (returns raw segments without speakers).
_DIARIZE_WARN_THRESHOLD_SEC = 14400  # 4 hours
MAX_DIARIZE_DURATION_SEC = int(os.environ.get("MAX_DIARIZE_DURATION_SEC", 0))


def _get_whisperx_device(unit_id: str) -> str:
    """Resolve the WhisperX device (cuda or cpu) based on the unit ID."""
    unit = next((u for u in config.HARDWARE_UNITS if u["id"] == unit_id), None)
    unit_type = unit["type"] if unit else "CPU"
    return "cuda" if unit_type == "CUDA" else "cpu"


def _cached_handle_for_current_generation(pool: dict, key: Any) -> str | None:
    """Return the pool's cached handle for `key` only if it was cached against
    the worker's *current* generation. A worker crash+respawn (see
    whisperx_worker_client.generation()) invalidates every handle from the
    prior process -- its `objects` dict no longer exists -- so a stale entry
    must be treated as a miss rather than sent to the new worker to fail
    there (or worse, silently and permanently degrade diarization/alignment
    for every subsequent request, since these pools are otherwise never
    cleared except by the explicit idle-cleanup path)."""
    cached = pool.get(key)
    if cached is None:
        return None
    handle, cached_generation = cached
    if cached_generation != worker.generation():
        return None
    return handle


def _get_align_model(lang_code: str, device: str, unit_id: str) -> str:
    """Load or retrieve the alignment model handle from the cache pool."""
    align_key = (unit_id, lang_code)
    cached = _cached_handle_for_current_generation(ALIGN_POOL, align_key)
    if cached is None:
        logger.info("[Diarization] Loading alignment model for language: %s on %s", lang_code, device)
        handle, generation = worker.call_with_generation("load_align_model", lang_code=lang_code, device=device)
        ALIGN_POOL[align_key] = (handle, generation)
        return handle
    return cached


def _get_diarize_pipeline(token: str, device: str, unit_id: str) -> str:
    """Load or retrieve the diarization pipeline handle from the cache pool."""
    cached = _cached_handle_for_current_generation(DIARIZE_POOL, unit_id)
    if cached is None:
        scheduler.update_task_progress(90, "Loading Diarization Model")
        logger.info("[Diarization] Loading diarization pipeline on %s...", device)
        handle, generation = worker.call_with_generation("load_diarization_pipeline", token=token, device=device)
        DIARIZE_POOL[unit_id] = (handle, generation)
        return handle
    return cached


def _format_diarized_segments(alignment_result: dict[str, Any]) -> list[dict[str, Any]]:
    """Format diarized segments back to the standard results format."""
    results = []
    for seg in alignment_result["segments"]:
        seg_dict = {
            "start": round(seg.get("start", 0.0), 2),
            "end": round(seg.get("end", 0.0), 2),
            "text": seg.get("text", "").strip(),
            "speaker": seg.get("speaker"),
        }
        if "words" in seg:
            seg_dict["words"] = seg["words"]
        results.append(seg_dict)
    return results


def run_diarization(
    *,
    processed_path: str,
    raw_segments: list[dict[str, Any]],
    info: Any,
    language: str | None,
    min_speakers: int | None,
    max_speakers: int | None,
    hf_token: str | None,
    unit_id: str,
) -> list[dict[str, Any]]:
    """Aligns segments and performs speaker diarization using whisperx.

    .. warning::

       For very long files (15 h+) this function will consume several GB of
       RAM because WhisperX loads the full audio into memory.  See module
       docstring for details and mitigation options.
    """
    audio_duration = getattr(info, "duration", 0) or 0

    skip_result = _maybe_skip_diarization_for_duration(audio_duration, raw_segments)
    if skip_result is not None:
        return skip_result

    resolved_hf_token = _resolve_hf_token(hf_token)
    if not resolved_hf_token:
        logger.warning("[Diarization] No Hugging Face token available; returning raw segments without speaker labels.")
        return _format_raw_segments_without_speakers(raw_segments)

    return _run_align_diarize_with_fallback(
        audio_duration,
        resolved_hf_token,
        unit_id,
        info=info,
        language=language,
        processed_path=processed_path,
        raw_segments=raw_segments,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )


#: See run_diarization's except clause below for why this tuple is this broad.
_DIARIZATION_FAILURE_TYPES = (
    ImportError,
    RuntimeError,
    OSError,
    ValueError,
    AttributeError,
    KeyError,
    TypeError,
    worker.WhisperXWorkerError,
)


def _run_align_diarize_with_fallback(
    audio_duration: float,
    resolved_hf_token: str,
    unit_id: str,
    *,
    info: Any,
    language: str | None,
    processed_path: str,
    raw_segments: list[dict[str, Any]],
    min_speakers: int | None,
    max_speakers: int | None,
) -> list[dict[str, Any]]:
    """Run the align/diarize pipeline, falling back to raw (non-diarized) segments on
    any expected failure. Split out of run_diarization purely to keep that function's
    cyclomatic complexity low.

    Populated by _run_alignment_step as soon as load_audio succeeds, independent of
    whether the subsequent align call raises -- so a failure mid-alignment still
    releases the audio handle instead of leaking it in the worker's objects pool
    (a bare `alignment_result, audio_handle = _run_alignment_step(...)` unpack would
    never assign audio_handle at all if _run_alignment_step raises before returning).
    """
    # Holds a (handle, generation) tuple, not just the bare handle -- generation is
    # recorded at the moment the handle is created so _release_worker_handle can
    # tell whether the worker that owns it is still the current one (see that
    # function for why this matters).
    audio_handle_holder: list[tuple[str, int]] = []
    try:
        alignment_result = _align_and_diarize(
            audio_duration,
            resolved_hf_token,
            unit_id,
            info=info,
            language=language,
            processed_path=processed_path,
            raw_segments=raw_segments,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            audio_handle_holder=audio_handle_holder,
        )
    except _DIARIZATION_FAILURE_TYPES as exc:
        logger.warning("[Diarization] Falling back to raw segments without speaker labels: %s", exc)
        return _format_raw_segments_without_speakers(raw_segments)
    finally:
        _release_worker_handle(audio_handle_holder[0] if audio_handle_holder else None)
    return _format_diarized_segments(alignment_result)


def _align_and_diarize(
    audio_duration: float,
    resolved_hf_token: str,
    unit_id: str,
    *,
    info: Any,
    language: str | None,
    processed_path: str,
    raw_segments: list[dict[str, Any]],
    min_speakers: int | None,
    max_speakers: int | None,
    audio_handle_holder: list[tuple[str, int]],
) -> dict[str, Any]:
    """Run the alignment -> diarization -> speaker-assignment sequence. Split out of
    run_diarization purely to keep that function's cyclomatic complexity low; all
    exception handling/fallback stays in the caller."""
    _warn_if_long_diarization(audio_duration)
    whisperx_device = _get_whisperx_device(unit_id)
    alignment_result, audio_handle = _run_alignment_step(
        whisperx_device,
        unit_id,
        info=info,
        language=language,
        processed_path=processed_path,
        raw_segments=raw_segments,
        audio_handle_holder=audio_handle_holder,
    )
    diarize_pipeline = _get_diarize_pipeline(resolved_hf_token, whisperx_device, unit_id)
    diarize_handle_and_generation = _run_diarization_step(diarize_pipeline, audio_handle, min_speakers, max_speakers)
    diarize_handle, _generation = diarize_handle_and_generation
    try:
        return _assign_speakers_step(diarize_handle, alignment_result)
    finally:
        _release_worker_handle(diarize_handle_and_generation)


def _release_worker_handle(handle_and_generation: tuple[str, int] | None) -> None:
    """Release a worker-side object handle -- but only if the worker that created it
    is still the current one. worker.call() transparently spawns a fresh worker
    process if none is running, so releasing a handle from an already-dead/respawned
    worker would needlessly spin up a brand-new process just to send a "release" for
    an object that doesn't exist in its (empty) objects pool anyway -- skip it."""
    if handle_and_generation is None:
        return
    handle, recorded_generation = handle_and_generation
    if recorded_generation != worker.generation():
        return
    try:
        worker.call("release", handle=handle)
    except worker.WhisperXWorkerError:
        pass


def _maybe_skip_diarization_for_duration(audio_duration: float, raw_segments: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    if not 0 < MAX_DIARIZE_DURATION_SEC < audio_duration:
        return None
    estimated_gb = _estimate_audio_ram_gb(audio_duration)
    logger.warning(
        "[Diarization] Skipping — audio duration (%s) exceeds MAX_DIARIZE_DURATION_SEC (%ds). "
        "WhisperX alignment would require ~%.1f GB RAM. Returning raw segments without speaker labels.",
        utils.format_duration(audio_duration),
        MAX_DIARIZE_DURATION_SEC,
        estimated_gb,
    )
    return _format_raw_segments_without_speakers(raw_segments)


def _warn_if_long_diarization(audio_duration: float) -> None:
    if audio_duration <= _DIARIZE_WARN_THRESHOLD_SEC:
        return
    logger.warning(
        "[Diarization] Long file detected (%s). WhisperX will load the full audio as float32 (~%.1f GB). "
        "Ensure sufficient RAM is available.",
        utils.format_duration(audio_duration),
        _estimate_audio_ram_gb(audio_duration),
    )


def _estimate_audio_ram_gb(audio_duration: float) -> float:
    return (audio_duration * 16000 * 4) / (1024**3)


def _format_raw_segments_without_speakers(raw_segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip(),
            **({"words": seg["words"]} if "words" in seg else {}),
        }
        for seg in raw_segments
    ]


def _run_alignment_step(
    whisperx_device: str,
    unit_id: str,
    *,
    info: Any,
    language: str | None,
    processed_path: str,
    raw_segments: list[dict[str, Any]],
    audio_handle_holder: list[tuple[str, int]],
) -> tuple[dict[str, Any], str]:
    scheduler.update_task_progress(83, "Loading Alignment Model")
    align_handle = _get_align_model(info.language or language or "en", whisperx_device, unit_id)
    scheduler.update_task_progress(85, "Aligning Transcription")
    logger.info("[Diarization] Aligning segments...")
    audio_handle, audio_generation = worker.call_with_generation("load_audio", path=processed_path)
    audio_handle_holder.append((audio_handle, audio_generation))
    alignment_result = worker.call(
        "align",
        raw_segments=raw_segments,
        align_handle=align_handle,
        audio_handle=audio_handle,
        device=whisperx_device,
    )
    return alignment_result, audio_handle


def _resolve_hf_token(hf_token: str | None) -> str | None:
    token = hf_token or config.DIARIZATION_HF_TOKEN
    return token or None


def _run_diarization_step(
    diarize_pipeline_handle: str,
    audio_handle: str,
    min_speakers: int | None,
    max_speakers: int | None,
) -> tuple[str, int]:
    scheduler.update_task_progress(93, "Diarizing Speakers")
    logger.info("[Diarization] Running speaker diarization...")
    return worker.call_with_generation(
        "run_diarization",
        pipeline_handle=diarize_pipeline_handle,
        audio_handle=audio_handle,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )


def _assign_speakers_step(diarize_handle: str, alignment_result: dict[str, Any]) -> dict[str, Any]:
    scheduler.update_task_progress(97, "Assigning Speakers")
    logger.info("[Diarization] Assigning speakers to segments...")
    return worker.call("assign_word_speakers", diarize_handle=diarize_handle, alignment_result=alignment_result)
