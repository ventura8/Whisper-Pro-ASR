"""Audio standardization and raw-PCM bypass helpers.

This module centralizes FFmpeg-based normalization and the raw PCM bypass contract
so that request route helpers stay focused and under size limits.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from modules.core import utils
from modules.inference.runtime import model_manager

logger = logging.getLogger(__name__)

ErrorInfo = tuple[str, int]


def _run_convert_to_wav(
    source_path: str,
    input_flags: list[str],
    *,
    stream_index: Optional[int] = None,
    delay_filter: Optional[str] = None,
) -> tuple[Optional[str], Optional[ErrorInfo]]:
    """Run FFmpeg normalization to standardized 16kHz mono WAV."""
    try:
        clean_wav = utils.convert_to_wav(
            source_path,
            input_flags=input_flags,
            stream_index=stream_index,
            delay_filter=delay_filter,
        )
        if not clean_wav:
            return None, ("FFmpeg conversion failed - invalid media format", 400)
        return clean_wav, None
    except (RuntimeError, ValueError, OSError) as exc:
        logger.warning("[Prep] FFmpeg conversion failed for %s: %s", source_path, exc)
        return None, ("FFmpeg conversion failed - invalid media format", 400)


def _resolve_input_flags(input_flags: Optional[list[str]]) -> Optional[list[str]]:
    if input_flags is not None:
        return input_flags
    return getattr(utils.THREAD_CONTEXT, "input_flags", None)


def _positive_audio_duration(path: str) -> float | None:
    try:
        duration = float(utils.get_audio_duration(path) or 0.0)
    except (TypeError, ValueError, OSError, RuntimeError):
        return None
    if duration <= 0:
        return None
    return duration


def _warn_on_truncated_standardization(source_path: str, clean_wav: str) -> None:
    """Warn if standardized WAV duration looks suspiciously smaller."""
    source_duration = _positive_audio_duration(source_path)
    wav_duration = _positive_audio_duration(clean_wav)
    if source_duration is None or wav_duration is None:
        return
    diff = source_duration - wav_duration
    threshold = max(1.0, source_duration * 0.05)
    if diff <= threshold:
        return
    logger.warning(
        "[Prep] Standardized WAV appears truncated: source=%.2fs wav=%.2fs diff=%.2fs",
        source_duration,
        wav_duration,
        diff,
    )


def _contains_only_null_bytes(source_path: str, *, chunk_size: int = 1024 * 1024) -> bool:
    """True iff the entire file is composed only of null bytes.

    This intentionally does *not* treat a silent prefix as corruption. Silent-prefix
    PCM can begin with many 0x00 bytes but still contain non-zero samples later.
    """
    try:
        with open(source_path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                # strip removes only leading/trailing 0x00; if any non-zero exists anywhere,
                # remaining bytes will keep the result non-empty.
                if chunk.strip(b"\x00"):
                    return False
        return True
    except tuple([OSError, IOError]):
        return False


def _is_file_corrupted(source_path: str) -> bool:
    try:
        if not os.path.exists(source_path):
            return False
        if os.path.getsize(source_path) == 0:
            return False
        return _contains_only_null_bytes(source_path)
    except tuple([OSError, IOError]):
        return False


def _corrupt_file_error(source_path: str) -> tuple[None, ErrorInfo] | None:
    if _is_file_corrupted(source_path):
        return None, ("Input file is corrupted (only null bytes).", 400)
    return None


def _log_stream_alignment_probe_start(source_path: str, language: Optional[str]) -> None:
    logger.info(
        "[Prep] Server-side Bazarr replication: probing %s for audio-track selection "
        "(target language=%s) and playback-delay correction, same as Bazarr's own "
        "client-side get_audio_delay/encode_audio_stream would do before uploading.",
        source_path,
        language or "none",
    )


def _log_stream_alignment_result(stream_index: Optional[int], delay_filter: Optional[str]) -> None:
    if stream_index is not None:
        _log_stream_selection(stream_index, delay_filter)
        return
    if delay_filter is not None:
        logger.info("[Prep] Server-side Bazarr replication result: delay_filter=%s (no stream remapping)", delay_filter)
    else:
        logger.info("[Prep] Server-side Bazarr replication result: no adjustment needed (single/default track, no delay).")


def _log_stream_selection(stream_index: int, delay_filter: Optional[str]) -> None:
    logger.info(
        "[Prep] Server-side Bazarr replication result: stream_index=%s (-map 0:%s), delay_filter=%s",
        stream_index,
        stream_index,
        delay_filter or "none",
    )


def _resolve_stream_alignment(
    source_path: str,
    language: Optional[str],
    apply_stream_alignment: bool,
) -> tuple[Optional[int], Optional[str]]:
    """Replicate real Bazarr's client-side audio-track selection + delay correction,
    but only for a raw local media file WE resolve directly (local_path/video_file) --
    never for an uploaded audio_file, which Bazarr has already corrected on its end.
    See utils.get_stream_alignment_directives / utils_helpers.build_stream_alignment_directives."""
    if not apply_stream_alignment:
        return None, None
    _log_stream_alignment_probe_start(source_path, language)
    stream_index, delay_filter = utils.get_stream_alignment_directives(source_path, language)
    _log_stream_alignment_result(stream_index, delay_filter)
    return stream_index, delay_filter


def get_clean_wav_or_error(
    source_path: str,
    input_flags: Optional[list[str]] = None,
    *,
    language: Optional[str] = None,
    apply_stream_alignment: bool = False,
) -> tuple[Optional[str], Optional[ErrorInfo]]:
    """Normalize input media to 16kHz mono WAV.

    input_flags (raw-PCM / encode=false path) is still run through FFmpeg here,
    not bypassed: faster-whisper's decode_audio() opens the file via PyAV's
    generic container auto-probing (av.open(), no format hint), which cannot
    identify headerless raw PCM on its own -- only FFmpeg, given the explicit
    `-f s16le -ar 16000 -ac 1` input flags Bazarr's encode=false implies, can
    correctly interpret it. A prior version of this function returned the raw
    .raw path unmodified for this case, which is what get_audio_duration()
    (which does thread input_flags through to ffprobe) already handled
    correctly, but the ASR engine received the same unmodified path with no
    input_flags applied anywhere downstream -- silently breaking real Bazarr
    raw-PCM transcription requests. FFmpeg still does effectively minimal work
    here since the input is already at the target 16kHz/mono/s16le format.

    apply_stream_alignment=True additionally probes the source for multi-audio-track
    selection and playback-delay correction (see _resolve_stream_alignment) -- callers
    must only set this for a server-resolved local_path/video_file source, never for
    genuine uploaded audio content.
    """
    flags = _resolve_input_flags(input_flags)
    model_manager.update_task_progress(0, "Standardizing Audio")
    logger.info("[Prep] Normalizing audio stream (FFmpeg)...")
    start = time.time()

    # The flags describe THIS call's raw input and nothing after it, so they are cleared on
    # every exit -- success, corruption and conversion failure alike. Clearing only on
    # success left them on the thread context after a failure, where the next request served
    # by that pooled thread inherited "-f s16le -ar 16000 -ac 1" and reinterpreted an
    # ordinary MP4 as headerless PCM.
    try:
        corrupt = _corrupt_file_error(source_path)
        if corrupt is not None:
            return corrupt

        stream_index, delay_filter = _resolve_stream_alignment(source_path, language, apply_stream_alignment)
        clean_wav, err = _run_convert_to_wav(source_path, flags or [], stream_index=stream_index, delay_filter=delay_filter)
        if err:
            return None, err

        _warn_on_truncated_standardization(source_path, clean_wav)
        logger.info(
            "[Prep] Standardization completed in %s",
            utils.format_duration(time.time() - start),
        )
        return clean_wav, None
    finally:
        utils.THREAD_CONTEXT.input_flags = None
