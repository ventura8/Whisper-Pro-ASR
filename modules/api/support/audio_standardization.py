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


def _run_convert_to_wav(source_path: str, input_flags: list[str]) -> tuple[Optional[str], Optional[ErrorInfo]]:
    """Run FFmpeg normalization to standardized 16kHz mono WAV."""
    try:
        clean_wav = utils.convert_to_wav(source_path, input_flags=input_flags)
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


def _raw_pcm_source_or_error(
    source_path: str,
) -> tuple[Optional[str], Optional[ErrorInfo]]:
    corrupt = _corrupt_file_error(source_path)
    if corrupt is not None:
        return corrupt
    return source_path, None


def get_clean_wav_or_error(
    source_path: str,
    input_flags: Optional[list[str]] = None,
) -> tuple[Optional[str], Optional[ErrorInfo]]:
    """Normalize input media to 16kHz mono WAV.

    If input_flags is truthy (raw-PCM / encode=false path), FFmpeg normalization is
    bypassed and the caller receives the source_path directly, after a corruption
    sanity check.
    """
    flags = _resolve_input_flags(input_flags)
    model_manager.update_task_progress(0, "Standardizing Audio")

    if flags:
        logger.info("[Prep] Skipping FFmpeg normalization (raw PCM / encode=false).")
        return _raw_pcm_source_or_error(source_path)

    logger.info("[Prep] Normalizing audio stream (FFmpeg)...")
    start = time.time()

    corrupt = _corrupt_file_error(source_path)
    if corrupt is not None:
        return corrupt

    clean_wav, err = _run_convert_to_wav(source_path, flags or [])
    if err:
        return None, err

    _warn_on_truncated_standardization(source_path, clean_wav)
    logger.info(
        "[Prep] Standardization completed in %s",
        utils.format_duration(time.time() - start),
    )
    return clean_wav, None
