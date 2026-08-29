"""
Voice Activity Detection (VAD) via Silero (Unified C++ implementation).

This module provides high-precision speech/silence segmentation using the
Silero VAD model via the optimized faster-whisper backend.
"""

import logging
import os
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from types import ModuleType
from typing import Any

from modules.core import config, process_exec, utils

try:
    import faster_whisper.audio as fw_audio
    import faster_whisper.vad as fw_vad

    FwOpts = fw_vad.VadOptions
    fw_get_ts = fw_vad.get_speech_timestamps
    fw_decode = fw_audio.decode_audio
except ImportError:
    FwOpts, fw_get_ts, fw_decode = None, None, None

# Lazy-loaded modules for hardware coordination
logger = logging.getLogger(__name__)


_VAD_STATE = {"wrapped": False, "wrapped_func": None}

AudioBuffer = Sequence[float]
VadTimestamp = Mapping[str, float | int]
VadTimestampList = list[VadTimestamp] | tuple[VadTimestamp, ...]
VadCallable = Callable[..., VadTimestampList]
DecodeAudioCallable = Callable[..., AudioBuffer]


def reset_vad_state():
    """Reset VAD module state. Intended for use in tests."""
    _VAD_STATE["wrapped"] = False
    _VAD_STATE["wrapped_func"] = None


def lazy_import_vad():
    """Return VAD components and apply monkeypatching for metrics logging."""
    module_obj = sys.modules[__name__]
    if module_obj.fw_get_ts is not None and not _VAD_STATE["wrapped"]:
        wrapped = _build_wrapped_get_speech_timestamps(module_obj.fw_get_ts)
        module_obj.fw_get_ts = wrapped
        _VAD_STATE["wrapped_func"] = wrapped
        _VAD_STATE["wrapped"] = True

    # Always ensure sys.modules is patched if the module is loaded
    _patch_loaded_faster_whisper_modules(_VAD_STATE["wrapped_func"])

    return module_obj.fw_get_ts, FwOpts, fw_decode


def _build_wrapped_get_speech_timestamps(orig_get_ts: VadCallable) -> VadCallable:
    def get_speech_timestamps_wrapped(audio: AudioBuffer, *args: Any, **kwargs: Any) -> VadTimestampList:
        res = orig_get_ts(audio, *args, **kwargs)
        _log_vad_statistics_safe(audio, res)
        return res

    return get_speech_timestamps_wrapped


def _log_vad_statistics_safe(audio: AudioBuffer, res: Any) -> None:
    try:
        if isinstance(res, (list, tuple)):
            _log_vad_statistics(audio, res)
    except tuple([Exception]) as e:
        logger.debug("[VAD] Failed to log VAD statistics: %s", e)


def _log_vad_statistics(audio: AudioBuffer, res: VadTimestampList) -> None:
    total_sec = len(audio) / 16000.0
    speech_sec = sum((ts["end"] - ts["start"]) / 16000.0 for ts in res)
    removed_sec = total_sec - speech_sec
    removed_pct = (removed_sec / total_sec) * 100 if total_sec > 0 else 0.0
    logger.info(
        "[VAD] Speech detection complete: processed %s of audio | VAD removed %s (%.1f%% silence)",
        utils.format_duration(total_sec),
        utils.format_duration(removed_sec),
        removed_pct,
    )


def _patch_loaded_faster_whisper_modules(wrapped_func: VadCallable | None) -> None:
    if wrapped_func is None:
        return
    try:
        for name, module in list(sys.modules.items()):
            if _is_patchable_faster_whisper_module(name, module):
                setattr(module, "get_speech_timestamps", wrapped_func)
    except (RuntimeError, KeyError, AttributeError, TypeError):
        pass


def _is_patchable_faster_whisper_module(name: str, module: ModuleType | object) -> bool:
    return bool(name.startswith("faster_whisper") and module and hasattr(module, "get_speech_timestamps"))


def decode_audio(audio_path: str, start_offset: float | None = None, duration: float | None = None) -> AudioBuffer:
    """
    Optimized audio decoding and 16kHz mono resampling using ffmpeg.
    Supports seeking to specific offsets and limiting duration.
    """
    _, _, fw_decode_audio = lazy_import_vad()
    if fw_decode_audio is None:
        raise ImportError("faster-whisper audio utilities not found.")

    if _should_decode_directly(start_offset, duration):
        return fw_decode_audio(audio_path, sampling_rate=16000)
    return _decode_audio_slice_with_ffmpeg(fw_decode_audio, audio_path, start_offset, duration)


def extract_slice_to_file(audio_path: str, start_offset: float, duration: float) -> str:
    """Extract [start_offset, start_offset+duration) into a new temp WAV file, returning its path.

    Unlike decode_audio (which extracts to a temp file internally and immediately deletes
    it after decoding to a numpy array), the caller here needs the path to persist:
    IsolatedEngine runs in a worker subprocess and can only be handed a path, never
    decoded samples -- passing an array raises TypeError there. The caller owns the
    returned file and must remove it.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=config.get_temp_dir()) as tmp:
        temp_path = tmp.name
    try:
        cmd = _build_ffmpeg_decode_cmd(audio_path, temp_path, start_offset=start_offset, duration=duration)
        process_exec.run_capture(cmd, check=True)
    except BaseException:
        # The caller owns the file only on success; a failed decode must not leave one
        # behind, because nothing downstream ever learns the path to clean it up.
        with suppress(OSError):
            os.remove(temp_path)
        raise
    return temp_path


def _should_decode_directly(start_offset: float | None = None, duration: float | None = None) -> bool:
    return start_offset is None and duration is None


def _decode_audio_slice_with_ffmpeg(
    fw_decode_audio: DecodeAudioCallable,
    audio_path: str,
    start_offset: float | None = None,
    duration: float | None = None,
) -> AudioBuffer:
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=config.get_temp_dir()) as tmp:
            temp_path = tmp.name
        cmd = _build_ffmpeg_decode_cmd(audio_path, temp_path, start_offset=start_offset, duration=duration)
        process_exec.run_capture(cmd, check=True)
        return fw_decode_audio(temp_path, sampling_rate=16000)
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                logger.debug("[VAD] Failed to remove temporary decode file: %s", temp_path)


def _build_ffmpeg_decode_cmd(
    audio_path: str,
    output_path: str,
    start_offset: float | None = None,
    duration: float | None = None,
) -> list[str]:
    ffmpeg_threads = str(max(1, int(getattr(config, "FFMPEG_THREADS", 1))))
    cmd = ["ffmpeg", "-threads", ffmpeg_threads, "-y"]
    if start_offset:
        cmd += ["-ss", str(start_offset)]
    if duration:
        cmd += ["-t", str(duration)]
    cmd += ["-i", audio_path, *utils.STANDARD_AUDIO_FLAGS, "-f", "wav", output_path]
    return cmd


def get_speech_timestamps(audio, threshold=0.35, min_silence_duration_ms=500, speech_pad_ms=500):
    """
    Identify regions of active speech within a numpy audio buffer (16kHz mono).
    Returns a list of dictionaries with 'start' and 'end' keys in seconds.
    """
    fw_get_vad_ts, vad_opts_class, _ = lazy_import_vad()
    if fw_get_vad_ts is None:
        return []

    try:
        # Configure VAD options
        vad_options = vad_opts_class(threshold=threshold, min_silence_duration_ms=min_silence_duration_ms, speech_pad_ms=speech_pad_ms)

        # Execute VAD inference
        # Input 'audio' must be 16kHz mono float32 numpy array
        speech_ts = fw_get_vad_ts(audio, vad_options=vad_options)

        # Convert sample counts (at 16kHz) to seconds
        return [{"start": round(ts["start"] / 16000, 3), "end": round(ts["end"] / 16000, 3)} for ts in speech_ts]
    except tuple([Exception]) as e:
        logger.warning("[VAD] Unified segment analysis failed: %s", e)
        return []


def get_speech_timestamps_from_path(audio_path, threshold=0.35, **kwargs):
    """Helper to run VAD directly on a file path."""
    # kwargs can contain: min_silence_duration_ms, speech_pad_ms, start_offset, duration
    min_silence = kwargs.get("min_silence_duration_ms", 500)
    speech_pad = kwargs.get("speech_pad_ms", 500)
    start_offset = kwargs.get("start_offset")
    duration = kwargs.get("duration")

    try:
        audio = decode_audio(audio_path, start_offset=start_offset, duration=duration)
        results = get_speech_timestamps(audio, threshold, min_silence, speech_pad)
        # If we had an offset, we must shift the results back to the original timeline
        if start_offset:
            for ts in results:
                ts["start"] += start_offset
                ts["end"] += start_offset
        return results
    except (ImportError, RuntimeError, process_exec.CommandExecutionError, process_exec.CommandTimeoutError, OSError, ValueError) as e:
        logger.error("[VAD] File decoding failed: %s", e)
        return []
