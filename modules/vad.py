"""
Voice Activity Detection (VAD) via Silero (Unified C++ implementation)

This module provides high-precision speech/silence segmentation using the
Silero VAD model via the optimized faster-whisper backend.
"""
import logging
import tempfile
import subprocess
from . import config
try:
    from faster_whisper.vad import get_speech_timestamps as fw_get_ts, VadOptions as fw_Opts
    from faster_whisper.audio import decode_audio as fw_decode
except ImportError:
    fw_get_ts, fw_Opts, fw_decode = None, None, None

# Lazy-loaded modules for hardware coordination
logger = logging.getLogger(__name__)


def _lazy_import_vad():
    """Return VAD components."""
    return fw_get_ts, fw_Opts, fw_decode


def decode_audio(audio_path, start_offset=None, duration=None):
    """
    Optimized audio decoding and 16kHz mono resampling using ffmpeg.
    Supports seeking to specific offsets and limiting duration.
    """
    _, _, fw_decode_audio = _lazy_import_vad()
    if fw_decode_audio is None:
        raise ImportError("faster-whisper audio utilities not found.")

    if start_offset is None and duration is None:
        return fw_decode_audio(audio_path, sampling_rate=16000)

    # If seeking is required, we use a manual ffmpeg call via temp file

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True, dir=config.get_temp_dir()) as tmp:
        cmd = ["ffmpeg", "-threads", "1", "-y"]
        if start_offset:
            cmd += ["-ss", str(start_offset)]
        if duration:
            cmd += ["-t", str(duration)]
        cmd += ["-i", audio_path, "-f", "wav", "-ar", "16000", "-ac", "1", tmp.name]

        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return fw_decode_audio(tmp.name, sampling_rate=16000)


def get_speech_timestamps(audio, threshold=0.35,
                          min_silence_duration_ms=500, speech_pad_ms=500):
    """
    Identify regions of active speech within a numpy audio buffer (16kHz mono).
    Returns a list of dictionaries with 'start' and 'end' keys in seconds.
    """
    fw_get_vad_ts, vad_opts_class, _ = _lazy_import_vad()
    if fw_get_vad_ts is None:
        return []

    try:
        # Configure VAD options
        vad_options = vad_opts_class(
            threshold=threshold,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms
        )

        # Execute VAD inference
        # Input 'audio' must be 16kHz mono float32 numpy array
        speech_ts = fw_get_vad_ts(audio, vad_options=vad_options)

        # Convert sample counts (at 16kHz) to seconds
        return [
            {'start': round(ts['start'] / 16000, 3),
             'end': round(ts['end'] / 16000, 3)}
            for ts in speech_ts
        ]
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning("[VAD] Unified segment analysis failed: %s", e)
        return []


def get_speech_timestamps_from_path(audio_path, threshold=0.35, **kwargs):
    """ Helper to run VAD directly on a file path. """
    # kwargs can contain: min_silence_duration_ms, speech_pad_ms, start_offset, duration
    min_silence = kwargs.get('min_silence_duration_ms', 500)
    speech_pad = kwargs.get('speech_pad_ms', 500)
    start_offset = kwargs.get('start_offset')
    duration = kwargs.get('duration')

    try:
        audio = decode_audio(audio_path, start_offset=start_offset, duration=duration)
        results = get_speech_timestamps(audio, threshold, min_silence, speech_pad)
        # If we had an offset, we must shift the results back to the original timeline
        if start_offset:
            for ts in results:
                ts['start'] += start_offset
                ts['end'] += start_offset
        return results
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("[VAD] File decoding failed: %s", e)
        return []
