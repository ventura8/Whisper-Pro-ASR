"""
Voice Activity Detection (VAD) via Silero (Unified C++ implementation)

This module provides high-precision speech/silence segmentation using the 
Silero VAD model via the optimized faster-whisper backend.
"""
# pylint: disable=broad-exception-caught, invalid-name
import logging

# Import the optimized functions from faster-whisper
try:
    from faster_whisper.vad import get_speech_timestamps as fw_get_vad_ts, VadOptions
    from faster_whisper.audio import decode_audio as fw_decode_audio
except ImportError:
    fw_get_vad_ts = None
    VadOptions = None
    fw_decode_audio = None

logger = logging.getLogger(__name__)


def decode_audio(audio_path):
    """
    Optimized audio decoding and 16kHz mono resampling using ffmpeg (via faster-whisper).
    """
    if fw_decode_audio is None:
        raise ImportError("faster-whisper audio utilities not found.")
    return fw_decode_audio(audio_path, sampling_rate=16000)


def get_speech_timestamps(audio, threshold=0.35,
                          min_silence_duration_ms=500, speech_pad_ms=500):
    """
    Identify regions of active speech within a numpy audio buffer (16kHz mono).
    Returns a list of dictionaries with 'start' and 'end' keys in seconds.
    """
    if fw_get_vad_ts is None:
        logger.error("[VAD] Unified ASR backend components missing.")
        return []

    try:
        # Configure VAD options
        vad_options = VadOptions(
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
    except Exception as e:
        logger.warning("[VAD] Unified segment analysis failed: %s", e)
        return []


def get_speech_timestamps_from_path(audio_path, threshold=0.35,
                                    min_silence_duration_ms=500, speech_pad_ms=500):
    """ Helper to run VAD directly on a file path. """
    try:
        audio = decode_audio(audio_path)
        return get_speech_timestamps(audio, threshold, min_silence_duration_ms, speech_pad_ms)
    except Exception as e:
        logger.error("[VAD] File decoding failed: %s", e)
        return []
