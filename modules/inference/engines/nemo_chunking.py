"""Cutting one decode request into the short pieces a NeMo model accepts.

The NeMo exports take at most 20-30 seconds per item, where Whisper takes a whole file and
windows it itself. The spans to decode come, in order of preference, from:

1. ``clip_timestamps`` -- the pipeline's own speech regions (speech_clips, run_decoding),
   and what resumable_decode sends to resume after a pause. Honouring them is what lets a
   NeMo engine take part in decoding by region and in pause/resume unchanged.
2. Silero VAD over the decoded audio, with the settings resumable_decode uses for the same
   job, so a first pass and a resumed one agree on where the speech is.
3. The whole file in fixed windows, when the caller turned VAD off.

Any span longer than the model takes is split at its quietest point in the back half of the
limit, so a cut lands in a pause rather than through a word whenever the speech allows.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any

import numpy as np

from modules.core import config
from modules.inference.pipeline import vad

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

#: Shorter spans are dropped. Measured on Canary-1B-v2: 50 ms of silence came back as a
#: confident full English sentence, the encoder-decoder's version of Whisper inventing speech
#: into silence. Nothing a VAD calls speech is this short; a resume trimmed to a sliver is.
MIN_CLIP_SEC = 0.1

#: Resolution of the quiet-point search when a span is split.
_FRAME_SEC = 0.1


def decode(audio: Any) -> np.ndarray:
    """16 kHz mono float32 samples from a path, or the samples a caller already decoded.

    A path is the normal case (an isolated engine only ever receives paths); samples arrive
    from the pipeline's in-process detection fallback.
    """
    if isinstance(audio, str):
        audio = vad.decode_audio(audio)
    return np.asarray(audio, dtype=np.float32)


def plan(audio: np.ndarray, *, clip_timestamps: Any, vad_filter: bool, vad_parameters: dict | None, max_sec: float) -> list:
    """The ``(start, end)`` spans, in seconds and in order, that this request decodes."""
    duration = len(audio) / SAMPLE_RATE
    spans = _requested_spans(audio, clip_timestamps, vad_filter, vad_parameters, duration)
    clamped = [(max(0.0, start), min(duration, end)) for start, end in spans]
    return [span for span in bounded(clamped, audio, max_sec) if span[1] - span[0] >= MIN_CLIP_SEC]


def _requested_spans(audio: np.ndarray, clip_timestamps: Any, vad_filter: bool, vad_parameters: dict | None, duration: float) -> list:
    """The caller's clips, else the VAD's speech, else the whole file -- see the module docstring."""
    spans = clips_from_kwargs(clip_timestamps, duration)
    if spans is not None:
        return spans
    return speech_spans(audio, vad_parameters) if vad_filter else [(0.0, duration)]


def clips_from_kwargs(clip_timestamps: Any, duration: float) -> list | None:
    """``clip_timestamps`` as ``(start, end)`` pairs, or None when there are none to honour.

    Faster-whisper's convention, which the pipeline writes to: a flat ``[s0, e0, s1, e1, ...]``
    list in seconds, where an odd length means the last clip runs to the end of the file. An
    empty list is "no clips", never "decode nothing" -- resumable_decode never sends one, and
    reading it as the latter would silently drop a whole request.
    """
    if not clip_timestamps or isinstance(clip_timestamps, str):
        return None
    flat = [float(value) for value in clip_timestamps]
    if len(flat) % 2:
        flat.append(duration)
    return list(zip(flat[0::2], flat[1::2]))


def speech_spans(audio: np.ndarray, vad_parameters: dict | None) -> list:
    """Silero's speech regions, with the request's threshold and pause length where it gave them."""
    params = vad_parameters or {}
    regions = vad.get_speech_timestamps(
        audio,
        params.get("threshold", config.VAD_THRESHOLD),
        params.get("min_silence_duration_ms", config.VAD_MIN_SILENCE_DURATION_MS),
        config.VAD_SPEECH_PAD_MS,
    )
    if not regions:
        logger.info("[NeMo] VAD found no speech in %.1fs of audio", len(audio) / SAMPLE_RATE)
    return [(float(region["start"]), float(region["end"])) for region in regions]


def bounded(spans: list, audio: np.ndarray, max_sec: float) -> list:
    """``spans`` with every one longer than ``max_sec`` split at quiet points until none is."""
    result: list = []
    for start, end in spans:
        while end - start > max_sec:
            cut = quietest_point(audio, start + max_sec / 2, start + max_sec)
            result.append((start, cut))
            start = cut
        result.append((start, end))
    return result


def quietest_point(audio: np.ndarray, earliest: float, latest: float) -> float:
    """The start of the lowest-energy frame between ``earliest`` and ``latest`` seconds.

    Falls back to ``latest`` when the window holds less than one frame, which only happens
    at the very end of the audio -- where there is nothing left to cut through.
    """
    frame = int(_FRAME_SEC * SAMPLE_RATE)
    first = int(earliest * SAMPLE_RATE)
    last = int(latest * SAMPLE_RATE)
    window = audio[first:last]
    count = len(window) // frame
    if count < 1:
        return latest
    usable = count * frame
    energy = np.square(window[:usable].astype(np.float32)).reshape(count, frame).mean(axis=1)
    return (first + int(np.argmin(energy)) * frame) / SAMPLE_RATE


def batches(items: list, size: int) -> list:
    """Consecutive groups of at most ``size``, so results come back in time order."""
    return [list(batch) for batch in itertools.batched(items, max(1, int(size)))]


def samples(audio: np.ndarray, span: tuple) -> np.ndarray:
    """The audio of one span, as the contiguous float32 array onnx-asr expects."""
    first, last = (int(seconds * SAMPLE_RATE) for seconds in span)
    return np.ascontiguousarray(audio[first:last], dtype=np.float32)
