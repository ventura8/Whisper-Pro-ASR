"""A decode that gives the worker back when it is paused, and picks up where it stopped.

Cooperative preemption pauses a transcription so a priority task -- language detection for a
subtitle client, typically -- can borrow its hardware unit. The pause used to happen *inside*
the decode's segment stream: the consumer blocked between two segments while the worker
channel stayed open, and an isolated engine holds the channel lock for the life of a stream.
On a host with one unit and one worker, the priority task then needed that same channel for
its own detection, and waited for it forever while the paused task waited for the priority
task. Measured on the Intel NUC (2026-09-12): one detect-language request from Bazarr
during a long-form transcription wedged the service for hours -- 41 requests queued, the
health check unanswered, load average 0.2.

The channel was built for the opposite: abandoning a stream cancels the worker-side decode
and keeps the model loaded, so a pause can cost the channel nothing. This module is that
contract's consumer. A decode asks, before each segment, whether a pause is pending; when
one is, it closes the stream, waits in the ordinary preemption flow, and then issues a new
call for what is left -- the remaining clips from the end of the last consumed segment, or,
for a whole-file decode, the speech the VAD finds after that point. Segments already in hand
are kept; nothing is decoded twice.
"""

from __future__ import annotations

import logging

from modules.core import config
from modules.inference.pipeline import vad
from modules.inference.runtime import model_segment_processing, speech_clips
from modules.inference.runtime.concurrency import _determine_preemption_needed, _get_current_task_info

logger = logging.getLogger(__name__)


def preemption_pending() -> bool:
    """Whether a priority task has asked for this task's unit -- without waiting for anything.

    The blocking check (:func:`concurrency._check_preemption`) is what a task calls once it
    holds no worker channel; this is what it asks while it still does.
    """
    _, _, unit_id, _, is_priority, _ = _get_current_task_info()
    if is_priority or not unit_id:
        return False
    return _determine_preemption_needed(unit_id)[0]


class _PausePending(Exception):
    """Raised from inside a worker stream to abandon it; the blocking wait follows outside."""


def _abandon_if_pause_pending() -> None:
    if preemption_pending():
        raise _PausePending


def separate(preprocessor, audio_path: str, *, force: bool, stage: str, preemption_check) -> str:
    """Run vocal isolation on ``preprocessor``, releasing its worker whenever a pause is pending.

    An isolated preprocessing manager streams its separation over a channel it holds for the
    stream's life, and its ``yield_cb`` contract is that a callback which *raises* cancels the
    worker-side separation and frees the channel. The pipeline's blocking check inside that
    stream is what deadlocked the NUC's second run (the priority task's own montage UVR needed
    the channel). So the yield only asks; when a pause is pending it abandons the stream, the
    ordinary wait runs with the channel free, and the separation is started again. An
    in-process manager has no channel and keeps the blocking yield it always had.
    """
    if getattr(preprocessor, "yield_may_block", True):
        return preprocessor.preprocess_audio(audio_path, force=force, yield_cb=preemption_check, stage=stage)
    while True:
        try:
            return preprocessor.preprocess_audio(audio_path, force=force, yield_cb=_abandon_if_pause_pending, stage=stage)
        except _PausePending:
            logger.info("[UVR] Paused during %s; the preprocessing worker is released for the priority task", stage)
            preemption_check()
            logger.info("[UVR] Resuming %s from the start", stage)


def decode(model, processed_path: str, call: dict, clips: list | None, *, consume: dict, progress_window=None) -> tuple:
    """Run one decoder call to completion, pausing and resuming as many times as it takes.

    ``call`` is the transcribe kwargs for the whole span; ``clips`` the padded regions it
    covers (None for a whole-file call). Returns ``(segments, info)`` with ``info`` from the
    first call, which is the one that detected the language.
    """
    results: list = []
    info = None
    remaining = None
    while True:
        segments, call_info = model.transcribe(processed_path, **_options(call, remaining))
        info = info or call_info
        part, paused = model_segment_processing.consume_until_pause(
            segments,
            info,
            consume["task"],
            diarize=consume["diarize"],
            pause_pending=consume.get("pause_pending") or (lambda: False),
            progress_window=progress_window,
        )
        results.extend(part)
        if not paused:
            return results, info
        _close(segments)
        logger.info("[ASR] Paused after %d segment(s); the worker is released for the priority task", len(results))
        consume["preemption_check"]()
        remaining = _remaining(_decoded_so_far(clips, remaining), results, info, processed_path)
        logger.info("[ASR] Resuming: %d clip(s) left to decode", len(remaining))


def _close(segments) -> None:
    """Abandon the stream: the channel cancels the worker-side decode and frees the lock."""
    close = getattr(segments, "close", None)
    if close:
        close()


def _decoded_so_far(clips: list | None, remaining: list | None) -> list | None:
    """The spans the last call was asked for: the original clips until a pause narrowed them."""
    return clips if remaining is None else remaining


def _options(call: dict, remaining: list | None) -> dict:
    """The call for what is left: the original one on the first pass -- it already carries its
    clips, so merging them again would only log the decode twice -- and the remainder after."""
    if remaining is None:
        return call
    return {**call, **speech_clips.decode_options(remaining)}


def _remaining(clips: list | None, results: list, info, processed_path: str) -> list:
    """What is still undecoded after ``results``: clips past the last consumed segment.

    A whole-file call has no clips to trim, so the remainder is scanned with the decode
    VAD's own settings -- the same speech ``vad_filter`` would have compacted -- and decoded
    as clips from the pause point on.
    """
    last_end = max((float(seg["end"]) for seg in results), default=0.0)
    duration = float(getattr(info, "duration", 0.0) or 0.0)
    spans = clips if clips is not None else _speech_after(processed_path, last_end, duration)
    # Nothing left but the file has not been fully consumed: decode to the end of the audio
    # rather than return an empty seek list, which the decoder reads as "the whole file".
    return _after(spans, last_end) or [{"start": last_end, "end": max(last_end, duration)}]


def _after(spans: list, last_end: float) -> list:
    """The spans past ``last_end``, the one straddling it trimmed to start there."""
    return [{"start": max(float(c["start"]), last_end), "end": float(c["end"])} for c in spans if float(c["end"]) > last_end]


def _speech_after(processed_path: str, start: float, duration: float) -> list:
    """The decode VAD's speech regions from ``start`` to the end of the file, padded."""
    regions = vad.get_speech_timestamps_from_path(
        processed_path,
        config.VAD_THRESHOLD,
        min_silence_duration_ms=config.VAD_MIN_SILENCE_DURATION_MS,
        speech_pad_ms=config.VAD_SPEECH_PAD_MS,
        start_offset=start,
        duration=max(duration - start, 0.0) or None,
    )
    return [{"start": float(r["start"]), "end": float(r["end"])} for r in regions]
