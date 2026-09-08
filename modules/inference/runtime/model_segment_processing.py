"""Segment consumption helpers for transcription and optional diarization."""

import logging

from modules.core import utils
from modules.inference import scheduler
from modules.inference.pipeline import diarization

logger = logging.getLogger(__name__)


def consume_transcription_segments(
    segments,
    info,
    task,
    *,
    diarize,
    min_speakers,
    max_speakers,
    hf_token,
    unit_id,
    processed_path,
    preemption_check,
):
    """Consume segment generator and optionally run diarization."""
    raw_segments = consume_segments(segments, info, task, diarize=diarize, preemption_check=preemption_check)
    if not diarize:
        return raw_segments
    return diarize_segments(
        raw_segments,
        info=info,
        processed_path=processed_path,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        hf_token=hf_token,
        unit_id=unit_id,
    )


def consume_segments(segments, info, task, *, diarize, preemption_check, progress_window=None) -> list:
    """Consume the segment generator into dicts, reporting live text and progress as it goes.

    ``diarize`` only sets the progress ceiling here -- 80 when diarization is still to run
    over the result, 95 otherwise. ``progress_window`` is ``(index, count)`` when this call
    is one of several decoding the same file -- one per language run group -- so the
    reported progress climbs through this call's share of the bar instead of falling back
    to zero at every call. Absent, the call owns the whole bar, as a single decode always did.

    ``preemption_check`` runs before every segment, *inside* the stream. That is only safe
    when it cannot block: a check that pauses here keeps the worker channel open while the
    priority task that asked for the pause waits for that same channel -- the deadlock
    measured on the NUC. A decode that may be paused uses :func:`consume_until_pause`.
    """
    raw_segments, _ = _consume(segments, info, task, diarize=diarize, before_each=preemption_check, progress_window=progress_window)
    return raw_segments


def consume_until_pause(segments, info, task, *, diarize, pause_pending, progress_window=None) -> tuple[list, bool]:
    """Consume segments until the stream ends or a pause is requested; ``(segments, paused)``.

    ``pause_pending`` is a non-blocking question -- has a priority task asked for this
    unit? -- so the caller can close the stream, release the worker channel, wait, and
    decode the rest afterwards (resumable_decode). Nothing waits while the stream is open.
    """
    return _consume(segments, info, task, diarize=diarize, stop_when=pause_pending, progress_window=progress_window)


def _consume(segments, info, task, *, diarize, before_each=None, stop_when=None, progress_window=None) -> tuple[list, bool]:
    raw_segments: list = []
    live_srt_blocks: list = []
    max_prog, window = _progress_bounds(diarize, progress_window)
    for segment in segments:
        if before_each:
            before_each()
        seg_dict = _process_single_segment(segment, raw_segments, live_srt_blocks, info=info, task=task, max_prog=max_prog, window=window)
        raw_segments.append(seg_dict)
        # Asked once the segment in hand is kept, so a pause never drops one the worker
        # already produced: the resume starts where this one ended.
        if stop_when and stop_when():
            return raw_segments, True
    return raw_segments, False


def _progress_bounds(diarize, progress_window) -> tuple[int, tuple]:
    """The ceiling this call climbs to and the share of the bar it owns."""
    return (80 if diarize else 95), (progress_window or (0, 1))


def diarize_segments(raw_segments, *, info, processed_path, min_speakers, max_speakers, hf_token, unit_id) -> list:
    """Run diarization once over a complete segment list, falling back to it unchanged.

    Public so that a decode made of several calls can merge their segments first and
    fingerprint the file once: diarization numbers speakers per call, and two calls' numbers
    would have no correspondence.
    """
    return _run_diarization_safe(
        processed_path,
        raw_segments,
        info=info,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        hf_token=hf_token,
        unit_id=unit_id,
    )


def _process_single_segment(segment, raw_segments, live_srt_blocks, *, info, task, max_prog, window) -> dict:
    seg_dict = _build_segment_dict(segment)
    seg_idx = len(raw_segments) + 1
    _update_live_srt_metadata(segment, seg_idx, live_srt_blocks)
    _update_segment_progress(segment, seg_idx, info, task, max_prog, window=window)
    _maybe_log_segment_progress(segment, seg_idx, info, task)
    return seg_dict


def _build_segment_dict(segment) -> dict:
    seg_dict = {"start": round(segment.start, 2), "end": round(segment.end, 2), "text": segment.text.strip()}
    timed = _timed_words(getattr(segment, "words", None))
    if timed:
        seg_dict["words"] = timed
    return seg_dict


def _timed_words(words) -> list[dict]:
    """The words that carry both timings, in the dict shape the API and gap_filling expect.

    Words with no start or end are dropped rather than passed on: gap_filling formats these
    with "%.1f", so a single untimed word raised TypeError from inside gap detection and lost
    the whole transcription. An empty result means the caller omits the key entirely -- an
    empty "words": [] reads as "timestamps were requested and this segment genuinely has
    none", which is a different claim.
    """
    return [w for w in (_word_dict(word) for word in words or ()) if _is_timed(w)]


def _is_timed(word: dict) -> bool:
    """Whether a converted word carries both of the timings gap_filling will format."""
    return word["start"] is not None and word["end"] is not None


def _word_dict(word) -> dict:
    """Normalize one word timing to the dict shape gap_filling and the API expect.

    Words arrive in two shapes. In-process engines yield objects with ``.start``/``.word``
    attributes; an IsolatedEngine's segments come back over a pipe, so its words are already
    plain dicts -- and attribute access on those raised AttributeError, failing every
    isolated transcription that asked for word timestamps. Both are accepted; the output
    shape is unchanged either way, because gap_filling reads these by key.
    """
    if isinstance(word, dict):
        # A dict word missing "start"/"end" yielded None for them, and gap_filling formats
        # those with "%.1f" -- so one untimed word raised TypeError from inside gap detection
        # and lost the whole transcription. Dropping the timings here lets the caller skip the
        # word instead; see _build_segment_dict, which filters them out.
        return {
            "start": word.get("start"),
            "end": word.get("end"),
            "word": word.get("word"),
            "probability": word.get("probability", 1.0),
        }
    return {"start": word.start, "end": word.end, "word": word.word, "probability": getattr(word, "probability", 1.0)}


def _update_live_srt_metadata(segment, seg_idx: int, live_srt_blocks: list):
    block = utils.format_single_srt_block(idx=seg_idx, start_ts=segment.start, end_ts=segment.end, text=segment.text)
    live_srt_blocks.append(block)
    scheduler.update_task_metadata(live_text="".join(live_srt_blocks), current_position=segment.end)


def _update_segment_progress(segment, seg_idx: int, info, task, max_prog: int, *, window=(0, 1)):
    if info.duration <= 0:
        return
    pct = _segment_progress_pct(segment.end, info.duration, max_prog, window)
    scheduler.update_task_progress(
        min(max_prog, pct),
        f"{_task_verb(task)} (Seg {seg_idx} | {utils.format_duration(segment.end)} / {utils.format_duration(info.duration)})",
    )


def _segment_progress_pct(segment_end: float, duration: float, max_prog: int, window=(0, 1)) -> int:
    """Progress through the bar, with this call's ``(index, count)`` share of it applied."""
    scale = 100 if max_prog == 95 else 80
    index, count = window
    return int((index + min(segment_end / duration, 1.0)) / count * scale)


def _maybe_log_segment_progress(segment, seg_idx: int, info, task):
    if seg_idx % 100 != 0 and seg_idx != 1:
        return
    logger.info(
        "[Engine] %s segment %d (Audio: %s / %s)",
        _task_verb(task),
        seg_idx,
        utils.format_duration(segment.end),
        utils.format_duration(info.duration),
    )


def _task_verb(task) -> str:
    return "Translating" if task == "translate" else "Transcribing"


def _run_diarization_safe(processed_path, raw_segments, *, info, min_speakers, max_speakers, hf_token, unit_id) -> list:
    if not raw_segments:
        return []
    try:
        return diarization.run_diarization(
            processed_path=processed_path,
            raw_segments=raw_segments,
            info=info,
            language=info.language,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            hf_token=hf_token,
            unit_id=unit_id,
        )
    except (ValueError, TypeError, KeyError, AttributeError, OSError, RuntimeError) as diarize_err:
        logger.error("[Diarization] Diarization failed: %s. Falling back to non-diarized output.", diarize_err)
        results = []
        for s in raw_segments:
            seg_dict = {"start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["text"].strip()}
            if "words" in s:
                seg_dict["words"] = s["words"]
            results.append(seg_dict)
        return results
