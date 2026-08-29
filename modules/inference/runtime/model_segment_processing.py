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
    raw_segments = []
    live_srt_blocks = []
    max_prog = 80 if diarize else 95

    for segment in segments:
        preemption_check()
        seg_dict = _process_single_segment(segment, raw_segments, live_srt_blocks, info=info, task=task, max_prog=max_prog)
        raw_segments.append(seg_dict)

    if not diarize:
        return raw_segments

    return _run_diarization_safe(
        processed_path,
        raw_segments,
        info=info,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        hf_token=hf_token,
        unit_id=unit_id,
    )


def _process_single_segment(segment, raw_segments, live_srt_blocks, *, info, task, max_prog) -> dict:
    seg_dict = _build_segment_dict(segment)
    seg_idx = len(raw_segments) + 1
    _update_live_srt_metadata(segment, seg_idx, live_srt_blocks)
    _update_segment_progress(segment, seg_idx, info, task, max_prog)
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


def _update_segment_progress(segment, seg_idx: int, info, task, max_prog: int):
    if info.duration <= 0:
        return
    pct = _segment_progress_pct(segment.end, info.duration, max_prog)
    scheduler.update_task_progress(
        min(max_prog, pct),
        f"{_task_verb(task)} (Seg {seg_idx} | {utils.format_duration(segment.end)} / {utils.format_duration(info.duration)})",
    )


def _segment_progress_pct(segment_end: float, duration: float, max_prog: int) -> int:
    scale = 100 if max_prog == 95 else 80
    return int((segment_end / duration) * scale)


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
