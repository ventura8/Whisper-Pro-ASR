"""Which language each transcribed segment was actually spoken in.

Until now the API reported one language for a whole file plus whatever gap filling happened to
label, so a code-switched response could not say where each language was found -- the recorded
`language` defect. No shipped engine offers the information: faster-whisper commits to a
language per decode window and never reports which, and its Segment carries no language field.

So it is measured here instead. Decoding by speech region already gives clips that map onto
segments, and detecting a language per clip costs one encoder pass each (~220ms measured on an
RTX 3080). That is real money, and the gate on it is simply whether there are clips: a file
with one speech region is never clipped, and gets the file's own language stamped on every
segment, which is both free and true.

There is deliberately no whole-file "is this multilingual" gate in front of it. Every such
signal measured reads a genuinely code-switching film as monolingual -- the montage samples a
handful of windows across two hours and the localised foreign passages fall between them, so
the top language holds 0.92-1.00 on real film whether or not it code-switches. Gating on one
would label the synthetic fixture and miss every real film the feature exists for.
"""

import logging

from modules.core import config

logger = logging.getLogger(__name__)


def spans_for(runs: list, segments: list, info) -> list:
    """Label the segments from the decode runs and return the spans to report.

    The runs are what the decoder was given, so their languages are the languages the
    segments were decoded in -- measured before decoding, grouped with hysteresis. Falls
    back to a single file-level span when no run carries a language, which is what the API
    reported before any of this and is still true, just coarse. ``ASR_SEGMENT_LANGUAGES``
    decides only whether the per-run detail is reported -- the runs exist either way -- and
    switched off it leaves the segments as they were before this release: no language on
    them, one span for the file.
    """
    if not config.ASR_SEGMENT_LANGUAGES:
        return [_file_span(info)]
    labelled = _reported(runs)
    stamp(segments, labelled, info.language)
    return labelled or [_file_span(info)]


def _reported(runs: list) -> list:
    """The runs that carry a language, in the shape the response reports."""
    return [
        {
            "start": round(float(r["start"]), 2),
            "end": round(float(r["end"]), 2),
            "language": r["language"],
            "confidence": round(float(r.get("confidence") or 0.0), 4),
        }
        for r in runs or []
        if r.get("language")
    ]


def _file_span(info) -> dict:
    return {"start": 0.0, "end": round(info.duration, 2), "language": info.language, "confidence": info.language_probability}


def for_clips(model, processed_path: str, regions: list, preemption_check=None) -> list:
    """One language per region, measured before decoding; ``[]`` when there is nothing to measure.

    Not gated on ``ASR_SEGMENT_LANGUAGES`` any more: these labels are what groups regions
    into decode runs, and without them decoding by region fractures monolingual film. The
    switch now only decides whether the labels are reported.

    ``[]`` on any failure rather than raising: with no labels every region becomes its own
    run, which is the pre-hysteresis behaviour, and losing the transcription over a lost
    label would be a far worse trade.
    """
    if not regions:
        return []
    try:
        return _collect(model, processed_path, regions, preemption_check)
    except _DetectionFailed:
        return []


class _DetectionFailed(Exception):
    """The detector itself failed; already logged, the labels are simply not available."""


#: Regions per worker call. The detection is a stream that holds the worker channel for its
#: life, so a preemption check taken *inside* it -- which blocks when a priority task wants
#: the unit -- kept the channel from the very task that asked, and deadlocked a single-unit
#: host (the NUC, 2026-09-12). Regions go to the worker in chunks instead, each consumed to
#: the end, with the check between chunks and no stream open when a pause happens. Sixteen
#: is ~80 s between checks on a CPU decoding at 5 s a region, ~4 s on a GPU; the worker keeps
#: the decoded audio between calls, so the chunking itself costs nothing.
DETECTION_CHUNK = 16


def _collect(model, processed_path: str, regions: list, preemption_check) -> list:
    """One labelled span per clip the detector could name a language for.

    Only the detection is allowed to fail quietly. The preemption callback runs between
    chunks, outside that guard and outside any stream: what it raises is the scheduler's
    business, and swallowing it here would keep labelling audio the task was told to give
    up; and it may block, which is only safe once the worker channel is free.
    """
    labelled = []
    for start in range(0, len(regions), DETECTION_CHUNK):
        stop = start + DETECTION_CHUNK
        chunk = regions[start:stop]
        labelled.extend(span for span in map(_labelled_span, _guarded_events(model, processed_path, chunk)) if span)
        if preemption_check:
            preemption_check()
    return labelled


def _guarded_events(model, processed_path: str, regions: list):
    """The detection events, with any failure of the detector logged and folded into one
    exception the caller can tell apart from anything raised while consuming them."""
    try:
        yield from _detection_events(model, processed_path, regions)
    except tuple([Exception]) as exc:
        logger.warning("[ASR] Per-clip language detection failed; segments keep the file language: %s", exc)
        raise _DetectionFailed from exc


def _labelled_span(event: dict):
    """The span this detection event describes, or None when it named no language."""
    result = event.get("result") or {}
    language = result.get("detected_language")
    if not language:
        return None
    return {
        "start": round(float(event["start"]), 2),
        "end": round(float(event["end"]), 2),
        "language": language,
        "confidence": round(float(result.get("confidence") or 0.0), 4),
    }


def _detection_events(model, processed_path: str, regions: list):
    """Stream detections from the worker when isolated, else detect in this process."""
    if hasattr(type(model), "detect_language_regions"):
        return model.detect_language_regions(processed_path, regions)
    return _detect_in_process(model, processed_path, regions)


def _detect_in_process(model, processed_path: str, regions: list):
    """Slice the decoded audio here, for an engine that is not behind a worker."""
    vad = __import__("importlib").import_module("modules.inference.pipeline.vad")
    core = __import__("importlib").import_module("modules.inference.pipeline.language_detection_core")
    audio = vad.decode_audio(processed_path)
    for region in regions:
        # Same clamping as the worker: the span reported is the span sliced.
        start = max(0, int(float(region["start"]) * 16000))
        end = min(int(float(region["end"]) * 16000), len(audio))
        if start >= end:
            continue
        yield {
            "start": start / 16000,
            "end": end / 16000,
            "result": core.run_language_detection_core(model, audio[start:end].copy(), skip_vad=True),
        }


def stamp(segments: list, labelled: list, fallback: str) -> None:
    """Give every segment the language of the clip it falls inside.

    Matched by midpoint, the same rule the long-form test uses to attribute text to a window,
    so a segment is labelled by where it actually sits rather than by whichever clip it starts
    to overlap. Anything unmatched keeps the file-level language: a wrong label is worse than a
    coarse one, and the file language is at least true of the file.
    """
    for segment in segments:
        midpoint = (float(segment["start"]) + float(segment["end"])) / 2
        segment["language"] = _language_at(labelled, midpoint) or fallback


def _language_at(labelled: list, moment: float):
    for entry in labelled:
        if entry["start"] <= moment < entry["end"]:
            return entry["language"]
    return None
