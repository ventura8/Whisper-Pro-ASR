"""Re-transcribe speech the first pass left uncovered, each gap in its own language.

Forcing one language across a whole file frequently makes the decoder stop emitting
segments once it reaches audio that does not match, rather than continue -- which is the
recorded dropped-code-switched-legs defect, and the related "segments stop short of the
clip end" defect, both of which are coverage holes. With 30-second chunk-grained detection
neither can be seen ahead of time on a clip under 30 seconds, which is always exactly one
chunk regardless of how many languages it contains.

A well-behaved single-language transcript has zero gaps, so this costs one VAD pass (cheap:
no decoding) and nothing further in the common case.

A gap takes the language of the run it sits in, or the nearest run, rather than detecting
its own. A gap is a slice of a second or two, which is exactly the audio language detection
is worst on: with every gap detected on its own, 21 real monolingual films measured 7-50
gaps each and the foreign share climbed back toward the per-region fracture the runs exist
to prevent (Japanese film: 8.7% -> 19.9%). Detection is kept only for the case where no
run carries a language at all.
"""

from __future__ import annotations

import logging
import os

from modules.core import config
from modules.inference.pipeline import language_detection_core, vad
from modules.inference.runtime.model_segment_processing import consume_transcription_segments

logger = logging.getLogger(__name__)


def _with_decoded_spans(segments: list, decoded_spans: list | None) -> list:
    """Segments plus the spans already handed to the decoder, as covered intervals.

    A decoded span counts as covered whether or not text came back from it: the decoder did
    look, and the silence between spans was excluded on purpose. Without this the gap scan --
    which runs at the looser LD_VAD_THRESHOLD -- treated every inter-clip pause as
    untranscribed speech and re-transcribed it, inventing "Thank you very much." and
    "Bye-bye." into the silence.
    """
    return segments + [{"start": span["start"], "end": span["end"], "text": ""} for span in decoded_spans or []]


def fill_language_gaps(
    model,
    processed_path,
    segments: list,
    segment_languages: list,
    *,
    decoded_spans: list | None = None,
    runs: list | None = None,
    options: dict,
    duration_sec: float,
    unit_id,
    preemption_check,
) -> tuple[list, list]:
    """Find speech VAD confirms but no segment covers, and re-transcribe it with its own
    detected language.

    Fixes the recorded dropped-code-switched-legs defect (and the related "segments stop
    short of the clip end" defect as a side effect, since both are coverage holes):
    forcing one language for the whole file frequently makes the decoder stop emitting
    segments once it reaches audio that does not match, rather than continue -- with 30s
    chunk-grained detection this cannot even be seen ahead of time on a clip under 30s,
    which is always exactly one chunk regardless of how many languages it contains.
    Verified against the failure this replaces: a 4.3s two-language fixture whose second
    (English) sentence was silently dropped by the language-run approach.

    A well-behaved single-language transcript has zero gaps, so this costs one VAD pass
    (cheap: no decoding) and nothing further in the common case.

    ``decoded_spans`` are regions already handed to the decoder as explicit clips. They count
    as covered whether or not text came back from them, because the decoder did look and the
    silence between them was excluded on purpose. Without this the scan -- which runs at the
    looser LD_VAD_THRESHOLD -- treated every inter-clip silence as an untranscribed gap and
    re-transcribed it, inventing "Thank you very much." and "Bye-bye." into the pauses.

    ``runs`` are the language runs the decode was made of; a gap is decoded in the language
    of the run it falls in or lies nearest to, and only detects its own when no run has one.
    """
    speech_ts = _scan_speech(processed_path)
    if speech_ts is None:
        return segments, segment_languages

    gaps = language_detection_core.find_uncovered_speech_gaps(_with_decoded_spans(segments, decoded_spans), speech_ts, duration_sec)
    if not gaps:
        return segments, segment_languages

    logger.info(
        "[ASR] %d uncovered speech gap(s) on hardware unit %s: %s",
        len(gaps),
        unit_id,
        ", ".join(f"{g['start']:.1f}-{g['end']:.1f}s" for g in gaps),
    )

    for gap in gaps:
        preemption_check()
        _fill_gap_from_slice(
            model,
            {**gap, "run": _run_for(gap, runs)},
            processed_path=processed_path,
            segments=segments,
            segment_languages=segment_languages,
            options=options,
            unit_id=unit_id,
        )

    segment_languages.sort(key=lambda r: r["start"])
    return segments, segment_languages


def _run_for(gap: dict, runs: list | None):
    """The labelled run the gap sits in, else the nearest one; None when no run has a language."""
    labelled = [run for run in runs or [] if run.get("language")]
    if not labelled:
        return None
    midpoint = (float(gap["start"]) + float(gap["end"])) / 2
    return min(labelled, key=lambda run: max(float(run["start"]) - midpoint, midpoint - float(run["end"]), 0.0))


def _gap_language(model, gap: dict, slice_path: str):
    """``(language, confidence)`` for the gap: its run's, or detected when there is no run."""
    if gap.get("run"):
        return gap["run"]["language"], gap["run"].get("confidence")
    try:
        gap_lang, gap_confidence, _ = model.detect_language(slice_path)
    except (RuntimeError, ValueError, ImportError, OSError, EOFError) as e:
        logger.warning("[ASR] Gap language detection failed for %.1f-%.1fs: %s", gap["start"], gap["end"], e)
        return None, None
    return gap_lang, gap_confidence


def _fill_gap_from_slice(model, gap: dict, *, processed_path, segments, segment_languages, options, unit_id) -> None:
    """Extract one gap to a temporary file, re-transcribe it, and clean the file up.

    A gap that cannot be extracted is skipped rather than fatal: the remaining gaps are
    still worth filling, and _extract_gap_slice has already said why this one was not.
    """
    slice_path = _extract_gap_slice(processed_path, gap)
    if slice_path is None:
        return
    try:
        _fill_one_gap(
            model,
            gap,
            slice_path=slice_path,
            processed_path=processed_path,
            segments=segments,
            segment_languages=segment_languages,
            options=options,
            unit_id=unit_id,
        )
    finally:
        _remove_quietly(slice_path)


def _scan_speech(processed_path):
    """VAD-scan the whole file, or None when the scan itself could not run.

    None rather than an empty list: "no speech regions" and "the scan failed" lead to
    different conclusions, and only the first of them means there is nothing to fill.
    """
    try:
        full_audio = vad.decode_audio(processed_path)
        return vad.get_speech_timestamps(
            full_audio,
            threshold=config.LD_VAD_THRESHOLD,
            min_silence_duration_ms=config.VAD_MIN_SILENCE_DURATION_MS,
            speech_pad_ms=config.VAD_SPEECH_PAD_MS,
        )
    except (ImportError, RuntimeError, OSError, ValueError) as e:
        logger.warning("[ASR] Gap-fill VAD scan failed, skipping: %s", e)
        return None


def _extract_gap_slice(processed_path: str, gap: dict):
    """Write one gap to its own file, or None when it cannot be extracted.

    A file path, not decoded samples: IsolatedEngine runs in a worker subprocess and can
    only be handed a path across that boundary -- passing an array raises TypeError there.
    Verified on hardware: this failed with exactly that error against the isolated
    FasterWhisperEngine before switching to a path.
    """
    try:
        return vad.extract_slice_to_file(processed_path, gap["start"], gap["end"] - gap["start"])
    except (RuntimeError, OSError, ValueError) as e:
        logger.warning("[ASR] Could not extract gap slice %.1f-%.1fs: %s", gap["start"], gap["end"], e)
        return None


def _remove_quietly(path: str) -> None:
    """Delete a temporary slice, tolerating a file that is already gone."""
    try:
        os.remove(path)
    except OSError:
        pass


def _fill_one_gap(
    model,
    gap: dict,
    *,
    slice_path: str,
    processed_path: str,
    segments: list,
    segment_languages: list,
    options: dict,
    unit_id,
) -> None:
    """Detect and transcribe one gap slice, appending its results in place.

    ``options`` carries the decode settings (task, initial_prompt, vad_filter,
    word_timestamps, and the file's resolved language) as one value; they always travel
    together and are passed straight through to the engine -- except the prompt, which
    reaches only a gap in the file's own language, as run_decoding gives it only to that
    language's call: a prompt written in one language pulls the decoder toward it.
    """
    gap_lang, gap_confidence = _gap_language(model, gap, slice_path)
    if not gap_lang:
        return

    try:
        trans_res = model.transcribe(
            slice_path,
            language=gap_lang,
            task=_task_for(gap_lang, options),
            beam_size=config.DEFAULT_BEAM_SIZE,
            initial_prompt=_prompt_for(gap_lang, options),
            vad_filter=options["vad_filter"],
            word_timestamps=options["word_timestamps"],
            vad_parameters={
                "min_silence_duration_ms": config.VAD_MIN_SILENCE_DURATION_MS,
                "threshold": config.VAD_THRESHOLD,
            },
        )
    except (RuntimeError, ValueError, ImportError, OSError, EOFError) as e:
        logger.warning("[ASR] Gap re-transcription failed for %.1f-%.1fs: %s", gap["start"], gap["end"], e)
        return

    # A gap slice is a second or two and is consumed whole. The preemption check runs
    # between gaps (fill_language_gaps), never inside this stream: a pause taken here would
    # hold the worker channel the priority task needs -- the deadlock resumable_decode
    # describes -- for a gain of at most one short slice.
    gap_segments = consume_transcription_segments(
        trans_res[0],
        trans_res[1],
        options["task"],
        diarize=False,
        min_speakers=None,
        max_speakers=None,
        hf_token=None,
        unit_id=unit_id,
        processed_path=processed_path,
        preemption_check=_never_pauses,
    )
    gap_segments = _sanitize_gap_segments(gap_segments, gap["end"] - gap["start"])
    if not gap_segments:
        return
    _offset_segment_times(gap_segments, gap["start"])
    segments.extend(_labelled(gap_segments, gap_lang))
    segments.sort(key=lambda s: s["start"])
    if _already_reported(gap, gap_lang, segment_languages):
        return
    segment_languages.append(
        {
            "start": round(gap["start"], 2),
            "end": round(gap["end"], 2),
            "language": gap_lang,
            "confidence": _rounded_confidence(gap_confidence),
        }
    )


def _task_for(gap_lang: str, options: dict) -> str:
    """The request's task -- or a translation, for a gap in another language when
    transcription is forced, as run_decoding does for the runs."""
    file_language = options.get("language")
    if options.get("force_transcription") and file_language and gap_lang != file_language:
        return "translate"
    return options["task"]


def _prompt_for(gap_lang: str, options: dict):
    """The caller's prompt for a gap in the file's language, nothing for one in another.

    With no file language in ``options`` the prompt is kept: there is nothing to say the gap
    is foreign, and the callers that omit the key are the ones that never set a prompt.
    """
    file_language = options.get("language")
    if file_language and gap_lang != file_language:
        return None
    return options.get("initial_prompt")


def _never_pauses() -> None:
    """The in-stream check for a gap slice: nothing to do, and nothing that could block."""


def _already_reported(gap: dict, gap_lang: str, segment_languages: list) -> bool:
    """Whether a reported span already covers the gap *in the gap's language*.

    A gap inside its own run needs no entry of its own. A gap that detected its own language
    -- no run had one, so the report holds only the file-level span -- must be reported even
    when that span covers it: its segments carry the detected language, and the report would
    otherwise say the whole file was one language while the transcript says otherwise.
    """
    midpoint = (float(gap["start"]) + float(gap["end"])) / 2
    return any(float(span["start"]) <= midpoint < float(span["end"]) and span.get("language") == gap_lang for span in segment_languages)


def _rounded_confidence(value) -> float:
    """A gap's detection confidence as a number, or 0.0 when the engine did not give one.

    ``detect_language`` is an engine method, and not every engine returns a float here --
    an isolated worker can hand back None over the pipe. ``round(None, 4)`` is a TypeError
    raised after the gap's segments were already merged into the transcript, so it aborted
    every remaining gap over a value used for nothing but reporting.
    """
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return 0.0


def _labelled(gap_segments: list, gap_lang: str) -> list:
    """Stamp the language this gap was re-transcribed in onto its segments.

    Gap filling is the one producer that already knows its own answer -- it detected the
    language before decoding the slice -- so it should say so rather than let the file-level
    language stand in for it. Only while per-segment languages are reported at all: with
    ``ASR_SEGMENT_LANGUAGES`` off a response has one shape, no language on any segment, the
    decoded ones included.
    """
    if not config.ASR_SEGMENT_LANGUAGES:
        return gap_segments
    for seg in gap_segments:
        seg["language"] = gap_lang
    return gap_segments


def _sanitize_gap_segments(segments: list, slice_duration: float) -> list:
    """Discard and clamp artifacts of transcribing a short, padded slice.

    Whisper always processes audio in an internal 30-second window, padding anything
    shorter. Found on hardware: a 0.64s gap slice produced a segment with end=~30s and
    empty text -- the window's own length, not the slice's, leaking into the reported
    timestamp. Empty-text segments are pure padding artifacts and are dropped outright;
    any remaining segment's end is clamped to the slice's real duration so a similar
    partial artifact cannot report time that was never in the gap.
    """
    sanitized = []
    for seg in segments:
        if not seg.get("text", "").strip():
            continue
        seg["end"] = min(seg["end"], slice_duration)
        if seg["end"] > seg["start"]:
            # The words carry the same 30-second window artifact as the segment that holds
            # them, and _offset_segment_times shifts both -- so an unclamped word kept a
            # timestamp past the end of its own segment, and past the gap it came from.
            _clamp_word_times(seg, slice_duration)
            sanitized.append(seg)
    return sanitized


def _clamp_word_times(seg: dict, slice_duration: float) -> None:
    """Clamp one segment's word timestamps to the slice they were decoded from."""
    for word in seg.get("words") or []:
        word["start"] = min(word["start"], slice_duration)
        word["end"] = min(word["end"], slice_duration)


def _offset_segment_times(segments: list, offset: float) -> None:
    """Shift a run's segment (and word) timestamps from slice-relative to file-relative."""
    for seg in segments:
        seg["start"] = round(seg["start"] + offset, 2)
        seg["end"] = round(seg["end"] + offset, 2)
        for word in seg.get("words") or []:
            word["start"] = round(word["start"] + offset, 2)
            word["end"] = round(word["end"] + offset, 2)
