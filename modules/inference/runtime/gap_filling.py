"""Re-transcribe speech the first pass left uncovered, each gap in its own language.

Forcing one language across a whole file frequently makes the decoder stop emitting
segments once it reaches audio that does not match, rather than continue -- which is the
recorded dropped-code-switched-legs defect, and the related "segments stop short of the
clip end" defect, both of which are coverage holes. With 30-second chunk-grained detection
neither can be seen ahead of time on a clip under 30 seconds, which is always exactly one
chunk regardless of how many languages it contains.

A well-behaved single-language transcript has zero gaps, so this costs one VAD pass (cheap:
no decoding) and nothing further in the common case.
"""

from __future__ import annotations

import logging
import os

from modules.core import config
from modules.inference.pipeline import language_detection_core, vad
from modules.inference.runtime.model_segment_processing import consume_transcription_segments

logger = logging.getLogger(__name__)


def fill_language_gaps(
    model,
    processed_path,
    segments: list,
    segment_languages: list,
    *,
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
    """
    speech_ts = _scan_speech(processed_path)
    if speech_ts is None:
        return segments, segment_languages

    gaps = language_detection_core.find_uncovered_speech_gaps(segments, speech_ts, duration_sec)
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
            gap,
            processed_path=processed_path,
            segments=segments,
            segment_languages=segment_languages,
            options=options,
            unit_id=unit_id,
            preemption_check=preemption_check,
        )

    segment_languages.sort(key=lambda r: r["start"])
    return segments, segment_languages


def _fill_gap_from_slice(model, gap: dict, *, processed_path, segments, segment_languages, options, unit_id, preemption_check) -> None:
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
            preemption_check=preemption_check,
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
    preemption_check,
) -> None:
    """Detect and transcribe one gap slice, appending its results in place.

    ``options`` carries the decode settings (task, initial_prompt, vad_filter,
    word_timestamps) as one value; they always travel together and are passed straight
    through to the engine.
    """
    try:
        gap_lang, gap_confidence, _ = model.detect_language(slice_path)
    except (RuntimeError, ValueError, ImportError) as e:
        logger.warning("[ASR] Gap language detection failed for %.1f-%.1fs: %s", gap["start"], gap["end"], e)
        return
    if not gap_lang:
        return

    try:
        trans_res = model.transcribe(
            slice_path,
            language=gap_lang,
            task=options["task"],
            beam_size=config.DEFAULT_BEAM_SIZE,
            initial_prompt=options["initial_prompt"],
            vad_filter=options["vad_filter"],
            word_timestamps=options["word_timestamps"],
            vad_parameters={
                "min_silence_duration_ms": config.VAD_MIN_SILENCE_DURATION_MS,
                "threshold": config.VAD_THRESHOLD,
            },
        )
    except (RuntimeError, ValueError, ImportError) as e:
        logger.warning("[ASR] Gap re-transcription failed for %.1f-%.1fs: %s", gap["start"], gap["end"], e)
        return

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
        preemption_check=preemption_check,
    )
    gap_segments = _sanitize_gap_segments(gap_segments, gap["end"] - gap["start"])
    if not gap_segments:
        return
    _offset_segment_times(gap_segments, gap["start"])
    segments.extend(gap_segments)
    segments.sort(key=lambda s: s["start"])
    segment_languages.append(
        {
            "start": round(gap["start"], 2),
            "end": round(gap["end"], 2),
            "language": gap_lang,
            "confidence": round(gap_confidence, 4),
        }
    )


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
            sanitized.append(seg)
    return sanitized


def _offset_segment_times(segments: list, offset: float) -> None:
    """Shift a run's segment (and word) timestamps from slice-relative to file-relative."""
    for seg in segments:
        seg["start"] = round(seg["start"] + offset, 2)
        seg["end"] = round(seg["end"] + offset, 2)
        for word in seg.get("words") or []:
            word["start"] = round(word["start"] + offset, 2)
            word["end"] = round(word["end"] + offset, 2)
