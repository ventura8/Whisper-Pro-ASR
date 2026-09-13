"""Decoding the language runs: one decoder call per language, each told which language it is.

A run is a stretch of speech the pre-decode detection, with hysteresis, assigned to one
language (language_runs). Until this module the runs were handed to a single ``transcribe``
call as clips with ``multilingual=True``, which meant the decoder re-detected the language of
every clip itself from that clip's audio and the run's label was only ever *reported*. Two
things were wrong with that. The decoder's detection is the same short-audio detection the
hysteresis exists to correct, so the transcript could be in a language the response did not
say it was in. And the only way to give the decoder a run's worth of audio to detect on was
to make the run one clip, silence and all -- which put the passage between two scenes back
inside a decode window, the invented-text failure decoding by region had closed.

So the label is now what the decoder is told. Runs are grouped by language and each group is
decoded in one call with ``language`` fixed and per-window re-detection off; a monolingual
file is one call, exactly as before, and a file with ten languages is ten. The clips in each
call are the run's *regions*, one line each, never the run as a whole: a window holding a
run's worth of lines dropped some of them (5 of 46 on the film fixture, against none with
region-sized clips) and carried the pause between scenes. Each call decodes the whole file's
features again, which is cheap next to the encoder passes it then skips, and only the call
for the file's own language carries the caller's ``initial_prompt``: a prompt written in one
language pulls the decoder toward it, which is what a run in another language must not have.

When no run carries a language -- the detection failed outright -- the clips go to a single
call with per-window re-detection on, the behaviour before runs existed. With no runs at all
(an explicit ``language=``, an engine that cannot take clips, a file with one speech region)
the call is the plain whole-file one it always was.
"""

import logging

from modules.inference.runtime import model_segment_processing, resumable_decode, speech_clips

logger = logging.getLogger(__name__)


def decode(model, processed_path: str, runs: list, *, options: dict, consume: dict) -> tuple:
    """Decode ``runs`` and return ``(segments, info, clips)``.

    ``options`` are the transcribe kwargs shared by every call -- ``language`` (the file's,
    already resolved), ``multilingual`` (whether it was auto-detected), ``initial_prompt``,
    ``vad_filter`` and the rest. ``consume`` holds the keyword arguments for
    :func:`model_segment_processing.consume_transcription_segments` other than the segments
    and info. ``clips`` are the padded spans that were decoded, for gap filling to treat as
    covered; ``info`` is the dominant call's.
    """
    clips = speech_clips.pad([region for run in runs for region in run["regions"]])
    groups = _grouped(runs, clips)
    if len(groups) <= 1:
        return _single_call(model, processed_path, groups, options=options, consume=consume) + (clips,)
    results, info = _one_call_per_language(model, processed_path, groups, options=options, consume=consume)
    if consume.get("diarize"):
        results = model_segment_processing.diarize_segments(results, info=info, **_diarization_kwargs(consume))
    return results, info, clips


def _one_call_per_language(model, processed_path: str, groups: list, *, options: dict, consume: dict) -> tuple:
    """Decode every group in turn and return the merged segments with the first group's info."""
    results = []
    infos = []
    for index, (language, group_clips) in enumerate(groups):
        call = _call_options(options, language, group_clips)
        window = (index, len(groups))
        part, info = resumable_decode.decode(model, processed_path, call, group_clips, consume=consume, progress_window=window)
        results.extend(part)
        infos.append(info)
    results.sort(key=lambda seg: (float(seg["start"]), float(seg["end"])))
    return results, infos[0]


def _single_call(model, processed_path: str, groups: list, *, options: dict, consume: dict) -> tuple:
    """The one-call cases: no runs at all, or every run in one language.

    Through resumable_decode like the per-language calls, so a pause releases the worker
    here too; diarization runs once over the complete result, as it did.
    """
    call = _call_options(options, *groups[0]) if groups else _engine_options(options)
    clips = groups[0][1] if groups else None
    results, info = resumable_decode.decode(model, processed_path, call, clips, consume=consume)
    if consume.get("diarize"):
        results = model_segment_processing.diarize_segments(results, info=info, **_diarization_kwargs(consume))
    return results, info


def _grouped(runs: list, clips: list) -> list:
    """``[(language, clips), ...]`` with the language holding the most speech first.

    ``clips`` are the padded regions of every run in order, so each run takes the next
    ``len(run["regions"])`` of them. ``None`` -- runs the detection could not label -- is a
    group like any other and is decoded with re-detection on, as such regions always were.
    """
    seconds: dict = {}
    members: dict = {}
    remaining = list(clips)
    for run in runs:
        spoken = sum(float(region["end"]) - float(region["start"]) for region in run["regions"])
        seconds[run["language"]] = seconds.get(run["language"], 0.0) + spoken
        count = len(run["regions"])
        members.setdefault(run["language"], []).extend(remaining[:count])
        del remaining[:count]
    ordered = sorted(members, key=lambda language: -seconds[language])
    return [(language, members[language]) for language in ordered]


def _call_options(options: dict, language, clips: list) -> dict:
    """The transcribe kwargs for one language group.

    A labelled group is decoded in its language with per-window re-detection off; the
    caller's prompt reaches only the group in the file's own language. An unlabelled group
    keeps the shared options: the file's language, and re-detection as the request had it.
    """
    call = {**_engine_options(options), **speech_clips.decode_options(clips)}
    if language is None:
        return call
    call["language"] = language
    call["multilingual"] = False
    if language != options.get("language"):
        _for_another_language(call, options, language, len(clips))
    return call


def _engine_options(options: dict) -> dict:
    """The shared options as the engine takes them: the pipeline's own switches removed."""
    return {key: value for key, value in options.items() if key != "force_transcription"}


def _for_another_language(call: dict, options: dict, language: str, clips: int) -> None:
    """What a group outside the file's language does not get, and what it gets instead.

    The caller's prompt stays with the file's language. With transcription forced, the
    group is translated -- into English, the one target Whisper has -- so a film that
    switches comes back as its own language plus English, never as the other language
    decoded under the film's token.
    """
    if call.get("initial_prompt"):
        logger.info("[ASR] Decoding %d clip(s) in %s without the caller's prompt", clips, language)
    call["initial_prompt"] = None
    if options.get("force_transcription") and not _translates(call.get("task")):
        logger.info("[ASR] Translating %d clip(s) in %s: transcription is forced to %s", clips, language, options.get("language"))
        call["task"] = "translate"


def _translates(task) -> bool:
    return str(task).lower() == "translate"


def _diarization_kwargs(consume: dict) -> dict:
    return {key: consume[key] for key in ("processed_path", "min_speakers", "max_speakers", "hf_token", "unit_id")}
