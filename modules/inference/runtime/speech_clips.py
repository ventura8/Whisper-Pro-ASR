"""Deciding which spans of audio the decoder is given as explicit clips.

A decode window is the unit of language commitment: whatever it holds, the decoder settles on
one language for. Left to itself that window is 30 seconds of whatever VAD compacted together,
so on audio that changes language faster than that, the decoder picks one language for the
lot -- the recorded `windows` defect. The speech regions are where the language may change:
each is detected before the decode, grouped into runs of one language (language_runs), and
the runs go to the decoder as clips, one call per language (run_decoding).

This lives in the pipeline rather than inside the engine because two callers need the same
spans. The decoder gets them as clips, and gap filling needs to know that the silence between
them was skipped deliberately -- otherwise it re-transcribes every pause and invents speech
there, which is the `quiet` defect arriving through a different door.
"""

import logging

from modules.core import config, engine_registry
from modules.inference.pipeline import vad

logger = logging.getLogger(__name__)


def regions_for(processed_path: str, follows_switches: bool) -> list:
    """Unpadded speech regions to group into decode runs, or ``[]`` to decode the file as one span.

    ``follows_switches`` is whether this request's language may change from run to run: an
    auto-detected request, or any translation -- the output is English whatever was spoken,
    so a named language there is the main audio, not a constraint on the decode.

    Unpadded because two things downstream need real edges: the language-run grouping
    (``language_runs.build``) reads the silence between regions, and padding is applied to
    the *runs* by :func:`pad`, so a clip never cuts a word and never touches its neighbour.
    """
    if not _clipping_applies(follows_switches):
        return []
    # Unpadded on purpose: the merge below asks how much real silence separates two regions,
    # and padding destroys that answer before it can be asked. Padding is applied after.
    regions = vad.get_speech_timestamps_from_path(
        processed_path,
        config.VAD_THRESHOLD,
        min_silence_duration_ms=config.SEGMENT_SPLIT_MIN_SILENCE_MS,
        speech_pad_ms=0,
    )
    merged = _rejoin_split_utterances(regions)
    return merged if _regions_are_worth_clipping(merged) else []


def _rejoin_split_utterances(regions: list) -> list:
    """Rejoin regions separated by a pause too short to be a turn boundary.

    Silero splits at breaths *inside* an utterance, not only between them, and the sub-second
    fragment that produces is where per-window language detection is least reliable. Measured
    on nine real film excerpts: segments the decoder rendered in the wrong language had a
    1.36s median duration against 2.40s for correct ones, and 62% were under two seconds.

    The gaps here are real silence, because the VAD was asked not to pad. That matters more
    than the threshold: reading padded gaps instead measured 0.3s and merged at 0.7s, because
    Silero pads each side by ``speech_pad_ms`` and halves any silence shorter than twice it.
    On the long-form fixture that turned the tightest language boundary -- 0.403s, and every
    adjacent utterance there is a different language -- into a gap of 0.016s, so it merged.
    Measured: 7 clips spanned a language change, and 5 of the 118 ground-truth windows came
    back with an overlap of exactly 0.00, decoded in a neighbour's language.

    So the threshold means what it says again: silence shorter than this is a breath inside an
    utterance, anything longer is a turn. Nothing on the stress fixture qualifies, which is
    correct -- its utterances are cleanly separated and have no breaths to rejoin.
    """
    if not regions:
        return []
    merged = [dict(regions[0])]
    for region in regions[1:]:
        if region["start"] - merged[-1]["end"] <= config.SEGMENT_CLIP_MERGE_GAP_SEC:
            merged[-1]["end"] = region["end"]
        else:
            merged.append(dict(region))
    return merged


def pad(regions: list) -> list:
    """Widen each clip so a boundary never cuts a word, without letting two clips touch.

    Done here rather than by the VAD so the merge above can see real silence. Each side is
    clamped to the midpoint of the adjacent gap -- the same rule Silero applies -- so clips
    stay ordered and disjoint, which ``clip_timestamps`` requires: an overlapping or
    out-of-order pair makes the seek points non-monotonic.
    """
    margin = config.SEGMENT_SPLIT_PAD_MS / 1000.0
    padded = []
    for index, region in enumerate(regions):
        start = region["start"] - margin
        end = region["end"] + margin
        if index:
            start = max(start, (regions[index - 1]["end"] + region["start"]) / 2)
        if index < len(regions) - 1:
            end = min(end, (region["end"] + regions[index + 1]["start"]) / 2)
        padded.append({"start": round(max(0.0, start), 3), "end": round(end, 3)})
    return padded


def _clipping_applies(follows_switches: bool) -> bool:
    """Whether clips could help this request at all.

    A transcription in a language the caller named gets the single-language path untouched:
    that language is what they asked to read. A translation follows the switches even with a
    language named, since its output is English from whatever was spoken. An engine that
    would silently drop the clips must never be handed them: WhisperX accepts arbitrary kwargs
    and forwards none, so clips there decode the whole file as one span while the pipeline
    believes otherwise.
    """
    if not (follows_switches and config.ASR_SEGMENT_FIRST):
        return False
    return config.ASR_ENGINE == engine_registry.ENGINE_FASTER_WHISPER


def _regions_are_worth_clipping(regions: list) -> bool:
    """Whether the scan found something clips can actually improve on.

    One region commits to one language whatever is done to it, so there is nothing for a clip
    boundary to separate. Two or more regions can be separated regardless of how short the file
    is -- a 4.9s two-language clip is decoded as one window without clips and as two with them,
    which is the difference between dropping a leg and transcribing it.
    """
    return len(regions) >= config.SEGMENT_FIRST_MIN_REGIONS


def decode_options(regions: list) -> dict:
    """The transcribe kwargs these regions imply, or ``{}`` for the ordinary whole-file call.

    An empty clip list must never reach faster-whisper: it builds an odd-length seek-point
    list from it and decodes the entire file as one clip, which looks like success and is the
    single-language behaviour clips exist to replace.

    ``vad_filter`` goes off because the clips *are* the VAD output. Leaving it on would run
    Silero a second time and compact the audio again underneath the clips.
    """
    if not regions:
        return {}
    # Logged here rather than at the call site: this changes how the decoder windows the audio
    # and the only other evidence of it is a transcript that happens to be better.
    logger.info("[ASR] Decoding by speech region: %d clips from VAD", len(regions))
    flat = []
    for region in regions:
        flat.extend([round(region["start"], 3), round(region["end"], 3)])
    return {"clip_timestamps": flat, "vad_filter": False}
