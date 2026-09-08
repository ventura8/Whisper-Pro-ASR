"""Grouping speech regions into runs of one language, so a decode window is never one line.

Decoding by speech region caps each decode window at one VAD region, which is what lets
the language follow a code-switch. On real film that region is a single line of dialogue
-- 1.1 s median, measured across 24 monolingual films from the library -- and language
detection on one second of audio is confidently wrong about one time in five. Measured
2026-09-11 with regions decoded one per window and the decoder detecting each: on 19
monolingual films across 12 languages, a **median 20% of the speech** was labelled and
decoded as another language, almost always a neighbour (Russian as Ukrainian, Norwegian as
Swedish, Italian as Spanish) or as English. That is the fracture rate that killed the
coarse language map at 12-24%, arriving through the opposite door.

The fix is the plan's original "run-level hysteresis": detect per region *before*
decoding, then group consecutive regions of one language into a run, and decode each
language in its own call, told its language (run_decoding). A block of regions in a
different language becomes its own run only when it holds enough speech to be believed --
SEGMENT_RUN_MIN_SWITCH_SEC, with the detector confident -- otherwise it is absorbed into
the run around it. A scattered one-second mislabel on a Russian film disappears; a scene in
another language, which is many seconds long, does not.

A run decides the language; it is not the decode window. Each run carries its regions and
the decoder gets those, one clip per region: a window that held a whole run dropped lines
(measured on the film fixture: 5 of 46 lines gone from a forced-language decode of run-sized
clips, none from the same decode of region-sized ones) and put the pause between scenes
back inside the window.

The stress fixture is the boundary case in the other direction: every utterance is a
different language and every one is over 3 s, so each block survives as its own run and
the 0/118 result stands.
"""

from modules.core import config


def build(regions: list, labels: list, file_language: str | None = None) -> list:
    """Group ``regions`` into language runs using ``labels`` from the per-region detector.

    ``labels`` are the spans :func:`segment_languages.for_clips` produced -- a subset of
    the regions, matched by start time, each carrying a language and a confidence. A region
    the detector could not name takes the language of the run it sits in. With no labels at
    all (the detector failed outright), every region is its own run with no language, which
    is exactly the pre-hysteresis behaviour.

    The anchor -- the language a block may return to with no evidence at all -- is the one
    that holds the most labelled speech. For a film in one language that is the right
    language by construction, since four labels in five are right; for a film that switches,
    it is the dominant one, which is what a return should default to. It is deliberately
    *not* the whole-file detection: on a music-heavy excerpt with little dialogue the
    montage read a Polish film as English, and anchoring on that pulled the whole film the
    wrong way. ``file_language`` is only the fallback when nothing is labelled.
    """
    if not regions:
        return []
    labelled = _labelled(regions, labels)
    if not any(span for _, span in labelled):
        return _unlabelled(labelled)
    return _flatten(_merge(_blocks(labelled), _anchor(labelled) or file_language, _labelled_seconds(labelled)))


def _anchor(labelled: list) -> str | None:
    """The language holding the most labelled speech."""
    seconds: dict = {}
    for region, span in labelled:
        if span:
            seconds[span["language"]] = seconds.get(span["language"], 0.0) + float(region["end"]) - float(region["start"])
    return max(seconds, key=seconds.get) if seconds else None


def _labelled_seconds(labelled: list) -> float:
    return sum(float(region["end"]) - float(region["start"]) for region, span in labelled if span)


def _labelled(regions: list, labels: list) -> list:
    by_start = {round(float(span["start"]), 2): span for span in labels or []}
    return [(dict(region), by_start.get(round(float(region["start"]), 2))) for region in regions]


def _unlabelled(labelled: list) -> list:
    return [_run([region], None, 0.0) for region, _ in labelled]


def _flatten(runs: list) -> list:
    return [_run(run["regions"], run["language"], _mean(run["confidences"])) for run in runs]


def _blocks(labelled: list) -> list:
    """Maximal stretches of consecutive regions the detector gave the same language.

    A region with no label continues the block it follows; an unlabelled opening waits for
    the first label and takes it, so the first run is never languageless.
    """
    blocks: list = []
    for region, span in labelled:
        language = _language_of(span)
        if _continues(blocks, language):
            pass
        elif _awaiting_label(blocks):
            blocks[-1]["language"] = language
        else:
            blocks.append({"language": language, "regions": [], "confidences": []})
        _extend(blocks[-1], region, span)
    return blocks


def _language_of(span) -> str | None:
    return span["language"] if span else None


def _awaiting_label(blocks: list) -> bool:
    """An unlabelled opening block, still waiting for its first language."""
    return bool(blocks) and blocks[-1]["language"] is None


def _continues(blocks: list, language) -> bool:
    """Whether a region of ``language`` belongs to the block being built."""
    return bool(blocks) and (language is None or language == blocks[-1]["language"])


def _extend(block: dict, region: dict, span) -> None:
    """Add a region to the block; only a labelled one has a confidence to add.

    An unlabelled region used to count as a zero, which pulled a block's mean confidence
    under LD_MIN_CONFIDENCE: a foreign scene with a breath the detector could not name in
    the middle was judged unsure and absorbed, on evidence that was never measured.
    """
    block["regions"].append(region)
    if span:
        block["confidences"].append(float(span.get("confidence") or 0.0))


def _merge(blocks: list, file_language: str | None, total_speech: float) -> list:
    """Turn blocks into runs, absorbing any block too short or too unsure to be a switch.

    The opening block has no run before it to be absorbed into, so it is judged after the
    rest: a short, unsure block that opens the file -- one Ukrainian-labelled line before
    the Russian starts -- is absorbed into the run that follows it, exactly as it would be
    anywhere else in the file. A believable opening keeps its run.
    """
    runs: list = []
    for block in blocks:
        if runs and _belongs_to(runs[-1], block, file_language, total_speech):
            _absorb(runs[-1], block)
        else:
            runs.append({"language": block["language"], "confidences": list(block["confidences"]), "regions": list(block["regions"])})
    return _without_a_stray_opening(runs, file_language, total_speech)


def _without_a_stray_opening(runs: list, file_language: str | None, total_speech: float) -> list:
    """Fold a short, unsure opening run into the run that follows it."""
    if len(runs) > 1 and not _is_believable_switch(runs[0], file_language, total_speech):
        _absorb(runs[1], runs.pop(0), leading=True)
    return runs


def _belongs_to(run: dict, block: dict, file_language: str | None, total_speech: float) -> bool:
    """Same language as the run -- typically the return after an absorbed mislabel -- or not
    enough behind it to be a switch: either way the block belongs to this run."""
    return block["language"] == run["language"] or not _is_believable_switch(block, file_language, total_speech)


def _absorb(run: dict, block: dict, leading: bool = False) -> None:
    """Fold ``block`` into ``run`` -- after its regions, or before them for an opening block."""
    if leading:
        run["regions"] = block["regions"] + run["regions"]
        run["confidences"] = block["confidences"] + run["confidences"]
    else:
        run["regions"].extend(block["regions"])
        run["confidences"].extend(block["confidences"])


def _is_believable_switch(block: dict, file_language: str | None, total_speech: float) -> bool:
    """Whether a block may start a run of its own: the film's own language always, anything
    else only with enough confident speech behind it.

    "Enough" is SEGMENT_RUN_MIN_SWITCH_SEC on a film, where a two-second block is a slip one
    time in five; on a five-second clip a two-second block is half the file, and no detector
    mislabels half a file by accident. So the bar is the lesser of the seconds and a share of
    the labelled speech (SEGMENT_RUN_MIN_SWITCH_SHARE) -- the six code-switched fixtures are
    two legs of about two seconds each, and the bar in seconds alone absorbed the second leg
    into the first, which is exactly the dropped-leg failure they exist to catch.
    """
    if block["language"] == file_language:
        return True
    speech = sum(float(r["end"]) - float(r["start"]) for r in block["regions"])
    enough = min(config.SEGMENT_RUN_MIN_SWITCH_SEC, config.SEGMENT_RUN_MIN_SWITCH_SHARE * total_speech)
    return speech >= enough and _mean(block["confidences"]) >= config.LD_MIN_CONFIDENCE


def _run(regions: list, language, confidence: float) -> dict:
    """A run: its span and language for the report, its regions for the decoder."""
    return {
        "start": regions[0]["start"],
        "end": regions[-1]["end"],
        "language": language,
        "confidence": confidence,
        "regions": list(regions),
    }


def _mean(values: list) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0
