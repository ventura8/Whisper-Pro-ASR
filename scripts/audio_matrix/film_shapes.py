"""The passages a real title has around its dialogue, as shapes the film profile can take.

The ``film`` layout is mid-film dialogue: scenes, the pauses between them, a bed under all
of it. A real title is not only that. It opens on a logo and a cold open, cuts to a title
sequence that is music and no speech, and ends on a minute or more of credits -- and every
one of those is a stretch of loud, structured non-speech the decoder is handed, which is
where invented text comes from on real media. Dub tracks are a different case again: a
studio voice over the original mix, cleaner and louder than location dialogue.

A shape is a list of passages inserted into the film layout at a point in the clip: at the
start, after so many seconds of dialogue, or at the end. Each passage is silence in the
speech track (so it is a quiet window in the ground truth) with its own bed level, which
the film profile renders like any scene's bed. The numbers come from the library's
excerpts measured with ``acoustic_stats.py`` -- the same VAD the suite scores by -- and
nothing here is audio, text or a title from that library.
"""

from __future__ import annotations

import random

#: Measured 2026-09-11 on the library, medians per scenario (n): the first ten minutes of 7
#: films (OPEN), the last ten of 6 (END), the first thirty of 14 episodes (EP), ten mid-film
#: minutes of 6 dub tracks (DUB). Seconds of non-speech and the bed under it in dB *below*
#: the dialogue (negative: above it).
#:
#:   ==========================  =======  =======  =======  =======
#:   property (median)           OPEN     END      EP       DUB
#:   ==========================  =======  =======  =======  =======
#:   speech density              0.28     0.04     0.66     0.41
#:   non-speech before line 1    105 s    1.3 s    10.5 s   1.1 s
#:     bed under it              1.8 dB   3.0 dB   0.8 dB   5.9 dB
#:   non-speech after last line  0 s      265 s    --       0.7 s
#:     bed under it              --       1.6 dB   --       4.8 dB
#:   longest passage inside      82 s     66 s     23 s     62 s
#:     bed under it              1.6 dB   -0.4 dB  -2.8 dB  1.6 dB
#:   line length, median         0.99 s   0.97 s   1.87 s   1.28 s
#:   pause between lines         0.90 s   1.2 s    0.70 s   1.03 s
#:   ==========================  =======  =======  =======  =======
#:
#: OPEN ranges 7-330 s before the first line (a logo, a cold open, or a five-minute
#: prologue without dialogue); END 84-567 s of credits. EP is the first thirty minutes,
#: so its tail is mid-episode and says nothing about credits; its title sequence is the
#: passage inside, and it sits *above* the dialogue. DUB is within the mid-film range on
#: every axis -- the studio voice is a timbre, not a layout -- so it has no shape of its own.
FILM_STATS = {
    "OPEN": {"head_s": 105, "head_bed_db": 1.8, "longest_s": 82, "longest_bed_db": 1.6},
    "END": {"tail_s": 265, "tail_bed_db": 1.55, "longest_s": 66, "longest_bed_db": -0.4},
    "EP": {"head_s": 10.5, "head_bed_db": 0.8, "longest_s": 23, "longest_bed_db": -2.8, "density": 0.66},
}

#: Per shape: the passages -- where they go, how long they are (a range drawn per clip),
#: and how far under the dialogue their bed sits (negative: louder than the dialogue) --
#: and, where the dialogue itself sits differently from mid-film dialogue, the per-scene
#: bed levels or scene-gap bands to use instead of the film profile's own. ``film`` is the
#: measured mid-film layout and overrides nothing.
SHAPES: dict[str, dict] = {
    "film": {"passages": ()},
    # A film's two ends in one clip: the opening -- logo and title music before the first
    # line -- a long passage inside, and the credits. Lengths are drawn around the medians,
    # within the measured ranges; the beds sit where OPEN and END measured them, at or
    # just under dialogue level, which is what makes credits and title music the place
    # invented text comes from on real media.
    "bookends": {
        "passages": (
            {"at": "start", "seconds": (60.0, 150.0), "bed_db": 1.8, "bed": "music", "kind": "opening"},
            {"at": "after", "after_seconds": (420.0, 600.0), "seconds": (60.0, 90.0), "bed_db": -0.4, "bed": "both", "kind": "montage"},
            {"at": "end", "seconds": (180.0, 300.0), "bed_db": 1.55, "bed": "music", "kind": "credits"},
        ),
    },
    # A television episode: a short cold open, a title sequence whose music sits above the
    # dialogue, then acts with tighter scene gaps than a film (longest passage 23 s against
    # a film's 30-232 s) -- the first thirty minutes, before any credits.
    "episode": {
        "passages": (
            {"at": "start", "seconds": (5.0, 20.0), "bed_db": 0.8, "bed": "music", "kind": "cold-open"},
            {"at": "after", "after_seconds": (60.0, 180.0), "seconds": (18.0, 30.0), "bed_db": -2.8, "bed": "music", "kind": "title"},
        ),
        "scene_gap_bands": ((55, 3.0, 6.0), (30, 6.0, 12.0), (15, 12.0, 23.0)),
    },
}


def require(shape: str) -> None:
    """Reject a shape name ``SHAPES`` does not know, before anything indexes it.

    A manifest typo used to surface as a bare KeyError from inside the planner, after the
    sources had been rendered; the validation path and the planner now reject the same set.
    """
    if shape not in SHAPES:
        raise ValueError(f"unknown long-form shape {shape!r}; expected one of {', '.join(SHAPES)}")


def scene_gap_bands(shape: str) -> tuple | None:
    """The shape's own passage-between-scenes bands, or None to use the film profile's."""
    return SHAPES[shape].get("scene_gap_bands")


def bed_levels(shape: str) -> tuple | None:
    """The shape's own per-scene bed levels, or None to use the film profile's."""
    return SHAPES[shape].get("bed_levels_db")


def schedule(shape: str, rng: random.Random, target_seconds: float) -> list[dict]:
    """The shape's passages with an absolute ``at`` each, in the order they are due.

    ``at`` is when the planner inserts the passage: 0 for an opening, the drawn offset for
    one placed after some dialogue, and ``target_seconds`` minus its own length for a
    closing one, so the clip still ends near the target.
    """
    require(shape)
    due = []
    for passage in SHAPES[shape]["passages"]:
        seconds = round(rng.uniform(*passage["seconds"]), 3)
        if passage["at"] == "start":
            at = 0.0
        elif passage["at"] == "end":
            at = max(0.0, target_seconds - seconds)
        else:
            # Never past the target: a short clip would otherwise lay dialogue until the
            # drawn offset arrived, and end well over its length.
            at = min(round(rng.uniform(*passage["after_seconds"]), 3), target_seconds)
        due.append({"at": at, "seconds": seconds, "bed_db": passage["bed_db"], "bed": passage["bed"], "kind": passage["kind"]})
    return sorted(due, key=lambda entry: entry["at"])
