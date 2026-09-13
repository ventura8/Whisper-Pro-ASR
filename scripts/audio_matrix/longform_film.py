"""The ``film`` long-form profile: laid out from the acoustic structure of real screen dialogue.

The stress grid is a deliberate worst case and the natural profile reproduces real pause
spacing, but both are built from ~6s pangram recordings over near-silence. Real film is
neither. Measured on 2026-09-11 across nine 12-minute excerpts from the library -- five
monolingual, four code-switching, 16 kHz mono, Silero at 0.5 / 250 ms / no padding -- with
the same VAD the fixtures are later scored by, so the comparison is like for like:

    ==============================  ==========  =========  =======
    property (median of excerpts)   real film   natural    stress
    ==============================  ==========  =========  =======
    speech density                  0.42        0.68       0.55
    utterance length, median        1.28 s      2.53 s     2.82 s
    utterance length, p90           4.0 s       3.0 s      3.3 s
    pause between lines, median     0.86 s      0.58 s     0.83 s
    utterances per scene, median    3           8          6
    longest non-speech passage      30-232 s    16.5 s     14.8 s
    music/ambience below speech     2.6 dB      20.7 dB    20.3 dB
    ==============================  ==========  =========  =======

Two of those rows change what a fixture can find. Lines are short -- three-utterance
exchanges, not paragraphs -- which is where per-window language detection has the least
audio to work with. And the bed is *loud*: the score and the room sit 2.6 dB under the
dialogue on a median film, and above it on one of the nine. The synthetic fixtures' quiet
windows were effectively silent, so ``quiet`` at 0/25 was measured against a case real
media never presents.

No audio, text or title from the library is used here. The excerpts contributed numbers;
the voices, lines and beds are all synthesized.
"""

from __future__ import annotations

import random
from pathlib import Path

from scripts.audio_matrix import film_shapes, render

#: Measured medians, with the range across excerpts beside each. Constants below are
#: derived from these and are the only tunable surface; the numbers themselves are data.
FILM_STATS = {
    "speech_density": 0.42,  # 0.18 - 0.75
    "utterance_median_s": 1.28,  # p90 4.0
    "gap_median_s": 0.86,  # 2% under 0.3 s, 24% over 2 s
    "utterances_per_scene": 3,  # 2 - 14
    "silences_over_5s_per_minute": 13 / 12,  # median ~10 s, longest 30-232 s
    "bed_below_speech_db": 2.6,  # -6.1 (louder than speech) to +9.8
}

#: Pause between two lines inside a scene: (weight, low, high). Set below the measured
#: 0.86 s median on purpose: each rendered line carries ~0.4 s of its own leading and
#: trailing silence, which the VAD adds to every gap, so this lands on 0.86 once measured.
#: Capped under 3 s -- the scene-split threshold the measurement uses -- so a long pause
#: inside a scene does not read as a scene boundary.
LINE_GAP_BANDS = ((5, 0.05, 0.20), (50, 0.20, 0.60), (28, 0.60, 1.50), (17, 1.50, 2.90))

#: Non-speech passage between scenes: heavy-tailed, as film is. Half are a few seconds of
#: room; the long tail is credits, action, a montage. A 20-minute clip draws one or two
#: passages over a minute long, which nothing else in the matrix has.
SCENE_GAP_BANDS = ((50, 3.0, 6.0), (30, 6.0, 12.0), (15, 12.0, 30.0), (5, 30.0, 100.0))

#: Lines per scene. The median is 3 -- most scenes are a short exchange -- but the mean is
#: about twice that, because a handful of long dialogue scenes carry most of a film's
#: speech (one excerpt's median was 14). Both halves of that shape are needed: the
#: median for what per-window detection usually gets, the tail for the 0.42 density.
SCENE_LENGTH_WEIGHTS = (
    (1, 15),
    (2, 18),
    (3, 17),
    (4, 12),
    (5, 8),
    (7, 10),
    (11, 8),
    (18, 8),
    (28, 7),
)

#: Share of lines drawn from the short ``role: line`` clips. The remainder are the ~6 s
#: recordings, which supply the p90 tail of long lines that real dialogue also has.
SHORT_LINE_SHARE = 0.88

#: Share of scenes in the clip's dominant language; the rest are passages in another.
DOMINANT_SHARE = 0.65

#: The bed under each scene, as dB *below* the speech level, drawn per scene from the
#: measured range. Negative means the bed is louder than the dialogue, which one excerpt
#: in nine was.
BED_LEVELS_DB = ((-6.0, 1), (0.0, 2), (2.6, 4), (7.0, 2), (9.8, 1))

#: What the bed is made of, chosen per scene. Every source is synthesized by ffmpeg; the
#: chords vary so a 20-minute clip is not one sustained triad.
BED_CHORDS = (
    ("sine=f=220", "sine=f=277.18", "sine=f=329.63"),
    ("sine=f=196", "sine=f=246.94", "sine=f=293.66"),
    ("sine=f=174.61", "sine=f=220", "sine=f=261.63"),
)
BED_NOISE = "anoisesrc=color=brown:amplitude=0.6:seed={seed},lowpass=f=900"
BED_TREMOLO = "tremolo=f=0.35:d=0.5"
#: Every line's gain; see BED_REFERENCE_VOLUME. -6 dB leaves headroom for beds above speech.
LINE_GAIN = 0.25
#: Room noise is broadband and closer to speech than a chord is, so Silero fires on it at
#: levels where music passes; it sits 6 dB under the music in the same scene.
NOISE_RELATIVE = 0.35

#: The ffmpeg ``volume`` at which a bed sits 0 dB below a line rendered at LINE_GAIN.
#: Calibrated by rendering and re-measuring with the script that produced FILM_STATS:
#: 0.30 against gain-1.0 lines measured 16.3 dB below speech, 0.73 against 0.5 measured
#: 9.6, 1.0 against 0.35 measured 5.7, and 1.0 against 0.25 measured 3.4 -- within a
#: decibel of the real median. Lines are rendered quieter rather than the bed louder
#: because three full-scale tones peak at exactly 1.0 here and anything above clips; the
#: ratio is what real film has, and the stress grid already proves lines at 0.25 gain
#: transcribe cleanly.
BED_REFERENCE_VOLUME = 1.0


def group_by_language(sources: list[dict]) -> dict[str, list[dict]]:
    """Clips keyed by language, in source order; shared with the natural planner."""
    by_language: dict[str, list[dict]] = {}
    for clip in sources:
        by_language.setdefault(clip["language"], []).append(clip)
    return by_language


def draw_band(rng: random.Random, bands: tuple) -> float:
    """Draw a value from ``(weight, low, high)`` bands; shared with the natural planner."""
    total = sum(band[0] for band in bands)
    pick = rng.uniform(0, total)
    upto = 0.0
    for weight, low, high in bands:
        upto += weight
        if pick <= upto:
            return round(rng.uniform(low, high), 3)
    return round(rng.uniform(*bands[-1][1:]), 3)


def _scene_length(rng: random.Random) -> int:
    """How many lines this scene has."""
    total = sum(weight for _, weight in SCENE_LENGTH_WEIGHTS)
    pick = rng.uniform(0, total)
    upto = 0
    for count, weight in SCENE_LENGTH_WEIGHTS:
        upto += weight
        if pick <= upto:
            return count
    return SCENE_LENGTH_WEIGHTS[-1][0]


def _bed_level_db(rng: random.Random, levels: tuple = BED_LEVELS_DB) -> float:
    """How far under the dialogue this scene's bed sits, drawn from ``levels``."""
    total = sum(weight for _, weight in levels)
    pick = rng.uniform(0, total)
    upto = 0
    for level, weight in levels:
        upto += weight
        if pick <= upto:
            return level
    return levels[-1][0]


def _split_pool(pool: list[dict]) -> tuple[list[dict], list[dict]]:
    """The short ``role: line`` clips and the long recordings, as two pools."""
    short = [c for c in pool if c.get("role") == "line"]
    long = [c for c in pool if c.get("role") != "line"]
    return short, long


def _pick_line(short: list[dict], long: list[dict], rng: random.Random, index: int) -> dict:
    """A short line most of the time, a long one for the tail."""
    pool = short if (short and (not long or rng.random() < SHORT_LINE_SHARE)) else long
    return pool[index % len(pool)]


def plan(sources: list[dict], rng: random.Random, target_seconds: float, shape: str = "film") -> tuple[list[dict], list[dict]]:
    """Return the block layout and the scene spans the beds are built from.

    Blocks are what :func:`longform._render_blocks` consumes -- speech and gaps, in order.
    Scenes carry ``start``/``end``/``bed_db``/``bed`` so the bed can be rendered per scene
    afterwards, spanning each scene *and the silence that follows it*: the room does not go
    quiet between lines, and the music does not stop when the dialogue does.

    ``shape`` names the passages a real title has around its dialogue -- an opening, a title
    sequence, credits (film_shapes) -- each laid down as silence in the speech track with its
    own bed, at the point in the clip it is due. The plain ``film`` shape has none. A scene
    stops taking lines once the next passage is due, so a long scene cannot carry the clip
    past the credits, and every scheduled passage is laid down even when the dialogue has
    already reached the target: the credits close the clip, whatever the last scene did.
    """
    by_language = group_by_language(sources)
    if not by_language:
        return [], []
    blocks: list[dict] = []
    scenes: list[dict] = []
    due = film_shapes.schedule(shape, rng, target_seconds)
    style = _style(shape)
    cursor = 0.0
    index = 0
    while cursor < target_seconds or due:
        if _passage_due(due, cursor):
            cursor = _passage(blocks, scenes, due.pop(0), cursor)
        else:
            pool = by_language[_scene_language(sorted(by_language), rng)]
            cursor, index, scene = _scene(blocks, pool, rng, cursor=cursor, index=index, limit=_limit(due, target_seconds), style=style)
            scenes.append(scene)
    return blocks, scenes


def _limit(due: list[dict], target_seconds: float) -> float:
    """Where the scene being laid out must stop taking lines: the next passage, or the end."""
    return min(target_seconds, due[0]["at"]) if due else target_seconds


def _style(shape: str) -> dict:
    """The shape's bed levels and scene-gap bands, defaulting to mid-film dialogue's."""
    return {"levels": film_shapes.bed_levels(shape) or BED_LEVELS_DB, "gaps": film_shapes.scene_gap_bands(shape) or SCENE_GAP_BANDS}


def _passage_due(due: list[dict], cursor: float) -> bool:
    return bool(due) and cursor >= due[0]["at"]


def _scene_language(languages: list[str], rng: random.Random) -> str:
    """The dominant language most of the time, another one for the rest."""
    dominant, others = languages[0], languages[1:] or [languages[0]]
    return dominant if rng.random() < DOMINANT_SHARE else rng.choice(others)


def _passage(blocks: list[dict], scenes: list[dict], passage: dict, cursor: float) -> float:
    """Lay down one shape passage: no speech, its own bed, and the ground truth to match."""
    blocks.append({"kind": "gap", "duration": passage["seconds"]})
    end = cursor + passage["seconds"]
    scenes.append(
        {"start": round(cursor, 3), "end": round(end, 3), "bed_db": passage["bed_db"], "bed": passage["bed"], "kind": passage["kind"]}
    )
    return end


def _scene(
    blocks: list[dict], pool: list[dict], rng: random.Random, *, cursor: float, index: int, limit: float, style: dict | None = None
) -> tuple[float, int, dict]:
    """Append one scene -- its lines, their pauses, and the passage after -- and describe it.

    ``style`` carries a shape's own bed levels and scene-gap bands; without it the scene is
    laid out as mid-film dialogue measured.
    """
    style = style or {"levels": BED_LEVELS_DB, "gaps": SCENE_GAP_BANDS}
    start = cursor
    cursor, index = _lines(blocks, pool, rng, cursor=cursor, index=index, limit=limit)
    rest = draw_band(rng, style["gaps"])
    blocks.append({"kind": "gap", "duration": rest})
    cursor += rest
    scene = {
        "start": round(start, 3),
        "end": round(cursor, 3),
        "bed_db": _bed_level_db(rng, style["levels"]),
        "bed": rng.choice(("music", "room", "both")),
    }
    return cursor, index, scene


def _lines(blocks: list[dict], pool: list[dict], rng: random.Random, *, cursor: float, index: int, limit: float) -> tuple[float, int]:
    """Append the lines of one scene, each followed by an in-scene pause."""
    short, long = _split_pool(pool)
    for _ in range(_scene_length(rng)):
        if cursor >= limit:
            break
        clip = _pick_line(short, long, rng, index)
        gap = draw_band(rng, LINE_GAP_BANDS)
        blocks.append({**clip, "kind": "speech", "gain": LINE_GAIN})
        blocks.append({"kind": "gap", "duration": gap})
        cursor += clip["duration"] + gap
        index += 1
    return cursor, index


def _bed_segment(scene: dict, seconds: float, dest: Path, rate: int, seed: int) -> Path:
    """Render one scene's bed at its level, as long as the scene."""
    sources: list[str] = []
    if scene["bed"] in ("music", "both"):
        sources.extend(BED_CHORDS[seed % len(BED_CHORDS)])
    if scene["bed"] in ("room", "both"):
        sources.append(f"{BED_NOISE.format(seed=seed)},volume={NOISE_RELATIVE}")
    # amix sums without normalising, so the level is set per source: three full-scale tones
    # would otherwise peak at 3.0 before the volume stage. The limiter is the backstop for
    # the scenes whose bed sits above the dialogue.
    volume = BED_REFERENCE_VOLUME * (10 ** (-scene["bed_db"] / 20.0)) / len(sources)
    tail = f"{BED_TREMOLO},volume={volume:.4f},alimiter=limit=0.95"
    render.mix(sources, dest, rate, tail=tail, seconds=seconds)
    return dest


def beds(scenes: list[dict], total: float, context: dict, temporaries: list[Path]) -> list[Path]:
    """One continuous bed for the whole clip, built scene by scene.

    Returned as a single-track list so :func:`longform._mix_final` treats it like the
    fixed beds of the other profiles. Every segment path is registered before it is
    rendered, so a failure part-way leaves nothing behind.
    """
    root, rate = context["root"], context["rate"]
    segments: list[Path] = []
    covered = 0.0
    for number, scene in enumerate(scenes):
        seconds = max(min(scene["end"], total) - covered, 0.0)
        if seconds <= 0.0:
            continue
        path = root / f"_lf_bed_{number:04d}.wav"
        temporaries.append(path)
        segments.append(_bed_segment(scene, seconds, path, rate, number))
        covered += seconds
    if covered < total:
        if not scenes:
            raise ValueError(f"no scenes to render a bed from, yet {total:.1f}s of audio to cover")
        path = root / "_lf_bed_tail.wav"
        temporaries.append(path)
        segments.append(_bed_segment(scenes[-1], total - covered, path, rate, len(scenes)))
    bed = root / "_lf_bed.wav"
    temporaries.append(bed)
    render.concat(segments, bed, rate)
    return [bed]
