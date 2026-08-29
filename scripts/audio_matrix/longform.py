"""Builder for the 20-minute long-form stress clip.

A twenty-minute file is not just a short file repeated: it exercises chunking, VAD across
long pauses, memory over time, and -- most usefully -- the failure mode where a model fills
silence or music with confident hallucinated text. So the timeline deliberately contains
long pauses, short pauses, synthesized music, broadband noise, an ambient hum, varying
loudness, and speech in several languages.

Everything is emitted alongside a ground-truth sidecar (``*.timeline.json``) giving the
exact window each utterance occupies and which windows contain no speech at all, so the
test can assert *where* text should and should not appear rather than eyeballing a blob.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from scripts.audio_matrix import render

TARGET_SECONDS = 1200.0
# Fixed seed: the layout must be irregular (a uniform grid would not stress chunk
# boundaries) but identical on every machine and every run.
LAYOUT_SEED = 20240901
GAIN_CYCLE = (1.0, 0.55, 0.25, 0.8)
SHORT_GAP = (0.4, 1.5)
LONG_GAP = (5.0, 15.0)
# A pause must be comfortably longer than a normal breath before absence of text there is
# evidence of anything.
QUIET_WINDOW_MIN_SECONDS = 8.0

# --- Layout profiles -------------------------------------------------------------------
#
# "stress" is the original layout: every utterance switches language and pauses are
# generous. It is a deliberate worst case and stays as it is -- but it is NOT what real
# media looks like, and reading it as representative produced two wrong conclusions.
#
# "natural" is measured. Subtitle tracks of two genuinely code-switching films
# (Life Is Beautiful, 1549 Italian cues over 116 min; Aferim!, 1386 Romanian cues over
# 98 min) give the real shape of screen dialogue:
#
#   utterance length   median 2.0-2.3s          (stress layout: 6.3s -- 3x too long)
#   gap between lines  median 0.30-0.37s        (stress layout: 0.4-1.5s minimum)
#   gaps >= 0.5s       37-40%                   (so ~60% of real turns are TIGHTER
#                                                than the VAD split threshold)
#   gaps >= 2.0s       14-15%
#   gaps >= 8.0s       5%
#   speech density     48-57% of runtime
#   language switches  sparse and scene-shaped: a film is mostly one language with
#                      localized passages in another, NOT an alternation per line.
#
# Both differences matter and neither is cosmetic. The tight-gap distribution decides
# whether utterance-level VAD segmentation is even possible (at a 500ms split threshold,
# most real turns merge rather than separate). The scene shape decides whether a
# whole-file "is this multilingual" gate is the right abstraction at all.
NATURAL_GAP_BANDS = (
    # (weight, low, high) -- reproduces the measured percentiles above.
    (60, 0.10, 0.50),
    (23, 0.50, 2.00),
    (12, 2.00, 8.00),
    (5, 8.00, 20.00),
)
# Utterances per scene before the language may change. Real films hold a language for a
# scene, not a line; a 20-minute clip should contain a handful of switches, not a hundred.
NATURAL_SCENE_UTTERANCES = (12, 30)
# Share of scenes spoken in the clip's dominant language, mirroring a film that is mostly
# one language with passages in others.
NATURAL_DOMINANT_SHARE = 0.65

MUSIC_TONES = ("sine=f=220", "sine=f=277.18", "sine=f=329.63")
MUSIC_TAIL = "tremolo=f=0.4:d=0.6,volume=0.14"
NOISE_SOURCE = "anoisesrc=color=brown:amplitude=0.6:seed=101,lowpass=f=900,volume=0.18"
HUM_SOURCE = "sine=f=50,volume=0.05"
MUSIC_BED_SECONDS = 45.0
NOISE_BED_SECONDS = 60.0


def _silence(seconds: float, dest: Path, rate: int) -> Path:
    """Render ``seconds`` of digital silence."""
    render.lavfi(f"anullsrc=r={rate}:cl=mono", dest, rate, seconds=seconds)
    return dest


def _speech_block(clip: Path, gain: float, dest: Path, rate: int) -> Path:
    """Render one utterance at a given loudness."""
    render.apply_gain(clip, dest, gain, rate)
    return dest


def _gap_seconds(rng: random.Random, index: int) -> float:
    """Return the next pause length, alternating between short and long pauses."""
    low, high = LONG_GAP if index % 3 == 2 else SHORT_GAP
    return round(rng.uniform(low, high), 3)


def _natural_gap_seconds(rng: random.Random) -> float:
    """Return a pause drawn from the gap distribution measured on real film dialogue."""
    total = sum(band[0] for band in NATURAL_GAP_BANDS)
    pick = rng.uniform(0, total)
    upto = 0.0
    for weight, low, high in NATURAL_GAP_BANDS:
        upto += weight
        if pick <= upto:
            return round(rng.uniform(low, high), 3)
    return round(rng.uniform(*NATURAL_GAP_BANDS[-1][1:]), 3)


def _plan(sources: list[dict], rng: random.Random) -> list[dict]:
    """Return the block layout: alternating speech and pauses until the target length."""
    blocks: list[dict] = []
    cursor = 0.0
    index = 0
    while cursor < TARGET_SECONDS:
        clip = sources[index % len(sources)]
        # The cycled gain must come after **clip: the source entry carries its own "gain"
        # (1.0 for a normally rendered clip) and would otherwise silence the variation.
        blocks.append({**clip, "kind": "speech", "gain": GAIN_CYCLE[index % len(GAIN_CYCLE)]})
        cursor += clip["duration"]
        gap = _gap_seconds(rng, index)
        blocks.append({"kind": "gap", "duration": gap})
        cursor += gap
        index += 1
    return blocks


def _plan_natural(sources: list[dict], rng: random.Random) -> list[dict]:
    """Return a scene-shaped layout with the pause distribution of real screen dialogue.

    Differs from :func:`_plan` in the two ways real media differs from a stress grid:
    language is held for a whole scene rather than swapped every line, and pauses come
    from the measured distribution -- most of them tighter than any plausible VAD split
    threshold. See NATURAL_GAP_BANDS for where the numbers come from.
    """
    by_language: dict[str, list[dict]] = {}
    for clip in sources:
        by_language.setdefault(clip["language"], []).append(clip)
    languages = sorted(by_language)
    if not languages:
        return []
    dominant = languages[0]
    others = languages[1:] or [dominant]

    blocks: list[dict] = []
    cursor = 0.0
    index = 0
    while cursor < TARGET_SECONDS:
        pool = by_language[_pick_scene_language(dominant, others, rng)]
        cursor, index = _append_scene(blocks, pool, cursor=cursor, index=index, rng=rng)
    return blocks


def _pick_scene_language(dominant: str, others: list[str], rng: random.Random) -> str:
    """The language this scene is spoken in: usually the dominant one, sometimes another."""
    return dominant if rng.random() < NATURAL_DOMINANT_SHARE else rng.choice(others)


def _append_scene(blocks: list[dict], pool: list[dict], *, cursor: float, index: int, rng: random.Random) -> tuple[float, int]:
    """Append one scene's utterances and their gaps, returning the advanced cursor and index.

    A scene is several consecutive utterances in one language -- which is what separates
    this layout from the stress grid, where the language changes every line.
    """
    for _ in range(rng.randint(*NATURAL_SCENE_UTTERANCES)):
        if cursor >= TARGET_SECONDS:
            break
        clip = pool[index % len(pool)]
        blocks.append({**clip, "kind": "speech", "gain": GAIN_CYCLE[index % len(GAIN_CYCLE)]})
        gap = _natural_gap_seconds(rng)
        blocks.append({"kind": "gap", "duration": gap})
        cursor += clip["duration"] + gap
        index += 1
    return cursor, index


def _render_blocks(blocks: list[dict], context: dict) -> tuple[list[Path], list[dict], list[dict]]:
    """Render every block, returning the parts plus speech and quiet ground truth."""
    parts: list[Path] = []
    speech: list[dict] = []
    quiet: list[dict] = []
    cursor = 0.0
    for index, block in enumerate(blocks):
        dest = context["root"] / f"_lf_{index:04d}.wav"
        if block["kind"] == "speech":
            parts.append(_speech_block(Path(block["path"]), block["gain"], dest, context["rate"]))
            speech.append({"start": round(cursor, 3), "end": round(cursor + block["duration"], 3), **_speech_meta(block)})
        else:
            parts.append(_silence(block["duration"], dest, context["rate"]))
            if block["duration"] >= QUIET_WINDOW_MIN_SECONDS:
                quiet.append({"start": round(cursor, 3), "end": round(cursor + block["duration"], 3), "kind": "pause"})
        cursor += block["duration"]
    return parts, speech, quiet


def _speech_meta(block: dict) -> dict:
    """Return the ground-truth fields describing one spoken block."""
    return {"language": block["language"], "text": block["text"], "expect_words": block["expect_words"], "gain": block["gain"]}


def _bed(source: str, seconds: float, dest: Path, rate: int) -> Path:
    """Render one background bed, bounded on the output side."""
    render.lavfi(source, dest, rate, seconds=seconds)
    return dest


def _music_bed(seconds: float, dest: Path, rate: int) -> Path:
    """Render the synthesized music bed: a slow tremolo triad, not a copyrighted recording."""
    render.mix(list(MUSIC_TONES), dest, rate, tail=MUSIC_TAIL, seconds=seconds)
    return dest


def _placed_bed(body: Path, start: float, total: float, dest: Path, root: Path, rate: int) -> Path:
    """Place ``body`` at ``start`` inside an otherwise silent full-length track."""
    head = _silence(start, root / f"{dest.stem}_head.wav", rate)
    tail_seconds = max(total - start - render.probe_duration(body), 0.01)
    tail = _silence(tail_seconds, root / f"{dest.stem}_tail.wav", rate)
    render.concat([head, body, tail], dest, rate)
    head.unlink(missing_ok=True)
    tail.unlink(missing_ok=True)
    return dest


def _build_beds(total: float, context: dict) -> list[Path]:
    """Render the music, noise and hum beds, positioned over the timeline."""
    root, rate = context["root"], context["rate"]
    music_body = _music_bed(MUSIC_BED_SECONDS, root / "_lf_music_body.wav", rate)
    noise_body = _bed(NOISE_SOURCE, NOISE_BED_SECONDS, root / "_lf_noise_body.wav", rate)
    hum = _bed(HUM_SOURCE, total, root / "_lf_hum.wav", rate)
    music = _placed_bed(music_body, total * 0.25, total, root / "_lf_music.wav", root, rate)
    noise = _placed_bed(noise_body, total * 0.6, total, root / "_lf_noise.wav", root, rate)
    music_body.unlink(missing_ok=True)
    noise_body.unlink(missing_ok=True)
    return [music, noise, hum]


def _mix_final(speech_track: Path, beds: list[Path], dest: Path, rate: int) -> None:
    """Mix the speech track with the beds and limit the result to avoid clipping."""
    inputs: list[str] = []
    for path in [speech_track, *beds]:
        inputs.extend(["-i", str(path)])
    chain = f"amix=inputs={len(beds) + 1}:duration=first:normalize=0,alimiter=limit=0.95"
    render.run_ffmpeg([*inputs, "-filter_complex", chain, "-ac", "1", "-ar", str(rate), "-c:a", "pcm_s16le", str(dest)])


def build(sources: list[dict], dest: Path, context: dict, profile: str = "stress") -> dict:
    """Build the long-form clip and return its ground-truth timeline.

    ``profile`` selects the layout: "stress" for the original worst case (a language
    change on every utterance, generous pauses) or "natural" for the scene-shaped,
    tight-pause layout measured from real film subtitle tracks. Keep both -- the stress
    layout is where a decoder's language handling breaks most visibly, and the natural
    layout is the only one that says whether a fix survives contact with real dialogue.
    """
    rng = random.Random(LAYOUT_SEED)
    planner = _plan_natural if profile == "natural" else _plan
    parts, speech, quiet = _render_blocks(planner(sources, rng), context)
    speech_track = context["root"] / "_lf_speech.wav"
    render.concat(parts, speech_track, context["rate"])
    total = render.probe_duration(speech_track)
    beds = _build_beds(total, context)
    _mix_final(speech_track, beds, dest, context["rate"])
    for path in [*parts, *beds, speech_track]:
        path.unlink(missing_ok=True)
    timeline = {"duration": round(total, 3), "speech": speech, "quiet_windows": quiet}
    dest.with_suffix(".timeline.json").write_text(json.dumps(timeline, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return timeline
