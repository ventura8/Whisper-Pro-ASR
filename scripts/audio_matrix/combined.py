"""Builder for code-switched clips: one file, several languages.

Real recordings switch language mid-sentence far more often than test suites admit -- a
bilingual speaker, a dubbed interview, a support call. Each leg is rendered with its own
voice and concatenated with a short joining silence, and the leg boundaries are written to
a sidecar so the tests can assert *which* language appeared *where* rather than only that
something was transcribed.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.audio_matrix import piper, render

JOIN_SILENCE_SECONDS = 0.6


def _render_leg(leg: dict, index: int, context: dict) -> Path:
    """Render one language leg to its own PCM file."""
    dest = context["root"] / f"_leg_{context['entry_id']}_{index}.wav"
    model = piper.ensure_voice(context["root"], leg["voice"], leg.get("voice_md5", ""))
    raw = dest.with_name(dest.name + ".raw.wav")
    piper.synth(leg["text"], model, raw, context["pins"])
    render.to_pcm16_mono(raw, dest, context["rate"])
    raw.unlink(missing_ok=True)
    return dest


def _joiner(context: dict) -> Path:
    """Render the short silence placed between legs."""
    dest = context["root"] / f"_join_{context['entry_id']}.wav"
    render.lavfi(f"anullsrc=r={context['rate']}:cl=mono", dest, context["rate"], seconds=JOIN_SILENCE_SECONDS)
    return dest


def _leg_bounds(parts: list[Path], legs: list[dict]) -> list[dict]:
    """Return the [start, end] window each leg occupies in the concatenated file."""
    bounds: list[dict] = []
    cursor = 0.0
    for index, leg in enumerate(legs):
        # parts alternates leg, joiner, leg, ... so a leg still sits at every even index;
        # only the trailing joiner is gone, which is past the last leg either way.
        duration = render.probe_duration(parts[index * 2])
        bounds.append({"start": round(cursor, 3), "end": round(cursor + duration, 3), "language": leg["language"], "text": leg["text"]})
        cursor += duration + JOIN_SILENCE_SECONDS
    return bounds


def build(entry: dict, dest: Path, context: dict) -> None:
    """Render every leg of a code-switched entry and concatenate them."""
    context = {**context, "entry_id": entry["id"]}
    legs = entry["legs"]
    joiner = _joiner(context)
    parts: list[Path] = []
    # Cleanup is unconditional: `parts` holds only the legs actually rendered, so a failure
    # part-way through still removes them, and `joiner` is removed separately because a
    # single-leg entry never appends it to `parts` and it would otherwise be left behind in
    # the cache directory on every run.
    try:
        for index, leg in enumerate(legs):
            parts.append(_render_leg(leg, index, context))
            # Between legs only. Appending after the last one padded every code-switched clip
            # with a trailing 0.6s of silence that the sidecar's leg bounds do not describe, so
            # a test asserting a leg reaches the end of the file was measuring against a
            # duration the manifest never accounted for.
            if index < len(legs) - 1:
                parts.append(joiner)
        render.concat(parts, dest, context["rate"])
        sidecar = dest.with_suffix(".legs.json")
        sidecar.write_text(json.dumps({"legs": _leg_bounds(parts, legs)}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    finally:
        for part in {*parts, joiner}:
            part.unlink(missing_ok=True)
