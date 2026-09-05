#!/usr/bin/env python3
"""Generate the agent-facing catalog of every audio file this repo ships or generates.

Writes ``docs/AUDIO_CATALOG.md`` from ``tests/e2e/fixtures/audio_matrix/manifest.json``
plus whatever is currently on disk. Run ``--check`` to fail when the committed catalog is
stale. See ``tests/e2e/fixtures/audio_matrix/README.md`` for the generator workflow.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("audio_catalog")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Imported after the path bootstrap above: this file runs as a script, so `scripts` is not
# an importable package until ROOT is on sys.path.
from scripts.audio_matrix.render import FFMPEG_TIMEOUT_SEC  # noqa: E402  # pylint: disable=wrong-import-position

MANIFEST_PATH = ROOT / "tests" / "e2e" / "fixtures" / "audio_matrix" / "manifest.json"
COMMITTED_DIR = ROOT / "tests" / "e2e" / "fixtures" / "audio_matrix" / "core"
GENERATED_DIR = ROOT / "test_data" / "audio_matrix"
OUTPUT_PATH = ROOT / "docs" / "AUDIO_CATALOG.md"

# Loose fixtures that predate the matrix and are not described by the manifest.
STANDALONE = [
    {
        "path": "tests/e2e/fixtures/speech_known_text.wav",
        "what": (
            "Real English speech saying *The quick brown fox jumps over the lazy dog.* and "
            "*Whisper Pro ASR is running a hardware acceleration test on this machine.*"
        ),
        "used_by": "tests/integration/test_transcription_accuracy.py (RUN_REAL_ASR=1)",
    },
    {
        "path": "tests/e2e/fixtures/silence.wav",
        "what": "A short silent clip used as an upload payload where the transcript is irrelevant.",
        "used_by": "tests/e2e/real/dashboard-lifecycle-real.spec.cjs",
    },
]

# What each adversarial builder actually produces, so an agent picking a fixture does not
# have to read scripts/audio_matrix/adversarial.py first.
BUILDER_DESCRIPTIONS = {
    "silence": "Digital silence for {seconds}s -- must not hang or hallucinate speech.",
    "noise": "White noise for {seconds}s -- decoder must not lock up on non-speech.",
    "tones": "Pure tones for {seconds}s -- musical, speech-free input.",
    "clipped": "`{source}` amplified {gain}x into hard clipping -- distorted speech.",
    "quiet": "`{source}` attenuated to {gain} of full scale -- near-inaudible speech.",
    "telephone": "`{source}` band-limited to {rate} Hz telephone quality.",
    "stereo": "`{source}` duplicated to two channels -- exercises downmixing.",
    "resampled": "`{source}` resampled to {rate} Hz -- exercises rate conversion.",
    "speech_after_silence": "{lead_seconds}s of silence followed by `{source}` -- late speech onset.",
    "tiny": "A {seconds}s clip, shorter than one decode window.",
    "truncated_header": "`{source}` with a truncated WAV header -- malformed container.",
    "zero_byte": "A zero-byte file with a .wav name -- empty upload.",
    "mp3_named_wav": "`{source}` encoded as MP3 but named .wav -- extension/content mismatch.",
}


def _probe(path: Path) -> dict | None:
    """Return duration, sample rate and channels for ``path``, or None if unreadable."""
    if not path.is_file() or not shutil.which("ffprobe"):
        return None
    raw = _ffprobe_json(path)
    if raw is None:
        return None
    data = json.loads(raw)
    stream = (data.get("streams") or [{}])[0]
    return {
        "duration": _format_seconds(data.get("format", {}).get("duration")),
        "rate": stream.get("sample_rate", "-"),
        "channels": stream.get("channels", "-"),
        "size": f"{path.stat().st_size / 1024:.0f} KB",
    }


def _ffprobe_json(path: Path) -> str | None:
    """Raw ffprobe JSON for one file, or None when it cannot be read.

    Bounded by the same ceiling every other ffmpeg call here uses: an unbounded ffprobe on a
    malformed or truncated fixture hangs the catalog run forever rather than reporting the
    one file it could not measure. Empty output is a failure too -- json.loads would raise
    on it, turning an unreadable file into a traceback instead of a dash in the table.
    """
    try:
        raw = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=sample_rate,channels:format=duration",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=FFMPEG_TIMEOUT_SEC,
        ).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return raw if raw.strip() else None


def _format_seconds(duration: str | None) -> str:
    """A duration rendered for the table, or a dash when ffprobe reported none."""
    return f"{float(duration):.1f}s" if duration else "-"


def _locate(entry_id: str, committed: bool) -> tuple[Path, str]:
    """Return the clip's path and whether it is committed or rendered on demand."""
    if committed:
        return COMMITTED_DIR / f"{entry_id}.flac", "committed"
    return GENERATED_DIR / f"{entry_id}.wav", "on demand"


def _facts(path: Path, committed: bool = True) -> str:
    """Return a one-cell summary of a clip's measured properties.

    Only committed clips are probed. Generated clips are absent on a fresh checkout, so
    measuring them would make the catalog differ per machine and ``--check`` meaningless.
    """
    probed = _probe(path) if committed else None
    if not probed:
        return "-"
    return f"{probed['duration']}, {probed['rate']} Hz, {probed['channels']} ch, {probed['size']}"


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _describe_adversarial(entry: dict) -> str:
    template = BUILDER_DESCRIPTIONS.get(entry["builder"], entry["builder"])
    try:
        return template.format(**entry.get("params", {}))
    except KeyError:
        return template


def _flag_notes(entry: dict) -> str:
    notes = []
    if entry.get("smoke"):
        notes.append("smoke")
    if entry.get("xfail_reason"):
        notes.append(f"xfail: {entry['xfail_reason']}")
    return "; ".join(notes) or "-"


def _cell(text: str) -> str:
    """Escape a value so it cannot break out of a Markdown table row."""
    return str(text).replace("|", "\\|").replace("\n", " ")


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(_cell(value) for value in row) + " |" for row in rows]
    return lines + [""]


def _speech_rows(clips: list[dict]) -> list[list[str]]:
    rows = []
    for entry in clips:
        path, state = _locate(entry["id"], entry.get("committed", False))
        if entry.get("voice") is None:
            reason = entry.get("unsupported_reason", "no Piper voice")
            rows.append(
                [
                    entry["id"],
                    entry["language"],
                    entry.get("tier", "-"),
                    "unavailable",
                    f"Coverage gap, no clip rendered: {reason}.",
                    "-",
                    "-",
                ]
            )
            continue
        rows.append(
            [
                entry["id"],
                entry["language"],
                entry.get("tier", "-"),
                state,
                f'Spoken by `{entry["voice"]}`: "{entry["text"]}"',
                _facts(path, entry.get("committed", False)),
                _flag_notes(entry),
            ]
        )
    return rows


def _combined_rows(entries: list[dict]) -> list[list[str]]:
    rows = []
    for entry in entries:
        path, state = _locate(entry["id"], entry.get("committed", False))
        legs = " then ".join(f'{leg["language"]}: "{leg["text"]}"' for leg in entry["legs"])
        rows.append([entry["id"], state, legs, _facts(path, entry.get("committed", False)), _flag_notes(entry)])
    return rows


def _adversarial_rows(entries: list[dict]) -> list[list[str]]:
    rows = []
    for entry in entries:
        path, state = _locate(entry["id"], entry.get("committed", False))
        expect = entry.get("expect", {})
        accepted = ", ".join(str(code) for code in expect.get("status_in", []))
        rows.append(
            [
                entry["id"],
                state,
                _describe_adversarial(entry),
                _facts(path, entry.get("committed", False)),
                f"HTTP {accepted}; text {expect.get('text_policy', 'any')}",
                _flag_notes(entry),
            ]
        )
    return rows


def _standalone_rows() -> list[list[str]]:
    rows = []
    for item in STANDALONE:
        path = ROOT / item["path"]
        rows.append(
            [
                f"`{item['path']}`",
                "present" if path.is_file() else "missing",
                item["what"],
                _facts(path),
                item["used_by"],
            ]
        )
    return rows


def _split_by_tier(clips: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split the clip list into the committed tier-A core and the generated long tail."""
    return [c for c in clips if c.get("tier") == "A"], [c for c in clips if c.get("tier") != "A"]


def render(data: dict) -> str:
    """Return the full catalog document."""
    clips = data["clips"]
    core, tail = _split_by_tier(clips)
    longform = data["longform"]

    lines = [
        "# Audio catalog",
        "",
        "<!-- Generated by scripts/audio_catalog.py. Do not edit by hand; run the script. -->",
        "",
        "Every audio file this repository ships or generates, and what is in it. Written for",
        "agents choosing a fixture: pick from here rather than guessing a filename or reading",
        "the generator.",
        "",
        "- Committed clips live in `tests/e2e/fixtures/audio_matrix/core/` (FLAC) and are always present.",
        "- Everything else renders on demand into `test_data/audio_matrix/` (WAV, gitignored) via",
        "  `scripts/generate_fixtures_docker.sh all`, and `scripts/generate_fixtures_docker.sh verify`",
        "  reports language coverage. Both run the generator in the test image, which is the",
        "  supported path: the TTS toolchain is heavy, and on a host with an externally-managed",
        "  system Python `poetry install --with tools` fails outright trying to uninstall pip.",
        "- `python3 scripts/generate_audio_matrix.py all` is the same generator run directly, and",
        "  works only where piper-tts, transformers+torch and ffmpeg are already installed.",
        "- Expectations are data: tune a language by editing `manifest.json`, never a test.",
        '- `smoke` marks the representative subset run by `-m "real_audio and smoke"`.',
        "- The Measured column is filled in for committed clips only; on-demand clips are absent",
        "  on a fresh checkout, so measuring them would make this file machine-dependent.",
        "",
        "## Standalone fixtures",
        "",
    ]
    lines += _table(["File", "State", "Contents", "Measured", "Used by"], _standalone_rows())

    lines += ["## Core speech clips (tier A, committed)", ""]
    lines += _table(["ID", "Lang", "Tier", "State", "Contents", "Measured", "Notes"], _speech_rows(core))

    lines += [
        "## Tail speech clips (generated on demand)",
        "",
        "One clip per additional supported language. Entries with no Piper voice are listed so",
        "the coverage gap stays visible.",
        "",
    ]
    lines += _table(["ID", "Lang", "Tier", "State", "Contents", "Measured", "Notes"], _speech_rows(tail))

    lines += [
        "## Code-switched clips",
        "",
        "Two languages spoken back to back in one file, for language-detection behaviour.",
        "",
    ]
    lines += _table(["ID", "State", "Legs", "Measured", "Notes"], _combined_rows(data["combined"]))

    lines += [
        "## Adversarial and degraded clips",
        "",
        "Non-speech, damaged and awkwardly encoded audio. These assert the service responds and",
        "does not hang; a correct transcript is not always expected.",
        "",
    ]
    lines += _table(
        ["ID", "State", "Contents", "Measured", "Accepted response", "Notes"],
        _adversarial_rows(data["adversarial"]),
    )

    path, _ = _locate(longform["id"], longform.get("committed", False))
    lines += [
        "## Long-form stress clip",
        "",
        f"`{longform['id']}` -- rendered on demand. About {longform['target_seconds'] // 60} minutes of speech in "
        f"{len(longform['languages'])} languages ({', '.join(longform['languages'])}), interleaved with "
        "silence, music and noise beds. Rendered to " + _rel(path) + ".",
        "",
        "Known defects it currently reproduces:",
        "",
    ]
    lines += [f"- **{name}** -- {detail}" for name, detail in longform["known_defects"].items()]
    lines += [""]
    return "\n".join(lines)


def run() -> int:
    """Write the catalog, or verify the committed copy is current."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit 1 if the catalog is stale")
    args = parser.parse_args()

    # Checked here rather than left to _probe, which answers None per file. Without ffprobe
    # every Measured cell becomes "-", so writing would silently strip the measurements from
    # the committed catalog and --check would fail with "stale" -- naming the wrong problem.
    if not shutil.which("ffprobe"):
        logger.error("ffprobe is not on PATH; the catalog's Measured column cannot be filled in.")
        logger.error("Install ffmpeg, or run this through scripts/generate_fixtures_docker.sh.")
        return 1

    document = render(json.loads(MANIFEST_PATH.read_text(encoding="utf-8")))
    if args.check:
        current = OUTPUT_PATH.read_text(encoding="utf-8") if OUTPUT_PATH.is_file() else ""
        if current != document:
            logger.error("%s is stale; run python3 %s", _rel(OUTPUT_PATH), _rel(Path(__file__)))
            return 1
        logger.info("%s is up to date", _rel(OUTPUT_PATH))
        return 0
    OUTPUT_PATH.write_text(document, encoding="utf-8")
    logger.info("wrote %s", _rel(OUTPUT_PATH))
    return 0


if __name__ == "__main__":
    sys.exit(run())
