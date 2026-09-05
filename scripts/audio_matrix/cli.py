"""Command-line orchestration for the audio-matrix generator."""

from __future__ import annotations

import argparse
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

from scripts.audio_matrix import adversarial, cache, combined, longform, manifest, mms, piper, render

logger = logging.getLogger("audio_matrix")

# Sections are generated in this order because later ones consume earlier output: the
# adversarial builders degrade an already-rendered clip, and the long-form timeline is
# assembled from them.
SECTIONS = ("clips", "combined", "adversarial")
COMMANDS = ("all", "core", *SECTIONS, "longform", "verify")


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the generator."""
    parser = argparse.ArgumentParser(description="Generate the multilingual real-audio test matrix.")
    parser.add_argument("command", nargs="?", default="all", choices=COMMANDS, help="what to generate (default: all)")
    parser.add_argument("--out", default=None, help="cache directory (default: $ASR_AUDIO_MATRIX_DIR or test_data/audio_matrix)")
    parser.add_argument("--only", default=None, help="generate a single clip id")
    parser.add_argument("--tier", default=None, choices=("A", "B"), help="restrict to one tier")
    parser.add_argument("--force", action="store_true", help="regenerate even when the cache is fresh")
    parser.add_argument("--strict", action="store_true", help="exit non-zero when tooling is missing")
    return parser


def _tool_versions() -> dict[str, str]:
    """Return the toolchain versions that participate in the cache digest."""
    return {"piper": piper.piper_version(), "mms": mms.mms_version(), "ffmpeg": render.ffmpeg_version()}


def _context(data: dict[str, Any], args: argparse.Namespace) -> dict:
    """Return the shared rendering context: cache root and the pinned audio settings."""
    root = cache.cache_root(args.out)
    root.mkdir(parents=True, exist_ok=True)
    defaults = data["defaults"]
    # Every entry by id, so a derived clip's digest can include the spec of the clip it is
    # built from; see _source_spec.
    by_id = {entry["id"]: entry for section in (*SECTIONS, "clips") for entry in (data.get(section) or []) if entry.get("id")}
    return {"root": root, "rate": int(defaults["sample_rate"]), "pins": defaults["pins"], "defaults": defaults, "by_id": by_id}


def _filters(args: argparse.Namespace, section: str) -> list[Callable[[dict], bool]]:
    """Return the predicates the command-line flags ask an entry to satisfy."""
    checks: list[Callable[[dict], bool]] = []
    if section == "clips":
        checks.append(lambda entry: bool(entry.get("voice")))
    if args.only:
        checks.append(lambda entry: entry["id"] == args.only)
    if args.tier:
        checks.append(lambda entry: entry.get("tier") == args.tier)
    if args.command == "core":
        checks.append(lambda entry: bool(entry.get("committed")))
    return checks


def _selected(entries: list[dict], args: argparse.Namespace, section: str) -> list[dict]:
    """Return the entries the command-line flags select."""
    checks = _filters(args, section)
    return [entry for entry in entries if all(check(entry) for check in checks)]


def _synthesize(entry: dict, raw: Path, context: dict) -> None:
    """Render one clip with whichever engine the manifest assigns it.

    Piper is the default and covers most of the matrix; MMS-TTS supplies the languages
    Piper has no voice for. Both are VITS models reading the same determinism pins, so a
    clip's engine is a property of its language, not a difference in fidelity.
    """
    engine = entry.get("engine", "piper")
    if engine == "mms":
        mms.synth(entry["text"], entry["voice"], raw, context["pins"])
        return
    if engine != "piper":
        raise ValueError(f"Unknown synthesis engine {engine!r} for clip {entry.get('id')!r}")
    model = piper.ensure_voice(context["root"], entry["voice"], entry.get("voice_md5", ""))
    piper.synth(entry["text"], model, raw, context["pins"])


def _render_clip(entry: dict, dest: Path, context: dict) -> None:
    """Synthesize one spoken clip and write it as 16 kHz mono PCM."""
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.wav"
        _synthesize(entry, raw, context)
        gain = float(entry.get("gain", 1.0))
        if gain == 1.0:
            render.to_pcm16_mono(raw, dest, context["rate"])
        else:
            render.apply_gain(raw, dest, gain, context["rate"])


def _render_combined(entry: dict, dest: Path, context: dict) -> None:
    """Build one code-switched clip."""
    combined.build(entry, dest, context)


def _render_adversarial(entry: dict, dest: Path, context: dict) -> None:
    """Build one degraded or malformed artifact."""
    adversarial.build(entry, dest, context)


_RENDERERS: dict[str, Callable[[dict, Path, dict], None]] = {
    "clips": _render_clip,
    "combined": _render_combined,
    "adversarial": _render_adversarial,
}


def _publish_committed(entry: dict, source: Path, context: dict) -> None:
    """Write the committed FLAC copy for an entry marked ``committed``."""
    if not entry.get("committed"):
        return
    manifest.CORE_DIR.mkdir(parents=True, exist_ok=True)
    render.to_flac(source, manifest.CORE_DIR / f"{entry['id']}.flac", context["rate"])


def _committed_copy_is_missing(entry: dict) -> bool:
    """Whether a committed entry's FLAC is absent from the tracked core directory.

    A fresh cache hit returned "cached" without ever looking at the committed copy, so a
    deleted core/*.flac stayed deleted: the generator reported nothing to do and the tests
    that resolve a committed clip skipped. The stamp only says the .wav is current.
    """
    if not entry.get("committed"):
        return False
    return not (manifest.CORE_DIR / f"{entry['id']}.flac").exists()


def _source_spec(entry: dict, context: dict) -> dict | None:
    """Return the manifest entry a derived clip is built from, when it names one.

    An adversarial entry degrades an already-rendered clip named by ``params.source``. Its
    own manifest entry says nothing about what that clip contains, so a digest over the
    entry alone left the derived artifact "fresh" after its source's text or voice changed
    -- a stale degraded clip checked against updated expectations.
    """
    source_id = (entry.get("params") or {}).get("source")
    if not source_id:
        return None
    return context["by_id"].get(source_id)


def _build_one(section: str, entry: dict, context: dict, force: bool) -> str:
    """Generate one entry when stale, returning a one-word status."""
    dest = context["root"] / f"{entry['id']}.wav"
    spec = {"entry": entry, "defaults": context["defaults"]}
    source_spec = _source_spec(entry, context)
    if source_spec is not None:
        spec["source"] = source_spec
    digest = cache.spec_digest(spec, _tool_versions())
    if not force and cache.is_fresh(dest, digest):
        if _committed_copy_is_missing(entry):
            _publish_committed(entry, dest, context)
            logger.info("restored %s", entry["id"])
            return "built"
        return "cached"
    _RENDERERS[section](entry, dest, context)
    cache.write_stamp(dest, digest, {"section": section})
    _publish_committed(entry, dest, context)
    logger.info("built    %s", entry["id"])
    return "built"


def _error_text(error: Exception) -> str:
    """Return a subprocess failure's captured stderr, or the exception's own message."""
    stderr = getattr(error, "stderr", None)
    if isinstance(stderr, bytes):
        return stderr.decode("utf-8", "replace")
    return str(stderr or error)


def _brief(error: Exception) -> str:
    """Return the most useful single line from a subprocess or filesystem failure."""
    lines = [line for line in _error_text(error).splitlines() if line.strip()]
    return lines[-1] if lines else str(error)


def _try_build(section: str, entry: dict, context: dict, force: bool) -> str:
    """Build one entry, reporting a failure instead of aborting the whole run.

    One language whose voice needs an extra phonemizer must not cost you the other forty:
    the run reports what failed and exits non-zero once everything else is done.

    RuntimeError is in the list because that is what mms._prepare_text raises when a
    language needs uroman romanization that is unavailable -- a per-entry tooling gap,
    which was aborting the entire matrix.
    """
    try:
        return _build_one(section, entry, context, force)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        logger.error("FAILED   %s: %s", entry["id"], _brief(error))
        return "failed"


def _build_section(section: str, data: dict[str, Any], args: argparse.Namespace, context: dict) -> int:
    """Build every selected entry in one section, returning the failure count."""
    entries = _selected(list(data.get(section) or []), args, section)
    statuses = [_try_build(section, entry, context, args.force) for entry in entries]
    failed = statuses.count("failed")
    logger.info("%-12s %d ready, %d failed", section, len(entries) - failed, failed)
    return failed


def _longform_sources(data: dict[str, Any], context: dict) -> list[dict]:
    """Return the rendered tier-A clips the long-form timeline is assembled from."""
    wanted = set(data.get("longform", {}).get("languages") or [])
    sources = []
    for entry in manifest.clips(data):
        path = context["root"] / f"{entry['id']}.wav"
        if entry["language"] in wanted and path.exists():
            sources.append({**entry, "path": str(path), "duration": render.probe_duration(path)})
    return sources


def _build_longform(data: dict[str, Any], context: dict) -> int:
    """Build the long-form stress clip, returning the failure count."""
    spec = data.get("longform") or {}
    if not spec:
        logger.info("longform     not configured")
        return 0
    sources = _longform_sources(data, context)
    if not sources:
        logger.error("FAILED   longform: no rendered source clips; generate the clips section first")
        return 1
    # Routed through _try_build like every other entry: a missing spec field or an ffmpeg
    # failure used to escape run() as a traceback instead of the one-line failure the rest
    # of the matrix reports, which also lost the sections that had already succeeded.
    return 1 if _try_build_longform(spec, sources, context) == "failed" else 0


def _try_build_longform(spec: dict, sources: list[dict], context: dict) -> str:
    """Build the long-form timeline, reporting a failure instead of aborting the run.

    Cached like every other entry. The digest covers the spec *and* the source clips it is
    assembled from, because the timeline is 20 minutes of those clips: a changed voice or
    text in any of them makes the existing render stale, while an unchanged set makes
    rebuilding it a pure ~20-minute cost for a byte-identical file.
    """
    dest = context["root"] / f"{spec['id']}.wav"
    digest = cache.spec_digest(
        {"entry": spec, "defaults": context["defaults"], "sources": [s.get("id") for s in sources]},
        _tool_versions(),
    )
    if cache.is_fresh(dest, digest):
        return "cached"
    try:
        timeline = longform.build(sources, dest, context)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        logger.error("FAILED   %s: %s", spec.get("id", "longform"), _brief(error))
        return "failed"
    cache.write_stamp(dest, digest, {"section": "longform"})
    logger.info("built    %s (%.1fs, %d utterances)", spec["id"], timeline["duration"], len(timeline["speech"]))
    return "built"


def _sections_for(command: str) -> tuple[str, ...]:
    """Return the manifest sections a command builds."""
    if command in ("all", "core"):
        return SECTIONS
    return tuple(section for section in (command,) if section in SECTIONS)


def _generate(data: dict[str, Any], args: argparse.Namespace) -> int:
    """Generate everything the command selects, returning a process exit code."""
    context = _context(data, args)
    failed = sum(_build_section(section, data, args, context) for section in _sections_for(args.command))
    if _should_build_longform(args):
        failed += _build_longform(data, context)
    return 1 if failed else 0


def _should_build_longform(args: argparse.Namespace) -> bool:
    """Whether this invocation should build the long-form timeline.

    `all --only <clip>` must not: --only names a single clip, and the long-form build is a
    ~20-minute assembly over every rendered tier-A clip. Regenerating one clip therefore
    paid for the whole timeline, and on a cache with only that clip present it failed with
    "no rendered source clips" -- an error about a thing the caller never asked for.
    An explicit `longform` command is always honoured.
    """
    if args.command == "longform":
        return True
    return args.command == "all" and not args.only


def _covered_languages(entries: list[dict]) -> set[str]:
    """Return the languages the matrix can actually render."""
    return {entry["language"] for entry in entries if entry.get("voice")}


def _declared_gaps(entries: list[dict]) -> dict[str, str]:
    """Return the languages the manifest records as uncoverable, with the reason."""
    return {entry["language"]: entry.get("unsupported_reason", "") for entry in entries if not entry.get("voice")}


def _log_gaps(gaps: dict[str, str]) -> None:
    """Log each declared coverage gap."""
    for language in sorted(gaps):
        logger.info("  gap %-6s %s", language, gaps[language])


def _verify(data: dict[str, Any], _args: argparse.Namespace) -> int:
    """Report language coverage against the service's supported languages."""
    entries = manifest.clips(data)
    covered = _covered_languages(entries)
    gaps = _declared_gaps(entries)
    logger.info("covered languages: %d", len(covered))
    logger.info("declared gaps:     %d", len(gaps))
    _log_gaps(gaps)
    unlisted = sorted(manifest.known_languages() - covered - set(gaps))
    logger.info("not in the manifest at all (%d): %s", len(unlisted), ", ".join(unlisted))
    return 0


def _mms_is_selected(data: dict[str, Any], args: argparse.Namespace) -> bool:
    """Whether any entry this invocation would build is rendered by MMS-TTS."""
    return any(entry.get("engine") == "mms" for entry in _selected(list(data.get("clips") or []), args, "clips"))


def _missing_tools(data: dict[str, Any], args: argparse.Namespace) -> list[str]:
    """Return the names of required tools that are unavailable for this invocation.

    MMS supplies only the languages Piper has no voice for, so a Piper-only regeneration
    (`--only en_core`, or a tier that has no MMS entries) must not be blocked by a missing
    transformers/torch install it would never call.
    """
    missing = [] if piper.piper_available() else ["piper-tts"]
    if _mms_is_selected(data, args) and not mms.mms_available():
        missing.append("transformers+torch (MMS-TTS voices)")
    return missing + ([] if render.ffmpeg_available() else ["ffmpeg"])


def _report_manifest_errors(errors: list[str]) -> None:
    """Log every manifest validation error."""
    for message in errors:
        logger.error("manifest: %s", message)


def run(argv: list[str] | None = None) -> int:
    """Entry point: parse arguments, validate the manifest, dispatch."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    data = manifest.load()
    errors = manifest.validate(data)
    if errors:
        _report_manifest_errors(errors)
        return 2
    if args.command == "verify":
        return _verify(data, args)
    missing = _missing_tools(data, args)
    if missing:
        # Missing tooling is not an error by default: the committed core tier still works
        # without it, and the tests skip the rest with an actionable message.
        logger.warning("skipping generation; missing tooling: %s", ", ".join(missing))
        logger.warning("install it with: poetry install --with tools")
        return 1 if args.strict else 0
    return _generate(data, args)
