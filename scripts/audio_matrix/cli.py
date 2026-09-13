"""Command-line orchestration for the audio-matrix generator."""

from __future__ import annotations

import argparse
import functools
import importlib
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


@functools.lru_cache(maxsize=1)
def _tool_versions() -> dict[str, str]:
    """Return the toolchain versions that participate in the cache digest.

    Every library whose output ends up in a rendered clip belongs here, or a toolchain
    upgrade that changes the audio leaves the cache reporting it as fresh. torch and uroman
    are in the list because MMS synthesis runs the VITS model through torch and romanizes
    its input with uroman -- both shape the waveform, and neither is covered by the
    transformers version that ``mms.mms_version()`` reports.
    """
    return {
        "piper": piper.piper_version(),
        "mms": mms.mms_version(),
        "torch": _module_version("torch"),
        "uroman": _module_version("uroman"),
        "ffmpeg": render.ffmpeg_version(),
    }


def _module_version(name: str) -> str:
    """An installed module's version, or "missing" when it is not present.

    Absence is recorded rather than skipped: a clip rendered without uroman is not the same
    artifact as one rendered with it, so the two must not share a digest.
    """
    try:
        return str(getattr(importlib.import_module(name), "__version__", "unknown"))
    except ImportError:
        return "missing"


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
    _render_atomically(section, entry, dest, context)
    cache.write_stamp(dest, digest, {"section": section})
    _publish_committed(entry, dest, context)
    logger.info("built    %s", entry["id"])
    return "built"


def _render_atomically(section: str, entry: dict, dest: Path, context: dict) -> None:
    """Render to a staging file and move it into place only once it is complete.

    A renderer that fails part-way used to leave a truncated file at ``dest``. The stamp is
    written after, so the cache correctly called it stale -- but every consumer that resolves
    a clip by path (the tests, and the adversarial builders that degrade an existing clip)
    read the partial audio in the meantime. A rename within one directory is atomic, so a
    reader sees either the previous render or the new one, never a half-written one.
    """
    # ".partial" goes BEFORE the extension, not after. ffmpeg chooses its muxer from the
    # filename extension, so a staging path of "<id>.wav.partial" made it fail every render
    # with "Unable to choose an output format ... use a standard extension" -- which is
    # exactly what it did: atomic staging landed after the fixtures were last generated, so
    # nothing exercised it until the next regeneration, and then every single clip failed.
    staged = dest.with_name(f"{dest.stem}.partial{dest.suffix}")
    # A renderer may also emit a sidecar (combined.build writes the code-switched leg
    # bounds). It is staged beside the audio and published only after it, so a reader can
    # never see a sidecar describing audio that was not written -- and a failed render
    # leaves neither behind.
    staged_sidecar = staged.with_name(staged.name + ".legs.json")
    try:
        _RENDERERS[section](entry, staged, context)
        staged.replace(dest)
        if staged_sidecar.exists():
            staged_sidecar.replace(dest.with_suffix(".legs.json"))
    finally:
        staged.unlink(missing_ok=True)
        staged_sidecar.unlink(missing_ok=True)


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


def _longform_sources(data: dict[str, Any], context: dict, *, profile: str, languages: list[str] | None = None) -> list[dict]:
    """Return the rendered clips the long-form timeline is assembled from.

    The stress profile takes exactly one clip per language and the natural profile takes all
    of them, and that difference is load-bearing rather than cosmetic.

    Stress alternates language on every utterance, which it does by cycling its source list --
    so a list holding three clips per language stops alternating and starts repeating a
    language two or three times in a row. That happened: adding clips took the stress fixture
    from 118 utterances switching at every boundary to 117 switching at 78 of 116, silently
    changing the artifact every recorded `windows` measurement is stated against. One clip per
    language keeps that fixture byte-identical to the one those numbers describe.

    Natural wants the opposite. Its scenes are a dozen to thirty consecutive utterances in one
    language, so with a single source a scene is one sentence repeated up to thirty times --
    which trips the repetition filter and measures that filter rather than language handling.

    Film takes everything, including the ``role: line`` clips -- short single lines of
    dialogue that reproduce the ~1.3s utterances real film is made of. The other two profiles
    exclude those on purpose: natural takes every clip of a language, so adding the lines to
    the manifest would otherwise have changed the natural fixture under every number recorded
    against it, exactly as adding the scene clips once changed the stress grid.

    ``languages`` is the variant's own list; without one the top-level spec's applies. The
    three original variants all name the same ten, so they are unchanged by this -- it exists
    so a variant can be built in one language, which is what most of the library is.
    """
    wanted = _top_level_languages(data) if languages is None else list(languages)
    sources = _eligible(profile, _rendered_clips_in(data, context, wanted))
    _require_every_language(wanted, sources)
    return sources


def _eligible(profile: str, sources: list[dict]) -> list[dict]:
    """The rendered clips a profile lays out from: film takes all, the others no lines,
    and stress one clip per language."""
    if profile == "film":
        return sources
    sources = [clip for clip in sources if clip.get("role") != "line"]
    return sources if profile == "natural" else _one_per_language(sources)


def _require_every_language(wanted: list[str], sources: list[dict]) -> None:
    """A language the spec asks for and no eligible rendered clip supplies is an error.

    Silently dropped, the clip laid out without it would still carry the variant's name, and
    every number stated against the fixture would describe audio that lacks a language the
    manifest says it has.
    """
    missing = sorted(set(wanted) - {clip["language"] for clip in sources})
    if missing:
        raise ValueError(f"no eligible rendered clip for long-form language(s): {', '.join(missing)}")


def _rendered_clips_in(data: dict[str, Any], context: dict, languages: list[str] | None = None) -> list[dict]:
    """Every rendered clip whose language the long-form spec, or ``languages``, asks for.

    A variant's own list wins whenever it gives one, an empty list included: ``[]`` asks for
    no languages and gets no sources, so the build fails where it says why, rather than
    quietly rendering the top-level set under the variant's name.
    """
    wanted = set(_top_level_languages(data) if languages is None else languages)
    rendered = []
    for entry in manifest.clips(data):
        path = context["root"] / f"{entry['id']}.wav"
        if entry["language"] in wanted and path.exists():
            rendered.append({**entry, "path": str(path), "duration": render.probe_duration(path)})
    return rendered


def _top_level_languages(data: dict[str, Any]) -> list[str]:
    """The languages the top-level long-form spec names; what a variant without its own gets."""
    return list(data.get("longform", {}).get("languages") or [])


def _one_per_language(sources: list[dict]) -> list[dict]:
    """The first rendered clip of each language, in the manifest's own order."""
    seen: dict[str, dict] = {}
    for entry in sources:
        seen.setdefault(entry["language"], entry)
    return list(seen.values())


def _build_longform(data: dict[str, Any], context: dict) -> int:
    """Build every long-form variant, returning the total failure count.

    The stress grid and the scene-shaped natural clip measure different things and neither
    replaces the other: stress switches language on every utterance, which is a deliberate
    worst case, while natural reproduces the pause distribution and scene shape of real
    screen dialogue.
    """
    spec = data.get("longform") or {}
    if not spec:
        logger.info("longform     not configured")
        return 0
    failures = 0
    for entry in [spec, *(spec.get("variants") or [])]:
        failures += _build_one_longform(entry, data, context)
    return failures


def _build_one_longform(spec: dict, data: dict[str, Any], context: dict) -> int:
    """Build a single long-form variant, returning the failure count."""
    try:
        sources = _longform_sources(data, context, profile=str(spec.get("profile") or "stress"), languages=spec.get("languages"))
    except ValueError as exc:
        # A language the spec names with nothing rendered for it: reported as this variant's
        # failure, like every other one, rather than as a traceback that loses the sections
        # and variants already built.
        logger.error("FAILED   longform %s: %s", spec.get("id", "<no id>"), exc)
        return 1
    if not sources:
        logger.error("FAILED   longform: no rendered source clips; generate the clips section first")
        return 1
    # Routed through _try_build like every other entry: a missing spec field or an ffmpeg
    # failure used to escape run() as a traceback instead of the one-line failure the rest
    # of the matrix reports, which also lost the sections that had already succeeded.
    return 1 if _try_build_longform(spec, sources, context) == "failed" else 0


def _longform_digest(spec: dict, sources: list[dict], context: dict) -> str:
    """The cache digest for the long-form timeline.

    Over the full source specs, not their ids. Ids are stable across every edit that matters
    here -- changing a clip's text, voice or gain leaves its id alone -- so an id-only digest
    reported a 20-minute timeline as fresh while the utterances inside it had changed, and
    the ground-truth sidecar then described audio that was no longer there.

    ``path`` and ``duration`` are dropped: they are properties of this machine's cache
    directory, so including them would make the digest differ between checkouts of the same
    manifest and force a needless 20-minute rebuild on every fresh clone. ``variants`` is
    dropped too: the top-level spec carries the list of every other variant, none of which
    reaches ``longform.build`` for the base clip, so adding a variant was rebuilding the
    stress grid for a byte-identical file.
    """
    entry = _without(spec, ("variants",))
    source_specs = [_without(source, ("path", "duration")) for source in sources]
    return cache.spec_digest({"entry": entry, "defaults": context["defaults"], "sources": source_specs}, _tool_versions())


def _without(mapping: dict, keys: tuple[str, ...]) -> dict:
    """A copy of ``mapping`` without ``keys``."""
    return {key: value for key, value in mapping.items() if key not in keys}


def _try_build_longform(spec: dict, sources: list[dict], context: dict) -> str:
    """Build the long-form timeline, reporting a failure instead of aborting the run.

    Cached like every other entry. The digest covers the spec *and* the source clips it is
    assembled from, because the timeline is 20 minutes of those clips: a changed voice or
    text in any of them makes the existing render stale, while an unchanged set makes
    rebuilding it a pure ~20-minute cost for a byte-identical file.
    """
    dest = context["root"] / f"{spec['id']}.wav"
    digest = _longform_digest(spec, sources, context)
    if cache.is_fresh(dest, digest):
        return "cached"
    try:
        # The manifest's profile, not longform.build's "stress" default: a spec asking for
        # the "natural" scene-shaped layout was silently rendered as the stress layout, so
        # the ground-truth sidecar described a timeline the audio did not have.
        timeline = longform.build(sources, dest, context, profile=spec.get("profile") or "stress", shape=spec.get("shape") or "film")
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


def _selected_clips(data: dict[str, Any], args: argparse.Namespace) -> list[dict]:
    """The clip entries this invocation would build, empty when it builds no clips at all.

    Gated on _sections_for: `adversarial` and `longform` never render a clip, so reading the
    clips section for them reported a synthesis engine as required and refused to run
    without piper-tts or transformers installed -- for a command that would not have called
    either.
    """
    if "clips" not in _sections_for(args.command):
        return []
    return _selected(list(data.get("clips") or []), args, "clips")


def _mms_is_selected(data: dict[str, Any], args: argparse.Namespace) -> bool:
    """Whether any entry this invocation would build is rendered by MMS-TTS."""
    return any(entry.get("engine") == "mms" for entry in _selected_clips(data, args))


def _piper_is_selected(data: dict[str, Any], args: argparse.Namespace) -> bool:
    """Whether this invocation would render anything with Piper.

    Every ``combined`` leg is Piper-rendered, so a selection touching that section needs
    Piper regardless of which clips are chosen. Among the clips, "piper" is the default
    engine, so only an explicit ``"engine": "mms"`` opts out.
    """
    if "combined" in _sections_for(args.command) and _selected(list(data.get("combined") or []), args, "combined"):
        return True
    return any(entry.get("engine", "piper") == "piper" for entry in _selected_clips(data, args))


def _missing_tools(data: dict[str, Any], args: argparse.Namespace) -> list[str]:
    """Return the names of required tools that are unavailable for this invocation.

    Each engine is required only when something selected actually uses it. MMS supplies only
    the languages Piper has no voice for, so a Piper-only regeneration must not be blocked by
    a missing transformers/torch install it would never call -- and the converse holds too:
    `--only ta_mms` renders through MMS alone, so a missing piper-tts blocked a run that
    would never have invoked Piper.
    """
    engines = (
        ("piper-tts", _piper_is_selected, piper.piper_available),
        ("transformers+torch (MMS-TTS voices)", _mms_is_selected, mms.mms_available),
    )
    missing = [name for name, is_selected, is_available in engines if is_selected(data, args) and not is_available()]
    # ffmpeg is unconditional: every render path ends in a transcode.
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
