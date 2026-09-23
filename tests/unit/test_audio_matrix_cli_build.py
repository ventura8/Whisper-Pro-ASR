"""Tests for the per-entry build path in scripts/audio_matrix/cli.py.

The generator is content-addressed and idempotent, so the properties that matter are what
goes into a digest, what happens when a render fails part-way, and that one failing entry
does not cost the other forty. The command dispatch is in test_audio_matrix_cli_run.py.
"""

import argparse
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from scripts.audio_matrix import cli


@pytest.fixture(autouse=True)
def _clear_tool_version_cache():
    """_tool_versions is lru_cached; stop a patched toolchain leaking between tests."""
    cli._tool_versions.cache_clear()
    yield
    cli._tool_versions.cache_clear()


@pytest.fixture(name="versions")
def _versions():
    """Pin the toolchain versions so digests are stable across machines."""
    pinned = {"piper": "1.8.0", "mms": "5.17.0", "torch": "2.13.0", "uroman": "1.3.1.1", "ffmpeg": "6.1"}
    with mock.patch.object(cli, "_tool_versions", return_value=pinned):
        yield pinned


def _args(**overrides):
    """A parsed argument namespace with the generator's defaults."""
    values = {"command": "all", "out": None, "only": None, "tier": None, "force": False, "strict": False}
    values.update(overrides)
    return argparse.Namespace(**values)


def _data(**overrides):
    """A minimal manifest."""
    data = {
        "defaults": {"sample_rate": 16000, "pins": {"length_scale": 1.0}},
        "clips": [{"id": "en_core", "language": "en", "tier": "A", "voice": "en_US-amy", "text": "hello", "committed": True}],
        "combined": [],
        "adversarial": [],
    }
    data.update(overrides)
    return data


@pytest.fixture(name="context")
def _context(tmp_path):
    """A rendering context rooted at a temporary cache."""
    return {
        "root": tmp_path,
        "rate": 16000,
        "pins": {"length_scale": 1.0},
        "defaults": {"sample_rate": 16000},
        "by_id": {},
    }


def test_module_version_reports_missing_rather_than_skipping():
    """A clip rendered without uroman is not the same artifact as one rendered with it, so
    absence has to change the digest rather than be omitted from it."""
    with mock.patch.object(cli.importlib, "import_module", side_effect=ImportError):
        assert cli._module_version("uroman") == "missing"


def test_module_version_reports_unknown_when_there_is_no_version_attribute():
    """Still a distinct value, so the digest moves if the module appears or disappears."""
    with mock.patch.object(cli.importlib, "import_module", return_value=object()):
        assert cli._module_version("uroman") == "unknown"


def test_tool_versions_covers_every_library_that_shapes_a_waveform():
    """torch and uroman are in the digest because MMS runs VITS through torch and
    romanizes with uroman -- neither is covered by the transformers version."""
    with mock.patch.object(cli.piper, "piper_version", return_value="1.8.0"):
        with mock.patch.object(cli.mms, "mms_version", return_value="5.17.0"):
            with mock.patch.object(cli.render, "ffmpeg_version", return_value="6.1"):
                with mock.patch.object(cli, "_module_version", side_effect=lambda name: f"{name}-v"):
                    versions = cli._tool_versions()

    assert set(versions) == {"piper", "mms", "torch", "uroman", "ffmpeg"}


def test_context_indexes_entries_by_id(tmp_path):
    """A derived clip's digest includes the spec of the clip it is built from."""
    data = _data(adversarial=[{"id": "clipped", "builder": "clipped", "params": {"source": "en_core"}}])

    context = cli._context(data, _args(out=str(tmp_path)))

    assert context["by_id"]["en_core"]["text"] == "hello"
    assert context["rate"] == 16000


def test_filters_skip_voiceless_clips():
    """A clip with no voice cannot be rendered, so it is never selected."""
    entries = [{"id": "a", "voice": "v"}, {"id": "b", "voice": None}]

    assert [e["id"] for e in cli._selected(entries, _args(), "clips")] == ["a"]


def test_filters_honour_only_and_tier():
    """The flags compose: --only picks one id, --tier restricts the set."""
    entries = [{"id": "a", "voice": "v", "tier": "A"}, {"id": "b", "voice": "v", "tier": "B"}]

    assert [e["id"] for e in cli._selected(entries, _args(only="b"), "clips")] == ["b"]
    assert [e["id"] for e in cli._selected(entries, _args(tier="A"), "clips")] == ["a"]


def test_the_core_command_selects_only_committed_entries():
    """`core` regenerates the committed tier, which is what ships in the repository."""
    entries = [{"id": "a", "voice": "v", "committed": True}, {"id": "b", "voice": "v"}]

    assert [e["id"] for e in cli._selected(entries, _args(command="core"), "clips")] == ["a"]


def test_synthesize_routes_mms_entries_to_mms(context):
    """MMS supplies the languages Piper has no voice for."""
    entry = {"id": "ta_core", "engine": "mms", "voice": "tam", "text": "hello"}

    with mock.patch.object(cli.mms, "synth") as synth:
        cli._synthesize(entry, Path("/tmp/raw.wav"), context)

    synth.assert_called_once()


def test_synthesize_defaults_to_piper(context):
    """Piper is the default and covers most of the matrix."""
    entry = {"id": "en_core", "voice": "en_US-amy", "text": "hello"}

    with mock.patch.object(cli.piper, "ensure_voice", return_value="model.onnx"):
        with mock.patch.object(cli.piper, "synth") as synth:
            cli._synthesize(entry, Path("/tmp/raw.wav"), context)

    synth.assert_called_once()


def test_synthesize_rejects_an_unknown_engine(context):
    """A typo must name itself rather than silently rendering with the default."""
    entry = {"id": "x", "engine": "espeak", "voice": "v", "text": "hello"}

    with pytest.raises(ValueError, match="Unknown synthesis engine"):
        cli._synthesize(entry, Path("/tmp/raw.wav"), context)


def test_render_clip_transcodes_when_no_gain_is_requested(context):
    """The common path: straight to 16 kHz mono PCM."""
    with mock.patch.object(cli, "_synthesize"):
        with mock.patch.object(cli.render, "to_pcm16_mono") as pcm:
            with mock.patch.object(cli.render, "apply_gain") as gain:
                cli._render_clip({"id": "a"}, Path("/tmp/a.wav"), context)

    pcm.assert_called_once()
    gain.assert_not_called()


def test_render_clip_applies_a_requested_gain(context):
    """An entry asking for a gain takes the gain path instead."""
    with mock.patch.object(cli, "_synthesize"):
        with mock.patch.object(cli.render, "apply_gain") as gain:
            cli._render_clip({"id": "a", "gain": 0.5}, Path("/tmp/a.wav"), context)

    gain.assert_called_once()


def test_publish_committed_writes_flac_only_for_committed_entries(tmp_path, context):
    """Only the committed tier is tracked in the repository."""
    with mock.patch.object(cli.manifest, "CORE_DIR", tmp_path / "core"):
        with mock.patch.object(cli.render, "to_flac") as to_flac:
            cli._publish_committed({"id": "a"}, tmp_path / "a.wav", context)
            to_flac.assert_not_called()

            cli._publish_committed({"id": "a", "committed": True}, tmp_path / "a.wav", context)
            to_flac.assert_called_once()


def test_a_deleted_committed_copy_is_detected(tmp_path):
    """A fresh cache hit returned "cached" without looking at the committed copy, so a
    deleted core/*.flac stayed deleted and the tests resolving it skipped. The stamp only
    says the .wav is current."""
    with mock.patch.object(cli.manifest, "CORE_DIR", tmp_path):
        assert cli._committed_copy_is_missing({"id": "a", "committed": True}) is True

        (tmp_path / "a.flac").write_bytes(b"fLaC")
        assert cli._committed_copy_is_missing({"id": "a", "committed": True}) is False


def test_an_uncommitted_entry_never_reports_a_missing_copy():
    """Nothing is expected in the core directory for it."""
    assert cli._committed_copy_is_missing({"id": "a"}) is False


def test_source_spec_resolves_a_derived_clip(context):
    """An adversarial entry degrades a clip named by params.source.

    Its own entry says nothing about what that clip contains, so a digest over the entry
    alone left the derived artifact fresh after its source's text or voice changed -- a
    stale degraded clip checked against updated expectations.
    """
    context["by_id"] = {"en_core": {"id": "en_core", "text": "hello"}}

    assert cli._source_spec({"params": {"source": "en_core"}}, context)["text"] == "hello"
    assert cli._source_spec({"params": {}}, context) is None
    assert cli._source_spec({}, context) is None


def test_build_one_reports_a_fresh_entry_as_cached(tmp_path, context, versions):
    """Re-running with nothing changed must do no work."""
    dest = tmp_path / "en_core.wav"
    entry = {"id": "en_core", "text": "hello"}
    digest = cli.cache.spec_digest({"entry": entry, "defaults": context["defaults"]}, versions)
    dest.write_bytes(b"RIFF")
    cli.cache.write_stamp(dest, digest)

    assert cli._build_one("clips", entry, context, force=False) == "cached"


def test_build_one_rebuilds_when_forced(tmp_path, context, versions):
    """--force exists for the case where the digest cannot see what changed."""
    dest = tmp_path / "en_core.wav"
    entry = {"id": "en_core", "text": "hello"}
    dest.write_bytes(b"RIFF")
    cli.cache.write_stamp(dest, cli.cache.spec_digest({"entry": entry, "defaults": context["defaults"]}, versions))

    with mock.patch.object(cli, "_render_atomically") as render_call:
        assert cli._build_one("clips", entry, context, force=True) == "built"

    render_call.assert_called_once()


def test_build_one_restores_a_deleted_committed_copy(tmp_path, context, versions):
    """A cache hit whose FLAC is gone republishes rather than reporting nothing to do."""
    dest = tmp_path / "en_core.wav"
    entry = {"id": "en_core", "text": "hello", "committed": True}
    dest.write_bytes(b"RIFF")
    cli.cache.write_stamp(dest, cli.cache.spec_digest({"entry": entry, "defaults": context["defaults"]}, versions))

    with mock.patch.object(cli.manifest, "CORE_DIR", tmp_path / "core"):
        with mock.patch.object(cli.render, "to_flac") as to_flac:
            assert cli._build_one("clips", entry, context, force=False) == "built"

    to_flac.assert_called_once()


def test_build_one_stamps_the_digest_after_rendering(tmp_path, context, versions):
    """The stamp is what makes the next run a no-op."""
    entry = {"id": "en_core", "text": "hello"}

    with mock.patch.object(cli, "_render_atomically"):
        assert cli._build_one("clips", entry, context, force=False) == "built"

    assert cli.cache.read_stamp(tmp_path / "en_core.wav")


def test_render_atomically_stages_before_publishing(tmp_path, context):
    """A renderer that failed part-way used to leave a truncated file at dest.

    The stamp is written after, so the cache correctly called it stale -- but every
    consumer resolving a clip by path read the partial audio in the meantime.
    """
    dest = tmp_path / "en_core.wav"
    seen = {}

    def _render(entry, staged, ctx):
        seen["staged"] = Path(staged).name
        Path(staged).write_bytes(b"RIFF")

    with mock.patch.dict(cli._RENDERERS, {"clips": _render}):
        cli._render_atomically("clips", {"id": "en_core"}, dest, context)

    assert seen["staged"] == "en_core.partial.wav"
    assert dest.read_bytes() == b"RIFF"


def test_the_staging_suffix_keeps_an_extension_ffmpeg_recognises(tmp_path, context):
    """ffmpeg chooses its muxer from the extension, so "<id>.wav.partial" made every
    render fail with "Unable to choose an output format"."""
    staged_names = []

    def _render(entry, staged, ctx):
        staged_names.append(Path(staged).suffix)
        Path(staged).write_bytes(b"RIFF")

    with mock.patch.dict(cli._RENDERERS, {"clips": _render}):
        cli._render_atomically("clips", {"id": "en_core"}, tmp_path / "en_core.wav", context)

    assert staged_names == [".wav"]


def test_render_atomically_leaves_nothing_behind_when_the_render_fails(tmp_path, context):
    """A failed render must leave neither the staged audio nor a staged sidecar."""

    def _render(entry, staged, ctx):
        Path(staged).write_bytes(b"partial")
        Path(str(staged) + ".legs.json").write_text("{}", encoding="utf-8")
        raise RuntimeError("ffmpeg exploded")

    with mock.patch.dict(cli._RENDERERS, {"clips": _render}):
        with pytest.raises(RuntimeError):
            cli._render_atomically("clips", {"id": "en_core"}, tmp_path / "en_core.wav", context)

    assert not list(tmp_path.iterdir())


def test_a_sidecar_is_published_after_the_audio(tmp_path, context):
    """A reader must never see a sidecar describing audio that was not written."""
    dest = tmp_path / "mix.wav"

    def _render(entry, staged, ctx):
        Path(staged).write_bytes(b"RIFF")
        Path(str(staged) + ".legs.json").write_text('{"legs": []}', encoding="utf-8")

    with mock.patch.dict(cli._RENDERERS, {"combined": _render}):
        cli._render_atomically("combined", {"id": "mix"}, dest, context)

    assert dest.exists()
    assert (tmp_path / "mix.legs.json").exists()


def test_brief_prefers_a_subprocess_stderr_tail():
    """ffmpeg's last stderr line is the useful one; the rest is banner noise."""
    error = subprocess.CalledProcessError(1, "ffmpeg")
    error.stderr = b"configuration: ...\nError: Invalid argument\n"

    assert cli._brief(error) == "Error: Invalid argument"


def test_brief_falls_back_to_the_exception_message():
    """Not every failure carries stderr."""
    assert cli._brief(ValueError("bad manifest entry")) == "bad manifest entry"


def test_brief_handles_an_empty_stderr():
    """An empty capture must not produce an empty message."""
    error = subprocess.CalledProcessError(1, "ffmpeg")
    error.stderr = b"   \n"

    assert cli._brief(error)


def test_try_build_reports_a_failure_without_aborting(context):
    """One language whose voice needs an extra phonemizer must not cost you the other
    forty: the run reports what failed and exits non-zero once everything else is done."""
    with mock.patch.object(cli, "_build_one", side_effect=RuntimeError("uroman missing")):
        assert cli._try_build("clips", {"id": "am_core"}, context, force=False) == "failed"


@pytest.mark.parametrize(
    "error",
    [OSError("disk full"), ValueError("bad"), KeyError("missing"), subprocess.TimeoutExpired("piper", 600)],
)
def test_try_build_catches_every_per_entry_failure(context, error):
    """Each of these is a per-entry problem, not a reason to abandon the matrix."""
    with mock.patch.object(cli, "_build_one", side_effect=error):
        assert cli._try_build("clips", {"id": "x"}, context, force=False) == "failed"


def test_build_section_counts_failures(context, caplog):
    """The summary line is what a developer reads after a long run."""
    data = _data(clips=[{"id": "a", "voice": "v"}, {"id": "b", "voice": "v"}])

    with mock.patch.object(cli, "_try_build", side_effect=["built", "failed"]):
        failed = cli._build_section("clips", data, _args(), context)

    assert failed == 1
    assert "1 ready, 1 failed" in caplog.text
