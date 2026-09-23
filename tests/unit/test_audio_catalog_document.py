"""Tests for the row builders, document assembly and CLI of scripts/audio_catalog.py.

`--check` compares the rendered document byte for byte against the committed copy, so what
matters is that every section is emitted, that an entry the generator cannot render is
still listed rather than dropped, and that a missing ffprobe is reported as itself instead
of surfacing later as a spurious "stale".
"""

import json
from unittest import mock

import pytest

from scripts import audio_catalog


@pytest.fixture(name="no_probe")
def _no_probe():
    """Keep the Measured column deterministic without invoking ffprobe."""
    with mock.patch.object(audio_catalog, "_facts", return_value="-"):
        yield


def _clip(**overrides):
    """A spoken clip entry."""
    entry = {"id": "en_core", "language": "en", "tier": "A", "voice": "en_US-amy", "text": "hello", "committed": True}
    entry.update(overrides)
    return entry


def _manifest(**overrides):
    """A manifest with one entry in every section."""
    data = {
        "clips": [_clip(), _clip(id="fr_tail", language="fr", tier="B", committed=False)],
        "combined": [{"id": "mix_en_fr", "legs": [{"language": "en", "text": "hello"}, {"language": "fr", "text": "bonjour"}]}],
        "adversarial": [
            {"id": "silence", "builder": "silence", "params": {"seconds": 10}, "expect": {"status_in": [200], "text_policy": "empty"}}
        ],
        "longform": {"id": "lf_stress", "target_seconds": 1200, "languages": ["en", "fr"], "profile": "stress"},
    }
    data.update(overrides)
    return data


def test_speech_rows_render_the_spoken_text(no_probe):
    """The row says who speaks and what they say, which is the point of the catalog."""
    rows = audio_catalog._speech_rows([_clip()])

    assert rows[0][0] == "en_core"
    assert "en_US-amy" in rows[0][4] and "hello" in rows[0][4]


def test_a_voiceless_clip_is_listed_as_a_coverage_gap(no_probe):
    """18 languages have no Piper voice; listing them keeps the gap visible rather than
    silently shrinking the catalog."""
    entry = _clip(id="am_core", language="am", voice=None, unsupported_reason="no Piper voice for Amharic")

    rows = audio_catalog._speech_rows([entry])

    assert rows[0][3] == "unavailable"
    assert "Coverage gap" in rows[0][4] and "Amharic" in rows[0][4]


def test_a_voiceless_clip_without_a_reason_still_renders(no_probe):
    """Validation catches this, but the catalog must not crash on it."""
    rows = audio_catalog._speech_rows([_clip(voice=None)])

    assert "no Piper voice" in rows[0][4]


def test_combined_rows_show_the_legs_in_order(no_probe):
    """Which language appeared where is exactly what these clips are for."""
    rows = audio_catalog._combined_rows(_manifest()["combined"])

    assert rows[0][2] == 'en: "hello" then fr: "bonjour"'


def test_adversarial_rows_show_the_accepted_response(no_probe):
    """These assert the service responds and does not hang, so the accepted statuses are
    part of the description."""
    rows = audio_catalog._adversarial_rows(_manifest()["adversarial"])

    assert "HTTP 200" in rows[0][4]
    assert "text empty" in rows[0][4]


def test_adversarial_rows_default_the_text_policy(no_probe):
    """An entry with no policy accepts any text."""
    rows = audio_catalog._adversarial_rows([{"id": "x", "builder": "silence", "params": {}, "expect": {}}])

    assert "text any" in rows[0][4]


def test_standalone_rows_report_presence(no_probe):
    """The loose fixtures predate the matrix and are committed, so they should be present."""
    rows = audio_catalog._standalone_rows()

    assert len(rows) == len(audio_catalog.STANDALONE)
    assert all(row[1] in {"present", "missing"} for row in rows)
    assert all(row[0].startswith("`tests/e2e/fixtures/") for row in rows)


def test_render_emits_every_section(no_probe):
    """A section silently dropped would hide a whole class of fixture from the reader."""
    document = audio_catalog.render(_manifest())

    for heading in (
        "# Audio catalog",
        "## Standalone fixtures",
        "## Core speech clips",
        "## Tail speech clips",
        "## Code-switched clips",
        "## Adversarial and degraded clips",
        "## Long-form clips",
    ):
        assert heading in document


def test_render_warns_against_hand_editing(no_probe):
    """The file is generated; an edit would be lost and --check would then fail."""
    assert "Do not edit by hand" in audio_catalog.render(_manifest())


def test_render_splits_core_and_tail_by_tier(no_probe):
    """Tier A is committed and always present; the tail renders on demand."""
    document = audio_catalog.render(_manifest())
    core_section = document.split("## Tail speech clips", maxsplit=1)[0]

    assert "en_core" in core_section
    assert "fr_tail" not in core_section


def test_longform_variants_inherit_the_top_level_spec(no_probe):
    """A variant states only what differs and inherits the rest."""
    longform = {"id": "lf", "target_seconds": 1200, "languages": ["en", "fr"], "variants": [{"id": "lf_film", "profile": "film"}]}
    data = _manifest(longform=longform)

    document = audio_catalog.render(data)

    assert "`lf_film`" in document
    assert "2 languages (en, fr)" in document


def test_a_variant_does_not_inherit_recorded_defects(no_probe):
    """A scene-shaped clip does not carry the stress grid's defects."""
    data = _manifest(
        longform={
            "id": "lf",
            "target_seconds": 1200,
            "languages": ["en"],
            "known_defects": {"windows": "67 of 118 wrong"},
            "variants": [{"id": "lf_film", "profile": "film"}],
        }
    )

    document = audio_catalog.render(data)
    variant_section = document.split("`lf_film`", maxsplit=1)[1]

    assert "No known defects recorded" in variant_section


def test_longform_lines_describe_the_profile(no_probe):
    """The reader is choosing between layouts, so each is explained."""
    lines = audio_catalog._longform_lines({"id": "lf", "target_seconds": 1200, "languages": ["en"], "profile": "natural"})

    assert "scene-shaped runs" in lines[0]
    assert "About 20 minutes" in lines[0]


def test_longform_lines_describe_a_film_shape(no_probe):
    """Shapes add framing around the dialogue and are named separately from the profile."""
    lines = audio_catalog._longform_lines({"id": "lf", "target_seconds": 1200, "languages": ["en"], "profile": "film", "shape": "episode"})

    assert "television episode" in lines[0]


def test_longform_lines_pass_an_unknown_profile_through(no_probe):
    """Validation rejects these, but the catalog should not crash rendering one."""
    lines = audio_catalog._longform_lines({"id": "lf", "target_seconds": 60, "languages": ["en"], "profile": "brand_new"})

    assert "brand_new" in lines[0]


def test_run_refuses_without_ffprobe(caplog):
    """Checked up front rather than left to _probe, which answers None per file.

    Without ffprobe every Measured cell becomes "-", so writing would silently strip the
    measurements from the committed catalog and --check would then fail with "stale",
    naming the wrong problem entirely.
    """
    with mock.patch.object(audio_catalog.shutil, "which", return_value=None):
        with mock.patch.object(audio_catalog.sys, "argv", ["audio_catalog.py"]):
            assert audio_catalog.run() == 1

    assert "ffprobe is not on PATH" in caplog.text


def _run_with(tmp_path, data, argv, output_text=None):
    """Run the CLI against a temporary manifest and output file, returning (code, output)."""
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(data), encoding="utf-8")
    output = tmp_path / "AUDIO_CATALOG.md"
    if output_text is not None:
        output.write_text(output_text, encoding="utf-8")

    # ROOT and both clip directories move together. _rel() renders every path relative to
    # ROOT, and the clip directories are computed from ROOT at import -- patching one
    # without the others leaves a path under the real repository that _rel then refuses.
    patches = {
        "ROOT": tmp_path,
        # The stale message renders the script's own path relative to ROOT too.
        "__file__": str(tmp_path / "scripts" / "audio_catalog.py"),
        "OUTPUT_PATH": output,
        "MANIFEST_PATH": manifest_file,
        "COMMITTED_DIR": tmp_path / "core",
        "GENERATED_DIR": tmp_path / "generated",
    }
    with mock.patch.object(audio_catalog.shutil, "which", return_value="/usr/bin/ffprobe"):
        with mock.patch.multiple(audio_catalog, **patches):
            with mock.patch.object(audio_catalog.sys, "argv", argv):
                return audio_catalog.run(), output


def test_run_writes_the_catalog(tmp_path, no_probe, caplog):
    """The ordinary generation path."""
    code, output = _run_with(tmp_path, _manifest(), ["audio_catalog.py"])

    assert code == 0
    assert output.read_text(encoding="utf-8").startswith("# Audio catalog")
    assert "wrote" in caplog.text


def test_run_check_passes_when_the_catalog_is_current(tmp_path, no_probe):
    """--check is the CI gate; it must accept a file it would have written itself.

    Generated and verified in the same run rather than compared against a separately
    rendered string: the document embeds repository-relative paths, so rendering it
    outside the patched roots produces text the check would correctly call stale.
    """
    data = _manifest()
    write_code, _ = _run_with(tmp_path, data, ["audio_catalog.py"])
    check_code, _ = _run_with(tmp_path, data, ["audio_catalog.py", "--check"])

    assert (write_code, check_code) == (0, 0)


def test_run_check_fails_on_a_stale_catalog(tmp_path, no_probe, caplog):
    """A stale catalog must name the command that regenerates it."""
    code, _ = _run_with(tmp_path, _manifest(), ["audio_catalog.py", "--check"], output_text="# Audio catalog\n\nold\n")

    assert code == 1
    assert "is stale" in caplog.text


def test_run_check_treats_a_missing_catalog_as_stale(tmp_path, no_probe):
    """A catalog that was never written is not up to date."""
    code, _ = _run_with(tmp_path, _manifest(), ["audio_catalog.py", "--check"])

    assert code == 1


def test_the_committed_catalog_is_current():
    """The repo's own catalog must match what the generator produces from the manifest.

    This is the same comparison --check makes in CI, run without needing ffprobe: the
    Measured column comes from committed FLAC files that are present in the checkout.
    """
    data = json.loads(audio_catalog.MANIFEST_PATH.read_text(encoding="utf-8"))

    assert audio_catalog.render(data) == audio_catalog.OUTPUT_PATH.read_text(encoding="utf-8")
