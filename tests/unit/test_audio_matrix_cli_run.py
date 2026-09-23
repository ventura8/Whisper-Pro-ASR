"""Tests for command dispatch, long-form assembly and tooling checks in cli.py.

Most of these pin decisions the module documents at length, because the long-form
fixtures carry recorded measurements: which clips a profile lays out from, and which
commands need which synthesis engine, both changed a published artifact when they were
wrong. The per-entry build path is in test_audio_matrix_cli_build.py.
"""

import argparse
from unittest import mock

import pytest

from scripts.audio_matrix import cli


def _args(**overrides):
    """A parsed argument namespace with the generator's defaults."""
    values = {"command": "all", "out": None, "only": None, "tier": None, "force": False, "strict": False}
    values.update(overrides)
    return argparse.Namespace(**values)


def _clip(clip_id, language, **overrides):
    """A rendered clip as _rendered_clips_in returns it."""
    entry = {"id": clip_id, "language": language, "voice": "v", "text": "hello"}
    entry.update(overrides)
    return entry


def _data(**overrides):
    """A manifest with a long-form spec."""
    data = {
        "defaults": {"sample_rate": 16000, "pins": {}},
        "clips": [_clip("en_core", "en"), _clip("fr_core", "fr")],
        "combined": [],
        "adversarial": [],
        "longform": {"id": "lf", "languages": ["en", "fr"], "target_seconds": 1200},
    }
    data.update(overrides)
    return data


def test_build_parser_defaults_to_building_everything():
    """Running the generator with no arguments regenerates the whole matrix."""
    args = cli.build_parser().parse_args([])

    assert args.command == "all"
    assert args.force is False


def test_build_parser_rejects_an_unknown_command():
    """A typo must not fall through to the default."""
    parser = cli.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["nonsense"])


@pytest.mark.parametrize(
    ("command", "expected"),
    [("all", cli.SECTIONS), ("core", cli.SECTIONS), ("clips", ("clips",)), ("longform", ()), ("verify", ())],
)
def test_sections_for(command, expected):
    """`longform` and `verify` build no manifest sections."""
    assert cli._sections_for(command) == expected


def test_one_per_language_keeps_the_stress_grid_alternating():
    """Stress cycles its source list, so three clips per language stops it alternating.

    That happened: adding clips took the fixture from 118 utterances switching at every
    boundary to 117 switching at 78 of 116, silently changing the artifact every recorded
    `windows` measurement is stated against.
    """
    sources = [_clip("en_1", "en"), _clip("en_2", "en"), _clip("fr_1", "fr")]

    picked = cli._eligible("stress", sources)

    assert [clip["id"] for clip in picked] == ["en_1", "fr_1"]


def test_natural_keeps_every_clip_of_a_language():
    """With a single source a scene is one sentence repeated up to thirty times, which
    trips the repetition filter and measures that filter rather than language handling."""
    sources = [_clip("en_1", "en"), _clip("en_2", "en")]

    assert len(cli._eligible("natural", sources)) == 2


def test_film_takes_the_line_role_and_the_others_do_not():
    """Lines reproduce the ~1.3s utterances real film is made of.

    The other two profiles exclude them on purpose: natural takes every clip of a
    language, so adding lines would have changed the natural fixture under every number
    recorded against it.
    """
    sources = [_clip("en_core", "en"), _clip("en_line", "en", role="line")]

    assert len(cli._eligible("film", sources)) == 2
    assert [c["id"] for c in cli._eligible("natural", sources)] == ["en_core"]


def test_a_language_with_no_eligible_clip_is_an_error():
    """Silently dropped, the clip would still carry the variant's name and every number
    stated against the fixture would describe audio lacking a language it claims."""
    sources = [_clip("en_core", "en")]

    with pytest.raises(ValueError, match="no eligible rendered clip"):
        cli._require_every_language(["en", "de"], sources)


def test_rendered_clips_only_include_what_exists_on_disk(tmp_path):
    """The timeline is assembled from rendered audio, not from manifest entries."""
    (tmp_path / "en_core.wav").write_bytes(b"RIFF")
    context = {"root": tmp_path}

    with mock.patch.object(cli.render, "probe_duration", return_value=3.0):
        rendered = cli._rendered_clips_in(_data(), context, ["en", "fr"])

    assert [clip["id"] for clip in rendered] == ["en_core"]


def test_a_variant_language_list_wins_over_the_top_level_spec(tmp_path):
    """An empty list asks for no languages and must get none, so the build fails where it
    says why rather than quietly rendering the top-level set under the variant's name."""
    (tmp_path / "en_core.wav").write_bytes(b"RIFF")
    context = {"root": tmp_path}

    with mock.patch.object(cli.render, "probe_duration", return_value=3.0):
        assert not cli._rendered_clips_in(_data(), context, [])


def test_build_one_longform_reports_a_missing_language_as_this_variants_failure():
    """Reported like every other failure rather than as a traceback that loses the
    sections and variants already built."""
    with mock.patch.object(cli, "_longform_sources", side_effect=ValueError("no eligible rendered clip")):
        assert cli._build_one_longform({"id": "lf_film"}, _data(), {}) == 1


def test_build_one_longform_reports_having_no_sources_at_all():
    """A cache with no rendered clips cannot assemble a timeline."""
    with mock.patch.object(cli, "_longform_sources", return_value=[]):
        assert cli._build_one_longform({"id": "lf"}, _data(), {}) == 1


def test_build_one_longform_routes_through_try_build():
    """A missing spec field or an ffmpeg failure used to escape run() as a traceback,
    losing the sections that had already succeeded."""
    with mock.patch.object(cli, "_longform_sources", return_value=[_clip("en_core", "en")]):
        with mock.patch.object(cli, "_try_build_longform", return_value="failed") as attempt:
            assert cli._build_one_longform({"id": "lf"}, _data(), {}) == 1

    attempt.assert_called_once()


def test_build_longform_reports_an_absent_spec(caplog):
    """A manifest with no long-form section is not an error."""
    assert cli._build_longform({}, {}) == 0
    assert "not configured" in caplog.text


def test_build_longform_builds_the_spec_and_each_variant():
    """Variants are built alongside the top-level spec."""
    data = _data(longform={"id": "lf", "languages": ["en"], "variants": [{"id": "lf_film"}, {"id": "lf_ep"}]})

    with mock.patch.object(cli, "_build_one_longform", return_value=0) as build:
        assert cli._build_longform(data, {}) == 0

    assert build.call_count == 3


def test_try_build_longform_uses_the_manifest_profile(tmp_path):
    """A spec asking for the natural layout was silently rendered as stress, so the
    ground-truth sidecar described a timeline the audio did not have."""
    context = {"root": tmp_path, "rate": 16000}
    spec = {"id": "lf_nat", "profile": "natural", "shape": "film"}

    with mock.patch.object(cli, "_longform_digest", return_value="d"):
        with mock.patch.object(cli.longform, "build", return_value={"duration": 1200.0, "speech": []}) as build:
            with mock.patch.object(cli.cache, "write_stamp"):
                assert cli._try_build_longform(spec, [_clip("en_core", "en")], context) == "built"

    assert build.call_args.kwargs["profile"] == "natural"


def test_try_build_longform_reports_a_render_failure(tmp_path):
    """A 20-minute build that fails is one variant's failure, not the run's."""
    context = {"root": tmp_path, "rate": 16000}

    with mock.patch.object(cli, "_longform_digest", return_value="d"):
        with mock.patch.object(cli.longform, "build", side_effect=RuntimeError("ffmpeg exploded")):
            assert cli._try_build_longform({"id": "lf"}, [_clip("en_core", "en")], context) == "failed"


def test_try_build_longform_skips_a_fresh_timeline(tmp_path):
    """Rebuilding an unchanged timeline is a pure ~20-minute cost for an identical file."""
    context = {"root": tmp_path, "rate": 16000}

    with mock.patch.object(cli, "_longform_digest", return_value="d"):
        with mock.patch.object(cli.cache, "is_fresh", return_value=True):
            with mock.patch.object(cli.longform, "build") as build:
                assert cli._try_build_longform({"id": "lf"}, [], context) == "cached"

    build.assert_not_called()


def test_only_a_named_clip_does_not_pay_for_the_whole_timeline():
    """--only names a single clip, and long-form is a ~20-minute assembly over every
    rendered tier-A clip -- on a cache holding only that clip it failed with "no rendered
    source clips", an error about a thing the caller never asked for."""
    assert cli._should_build_longform(_args(command="all", only="en_core")) is False
    assert cli._should_build_longform(_args(command="all")) is True


def test_an_explicit_longform_command_is_always_honoured():
    """Even with --only, asking for longform builds longform."""
    assert cli._should_build_longform(_args(command="longform", only="en_core")) is True


def test_generate_returns_nonzero_when_anything_failed():
    """The exit code is what a build script keys on."""
    with mock.patch.object(cli, "_context", return_value={}):
        with mock.patch.object(cli, "_build_section", return_value=1):
            with mock.patch.object(cli, "_should_build_longform", return_value=False):
                assert cli._generate(_data(), _args(command="clips")) == 1


def test_generate_returns_zero_when_everything_succeeded():
    """The success path."""
    with mock.patch.object(cli, "_context", return_value={}):
        with mock.patch.object(cli, "_build_section", return_value=0):
            with mock.patch.object(cli, "_should_build_longform", return_value=False):
                assert cli._generate(_data(), _args(command="clips")) == 0


def test_verify_reports_coverage_and_gaps(caplog):
    """`verify` is how the coverage gap stays visible."""
    data = _data(clips=[_clip("en_core", "en"), _clip("am_core", "am", voice=None, unsupported_reason="no Piper voice")])

    with mock.patch.object(cli.manifest, "known_languages", return_value=frozenset({"en", "am", "zu"})):
        assert cli._verify(data, _args(command="verify")) == 0

    assert "covered languages: 1" in caplog.text
    assert "declared gaps:     1" in caplog.text
    assert "no Piper voice" in caplog.text
    assert "zu" in caplog.text


def test_adversarial_and_longform_commands_need_no_synthesis_engine():
    """Reading the clips section for them reported a synthesis engine as required and
    refused to run without piper-tts installed -- for a command that never calls it."""
    assert not cli._selected_clips(_data(), _args(command="adversarial"))
    assert not cli._selected_clips(_data(), _args(command="longform"))


def test_a_piper_only_run_does_not_require_mms():
    """MMS supplies only the languages Piper has no voice for."""
    data = _data(clips=[_clip("en_core", "en")])

    assert cli._mms_is_selected(data, _args()) is False
    assert cli._piper_is_selected(data, _args()) is True


def test_an_mms_only_run_does_not_require_piper():
    """`--only ta_mms` renders through MMS alone, so a missing piper-tts blocked a run
    that would never have invoked Piper."""
    data = _data(clips=[_clip("ta_core", "ta", engine="mms")])

    assert cli._mms_is_selected(data, _args()) is True
    assert cli._piper_is_selected(data, _args()) is False


def test_a_combined_selection_requires_piper():
    """Every code-switched leg is Piper-rendered, whichever clips are chosen."""
    data = _data(clips=[_clip("ta_core", "ta", engine="mms")], combined=[{"id": "mix", "legs": []}])

    assert cli._piper_is_selected(data, _args()) is True


def test_missing_tools_lists_only_what_this_run_needs():
    """ffmpeg is unconditional; every render path ends in a transcode."""
    data = _data(clips=[_clip("en_core", "en")])

    with mock.patch.object(cli.piper, "piper_available", return_value=False):
        with mock.patch.object(cli.mms, "mms_available", return_value=False):
            with mock.patch.object(cli.render, "ffmpeg_available", return_value=False):
                missing = cli._missing_tools(data, _args())

    assert "piper-tts" in " ".join(missing)
    assert "MMS-TTS" not in " ".join(missing)
    assert "ffmpeg" in missing


def test_run_reports_manifest_errors_and_exits_two(caplog):
    """A malformed manifest is distinguished from a failed render by its exit code."""
    with mock.patch.object(cli.manifest, "load", return_value=_data()):
        with mock.patch.object(cli.manifest, "validate", return_value=["clips/x: bad tier"]):
            assert cli.run([]) == 2

    assert "manifest: clips/x: bad tier" in caplog.text


def test_run_dispatches_verify():
    """`verify` short-circuits before any tooling check."""
    with mock.patch.object(cli.manifest, "load", return_value=_data()):
        with mock.patch.object(cli.manifest, "validate", return_value=[]):
            with mock.patch.object(cli, "_verify", return_value=0) as verify:
                assert cli.run(["verify"]) == 0

    verify.assert_called_once()


def test_missing_tooling_is_tolerated_by_default(caplog):
    """The committed core tier still works without it, and the tests skip the rest with
    an actionable message."""
    with mock.patch.object(cli.manifest, "load", return_value=_data()):
        with mock.patch.object(cli.manifest, "validate", return_value=[]):
            with mock.patch.object(cli, "_missing_tools", return_value=["piper-tts"]):
                assert cli.run([]) == 0

    assert "poetry install --with tools" in caplog.text


def test_strict_turns_missing_tooling_into_a_failure():
    """--strict is for the environments that are supposed to have everything."""
    with mock.patch.object(cli.manifest, "load", return_value=_data()):
        with mock.patch.object(cli.manifest, "validate", return_value=[]):
            with mock.patch.object(cli, "_missing_tools", return_value=["piper-tts"]):
                assert cli.run(["--strict"]) == 1


def test_run_generates_when_everything_is_available():
    """The ordinary path."""
    with mock.patch.object(cli.manifest, "load", return_value=_data()):
        with mock.patch.object(cli.manifest, "validate", return_value=[]):
            with mock.patch.object(cli, "_missing_tools", return_value=[]):
                with mock.patch.object(cli, "_generate", return_value=0) as generate:
                    assert cli.run([]) == 0

    generate.assert_called_once()
