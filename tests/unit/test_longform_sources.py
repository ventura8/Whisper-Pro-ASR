"""Which rendered clips each long-form variant is assembled from.

The three original variants name the same ten languages, so the top-level list served all of
them. A single-language variant -- what most of the library is -- needs its own list honoured,
and the others must keep drawing exactly what they drew before, because every number recorded
against them is stated for that source set.
"""

from unittest import mock

import pytest

from scripts.audio_matrix import cli

CLIPS = [
    {"id": "en_core", "language": "en"},
    {"id": "en_scene2", "language": "en"},
    {"id": "en_line1", "language": "en", "role": "line"},
    {"id": "ru_core", "language": "ru"},
    {"id": "ru_line1", "language": "ru", "role": "line"},
    {"id": "uk_core", "language": "uk"},
]


@pytest.fixture(name="context")
def context_fixture(tmp_path):
    """A cache root holding a rendered file for every clip."""
    for clip in CLIPS:
        (tmp_path / f"{clip['id']}.wav").write_bytes(b"RIFF")
    return {"root": tmp_path, "rate": 16000}


@pytest.fixture(autouse=True)
def _no_ffprobe():
    with mock.patch.object(cli.render, "probe_duration", return_value=1.5):
        yield


DATA = {"clips": CLIPS, "longform": {"languages": ["en", "ru"]}}


def _ids(sources):
    return sorted(source["id"] for source in sources)


class TestTheVariantsOwnLanguages:
    """A variant's list narrows the sources; without one the top-level list applies."""

    def test_a_single_language_variant_draws_only_that_language(self, context):
        """The film_mono case: one language, lines included."""
        assert _ids(cli._longform_sources(DATA, context, profile="film", languages=["ru"])) == ["ru_core", "ru_line1"]

    def test_an_explicit_empty_list_asks_for_nothing(self, context):
        """``[]`` is a variant's own answer, not the absence of one: it must reach the
        "no rendered source clips" failure rather than borrow the top-level languages."""
        assert not cli._longform_sources(DATA, context, profile="film", languages=[])

    def test_without_a_list_the_top_level_languages_apply(self, context):
        """What the three original variants get, unchanged."""
        assert _ids(cli._longform_sources(DATA, context, profile="film")) == ["en_core", "en_line1", "en_scene2", "ru_core", "ru_line1"]

    def test_a_language_the_top_level_does_not_name_is_still_reachable(self, context):
        """The variant's list is authoritative, not a filter over the top-level one."""
        assert _ids(cli._longform_sources(DATA, context, profile="film", languages=["uk"])) == ["uk_core"]

    def test_a_language_with_nothing_rendered_is_an_error(self, context):
        """Silently dropped, the clip would carry the variant's name without the language."""
        with pytest.raises(ValueError, match="no eligible rendered clip .*: ja"):
            cli._longform_sources(DATA, context, profile="film", languages=["ja"])

    def test_one_language_present_and_one_missing_names_the_missing_one(self, context):
        """The error names what is missing, not what was found."""
        with pytest.raises(ValueError, match=": ja$"):
            cli._longform_sources(DATA, context, profile="film", languages=["ru", "ja"])


class TestTheProfilesKeepTheirSourceRules:
    """The list changes which clips are eligible, never how a profile chooses among them."""

    def test_stress_still_takes_one_recording_per_language_and_no_lines(self, context):
        """One clip per language keeps the stress grid byte-identical to its recorded numbers:
        the second English recording is there for natural and film, never for stress."""
        assert _ids(cli._longform_sources(DATA, context, profile="stress", languages=["ru", "en"])) == ["en_core", "ru_core"]

    def test_natural_still_takes_every_recording_and_no_lines(self, context):
        """Every recording, so a scene is not one sentence repeated -- both English ones, no lines."""
        assert _ids(cli._longform_sources(DATA, context, profile="natural", languages=["ru", "en"])) == ["en_core", "en_scene2", "ru_core"]

    def test_film_still_takes_the_lines_too(self, context):
        """The short lines are what make film-length utterances."""
        assert _ids(cli._longform_sources(DATA, context, profile="film", languages=["ru"])) == ["ru_core", "ru_line1"]


class TestTheBaseDigestIgnoresTheVariantList:
    """The top-level spec carries every variant; none of them shapes the base clip."""

    def test_adding_a_variant_does_not_stale_the_base_clip(self, context):
        """Measured: adding `film_mono` rebuilt the stress grid to a byte-identical file."""
        base = {"id": "longform_stress", "target_seconds": 1200, "languages": ["en", "ru"], "variants": []}
        with mock.patch.object(cli, "_tool_versions", return_value={"piper": "1"}):
            before = cli._longform_digest(base, [], {"defaults": {}})
            after = cli._longform_digest({**base, "variants": [{"id": "longform_film_mono", "profile": "film"}]}, [], {"defaults": {}})
        assert before == after

    def test_a_change_to_the_base_spec_itself_still_stales_it(self, context):
        """Dropping the variant list must not make the digest blind to the spec's own fields."""
        base = {"id": "longform_stress", "target_seconds": 1200, "languages": ["en", "ru"]}
        with mock.patch.object(cli, "_tool_versions", return_value={"piper": "1"}):
            assert cli._longform_digest(base, [], {"defaults": {}}) != cli._longform_digest(
                {**base, "target_seconds": 600}, [], {"defaults": {}}
            )


class TestAVariantThatCannotBeBuilt:
    """A spec naming a language nothing rendered fails as one variant, not as the run."""

    def test_the_failure_is_counted_and_the_rest_of_the_run_survives(self, context, caplog):
        """The ValueError from the source selection becomes this variant's FAILED line."""
        spec = {"id": "longform_ja", "profile": "film", "languages": ["ja"], "target_seconds": 60}
        with mock.patch.object(cli, "_try_build_longform") as build:
            failures = cli._build_one_longform(spec, DATA, context)
        assert failures == 1
        assert not build.called, "nothing is laid out for a variant with no sources"
        assert any("longform_ja" in record.message and "ja" in record.message for record in caplog.records)
