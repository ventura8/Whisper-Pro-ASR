"""Tests for scripts/audio_matrix/manifest.py, the manifest contract and its validation.

The manifest is what the generator and the real-audio suites agree on, so validation's job
is to make a malformed entry fail fast with a precise message rather than surface later as
a mysterious assertion. Most cases below pin failures the module's comments record: a bare
string where an object belongs, a longform entry with no id, and an id colliding across
sections.
"""

import json
from unittest import mock

import pytest

from scripts.audio_matrix import manifest


@pytest.fixture(autouse=True)
def _clear_language_cache():
    """known_languages is lru_cached; keep patched languages from leaking between tests."""
    manifest.known_languages.cache_clear()
    yield
    manifest.known_languages.cache_clear()


@pytest.fixture(name="languages")
def _languages():
    """Pin the known-language set so tests do not depend on the service's full list."""
    with mock.patch.object(manifest, "known_languages", return_value=frozenset({"en", "fr", "de"})):
        yield


def _clip(**overrides):
    """A valid single-language clip entry."""
    entry = {"id": "en_core", "language": "en", "tier": "A", "voice": "en_US-amy"}
    entry.update(overrides)
    return entry


def test_load_reads_the_manifest_from_a_path(tmp_path):
    """An explicit path is honoured, which is what lets tests use a fixture manifest."""
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"clips": [_clip()]}), encoding="utf-8")

    assert manifest.load(path)["clips"][0]["id"] == "en_core"


def test_load_defaults_to_the_committed_manifest():
    """The real manifest must stay loadable and well formed."""
    data = manifest.load()

    assert isinstance(data.get("clips"), list)
    assert manifest.validate(data) == []


def test_known_languages_reads_the_service_language_list():
    """Loaded by path, not imported: the generator runs in a bare virtualenv holding only
    piper-tts, and importing modules.core would drag in the whole application package."""
    languages = manifest.known_languages()

    assert "en" in languages
    assert isinstance(languages, frozenset)


def test_known_languages_is_cached():
    """Validation calls this twice per entry; uncached it re-executed the module each time."""
    first = manifest.known_languages()
    second = manifest.known_languages()

    assert first is second


def test_clips_and_renderable_filter_as_documented():
    """`renderable` is what skips the entries with no voice to synthesize."""
    data = {"clips": [_clip(), _clip(id="am_core", voice=None, unsupported_reason="no Piper voice")]}

    assert len(manifest.clips(data)) == 2
    assert [entry["id"] for entry in manifest.renderable(manifest.clips(data))] == ["en_core"]


def test_clips_of_an_empty_manifest_is_empty():
    """A manifest with no clips section is not an error here."""
    assert not manifest.clips({})


def test_validate_accepts_a_well_formed_manifest(languages):
    """The baseline: no messages for a manifest that is correct."""
    assert manifest.validate({"clips": [_clip()]}) == []


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"id": None}, "missing fields"),
        ({"language": None}, "missing fields"),
        ({"tier": None}, "missing fields"),
        ({"tier": "C"}, "bad tier"),
        ({"language": "xx"}, "unknown language"),
        ({"expect_detect": ["en", "zz"]}, "unknown expect_detect"),
        ({"voice": None}, "unsupported_reason is missing"),
    ],
)
def test_validate_reports_each_clip_defect(languages, overrides, expected):
    """Each check names what is wrong rather than failing generically."""
    errors = manifest.validate({"clips": [_clip(**overrides)]})

    assert any(expected in message for message in errors), errors


def test_a_voiceless_entry_with_a_reason_is_valid(languages):
    """18 languages have no Piper voice; they are declared, not defects."""
    entry = _clip(id="am_core", voice=None, unsupported_reason="no Piper voice for Amharic")

    assert manifest.validate({"clips": [entry]}) == []


def test_validate_prefixes_messages_with_section_and_id(languages):
    """The message has to say which entry it is about."""
    errors = manifest.validate({"clips": [_clip(tier="C")]})

    assert errors[0].startswith("clips/en_core: ")


def test_an_entry_without_an_id_is_labelled_rather_than_anonymous(languages):
    """`<no id>` is more useful than a message naming nothing."""
    errors = manifest.validate({"clips": [_clip(id=None)]})

    assert any("<no id>" in message for message in errors)


def test_non_clip_sections_are_only_checked_for_an_id(languages):
    """A combined entry carries legs and an adversarial one a transformation, so neither
    has a language or tier to validate."""
    data = {"combined": [{"id": "mix_en_fr", "legs": []}], "adversarial": [{"id": "silence"}]}

    assert manifest.validate(data) == []


def test_a_non_object_entry_is_reported_readably(languages):
    """A bare string where an object belongs is an ordinary hand-edit slip.

    `entry.get` raised AttributeError from inside validation -- the one place whose entire
    job is to report a malformed manifest legibly.
    """
    errors = manifest.validate({"clips": ["en_core"]})

    assert any("expected an object, got str" in message for message in errors)


def test_a_non_object_entry_does_not_abort_duplicate_checking(languages):
    """The readable message must actually reach the caller.

    isinstance is checked before `.get` in the id collector too; without it the
    AttributeError aborted validate() and the message _section_errors had produced was
    never returned.
    """
    errors = manifest.validate({"clips": ["bare", _clip()]})

    assert any("expected an object" in message for message in errors)


def test_empty_entries_are_skipped(languages):
    """A null left by an edit is ignored rather than reported as a shapeless object."""
    assert manifest.validate({"clips": [None, _clip()]}) == []


def test_duplicate_ids_are_detected_across_sections(languages):
    """Uniqueness is global: every entry caches as <id>.wav in one directory, so a clip and
    an adversarial entry sharing an id overwrite each other's audio and stamp."""
    data = {"clips": [_clip(id="shared")], "adversarial": [{"id": "shared"}]}

    errors = manifest.validate(data)

    assert any("duplicate id 'shared'" in message for message in errors)


def test_a_missing_id_is_not_also_reported_as_a_duplicate(languages):
    """Repeating it under a duplicates heading would only add noise."""
    errors = manifest.validate({"clips": [_clip(id=None), _clip(id=None)]})

    assert not any("duplicate" in message for message in errors)


def test_longform_is_validated_like_every_other_section(languages):
    """It is generated and cached the same way, so a longform entry with no id is the same
    defect -- it used to reach the generator and raise a bare KeyError mid-build."""
    errors = manifest.validate({"longform": {"profile": "stress", "shape": "film"}})

    assert any("longform" in message and "missing field id" in message for message in errors)


def test_longform_variants_are_validated_too(languages):
    """Each variant names its own profile and shape."""
    data = {"longform": {"id": "lf", "variants": [{"id": "lf_film", "profile": "nonsense"}]}}

    errors = manifest.validate(data)

    assert any("unknown profile 'nonsense'" in message for message in errors)


def test_longform_rejects_an_unimplemented_shape(languages):
    """A shape nothing implements used to reach the planner after the sources were
    rendered."""
    errors = manifest.validate({"longform": {"id": "lf", "shape": "nonsense"}})

    assert any("unknown shape 'nonsense'" in message for message in errors)


def test_longform_defaults_profile_and_shape_when_absent(languages):
    """Omitting them selects the defaults rather than failing validation."""
    assert manifest.validate({"longform": {"id": "lf"}}) == []


def test_a_longform_id_colliding_with_a_clip_is_detected(languages):
    """It caches in the same directory, and was the one entry the check could not see."""
    data = {"clips": [_clip(id="shared")], "longform": {"id": "shared"}}

    errors = manifest.validate(data)

    assert any("duplicate id 'shared'" in message for message in errors)


def test_a_non_object_longform_spec_is_reported_rather_than_raising(languages):
    """Handed through as-is so _section_errors describes it instead of `.get` raising."""
    errors = manifest.validate({"longform": "film"})

    assert any("expected an object, got str" in message for message in errors)
