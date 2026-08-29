"""Requests that lie about, or disagree with, the audio they carry.

The audio in these tests is fine. What is wrong is the request around it: a `language`
hint naming a language the file does not contain, a language code the service has never
heard of, a translation request for audio that is already English, or the legacy parameter
spelling a client still sends.

The contract asserted here is narrow on purpose. Whether the service *obeys* a wrong hint
or overrides it is a product decision, not a correctness one; what must not happen is a
500, a hang, or a silently empty result.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tests.real_audio import matrix_support, service_client

# Cheap, high-value, and not manifest-parametrized: always part of the smoke set.
pytestmark = [*matrix_support.REAL_ASR_MARKS, pytest.mark.smoke]

REPO_ROOT = Path(__file__).resolve().parents[2]
ACCEPTED_OR_REJECTED = (200, 400, 415, 422)


def _known_languages() -> set[str]:
    """Return the service's supported language codes, loaded without importing the app."""
    spec = importlib.util.spec_from_file_location("_contract_languages", REPO_ROOT / "modules" / "core" / "languages.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return set(module.LANGUAGES)


def _assert_accepted_transcript(response, context: str) -> None:
    """A 200 must carry a real transcript, not merely a well-formed envelope.

    Asserting only that "text" is a key let a 200 with `{"text": null}` or `{"text": ""}`
    pass -- which is precisely the silently-empty result this module exists to rule out,
    listed in its own docstring alongside the 500 and the hang.
    """
    payload = response.json()
    assert "text" in payload, f"{context} produced a malformed response body: {payload!r}"
    text = payload["text"]
    assert isinstance(text, str), f"{context} returned a non-string transcript: {text!r}"
    assert text.strip(), f"{context} returned an empty transcript"


@pytest.fixture(name="spanish_clip")
def spanish_clip_fixture(clip_path) -> Path:
    """A clip of unambiguous Spanish speech, used to contradict the request's hint."""
    return clip_path("es_core")


@pytest.mark.usefixtures("service_ready")
def test_wrong_language_hint_does_not_break_the_request(spanish_clip):
    """A hint naming the wrong language must degrade the result, never crash the request."""
    response = service_client.post_audio(spanish_clip, data={"language": "ja"})
    assert response.status_code == 200, f"{response.status_code}: {response.text[:400]}"
    _assert_accepted_transcript(response, "a wrong language hint")


@pytest.mark.usefixtures("service_ready")
def test_unsupported_language_code_is_handled(spanish_clip):
    """A code the service does not know must be rejected or ignored, not fatal."""
    assert "xx" not in _known_languages(), "'xx' was chosen because it is not a supported code"
    response = service_client.post_audio(spanish_clip, data={"language": "xx"})
    assert response.status_code in ACCEPTED_OR_REJECTED, f"{response.status_code}: {response.text[:400]}"


@pytest.mark.usefixtures("service_ready")
def test_source_lang_alias_matches_language(spanish_clip):
    """`source_lang` is the legacy spelling of `language`; both must be accepted."""
    aliased = service_client.post_audio(spanish_clip, data={"source_lang": "es"})
    primary = service_client.post_audio(spanish_clip, data={"language": "es"})
    assert aliased.status_code == primary.status_code == 200, f"alias {aliased.status_code}, primary {primary.status_code}"
    _assert_accepted_transcript(aliased, "the source_lang alias")
    _assert_accepted_transcript(primary, "the language parameter")


#: Content words the English rendering of es_core is expected to contain. The clip says
#: "El veloz zorro marrón salta sobre el perro perezoso. Esta grabación verifica el
#: reconocimiento de voz en español." Wording varies between runs and engines, so the bar
#: is a handful of these rather than an exact transcript.
_EXPECTED_ENGLISH_WORDS = frozenset(
    {
        "quick",
        "fast",
        "brown",
        "fox",
        "jumps",
        "jumping",
        "over",
        "lazy",
        "dog",
        "this",
        "recording",
        "verifies",
        "checks",
        "speech",
        "voice",
        "recognition",
        "spanish",
    }
)

#: Spanish content words that must not survive translation. Their presence means the
#: service returned the source text rather than an English rendering.
_SOURCE_SPANISH_WORDS = frozenset({"zorro", "perezoso", "grabación", "grabacion", "reconocimiento", "español", "espanol"})


@pytest.mark.usefixtures("service_ready")
def test_translation_of_non_english_audio_returns_english(spanish_clip):
    """/v1/audio/translations must render non-English speech into English.

    Checked against known English content words rather than an ASCII ratio: Spanish is
    almost entirely ASCII too, so "grabacion verifica el reconocimiento" scored 1.0 on the
    old test and passed while being the untranslated source.
    """
    spoken = matrix_support.spoken_words(service_client.translate(spanish_clip))
    assert spoken, "translation returned no text"

    words = {word.strip(".,!?;:").lower() for word in spoken}
    matched = words & _EXPECTED_ENGLISH_WORDS
    leftover_spanish = words & _SOURCE_SPANISH_WORDS

    assert not leftover_spanish, f"translation still contains the Spanish source: {sorted(leftover_spanish)}"
    assert len(matched) >= 3, f"translation does not read as English (matched {sorted(matched)}): {' '.join(spoken)[:200]!r}"
