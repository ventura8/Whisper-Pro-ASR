"""Code-switched audio: one file, more than one language.

A dubbed interview, a bilingual speaker, a support call that switches halfway -- real media
does this constantly, and it is exactly where "detect one language for the whole file"
quietly returns the wrong answer for half the content. These clips are built by
concatenating legs rendered in different languages, with the leg boundaries recorded in a
sidecar so the assertions can be about *which* language appeared, not just that something
came back.
"""

from __future__ import annotations

import json

import pytest

from tests.real_audio import matrix_support, service_client

pytestmark = matrix_support.REAL_ASR_MARKS


@pytest.mark.usefixtures("service_ready")
@pytest.mark.parametrize("clip", matrix_support.clip_params("combined", scope="content"))
def test_mixed_language_clip_transcribes_both_halves(clip, clip_path):
    """Both languages' content must survive; dropping one half is the failure to catch."""
    payload = service_client.transcribe(clip_path(clip["id"]))
    spoken = matrix_support.spoken_words(payload)
    overlap = matrix_support.word_overlap(clip["expect_words"], spoken, clip.get("tokenizer", "words"))
    threshold = float(clip["min_word_overlap"])
    assert overlap >= threshold, f"{clip['id']}: overlap {overlap:.2f} < {threshold:.2f}; got {payload.get('text')!r}"


@pytest.mark.usefixtures("service_ready")
@pytest.mark.parametrize("clip", matrix_support.clip_params("combined", scope="translation"))
def test_mixed_language_clip_translates_both_legs_to_english(clip, clip_path):
    """Translation goes through the same per-language decode, so each leg must be *translated*
    in its own language: the English of both legs present, the foreign leg's own words gone.

    A leg transcribed instead of translated -- Spanish left in Spanish because the file was
    decoded under one language token -- fails the second bar even when the first is met by
    the English leg alone. An engine that drops a leg on transcription drops it here too,
    so those entries name both concerns in their ``xfail_scope``.
    """
    path = clip_path(clip["id"])
    payload = service_client.translate(path)
    segments = payload.get("segments") or []
    threshold = float(clip.get("min_translation_overlap", 0.5))
    # Judged leg by leg, in each leg's own stretch of the file. Scored on the whole
    # transcript, `mix_en_es` -- whose two legs say the same sentence -- passed on the English
    # leg alone with the Spanish one dropped entirely.
    windows = _legs(path)
    assert len(windows) == len(clip["legs"]), f"{clip['id']}: {len(windows)} legs in the sidecar, {len(clip['legs'])} in the manifest"
    for window, leg in zip(windows, clip["legs"]):
        expected = matrix_support.words(leg["text"] if leg["language"] == "en" else leg.get("translation", ""))
        assert expected, f"{clip['id']}: the {leg['language']} leg has no text or translation in the manifest"
        heard = matrix_support.words(matrix_support.text_in_window(segments, window["start"], window["end"]))
        overlap = matrix_support.word_overlap(expected, heard)
        assert overlap >= threshold, f"{clip['id']}: {leg['language']} leg translated at {overlap:.2f} < {threshold:.2f}; got {heard!r}"
        if leg["language"] != "en":
            as_spoken = matrix_support.word_overlap(matrix_support.words(leg["text"]), heard)
            assert as_spoken < 0.5, f"{clip['id']}: {leg['language']} leg came back as spoken ({as_spoken:.2f}); got {heard!r}"


@pytest.mark.usefixtures("service_ready")
@pytest.mark.parametrize("clip", matrix_support.clip_params("combined", scope="translation"))
def test_forced_transcription_keeps_the_named_language_and_translates_the_other(clip, clip_path):
    """`force_transcription=true` with a named language: that language's leg comes back as
    spoken, the other leg translated into English -- one subtitle in the film's language
    where a plain transcription would decode the foreign leg under the named token.

    The named language is the English leg's where the clip has one, else the first leg's,
    so both orders of the rule are exercised: a leg transcribed as itself and a leg
    translated, in each leg's own window.
    """
    path = clip_path(clip["id"])
    named = next((leg["language"] for leg in clip["legs"] if leg["language"] == "en"), clip["legs"][0]["language"])
    payload = service_client.transcribe(path, data={"language": named, "force_transcription": "true"})
    segments = payload.get("segments") or []
    threshold = float(clip.get("min_translation_overlap", 0.5))
    windows = _legs(path)
    assert len(windows) == len(clip["legs"]), f"{clip['id']}: {len(windows)} legs in the sidecar, {len(clip['legs'])} in the manifest"
    for window, leg in zip(windows, clip["legs"]):
        heard = matrix_support.words(matrix_support.text_in_window(segments, window["start"], window["end"]))
        as_spoken = matrix_support.word_overlap(matrix_support.words(leg["text"]), heard)
        if leg["language"] == named:
            assert as_spoken >= threshold, f"{clip['id']}: the {named} leg should be transcribed as spoken ({as_spoken:.2f}); got {heard!r}"
            continue
        translated = matrix_support.word_overlap(matrix_support.words(leg.get("translation", "")), heard)
        assert translated >= threshold, f"{clip['id']}: the {leg['language']} leg should be in English ({translated:.2f}); got {heard!r}"
        assert as_spoken < 0.5, f"{clip['id']}: the {leg['language']} leg came back as spoken ({as_spoken:.2f}); got {heard!r}"


def _legs(path) -> list[dict]:
    """The leg boundaries the generator wrote beside the clip -- where each language is."""
    sidecar = path.with_suffix(".legs.json")
    assert sidecar.exists(), f"missing leg sidecar {sidecar}; regenerate with scripts/generate_audio_matrix.py combined"
    legs = json.loads(sidecar.read_text(encoding="utf-8"))["legs"]
    assert legs, f"{sidecar} lists no legs"
    return legs


@pytest.mark.usefixtures("service_ready")
@pytest.mark.parametrize("clip", matrix_support.clip_params("combined"))
def test_mixed_language_detection_picks_a_language_present(clip, clip_path):
    """Detection must name one of the languages actually spoken, not a third one."""
    detected = matrix_support.detected_code(service_client.detect(clip_path(clip["id"])))
    assert detected in clip["expect_detect_any"], f"{clip['id']}: detected {detected!r}, expected one of {clip['expect_detect_any']}"


@pytest.mark.usefixtures("service_ready")
@pytest.mark.parametrize("clip", matrix_support.clip_params("combined"))
def test_mixed_language_clip_does_not_error(clip, clip_path):
    """Mixed-language audio must not be treated as a malformed request."""
    response = service_client.post_audio(clip_path(clip["id"]))
    assert response.status_code == 200, f"{response.status_code}: {response.text}"
