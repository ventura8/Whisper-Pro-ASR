"""Code-switched audio: one file, more than one language.

A dubbed interview, a bilingual speaker, a support call that switches halfway -- real media
does this constantly, and it is exactly where "detect one language for the whole file"
quietly returns the wrong answer for half the content. These clips are built by
concatenating legs rendered in different languages, with the leg boundaries recorded in a
sidecar so the assertions can be about *which* language appeared, not just that something
came back.
"""

from __future__ import annotations

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
