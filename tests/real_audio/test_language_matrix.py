"""Real-engine transcription and language detection across the language matrix.

Every other test in the suite mocks the ASR engine, so a regression that only affects
non-English audio -- a broken language hint, a detection model that never loads, an
accelerator path that decodes garbage for non-Latin scripts -- passes all of them. These
tests drive a **running** service over HTTP with real speech in each language.

Tier A asserts both content and identity: the transcript must contain the words actually
spoken, and detection must name the right language. Tier B is the long tail, held to the
weaker contract of "detected correctly, or at least transcribed to something", because
per-language accuracy varies more than a single threshold can express. Which tier a
language sits in is manifest data, never a branch here.

Run against a live stack (see docs/SETUP.md):

    RUN_REAL_ASR=1 python3 -m pytest tests/real_audio -m real_audio
"""

from __future__ import annotations

import pytest

from tests.real_audio import matrix_support, service_client

pytestmark = matrix_support.REAL_ASR_MARKS


def _overlap_for(clip: dict, payload: dict) -> float:
    """Return how much of the clip's expected wording the transcript actually contains."""
    spoken = matrix_support.spoken_words(payload)
    return matrix_support.word_overlap(clip["expect_words"], spoken, clip.get("tokenizer", "words"))


@pytest.mark.usefixtures("service_ready")
@pytest.mark.parametrize("clip", matrix_support.clip_params("clips", tier="A"))
def test_tier_a_transcribes_the_spoken_words(clip, clip_path):
    """A tier-A language must transcribe to the words the fixture actually says."""
    payload = service_client.transcribe(clip_path(clip["id"]))
    overlap = _overlap_for(clip, payload)
    threshold = float(clip["min_word_overlap"])
    assert overlap >= threshold, f"{clip['language']}: word overlap {overlap:.2f} < {threshold:.2f}; got {payload.get('text')!r}"


@pytest.mark.usefixtures("service_ready")
@pytest.mark.parametrize("clip", matrix_support.clip_params("clips", tier="A"))
def test_tier_a_detects_the_spoken_language(clip, clip_path):
    """Detection runs its own preprocessing path, so a good transcript does not imply it."""
    payload = service_client.detect(clip_path(clip["id"]))
    detected = matrix_support.detected_code(payload)
    assert detected in clip["expect_detect"], f"expected one of {clip['expect_detect']}, detected {detected!r}"


@pytest.mark.usefixtures("service_ready")
@pytest.mark.parametrize("clip", matrix_support.clip_params("clips", tier="B"))
def test_tier_b_is_recognized_or_transcribed(clip, clip_path):
    """A long-tail language must at least be identified, or produce some transcript."""
    path = clip_path(clip["id"])
    detected = matrix_support.detected_code(service_client.detect(path))
    transcript = " ".join(matrix_support.spoken_words(service_client.transcribe(path))).strip()
    assert detected in clip["expect_detect"] or transcript, f"{clip['language']}: detected {detected!r} and transcript was empty"


@pytest.mark.usefixtures("service_ready")
@pytest.mark.parametrize("clip", matrix_support.clip_params("clips"))
def test_segments_span_the_clip(clip, clip_path):
    """Timings must reach the end of the clip rather than stopping after the first phrase."""
    path = clip_path(clip["id"])
    payload = service_client.transcribe(path)
    segments = [s for s in (payload.get("segments") or []) if str(s.get("text", "")).strip()]
    if not segments:
        pytest.skip(f"{clip['id']}: no segments returned; content is asserted by the tier tests")
    last_end = max(float(seg["end"]) for seg in segments)
    try:
        duration = matrix_support.clip_duration(path)
    except FileNotFoundError:
        # clip_duration shells out to ffprobe, which the service has but the *test* image
        # need not. Without this the whole check errors with a bare "No such file or
        # directory: 'ffprobe'", which reads like the audio fixture is missing.
        pytest.skip("ffprobe is unavailable here, so the clip's duration cannot be measured")
    assert last_end > duration * 0.5, f"transcription stopped at {last_end:.1f}s of a {duration:.1f}s clip"
