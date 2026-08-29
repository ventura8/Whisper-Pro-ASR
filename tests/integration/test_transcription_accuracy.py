"""Real-engine transcription accuracy against a known-text speech fixture.

Every other test in the suite mocks the ASR engine, so a silently broken accelerator
path -- a wrong CUDA major, a missing ONNX Runtime, a model that loads but decodes
garbage -- still passes them all. This test drives a **running** service over HTTP and
asserts the transcript matches what the fixture actually says.

The fixture is synthesized speech (espeak-ng) of:

    "The quick brown fox jumps over the lazy dog. Whisper Pro ASR is running a
     hardware acceleration test on this machine."

It talks to a live container rather than an in-process app because the real engine needs
the per-vendor ONNX Runtime under /app/libs and a provisioned model_cache, neither of
which exists in the test image.

Run it after bringing the stack up (see docs/SETUP.md, "Local hardware validation"):

    docker compose -f docker-compose.yml -f docker-compose.nvidia.yml up -d
    RUN_REAL_ASR=1 python3 -m pytest tests/integration/test_transcription_accuracy.py

Point it elsewhere with WHISPER_BASE_URL. Skipped by default so CI stays fast.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.real_audio.matrix_support import spoken_words

FIXTURE = Path(__file__).resolve().parents[1] / "e2e" / "fixtures" / "speech_known_text.wav"
BASE_URL = os.environ.get("WHISPER_BASE_URL", "http://127.0.0.1:9000")

EXPECTED_PHRASES = (
    "the quick brown fox jumps over the lazy dog",
    "whisper pro asr is running a hardware acceleration test on this machine",
)

# A first start downloads the model while tasks wait in the queue, so allow for that.
REQUEST_TIMEOUT_SEC = float(os.environ.get("REAL_ASR_TIMEOUT", "900"))

pytestmark = [
    pytest.mark.real_asr,
    pytest.mark.skipif(
        os.environ.get("RUN_REAL_ASR") != "1",
        reason="Real-engine ASR test; start the stack and set RUN_REAL_ASR=1 to run.",
    ),
]


def _post_fixture(path: str = "/v1/audio/transcriptions?output=json", field: str = "audio_file"):
    """Submit the fixture to a live endpoint and return the response."""
    httpx = pytest.importorskip("httpx2", reason="httpx2 is required to drive the live service")
    with FIXTURE.open("rb") as handle:
        return httpx.post(
            f"{BASE_URL}{path}",
            files={field: ("speech_known_text.wav", handle, "audio/wav")},
            timeout=REQUEST_TIMEOUT_SEC,
        )


def _post_fixture_json(path: str = "/v1/audio/transcriptions?output=json", field: str = "audio_file") -> dict:
    """Submit the fixture and return the decoded JSON payload, asserting a 200."""
    response = _post_fixture(path, field)
    assert response.status_code == 200, f"{response.status_code}: {response.text}"
    return response.json()


def test_fixture_is_present_and_non_trivial():
    """Guard the fixture itself, so a missing or truncated file fails loudly here."""
    assert FIXTURE.exists(), f"missing speech fixture: {FIXTURE}"
    assert FIXTURE.stat().st_size > 100_000


def test_transcribes_known_speech_accurately():
    """The real engine must return the fixture's actual sentences, not garbage.

    This is the check that catches a broken accelerator: CPU fallback still produces
    correct text, but a half-working GPU path produces empty or nonsense output.
    """
    transcript = " ".join(spoken_words(_post_fixture_json()))
    for phrase in EXPECTED_PHRASES:
        assert phrase in transcript, f"expected {phrase!r} in transcript, got: {transcript!r}"


def test_segments_cover_the_full_clip():
    """Timings must span the ~8.3s clip rather than stopping after the first sentence."""
    segments = [s for s in (_post_fixture_json().get("segments") or []) if str(s.get("text", "")).strip()]
    assert segments, "expected at least one non-empty segment"

    last_end = max(float(seg["end"]) for seg in segments)
    assert last_end > 6.0, f"transcription stopped early at {last_end}s for an 8.3s clip"


def test_detect_language_identifies_english():
    """Language detection must identify the fixture's English speech.

    Detection runs its own preprocessing path (LD preprocessing, request coalescing), so a
    working /v1 transcription does not imply a working /detect-language.
    """
    payload = _post_fixture_json("/detect-language")

    language = str(payload.get("language") or payload.get("detected_language") or "").lower()
    code = str(payload.get("language_code") or "").lower()
    assert "en" in (language, code) or language.startswith("english"), f"expected English, got: {payload!r}"

    confidence = payload.get("confidence")
    if confidence is not None:
        assert float(confidence) > 0.5, f"low detection confidence on clear speech: {confidence}"


def test_detectlang_alias_matches_detect_language():
    """The /detectlang alias must behave identically to /detect-language."""
    primary = _post_fixture_json("/detect-language")
    alias = _post_fixture_json("/detectlang")

    def _lang(payload: dict) -> str:
        return str(payload.get("language") or payload.get("detected_language") or "").lower()

    assert _lang(alias) == _lang(primary)


def test_v1_transcriptions_returns_openai_shaped_payload():
    """The OpenAI-compatible v1 endpoint must return a populated `text` field."""
    payload = _post_fixture_json()

    assert "text" in payload, f"v1 response missing 'text': {sorted(payload)}"
    assert str(payload["text"]).strip(), "v1 response 'text' is empty"


def test_v1_translations_returns_english_text():
    """/v1/audio/translations must transcribe the already-English fixture to English."""
    transcript = " ".join(spoken_words(_post_fixture_json("/v1/audio/translations?output=json")))
    assert EXPECTED_PHRASES[0] in transcript, f"expected the fox sentence, got: {transcript!r}"


def test_v1_accepts_the_openai_file_field_name():
    """OpenAI clients send the field as `file`; both spellings must work."""
    payload = _post_fixture_json(field="file")
    assert str(payload.get("text", "")).strip(), "v1 rejected the OpenAI 'file' field name"


def test_legacy_asr_endpoint_still_transcribes():
    """The legacy /asr endpoint (used by Bazarr) must keep working."""
    response = _post_fixture("/asr?output=json")
    assert response.status_code == 200, f"{response.status_code}: {response.text}"
    assert str(response.json().get("text", "")).strip(), "legacy /asr returned empty text"
