"""HTTP client for driving a live Whisper Pro ASR service from the real-audio tests.

These tests talk to a running container rather than an in-process app for the same reason
``tests/integration/test_transcription_accuracy.py`` does: the real engine needs the
per-vendor ONNX Runtime under ``/app/libs`` and a provisioned ``model_cache``, neither of
which exists in the test image.
"""

from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from typing import Any

import pytest

BASE_URL = os.environ.get("WHISPER_BASE_URL", "http://127.0.0.1:9000")

# A first start downloads the model while tasks wait in the queue, so allow for that.
REQUEST_TIMEOUT_SEC = float(os.environ.get("REAL_ASR_TIMEOUT", "900"))

# Malformed input must be rejected promptly. A far tighter budget than the normal timeout
# turns "this corrupt file pinned a worker forever" into a test failure instead of a
# 15-minute hang that looks like success.
ADVERSARIAL_TIMEOUT_SEC = float(os.environ.get("REAL_ASR_ADVERSARIAL_TIMEOUT", "120"))

# The health check answers from memory, so it needs nothing like the request budget -- but
# it was the one timeout in this module written as a bare literal, which meant a slow or
# heavily loaded host could not be accommodated the way every other timeout here can.
HEALTH_TIMEOUT_SEC = float(os.environ.get("REAL_ASR_HEALTH_TIMEOUT", "10"))


def _httpx():
    """Return the httpx2 module, skipping the test when it is unavailable."""
    return pytest.importorskip("httpx2", reason="httpx2 is required to drive the live service")


def _mime_for(path: Path) -> str:
    """Return a best-effort content type for an upload."""
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def post_audio(
    path: Path,
    endpoint: str = "/v1/audio/transcriptions?output=json",
    field: str = "audio_file",
    data: dict[str, Any] | None = None,
    timeout: float | None = None,
):
    """Upload ``path`` to ``endpoint`` and return the raw response."""
    httpx = _httpx()
    with path.open("rb") as handle:
        return httpx.post(
            f"{BASE_URL}{endpoint}",
            files={field: (path.name, handle, _mime_for(path))},
            data=data or {},
            timeout=REQUEST_TIMEOUT_SEC if timeout is None else timeout,
        )


def post_audio_json(path: Path, endpoint: str = "/v1/audio/transcriptions?output=json", **kwargs) -> dict:
    """Upload ``path`` and return the decoded JSON payload, asserting a 200."""
    response = post_audio(path, endpoint, **kwargs)
    assert response.status_code == 200, f"{response.status_code}: {response.text}"
    return response.json()


def transcribe(path: Path, data: dict[str, Any] | None = None) -> dict:
    """Transcribe ``path`` through the OpenAI-compatible endpoint."""
    return post_audio_json(path, "/v1/audio/transcriptions?output=json", data=data)


def translate(path: Path, data: dict[str, Any] | None = None) -> dict:
    """Translate ``path`` to English through the OpenAI-compatible endpoint."""
    return post_audio_json(path, "/v1/audio/translations?output=json", data=data)


def detect(path: Path) -> dict:
    """Run language detection on ``path``."""
    return post_audio_json(path, "/detect-language")


def post_promptly(path: Path, endpoint: str = "/v1/audio/transcriptions?output=json", field: str = "audio_file"):
    """Upload ``path`` under the tight adversarial budget, failing the test on a hang.

    Malformed input that pins a worker until the normal 15-minute timeout looks like a slow
    success to a caller; here it is an explicit failure.
    """
    httpx = _httpx()
    try:
        return post_audio(path, endpoint, field=field, timeout=ADVERSARIAL_TIMEOUT_SEC)
    except httpx.TimeoutException as error:
        raise AssertionError(f"{path.name}: no response within {ADVERSARIAL_TIMEOUT_SEC}s; the request appears to have hung") from error


def service_is_up() -> bool:
    """Return whether the service answers at ``WHISPER_BASE_URL``.

    httpx raises its own transport errors -- ConnectError for a refused port, and
    TimeoutException for one that never answers -- and neither derives from OSError. With
    only OSError caught, "no service running" propagated instead of returning False, so the
    suite errored out where it was supposed to skip.
    """
    httpx = _httpx()
    try:
        return httpx.get(f"{BASE_URL}/status", timeout=HEALTH_TIMEOUT_SEC).status_code == 200
    except (httpx.HTTPError, OSError):
        return False
