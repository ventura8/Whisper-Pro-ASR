"""HTTP client for driving a live Whisper Pro ASR service from the real-audio tests.

These tests talk to a running container rather than an in-process app for the same reason
``tests/integration/test_transcription_accuracy.py`` does: the real engine needs the
per-vendor ONNX Runtime under ``/app/libs`` and a provisioned ``model_cache``, neither of
which exists in the test image.
"""

from __future__ import annotations

import importlib
import mimetypes
import os
from pathlib import Path
from typing import Any

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

# The engine probe runs during *collection*, once per manifest entry until the service
# answers, in every ordinary gate run with no service at all. A refused port fails at once,
# but a host that swallows packets would cost the health timeout per entry -- twenty
# minutes of collection -- so the probe gets its own, short budget.
ENGINE_PROBE_TIMEOUT_SEC = float(os.environ.get("REAL_ASR_ENGINE_PROBE_TIMEOUT", "2"))


def _httpx():
    """Return the httpx2 module, failing rather than skipping when it is unavailable.

    ``importorskip`` turned a broken test image into a green run: every module here is
    already gated behind RUN_REAL_ASR=1, so by the time this executes the operator has
    explicitly asked for the real-engine checks. Reporting "nothing to run" for a missing
    client library hides exactly what this suite exists to catch. Same reasoning, and same
    wording, as tests/integration/test_transcription_accuracy.py::_http_client.
    """
    try:
        return importlib.import_module("httpx2")
    except ImportError as exc:
        raise AssertionError(
            "RUN_REAL_ASR=1 was set but httpx2 is not installed, so the live service cannot be driven. "
            "Run this through the Docker test image (scripts/ci/build-and-test.sh), which ships it."
        ) from exc


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


#: The engine the live service reported, once it has answered. Only a successful answer is
#: kept: caching a transient failure -- the service still starting when collection ran --
#: would apply every engine-scoped mark to a healthy service for the rest of the session,
#: and a clip another engine holds strictly would report XFAIL instead of failing.
_ENGINE_CACHE: dict[str, str] = {}


def forget_running_engine() -> None:
    """Drop the cached engine, so the next call asks the service again."""
    _ENGINE_CACHE.clear()


def running_engine() -> str | None:
    """Return the ASR engine the live service reports, or None if it cannot be asked.

    Cached once known: manifest collection asks once per entry, which is over a hundred
    times, and the engine a running service reports cannot change inside one session. A
    failed lookup is not cached, so a service that answers later is still consulted.

    A defect can belong to an engine rather than to a clip: WHISPERX commits to one language
    for a whole file, so it returns one leg of a code-switched clip whatever the manifest
    says, while FASTER-WHISPER decoding by speech region returns both. Recording that in
    prose meant every such entry produced an XPASS on the engine that handles it; asking the
    service which engine is answering lets the manifest name the engine instead.

    None on any failure, so a missing or unreachable service degrades to "apply the mark
    everywhere" -- the behaviour before there was an engine to consult.

    That includes a missing client library, which is why ``_httpx`` is called inside the
    guard here and nowhere else in this module. Everywhere else it is right for it to raise:
    the caller has set RUN_REAL_ASR=1 and a broken image must not read as a green run. This
    one runs during *collection*, in every ordinary gate run, long before any test decides
    whether it wants a service -- so raising here would turn "no real-audio dependencies
    installed" into a collection error for the whole suite.
    """
    if "engine" in _ENGINE_CACHE:
        return _ENGINE_CACHE["engine"]
    engine = _ask_running_engine()
    if engine:
        _ENGINE_CACHE["engine"] = engine
    return engine


def _ask_running_engine() -> str | None:
    try:
        httpx = _httpx()
    except AssertionError:
        return None
    try:
        response = httpx.get(f"{BASE_URL}/status", timeout=ENGINE_PROBE_TIMEOUT_SEC)
        if response.status_code != 200:
            return None
        return _engine_in(response.json())
    except (httpx.HTTPError, OSError, ValueError):
        return None


def _engine_in(payload) -> str | None:
    """The engine a ``/status`` payload names, or None when it does not say.

    Only a non-empty string counts: a payload that is not an object (a proxy's error body
    with a 200 attached) or an ``asr_engine`` that is not a string would otherwise be
    cached as the running engine for the session.
    """
    engine = payload.get("asr_engine") if isinstance(payload, dict) else None
    return engine if isinstance(engine, str) and engine else None
