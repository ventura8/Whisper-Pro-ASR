"""Fixtures for the real-audio matrix tests.

Resolution order for every clip is: committed core-tier FLAC, then the local generation
cache, then on-demand generation. Generation is a *convenience*, never a requirement --
when piper-tts is not installed the clip is skipped with an actionable message rather than
failing, so a checkout with no TTS tooling still runs the whole committed core tier.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.real_audio import matrix_support, service_client

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR = REPO_ROOT / "scripts" / "generate_audio_matrix.py"

CACHE_ROOT = Path(os.environ.get("ASR_AUDIO_MATRIX_DIR", str(REPO_ROOT / "test_data" / "audio_matrix")))


@pytest.fixture(scope="session")
def manifest() -> dict:
    """The audio-matrix manifest: the single source of truth for clips and expectations."""
    return matrix_support.load_manifest()


@pytest.fixture(scope="session")
def audio_matrix_dir() -> Path:
    """Root of the local generation cache for non-committed clips."""
    return CACHE_ROOT


@pytest.fixture(scope="session")
def service_ready() -> str:
    """Return the live service's base URL, skipping when nothing answers there."""
    if not service_client.service_is_up():
        pytest.skip(f"no service at {service_client.BASE_URL}; bring the stack up first (see docs/SETUP.md)")
    return service_client.BASE_URL


def _committed_path(clip_id: str) -> Path:
    """Path a committed core-tier clip would occupy."""
    return matrix_support.MATRIX_DIR / "core" / f"{clip_id}.flac"


def _cached_path(clip_id: str) -> Path:
    """Path a generated clip would occupy in the local cache."""
    return CACHE_ROOT / f"{clip_id}.wav"


#: A single clip is TTS synthesis plus an ffmpeg pass. Bounded so a generator that wedges
#: -- waiting on a voice download that never completes, say -- fails the collection instead
#: of hanging the whole suite with no output.
GENERATE_TIMEOUT_SEC = float(os.environ.get("REAL_ASR_GENERATE_TIMEOUT", "600"))


def _generate(clip_id: str) -> str:
    """Ask the generator for one clip, returning what it said about any failure.

    A failure here is reported by the caller as a skip carrying the generator's own message,
    which is more useful than this function raising a CalledProcessError -- but only if the
    message survives. It used to be captured and dropped, so a clip that could not be
    generated skipped with a generic "install the tools" hint no matter what actually went
    wrong (a missing phonemizer, a voice checksum mismatch, a full disk).
    """
    if not GENERATOR.exists():
        return f"generator not found at {GENERATOR}"
    # Fixed argv list against a repo-local script; no shell, no user-controlled input.
    try:
        completed = subprocess.run(
            [sys.executable, str(GENERATOR), "--only", clip_id, "--out", str(CACHE_ROOT)],
            check=False,
            capture_output=True,
            text=True,
            timeout=GENERATE_TIMEOUT_SEC,
            cwd=str(REPO_ROOT),
        )
    except subprocess.TimeoutExpired:
        return f"the generator exceeded {GENERATE_TIMEOUT_SEC:.0f}s and was killed"
    output = f"{completed.stdout or ''}\n{completed.stderr or ''}".strip()
    lines = [line for line in output.splitlines() if line.strip()]
    return lines[-1] if lines else f"the generator exited {completed.returncode} with no output"


def resolve_clip(clip_id: str) -> Path:
    """Return a playable path for ``clip_id``, generating it on demand, or skip."""
    for candidate in (_committed_path(clip_id), _cached_path(clip_id)):
        if candidate.exists():
            return candidate
    reason = _generate(clip_id)
    cached = _cached_path(clip_id)
    if not cached.exists():
        hint = f"poetry install --with tools && python3 {GENERATOR.relative_to(REPO_ROOT)} all"
        pytest.skip(f"clip {clip_id!r} is not committed and could not be generated ({reason}); run: {hint}")
    return cached


@pytest.fixture(scope="session")
def clip_path():
    """Return a resolver that maps a manifest clip id to a playable file path."""
    return resolve_clip
