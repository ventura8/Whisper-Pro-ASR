"""Piper neural TTS adapter: the matrix's only speech source.

Two things here are load-bearing for reproducibility:

1. **Determinism.** Piper's CLI exposes no seed, and its VITS graph samples noise per run,
   so default settings produce a different waveform every time -- verified by hashing
   repeated renders. Pinning ``--noise-scale 0`` and ``--noise-w-scale 0`` removes the
   stochastic term entirely and makes output bit-identical across runs, which is what lets
   the committed fixtures round-trip with an empty ``git diff``.
2. **Voice pinning.** Voice models are downloaded, never committed. Each is verified
   against the ``md5_digest`` upstream publishes in ``voices.json``, so a re-download
   cannot silently swap the voice underneath a calibrated accuracy threshold.
"""

from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

VOICES_SUBDIR = "voices"

#: Ceiling for one Piper invocation. Synthesis of a two-sentence clip takes seconds and a
#: voice download a minute or two; a call still running after this is wedged, and without a
#: bound it hangs the whole matrix instead of failing one entry. cli._try_build catches the
#: resulting TimeoutExpired and reports that entry as failed.
PIPER_TIMEOUT_SEC = 600


def piper_available() -> bool:
    """Return whether the piper-tts package is importable."""
    return importlib.util.find_spec("piper") is not None


def piper_version() -> str:
    """Return the installed piper-tts version, or an empty string."""
    if not piper_available():
        return ""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("piper-tts")
    except PackageNotFoundError:
        return ""


def voice_dir(cache: Path) -> Path:
    """Return the directory holding downloaded voice models."""
    return cache / VOICES_SUBDIR


def voice_path(cache: Path, voice: str) -> Path:
    """Return the ONNX model path for ``voice``."""
    return voice_dir(cache) / f"{voice}.onnx"


def file_md5(path: Path) -> str:
    """Return the MD5 digest of ``path``.

    MD5 is used only to match the digests upstream publishes in ``voices.json``; it is an
    integrity check against a truncated or swapped download, not a security boundary.
    """
    digest = hashlib.md5(path.read_bytes(), usedforsecurity=False)
    return digest.hexdigest()


def download_voice(cache: Path, voice: str) -> None:
    """Download ``voice`` into the cache using piper's own downloader."""
    voice_dir(cache).mkdir(parents=True, exist_ok=True)
    args = [sys.executable, "-m", "piper.download_voices", "--download-dir", str(voice_dir(cache)), voice]
    subprocess.run(args, check=True, capture_output=True, timeout=PIPER_TIMEOUT_SEC)


def ensure_voice(cache: Path, voice: str, expected_md5: str) -> Path:
    """Return a verified local path for ``voice``, downloading it when absent."""
    model = voice_path(cache, voice)
    if not model.exists():
        download_voice(cache, voice)
    if expected_md5 and file_md5(model) != expected_md5:
        raise ValueError(f"voice {voice!r} failed checksum verification; delete {model} and retry")
    return model


def synth(text: str, model: Path, dest: Path, pins: dict) -> None:
    """Render ``text`` to ``dest`` with ``model``, using the determinism pins.

    ``pins`` comes from the manifest's ``defaults`` so the settings that make output
    reproducible are reviewable data rather than constants buried here.
    """
    args = [
        sys.executable,
        "-m",
        "piper",
        "--model",
        str(model),
        "--output-file",
        str(dest),
        "--length-scale",
        str(pins["length_scale"]),
        "--noise-scale",
        str(pins["noise_scale"]),
        "--noise-w-scale",
        str(pins["noise_w_scale"]),
        "--sentence-silence",
        str(pins["sentence_silence"]),
    ]
    subprocess.run(args, input=text, text=True, check=True, capture_output=True, timeout=PIPER_TIMEOUT_SEC)
