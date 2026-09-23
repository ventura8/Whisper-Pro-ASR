"""Tests for scripts/audio_matrix/piper.py, the TTS adapter behind every spoken fixture.

Two properties here are load-bearing for the committed fixtures and are what these tests
pin: the determinism arguments reach the CLI (Piper exposes no seed and samples noise per
run without them, so the fixtures would churn on every rebuild), and a voice is verified
against its published digest so a re-download cannot swap the voice underneath a
calibrated accuracy threshold.
"""

import hashlib
import subprocess
import sys
from importlib.metadata import PackageNotFoundError
from unittest import mock

import pytest

from scripts.audio_matrix import piper

PINS = {"length_scale": 1.0, "noise_scale": 0, "noise_w_scale": 0, "sentence_silence": 0.35}


def test_piper_available_reflects_the_import_spec():
    """Availability is a real import check, not a PATH guess."""
    with mock.patch.object(piper.importlib.util, "find_spec", return_value=object()):
        assert piper.piper_available() is True

    with mock.patch.object(piper.importlib.util, "find_spec", return_value=None):
        assert piper.piper_available() is False


def test_piper_version_is_empty_when_piper_is_absent():
    """The version feeds the cache digest, so an absent toolchain contributes nothing."""
    with mock.patch.object(piper, "piper_available", return_value=False):
        assert piper.piper_version() == ""


def test_piper_version_returns_the_installed_version():
    """A present piper-tts reports its version for the digest."""
    with mock.patch.object(piper, "piper_available", return_value=True):
        with mock.patch("importlib.metadata.version", return_value="1.8.0"):
            assert piper.piper_version() == "1.8.0"


def test_piper_version_tolerates_a_missing_distribution():
    """Importable but not installed as a distribution is not a crash."""
    with mock.patch.object(piper, "piper_available", return_value=True):
        with mock.patch("importlib.metadata.version", side_effect=PackageNotFoundError):
            assert piper.piper_version() == ""


def test_voice_paths_live_under_the_cache(tmp_path):
    """Voices are downloaded into the cache, never committed."""
    assert piper.voice_dir(tmp_path) == tmp_path / piper.VOICES_SUBDIR
    assert piper.voice_path(tmp_path, "en_US-amy").name == "en_US-amy.onnx"


def test_file_md5_matches_hashlib(tmp_path):
    """The digest is compared against what upstream publishes in voices.json."""
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"model-bytes")

    assert piper.file_md5(model) == hashlib.md5(b"model-bytes", usedforsecurity=False).hexdigest()


def test_download_voice_invokes_pipers_own_downloader(tmp_path):
    """Downloading goes through piper's module so the cache layout matches its expectations."""
    with mock.patch.object(piper.subprocess, "run") as run:
        piper.download_voice(tmp_path, "en_US-amy")

    args = run.call_args[0][0]
    assert args[:3] == [sys.executable, "-m", "piper.download_voices"]
    assert args[-1] == "en_US-amy"
    assert run.call_args.kwargs["timeout"] == piper.PIPER_TIMEOUT_SEC
    assert (tmp_path / piper.VOICES_SUBDIR).is_dir()


def _place_voice(tmp_path, payload=b"model-bytes"):
    """Write both halves of a cached voice and return the model path."""
    directory = piper.voice_dir(tmp_path)
    directory.mkdir(parents=True, exist_ok=True)
    model = directory / "en_US-amy.onnx"
    model.write_bytes(payload)
    model.with_suffix(".onnx.json").write_text("{}", encoding="utf-8")
    return model


def test_ensure_voice_skips_the_download_when_both_halves_are_cached(tmp_path):
    """A complete cache entry is used as is."""
    model = _place_voice(tmp_path)
    digest = piper.file_md5(model)

    with mock.patch.object(piper, "download_voice") as download:
        assert piper.ensure_voice(tmp_path, "en_US-amy", digest) == model

    download.assert_not_called()


def test_ensure_voice_redownloads_when_only_the_onnx_is_present(tmp_path):
    """Piper reads <voice>.onnx.json for the phoneme map and sample rate.

    A cache holding only the .onnx satisfied an earlier existence check and then failed
    inside synthesis -- once per clip, with an error naming neither the voice nor the
    missing sidecar.
    """
    directory = piper.voice_dir(tmp_path)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "en_US-amy.onnx").write_bytes(b"model-bytes")

    def _download(cache, voice):
        _place_voice(cache)

    with mock.patch.object(piper, "download_voice", side_effect=_download) as download:
        piper.ensure_voice(tmp_path, "en_US-amy", "")

    download.assert_called_once()


def test_ensure_voice_downloads_when_nothing_is_cached(tmp_path):
    """The ordinary first run."""
    with mock.patch.object(piper, "download_voice", side_effect=lambda cache, voice: _place_voice(cache)) as download:
        piper.ensure_voice(tmp_path, "en_US-amy", "")

    download.assert_called_once()


def test_ensure_voice_rejects_a_voice_failing_its_checksum(tmp_path):
    """A swapped or truncated download must not silently back a calibrated threshold."""
    _place_voice(tmp_path)

    with mock.patch.object(piper, "download_voice"):
        with pytest.raises(ValueError, match="failed checksum verification"):
            piper.ensure_voice(tmp_path, "en_US-amy", "0" * 32)


def test_ensure_voice_skips_verification_when_no_digest_is_published(tmp_path):
    """Some voices publish no md5; absence of a digest is not a failed check."""
    model = _place_voice(tmp_path)

    with mock.patch.object(piper, "download_voice"):
        assert piper.ensure_voice(tmp_path, "en_US-amy", "") == model


def test_synth_pins_the_stochastic_terms_to_zero(tmp_path):
    """Without these the VITS graph samples noise per run and the fixtures churn.

    The pins come from the manifest's defaults, so they are reviewable data rather than
    constants buried in this module -- the test asserts they reach the CLI.
    """
    with mock.patch.object(piper.subprocess, "run") as run:
        piper.synth("hello", tmp_path / "voice.onnx", tmp_path / "out.wav", PINS)

    args = run.call_args[0][0]
    assert args[args.index("--noise-scale") + 1] == "0"
    assert args[args.index("--noise-w-scale") + 1] == "0"
    assert args[args.index("--sentence-silence") + 1] == "0.35"
    assert run.call_args.kwargs["input"] == "hello"
    assert run.call_args.kwargs["timeout"] == piper.PIPER_TIMEOUT_SEC


def test_synth_propagates_a_timeout(tmp_path):
    """cli._try_build catches this and reports one failed entry rather than hanging."""
    with mock.patch.object(piper.subprocess, "run", side_effect=subprocess.TimeoutExpired("piper", 600)):
        with pytest.raises(subprocess.TimeoutExpired):
            piper.synth("hello", tmp_path / "voice.onnx", tmp_path / "out.wav", PINS)
