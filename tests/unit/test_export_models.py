"""Tests for scripts/export_models.py, the build-time model exporter.

This module is awkward to import on purpose: it installs a torchaudio.backend shim before
anything that needs it loads, and it creates /models at import. Both are stubbed here so the
module can be exercised at all -- which is why it had no coverage previously.
"""

import importlib
import subprocess
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

_INJECTED = (
    "scripts.export_models",
    "torchaudio",
    "torchaudio.backend",
    "torchaudio.backend.common",
    "torchaudio._backend",
    "audio_separator",
    "audio_separator.separator",
)


def _load(torchaudio_stub, extra_modules=None):
    """Import scripts.export_models fresh against a given torchaudio stub.

    The stubs are placed in sys.modules directly rather than through a context manager:
    the module under test stores references to them at import time, so tearing them down
    on the way out of a `with` block would undo the very state being asserted on. The
    autouse fixture removes them after each test instead.
    """
    for name in _INJECTED:
        sys.modules.pop(name, None)
    sys.modules["torchaudio"] = torchaudio_stub
    sys.modules.update(extra_modules or {})
    with mock.patch.object(Path, "mkdir"):
        return importlib.import_module("scripts.export_models")


def _bare_torchaudio():
    """A torchaudio without `backend`, which is what triggers the shim."""
    stub = types.ModuleType("torchaudio")
    stub.AudioMetaData = object
    return stub


@pytest.fixture(autouse=True)
def _cleanup_injected_modules():
    """Keep the stub modules out of sys.modules for everything that runs after."""
    yield
    for name in _INJECTED:
        sys.modules.pop(name, None)


def test_shim_creates_the_backend_module_when_torchaudio_lacks_one():
    """torchaudio 2.1+ dropped `backend`; libraries like demucs still import it."""
    module = _load(_bare_torchaudio())

    assert "torchaudio.backend" in sys.modules
    assert hasattr(module.torchaudio, "backend")
    assert sys.modules["torchaudio.backend.common"].AudioBackend is object


def test_shim_exposes_audio_metadata_from_torchaudio_when_available():
    """AudioMetaData is forwarded rather than stubbed, so callers get the real type."""

    class _Meta:
        pass

    stub = _bare_torchaudio()
    stub.AudioMetaData = _Meta

    _load(stub)

    assert sys.modules["torchaudio.backend.common"].AudioMetaData is _Meta


def test_shim_is_skipped_when_torchaudio_already_has_a_backend():
    """A torchaudio that still ships `backend` must be left alone."""
    stub = types.ModuleType("torchaudio")
    existing = types.ModuleType("torchaudio.backend")
    stub.backend = existing

    module = _load(stub)

    assert module.torchaudio.backend is existing


def test_export_whisper_invokes_optimum_cli_with_int8_openvino():
    """The export command is the contract with the image build; pin its shape."""
    module = _load(_bare_torchaudio())

    with mock.patch.object(module.subprocess, "run") as run:
        module.export_whisper("openai/whisper-large-v3")

    cmd = run.call_args[0][0]
    assert cmd[:3] == ["optimum-cli", "export", "openvino"]
    assert "--model" in cmd
    assert "openai/whisper-large-v3" in cmd
    assert "--weight-format" in cmd
    assert "int8" in cmd
    assert cmd[-1].endswith("whisper-openvino")
    assert run.call_args.kwargs["check"] is True


def test_export_whisper_exits_when_the_cli_fails():
    """A failed export must stop the build rather than producing a half-populated image."""
    module = _load(_bare_torchaudio())

    with mock.patch.object(module.subprocess, "run", side_effect=subprocess.CalledProcessError(2, "optimum-cli")):
        with pytest.raises(SystemExit) as exit_info:
            module.export_whisper("openai/whisper-large-v3")

    assert exit_info.value.code == 1


def _separator_module(recorder):
    """A stand-in audio_separator.separator whose Separator records its kwargs."""
    separator_mod = types.ModuleType("audio_separator.separator")

    class _Separator:
        """Records the kwargs it was constructed with and the model it was asked to load."""

        def __init__(self, **kwargs):
            recorder["init"] = kwargs

        def load_model(self, name):
            """Record the requested model instead of downloading it."""
            recorder["loaded"] = name

    separator_mod.Separator = _Separator
    package = types.ModuleType("audio_separator")
    package.separator = separator_mod
    return {"audio_separator": package, "audio_separator.separator": separator_mod}


def test_warmup_uvr_contains_separator_output_in_a_private_directory(monkeypatch):
    """The separator must not write into /tmp itself, which is world-writable.

    Nothing reads that output back -- the call exists to trigger the model download -- so
    containing it costs nothing and removes a swap-the-output-file opening on a build host.
    """
    recorder = {}
    module = _load(_bare_torchaudio(), _separator_module(recorder))
    monkeypatch.delenv("VOCAL_SEPARATION_MODEL", raising=False)

    with mock.patch.object(Path, "mkdir"):
        module.warmup_uvr()

    output_dir = recorder["init"]["output_dir"]
    assert output_dir not in ("/tmp", "/tmp/")
    assert "uvr-warmup-" in output_dir
    assert recorder["loaded"] == "UVR-MDX-NET-Inst_HQ_3.onnx"


def test_warmup_uvr_honours_the_configured_model(monkeypatch):
    """VOCAL_SEPARATION_MODEL selects which model the image pre-caches."""
    recorder = {}
    module = _load(_bare_torchaudio(), _separator_module(recorder))
    monkeypatch.setenv("VOCAL_SEPARATION_MODEL", "UVR-MDX-NET-Voc_FT.onnx")

    with mock.patch.object(Path, "mkdir"):
        module.warmup_uvr()

    assert recorder["loaded"] == "UVR-MDX-NET-Voc_FT.onnx"


def test_warmup_uvr_does_not_fail_the_build_when_precaching_fails(caplog):
    """A failed pre-cache is logged and tolerated: the runtime can still download later."""
    separator_mod = types.ModuleType("audio_separator.separator")

    class _Exploding:
        """A Separator whose construction fails, as it would with no network."""

        def __init__(self, **kwargs):
            raise RuntimeError("network unreachable")

    separator_mod.Separator = _Exploding
    package = types.ModuleType("audio_separator")
    package.separator = separator_mod
    module = _load(_bare_torchaudio(), {"audio_separator": package, "audio_separator.separator": separator_mod})

    with mock.patch.object(Path, "mkdir"):
        module.warmup_uvr()

    assert "pre-cache failed" in caplog.text
