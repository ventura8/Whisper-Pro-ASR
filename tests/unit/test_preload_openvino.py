"""Provisioning must not report success without a usable OpenVINO IR.

The Intel engine loads an OpenVINO IR directory. A previous fallback downloaded the raw
OpenAI weights into that same directory when the pre-converted download failed, logged
"source weights ready", and returned normally -- leaving a directory ov_genai cannot open.
Conversion needs optimum-cli, which the runtime image does not ship, so there was no route
from those weights to a usable IR: the run only looked like it had succeeded.
"""

import importlib.util
import sys
from pathlib import Path
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_preload_module():
    """Import scripts/preload_model.py by path, with its heavy imports stubbed out.

    The module imports torch and audio_separator at module scope; neither is needed to
    exercise the control flow under test, and both are absent from a lean environment.
    """
    stubs = {
        "torch": mock.MagicMock(),
        "audio_separator": mock.MagicMock(),
        "audio_separator.separator": mock.MagicMock(),
    }
    spec = importlib.util.spec_from_file_location("_preload_under_test", REPO_ROOT / "scripts" / "preload_model.py")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


@pytest.fixture(name="preload")
def preload_fixture():
    """The provisioning script, imported by path with its heavy imports stubbed."""
    return _load_preload_module()


def test_failed_genai_download_makes_preload_fail(preload):
    """With no IR obtainable, preload must exit non-zero rather than return quietly."""
    with (
        mock.patch.object(preload, "_ensure_ct2_whisper"),
        mock.patch.object(preload, "_openvino_model_already_available", return_value=False),
        mock.patch.object(preload, "_export_openvino_whisper", return_value=False),
        mock.patch.object(preload, "_download_openvino_genai", return_value=False) as genai,
    ):
        with pytest.raises(SystemExit) as exit_info:
            preload.preload_whisper()

    assert exit_info.value.code != 0, "a missing OpenVINO IR was reported as success"
    genai.assert_called_once()


def test_the_raw_source_fallback_is_gone(preload):
    """The unconverted-weights download must not exist to be reached by accident."""
    assert not hasattr(preload, "_download_openvino_source")


def test_a_successful_genai_download_still_succeeds(preload):
    """The working path must stay working: a usable IR means a clean return."""
    with (
        mock.patch.object(preload, "_ensure_ct2_whisper"),
        mock.patch.object(preload, "_openvino_model_already_available", return_value=False),
        mock.patch.object(preload, "_export_openvino_whisper", return_value=False),
        mock.patch.object(preload, "_download_openvino_genai", return_value=True),
    ):
        preload.preload_whisper()  # must not raise


def test_an_already_present_ir_short_circuits(preload):
    """An IR already on disk is used as-is, with nothing downloaded."""
    with (
        mock.patch.object(preload, "_ensure_ct2_whisper"),
        mock.patch.object(preload, "_openvino_model_already_available", return_value=True),
        mock.patch.object(preload, "_download_openvino_genai") as genai,
    ):
        preload.preload_whisper()

    genai.assert_not_called()


def test_skipping_intel_does_not_require_an_ir(preload):
    """--skip-intel-whisper provisions the other models without an OpenVINO IR."""
    with (
        mock.patch.object(preload, "_ensure_ct2_whisper"),
        mock.patch.object(preload, "SKIP_INTEL_WHISPER", True),
        mock.patch.object(preload, "_download_openvino_genai") as genai,
    ):
        preload.preload_whisper()

    genai.assert_not_called()
