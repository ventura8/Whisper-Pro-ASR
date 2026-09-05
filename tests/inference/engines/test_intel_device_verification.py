"""An Intel device must prove it can execute, not merely build a pipeline.

The NPU constructs a WhisperPipeline happily and then fails every generate() with
``L0 pfnAppendGraphExecute ... ZE_RESULT_ERROR_UNKNOWN``, because the exported IR is
dynamic-shaped and the NPU plugin requires static upper bounds. Without the warmup the
service starts healthy, reports "ASR Runtime: OpenVINO (NPU)", and returns HTTP 500 for
every request -- a broken service that looks correctly configured.

These pin the warmup and the CPU fallback it triggers, both of which run only on a real
Intel host and so would otherwise be exercised for the first time in production.
"""

# pylint: disable=protected-access
# The warmup and the fallback are internals by design: neither has a public entry point,
# and both run inside __init__, so reaching them by name is the only way to test the
# branches independently of a real OpenVINO pipeline.

from unittest import mock

import pytest

from modules.inference.engines import intel_engine


def _engine(device):
    """An engine with its constructor bypassed, so only the unit under test runs."""
    engine = intel_engine.IntelWhisperEngine.__new__(intel_engine.IntelWhisperEngine)
    engine.device = device
    engine.model_path = "/models/whisper-openvino"
    engine.pipeline = mock.MagicMock()
    return engine


def test_a_successful_warmup_keeps_the_npu():
    """The good path: the NPU executes, so nothing moves."""
    engine = _engine("NPU")

    engine._verify_device_executes(engine.model_path)

    engine.pipeline.generate.assert_called_once()
    # A 1-second buffer of 16 kHz silence: enough to reach the plugin, cheap enough to run
    # on every start.
    assert engine.pipeline.generate.call_args.args[0].shape == (16000,)
    assert engine.device == "NPU"


def test_a_failed_warmup_falls_back_to_the_cpu():
    """The recorded NPU defect: the pipeline builds, then cannot execute."""
    engine = _engine("NPU")
    engine.pipeline.generate.side_effect = RuntimeError("ZE_RESULT_ERROR_UNKNOWN")

    with mock.patch.object(intel_engine, "_init_intel_pipeline") as init:
        engine._verify_device_executes(engine.model_path)

    # CPU, not the iGPU: the GPU unit is already serving ASR, and sending this one there
    # too would serialise both units on one device.
    init.assert_called_once_with(engine.model_path, "CPU")
    assert engine.device == "CPU", "the reported device must match what actually runs"


def test_a_failed_warmup_with_no_usable_cpu_fallback_raises():
    """Serving 500s silently is the one outcome this must not produce."""
    engine = _engine("NPU")
    engine.pipeline.generate.side_effect = RuntimeError("ZE_RESULT_ERROR_UNKNOWN")

    with mock.patch.object(intel_engine, "_init_intel_pipeline", side_effect=RuntimeError("no CPU plugin")):
        with pytest.raises(RuntimeError, match="cannot execute this model"):
            engine._verify_device_executes(engine.model_path)


def test_the_gpu_pays_no_warmup_tax():
    """The GPU path is exercised constantly; a warmup on every start buys nothing."""
    engine = _engine("GPU.0")

    engine._verify_device_executes(engine.model_path)

    engine.pipeline.generate.assert_not_called()


def test_verification_can_be_switched_off():
    """VERIFY_RUNTIME=false skips the probe entirely, even on the NPU."""
    engine = _engine("NPU")

    with mock.patch.object(intel_engine.config, "VERIFY_RUNTIME", False):
        engine._verify_device_executes(engine.model_path)

    engine.pipeline.generate.assert_not_called()


def test_fallback_reports_failure_rather_than_claiming_the_cpu():
    """_fallback_to_cpu must not rewrite self.device when the load did not happen."""
    engine = _engine("NPU")

    with mock.patch.object(intel_engine, "_init_intel_pipeline", side_effect=OSError("plugin missing")):
        assert engine._fallback_to_cpu(engine.model_path) is False

    assert engine.device == "NPU"
