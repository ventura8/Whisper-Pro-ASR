"""Where a unit's work actually lands, as opposed to which unit it was dispatched to.

Every case here is a real divergence this project has hit on hardware: an engine with no
backend for the unit it was given, an NPU that builds a pipeline it cannot execute, and a
UVR separator that quietly fell back to the CPU provider.
"""

from __future__ import annotations

from unittest import mock

import pytest

from modules.monitoring import execution_status

CUDA = {"type": "CUDA", "id": "cuda:0", "name": "NVIDIA GPU 0"}
INTEL_GPU = {"type": "GPU", "id": "GPU.0", "name": "Intel Arc"}
INTEL_NPU = {"type": "NPU", "id": "NPU.0", "name": "Intel(R) AI Boost"}
HOST_CPU = {"type": "CPU", "id": "CPU", "name": "Host CPU"}


class _Engine:
    """An engine reporting the device it settled on, as the real ones do."""

    def __init__(self, device):
        self.device = device


class _Preprocessor:
    """A preprocessor whose separator reports its ONNX providers, as the real one does."""

    #: Distinguishes "no separator loaded" from "a separator whose providers read as None".
    #: Passing providers=None meant both, so the parametrised case below asserting that a
    #: None provider list is not trusted was in fact exercising the no-separator path.
    NO_SEPARATOR = object()

    def __init__(self, providers=NO_SEPARATOR, device_type=""):
        self.device_type = device_type
        self.separator = None if providers is _Preprocessor.NO_SEPARATOR else mock.MagicMock(onnx_execution_provider=providers)


class TestNormalizingDeviceStrings:
    """Engines disagree on spelling; the dashboard cannot show four names for one device."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("cuda", "CUDA"),
            ("cuda:0", "CUDA"),
            ("GPU.0", "GPU"),
            ("NPU.0", "NPU"),
            ("xpu", "XPU"),
            ("cpu", "CPU"),
            ("CPU", "CPU"),
            ("hip", "AMD"),
        ],
    )
    def test_known_spellings_normalize(self, raw, expected):
        """Known spellings normalize."""
        assert execution_status.normalize_device(raw) == expected

    @pytest.mark.parametrize("raw", [None, "", "   ", "something-else"])
    def test_unknown_values_are_empty_not_guessed(self, raw):
        """An unrecognised value must not be reported as a device."""
        assert execution_status.normalize_device(raw) == ""


class TestTranscriptionExecution:
    """The loaded engine wins, because it is the only thing that knows it moved."""

    def test_an_npu_engine_that_moved_itself_to_the_cpu_is_reported_as_a_fallback(self):
        """The case the dashboard exists to show.

        IntelWhisperEngine warms up on the NPU, hits the Level Zero failure, and rewrites
        its own device to CPU. Configuration still says the unit is an NPU, so only the
        engine can report this.
        """
        result = execution_status.asr_execution(INTEL_NPU, _Engine("CPU"))

        assert result["device"] == "CPU"
        assert result["fallback"] is True, "an accelerator unit whose work landed on the CPU"
        assert result["accelerated"] is False
        assert result["measured"] is True

    def test_an_engine_running_on_its_own_unit_is_not_a_fallback(self):
        """An engine running on its own unit is not a fallback."""
        result = execution_status.asr_execution(INTEL_GPU, _Engine("GPU.0"))

        assert (result["device"], result["fallback"], result["accelerated"]) == ("Intel GPU", False, True)

    def test_the_cpu_unit_running_on_the_cpu_is_not_a_fallback(self):
        """Nothing was given up: the CPU unit is doing exactly what it is for."""
        result = execution_status.asr_execution(HOST_CPU, _Engine("cpu"))

        assert result["fallback"] is False

    def test_with_nothing_loaded_the_answer_comes_from_the_engine_support_table(self):
        """CTranslate2 has no OpenVINO backend, so an Intel unit decodes on the CPU."""
        with mock.patch.object(execution_status.config, "engine_for_unit", return_value="FASTER-WHISPER"):
            result = execution_status.asr_execution(INTEL_GPU, None)

        assert result["device"] == "CPU"
        assert result["fallback"] is True
        assert result["measured"] is False, "predicted from configuration, not read off an engine"

    def test_a_supported_unit_with_nothing_loaded_predicts_the_unit_itself(self):
        """A supported unit with nothing loaded predicts the unit itself."""
        with mock.patch.object(execution_status.config, "engine_for_unit", return_value="FASTER-WHISPER"):
            result = execution_status.asr_execution(CUDA, None)

        assert (result["device"], result["accelerated"], result["measured"]) == ("CUDA", True, False)


class TestVocalIsolationExecution:
    """The ONNX provider is ground truth, and it is reported even from a worker process."""

    def test_a_cpu_provider_on_an_accelerator_unit_is_a_fallback(self):
        """The recorded defect: ASR_PREPROCESS_DEVICE=GPU ran UVR on CPUExecutionProvider."""
        result = execution_status.uvr_execution(INTEL_GPU, _Preprocessor(["CPUExecutionProvider"]))

        assert result["device"] == "CPU"
        assert result["fallback"] is True
        assert result["measured"] is True

    def test_the_openvino_provider_is_attributed_to_the_units_own_intel_device(self):
        """OpenVINO serves both Intel devices and its name does not say which."""
        gpu = execution_status.uvr_execution(INTEL_GPU, _Preprocessor(["OpenVINOExecutionProvider"]))
        npu = execution_status.uvr_execution(INTEL_NPU, _Preprocessor(["OpenVINOExecutionProvider"]))

        assert gpu["device"] == "Intel GPU"
        assert npu["device"] == "Intel NPU"

    def test_the_first_recognised_provider_wins(self):
        """ONNX Runtime lists providers by priority and always appends CPU as last resort.

        Reading any other entry would report the CPU for every accelerated separator.
        """
        result = execution_status.uvr_execution(CUDA, _Preprocessor(["CUDAExecutionProvider", "CPUExecutionProvider"]))

        assert result["device"] == "CUDA"
        assert result["fallback"] is False

    def test_an_unloaded_separator_falls_back_to_the_configured_device(self):
        """An unloaded separator falls back to the configured device."""
        result = execution_status.uvr_execution(INTEL_NPU, _Preprocessor(device_type="NPU"))

        assert (result["device"], result["measured"]) == ("Intel NPU", False)

    @pytest.mark.parametrize("providers", [None, "CPUExecutionProvider", 42])
    def test_a_provider_list_that_is_not_a_list_is_not_trusted(self, providers):
        """A string is iterable; treating one as a provider list would match nothing sane."""
        result = execution_status.uvr_execution(INTEL_GPU, _Preprocessor(providers, device_type="GPU"))

        assert result["measured"] is False
