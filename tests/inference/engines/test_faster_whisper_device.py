"""Regression coverage for the device a loaded faster-whisper engine reports.

The isolated proxy asks the engine which device it settled on, and FasterWhisperEngine had
no answer to give -- so with isolation on, which is the default, the dashboard's inference
chip was always a prediction from configuration rather than a measurement.
"""

from modules.inference.engines.faster_whisper_engine import FasterWhisperEngine


class _Handle:
    def __init__(self, device):
        self.device = device


class _Model:
    def __init__(self, device):
        self.model = _Handle(device)


class TestTheDeviceALoadedEngineReports:
    def _engine(self, model):
        engine = FasterWhisperEngine.__new__(FasterWhisperEngine)
        engine.model = model
        return engine

    def test_the_resolved_device_comes_from_the_ctranslate2_handle(self):
        engine = self._engine(_Model("cuda"))
        assert engine._loaded_device("auto") == "cuda"

    def test_a_request_served_on_the_cpu_is_reported_as_the_cpu(self):
        """A CUDA request on a build without GPU support quietly becomes the CPU."""
        engine = self._engine(_Model("cpu"))
        assert engine._loaded_device("cuda") == "cpu"

    def test_an_unreadable_handle_falls_back_to_the_requested_device(self):
        engine = self._engine(_Model(""))
        assert engine._loaded_device("cuda") == "cuda"

    def test_a_model_without_a_handle_falls_back_to_the_requested_device(self):
        engine = self._engine(object())
        assert engine._loaded_device("cpu") == "cpu"
