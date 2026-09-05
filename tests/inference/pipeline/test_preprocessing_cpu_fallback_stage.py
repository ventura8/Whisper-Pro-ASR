"""Regression coverage for caller-owned preprocessing stage labels."""

from modules.inference.pipeline import preprocessing


class TestCpuFallbackKeepsTheCallerStage:
    """A CPU fallback must retain the stage supplied by its caller."""

    def _manager(self):
        manager = preprocessing.PreprocessingManager.__new__(preprocessing.PreprocessingManager)
        manager._unit = None
        manager._device_id = "GPU"
        manager._device_type = "GPU"
        manager.separator = None
        return manager

    def test_stage_is_forwarded_to_the_cpu_run(self):
        manager = self._manager()
        seen = {}

        def fake_pipeline(audio_path, yield_cb=None, stage="Vocal Separation"):
            seen["stage"] = stage
            return audio_path

        manager._run_preprocess_pipeline = fake_pipeline
        manager._run_cpu_fallback("/tmp/clip.wav", None, RuntimeError("provider gone"), stage="Language Detection")
        assert seen["stage"] == "Language Detection"

    def test_the_device_is_restored_afterwards(self):
        manager = self._manager()
        manager._run_preprocess_pipeline = lambda audio_path, yield_cb=None, stage="Vocal Separation": audio_path
        manager._run_cpu_fallback("/tmp/clip.wav", None, RuntimeError("provider gone"), stage="Vocal Isolation")
        assert (manager._device_id, manager._device_type) == ("GPU", "GPU")
