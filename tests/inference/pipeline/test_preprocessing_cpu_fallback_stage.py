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

    def test_the_cpu_separator_does_not_outlive_the_fallback(self):
        """A transient failure must not demote the unit to CPU isolation permanently.

        The device fields were restored but the separator built during the fallback was
        not discarded, so it was reused for every later run on an accelerator unit -- and
        the dashboard, which reads the live separator, reported CPU ever after.
        """
        manager = self._manager()

        def fake_pipeline(audio_path, yield_cb=None, stage="Vocal Separation"):
            manager.separator = object()
            return audio_path

        manager._run_preprocess_pipeline = fake_pipeline
        manager._run_cpu_fallback("/tmp/clip.wav", None, RuntimeError("provider gone"), stage="Vocal Isolation")
        assert manager.separator is None

    def test_the_cpu_separator_is_dropped_even_when_the_fallback_fails(self):
        manager = self._manager()

        def failing_pipeline(audio_path, yield_cb=None, stage="Vocal Separation"):
            manager.separator = object()
            raise RuntimeError("empty stems")

        manager._run_preprocess_pipeline = failing_pipeline
        result = manager._run_cpu_fallback("/tmp/clip.wav", None, RuntimeError("provider gone"), stage="Vocal Isolation")
        assert result == "/tmp/clip.wav"
        assert manager.separator is None
