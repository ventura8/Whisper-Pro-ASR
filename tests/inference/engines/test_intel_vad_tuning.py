"""Regression coverage for VAD tuning reaching the Intel engine.

model_manager sends the nested ``vad_parameters`` dict that faster-whisper's API takes,
while this engine only read the flat names -- so a configured VAD_THRESHOLD was dropped on
INTEL-WHISPER and the engine ran at a hardcoded 0.35 that matched neither the request nor
config's own default.
"""

from unittest import mock

from modules.inference.engines.intel_engine import _vad_tuning


class TestResolvingVadTuning:
    def test_the_nested_vad_parameters_dict_is_honoured(self):
        threshold, min_silence, _pad = _vad_tuning({"vad_parameters": {"threshold": 0.8, "min_silence_duration_ms": 1234}})
        assert (threshold, min_silence) == (0.8, 1234)

    def test_the_configured_threshold_is_used_when_nothing_is_passed(self):
        with mock.patch("modules.inference.engines.intel_engine.config.VAD_THRESHOLD", 0.5):
            assert _vad_tuning({})[0] == 0.5

    def test_a_flat_name_wins_over_the_nested_one(self):
        """A caller naming this engine's own parameter is being specific about it."""
        assert _vad_tuning({"vad_threshold": 0.1, "vad_parameters": {"threshold": 0.9}})[0] == 0.1

    def test_an_empty_vad_parameters_dict_falls_back_to_config(self):
        with mock.patch("modules.inference.engines.intel_engine.config.VAD_THRESHOLD", 0.42):
            assert _vad_tuning({"vad_parameters": None})[0] == 0.42
