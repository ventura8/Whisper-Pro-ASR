"""Tests to increase coverage for modules/inference/intel_engine.py."""
import importlib
import types
from unittest import mock

import numpy as np

from modules.inference import intel_engine


def test_find_split_points_no_speech():
    """Test find_split_points when there are no speech timestamps."""
    audio_len = 950.0  # seconds
    speech_ts = []
    splits = intel_engine.find_split_points(audio_len, speech_ts, target_chunk_len=300.0)
    assert splits == [0.0, 300.0, 600.0, 900.0, 950.0]


def test_find_split_points_with_gaps():
    """Test find_split_points with speech timestamps and gaps."""
    audio_len = 1000.0
    speech_ts = [
        {"start": 10.0, "end": 50.0},
        {"start": 200.0, "end": 250.0},
        {"start": 400.0, "end": 450.0},
        {"start": 700.0, "end": 750.0},
    ]
    splits = intel_engine.find_split_points(audio_len, speech_ts, target_chunk_len=300.0)
    assert splits[0] == 0.0
    assert splits[-1] == audio_len
    assert len(splits) >= 4


def test_sanitize_audio_converts_and_contiguity():
    """Test sanitize_audio flattening and converting type/contiguity."""
    data = [0, 1, 2]
    arr = intel_engine.IntelWhisperEngine.sanitize_audio(intel_engine.IntelWhisperEngine, data)
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float32
    assert arr.flags['C_CONTIGUOUS']
    data2 = [[1, 2], [3, 4]]
    arr2 = intel_engine.IntelWhisperEngine.sanitize_audio(intel_engine.IntelWhisperEngine, data2)
    assert arr2.shape == (4,)


def test_apply_vad_no_speech_returns_zeros():
    """Test apply_vad when VAD results in no speech segment."""
    dummy_audio = np.arange(16000 * 2, dtype=np.float32)
    with mock.patch.object(intel_engine, "vad") as mock_vad:
        mock_vad.get_speech_timestamps.return_value = []
        engine = intel_engine.IntelWhisperEngine.__new__(intel_engine.IntelWhisperEngine)
        result = engine.apply_vad(dummy_audio, vad_filter=True)
        assert np.array_equal(result, np.zeros_like(dummy_audio))
        mock_vad.get_speech_timestamps.assert_called_once()


def test_unload_clears_pipeline():
    """Test unload clears pipeline instance and calls garbage collection."""
    engine = intel_engine.IntelWhisperEngine.__new__(intel_engine.IntelWhisperEngine)
    engine.pipeline = mock.MagicMock()
    engine.device = "NPU"
    with mock.patch.object(intel_engine, "gc") as mock_gc:
        engine.unload()
        assert engine.pipeline is None
        mock_gc.collect.assert_called_once()


def test_detect_language_error_path():
    """Test detect_language error path returns default language."""
    engine = intel_engine.IntelWhisperEngine.__new__(intel_engine.IntelWhisperEngine)
    dummy_pipeline = mock.MagicMock()
    dummy_pipeline.generate.side_effect = RuntimeError("boom")
    dummy_pipeline.get_generation_config.return_value = types.SimpleNamespace()
    engine.pipeline = dummy_pipeline
    lang, prob, probs = engine.detect_language(np.zeros(16000))
    assert lang == "en"
    assert prob == 0.0
    assert probs == [("en", 0.0)]


def test_engine_init_with_mock_pipeline(monkeypatch):
    """Test engine initialization triggers imports and WhisperPipeline call."""
    mock_genai = mock.MagicMock()
    mock_pipeline_instance = mock.MagicMock()
    mock_pipeline_instance.device = "NPU"
    mock_pipeline_instance.get_generation_config.return_value = types.SimpleNamespace()
    mock_genai.WhisperPipeline.return_value = mock_pipeline_instance

    def fake_import(name):
        """Mock importlib.import_module."""
        assert name == "openvino_genai"
        return mock_genai

    monkeypatch.setattr(importlib, "import_module", fake_import)
    engine = intel_engine.IntelWhisperEngine(model_path="/tmp/model", device="NPU")
    assert engine.pipeline == mock_pipeline_instance
    assert engine.pipeline.device == "NPU"
    engine.unload()
