"""Tests to increase coverage for modules/inference/intel_engine.py."""

import importlib
import os
import shutil
import types
from unittest import mock

import numpy as np
import pytest

from modules.core import model_integrity
from modules.inference.engines import intel_engine


def _sparse_file(path, size: int) -> None:
    """Create ``path`` reporting ``size`` bytes without allocating them."""
    path.touch()
    os.truncate(path, size)


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
    """Test _prepare_transcription_audio flattening and converting type/contiguity."""
    data = [0, 1, 2]
    func = getattr(intel_engine.IntelWhisperEngine, "_prepare_transcription_audio")
    arr = func(None, data)
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float32
    assert arr.flags["C_CONTIGUOUS"]
    data2 = [[1, 2], [3, 4]]
    arr2 = func(None, data2)
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
        getattr(engine, "unload")()
        assert engine.pipeline is None
        mock_gc.collect.assert_called_once()


def test_detect_language_error_path():
    """Test detect_language error path returns default language."""
    engine = intel_engine.IntelWhisperEngine.__new__(intel_engine.IntelWhisperEngine)
    dummy_pipeline = mock.MagicMock()
    dummy_pipeline.generate.side_effect = RuntimeError("boom")
    dummy_pipeline.get_generation_config.return_value = types.SimpleNamespace()
    engine.pipeline = dummy_pipeline
    lang, prob, probs = getattr(engine, "detect_language")(np.zeros(16000))
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
    getattr(engine, "unload")()


def test_intel_engine_init_corrupt_model_dir_purged(tmp_path):
    """Test corrupted Intel model directory is detected and purged after initial load fails."""
    corrupt_dir = tmp_path / "corrupt_ov"
    corrupt_dir.mkdir()
    (corrupt_dir / "openvino_encoder_model.xml").write_text("<xml/>")
    # Missing .bin files

    mock_genai = mock.MagicMock()
    mock_genai.WhisperPipeline.side_effect = RuntimeError("Invalid model format")

    # ensure_openvino_whisper is mocked out for two reasons: it is the step under test here,
    # and left real it would attempt a multi-gigabyte download from the network in a unit test.
    with mock.patch("importlib.import_module", return_value=mock_genai):
        with mock.patch.object(intel_engine.model_provisioning, "ensure_openvino_whisper", return_value=True) as provision:
            with pytest.raises(RuntimeError, match="Invalid model format"):
                intel_engine.IntelWhisperEngine(model_path=str(corrupt_dir), device="NPU")

    # Initial load failed, corruption was verified, the directory was purged, the IR was
    # re-fetched, and the one bounded retry then failed too -- so the original error stands.
    assert not corrupt_dir.exists()
    provision.assert_called_once()
    assert mock_genai.WhisperPipeline.call_count == 2


def test_intel_engine_does_not_retry_a_load_it_cannot_reprovision_for(tmp_path):
    """Purging deletes the weights, so a retry without re-provisioning cannot succeed.

    Retrying anyway spends a load attempt to reach a guaranteed "path does not exist",
    burying the real initialization error under a second, less informative one.
    """
    corrupt_dir = tmp_path / "corrupt_ov_no_reprovision"
    corrupt_dir.mkdir()
    (corrupt_dir / "openvino_encoder_model.xml").write_text("<xml/>")

    mock_genai = mock.MagicMock()
    mock_genai.WhisperPipeline.side_effect = RuntimeError("Invalid model format")

    with mock.patch("importlib.import_module", return_value=mock_genai):
        with mock.patch.object(intel_engine.model_provisioning, "ensure_openvino_whisper", return_value=False):
            with pytest.raises(RuntimeError, match="Invalid model format"):
                intel_engine.IntelWhisperEngine(model_path=str(corrupt_dir), device="NPU")

    assert not corrupt_dir.exists()
    assert mock_genai.WhisperPipeline.call_count == 1, "no second load attempt when there are no weights to load"


def test_intel_engine_init_successful_retry_after_recovery(tmp_path):
    """Test Intel engine succeeds on single bounded retry if model becomes valid."""
    model_dir = tmp_path / "ov_model_retry"
    model_dir.mkdir()

    (model_dir / "openvino_encoder_model.xml").write_text("<xml/>")
    # Initially missing bin files so invalid

    mock_genai = mock.MagicMock()
    call_count = 0
    mock_pipeline = mock.MagicMock()

    def fake_pipeline(_path, _device):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("Corrupted on first load")
        return mock_pipeline

    mock_genai.WhisperPipeline.side_effect = fake_pipeline

    with (
        mock.patch("importlib.import_module", return_value=mock_genai),
        mock.patch.object(intel_engine.model_integrity, "purge_corrupted_path") as mock_purge,
    ):

        def fake_purge(target_path, description=""):
            assert target_path == str(model_dir)
            assert description
            # Simulate purge and background restore/redownload with valid files.
            shutil.rmtree(model_dir)
            model_dir.mkdir()
            # The .bin files are created sparse and sized from the verifier's own
            # threshold: writing 51MB twice cost real time and disk, and a hardcoded size
            # would silently stop clearing the bar if the constant ever rose.
            (model_dir / "openvino_encoder_model.xml").write_text("<xml/>")
            _sparse_file(model_dir / "openvino_encoder_model.bin", model_integrity.MIN_OPENVINO_BIN_BYTES + 1)
            (model_dir / "openvino_decoder_model.xml").write_text("<xml/>")
            _sparse_file(model_dir / "openvino_decoder_model.bin", model_integrity.MIN_OPENVINO_BIN_BYTES + 1)
            (model_dir / "openvino_tokenizer.xml").write_text("<xml/>")
            (model_dir / "openvino_tokenizer.bin").write_bytes(b"x" * 10)
            (model_dir / "openvino_detokenizer.xml").write_text("<xml/>")
            (model_dir / "openvino_detokenizer.bin").write_bytes(b"x" * 10)
            (model_dir / "generation_config.json").write_text('{"max_length": 448}')
            return True

        mock_purge.side_effect = fake_purge

        engine = intel_engine.IntelWhisperEngine(model_path=str(model_dir), device="NPU")
        assert engine.pipeline is mock_pipeline
        assert call_count == 2
        mock_purge.assert_called_once()
