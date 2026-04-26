"""Tests for modules/inference/intel_engine.py using mocks."""
# pylint: disable=protected-access, import-error, no-member, reimported, wrong-import-order, wrong-import-position
from modules.inference import intel_engine
import sys
from unittest import mock
import numpy as np
import pytest
import importlib

# Mock OpenVINO GenAI before any imports
mock_genai = mock.MagicMock()
sys.modules['openvino_genai'] = mock_genai

importlib.reload(intel_engine)


@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset mocks before each test."""
    mock_genai.reset_mock()
    mock_genai.WhisperPipeline.side_effect = None
    mock_genai.WhisperPipeline.return_value = mock.MagicMock()
    yield


class TestIntelWhisperEngine:
    """Tests for IntelWhisperEngine class."""

    def test_init_success(self):
        """Test successful initialization."""
        engine = intel_engine.IntelWhisperEngine("/path/to/model", device="CPU")
        assert engine.pipeline is not None
        mock_genai.WhisperPipeline.assert_called_once_with("/path/to/model", "CPU")

    def test_init_failure(self):
        """Test initialization failure handles exception."""
        mock_genai.WhisperPipeline.side_effect = RuntimeError("OpenVINO Error")
        with mock.patch("os.path.exists", return_value=True):
            with mock.patch("os.listdir", return_value=[]):
                with pytest.raises(RuntimeError):
                    intel_engine.IntelWhisperEngine("/path/to/model")

    def test_transcribe_uninitialized(self):
        """Test transcribe raises if pipeline not initialized."""
        engine = intel_engine.IntelWhisperEngine("/path/to/model")
        engine.pipeline = None
        with pytest.raises(RuntimeError, match="not initialized"):
            engine.transcribe(np.zeros(10))

    def test_transcribe_with_path(self):
        """Test transcribe converts path to audio."""
        engine = intel_engine.IntelWhisperEngine("/path/to/model")

        with mock.patch("modules.inference.vad.decode_audio") as mock_decode:
            mock_decode.return_value = np.zeros(16000, dtype=np.float32)

            mock_result = mock.MagicMock()
            mock_result.text = "Hello"
            mock_result.chunks = []
            mock_result.language = "en"  # Set language on the mock result
            engine.pipeline.generate.return_value = mock_result

            segments, info = engine.transcribe("/path/audio.wav")
            mock_decode.assert_called_once_with("/path/audio.wav")
            assert info.language == "en"

    def test_transcribe_vad_suppression(self):
        """Test VAD suppression logic."""
        engine = intel_engine.IntelWhisperEngine("/path/to/model")
        audio = np.ones(16000, dtype=np.float32)

        msg = {'start': 0.0, 'end': 0.5}

        with mock.patch("modules.inference.vad.get_speech_timestamps", return_value=[msg]) as mock_get_timestamps:
            mock_result = mock.MagicMock()
            mock_result.chunks = []
            engine.pipeline.generate.return_value = mock_result

            engine.transcribe(audio, vad_filter=True, vad_threshold=0.5)

            mock_get_timestamps.assert_called_once()
            engine.pipeline.generate.assert_called_once()
            # Verify audio was masked (8000 samples @ 16kHz = 0.5s)
            assert np.all(audio[8000:] == 0.0)

    def test_transcribe_vad_no_speech(self):
        """Test early return when VAD finds no speech."""
        engine = intel_engine.IntelWhisperEngine("/path/to/model")
        with mock.patch("modules.inference.vad.get_speech_timestamps", return_value=[]):
            segments, info = engine.transcribe(np.zeros(16000), vad_filter=True)
            assert not list(segments)
            assert info.language == "en"

    def test_transcribe_language_resolution(self):
        """Test language resolution."""
        engine = intel_engine.IntelWhisperEngine("/path/to/model")
        mock_config = mock.MagicMock()
        mock_config.lang_to_id = {'<|en|>': 1, '<|fr|>': 2}
        engine.pipeline.get_generation_config.return_value = mock_config

        mock_result = mock.MagicMock()
        mock_result.chunks = []
        engine.pipeline.generate.return_value = mock_result

        engine.transcribe(np.zeros(10), language='fr')
        assert mock_config.language == '<|fr|>'

    def test_transcribe_tensor_sanitization(self):
        """Test tensor sanitization."""
        engine = intel_engine.IntelWhisperEngine("/path/to/model")
        audio = np.ones((2, 5000), dtype=np.float64).T

        mock_result = mock.MagicMock()
        mock_result.chunks = []
        engine.pipeline.generate.return_value = mock_result

        engine.transcribe(audio)

        call_args = engine.pipeline.generate.call_args[0]
        sanitized_audio = call_args[0]
        assert sanitized_audio.ndim == 1
        assert sanitized_audio.dtype == np.float32

    def test_transcribe_result_parsing(self):
        """Test full result parsing with segments."""
        engine = intel_engine.IntelWhisperEngine("/path/to/model")

        mock_segment = mock.MagicMock()
        mock_segment.start_ts = 0.5
        mock_segment.end_ts = 1.5
        mock_segment.text = " Chunk"

        mock_result = mock.MagicMock()
        mock_result.chunks = [mock_segment]
        mock_result.language = "fr"

        engine.pipeline.generate.return_value = mock_result

        segments, info = engine.transcribe(np.zeros(10))
        segments = list(segments)

        assert info.language == "fr"
        assert len(segments) == 1
        assert segments[0].text == " Chunk"
        assert segments[0].start == 0.5


def test_intel_engine_exceptions():
    """Cover various exception branches in Intel Engine."""
    with mock.patch("importlib.import_module") as mock_imp:
        mock_ov = mock.MagicMock()
        mock_imp.return_value = mock_ov
        engine = intel_engine.IntelWhisperEngine("path", "CPU")

        # transcribe failure
        engine.pipeline.generate.side_effect = RuntimeError("Infer Fail")
        with pytest.raises(RuntimeError):
            engine.transcribe(np.zeros(16000))

        # _apply_vad failure
        with mock.patch("modules.inference.vad.get_speech_timestamps", side_effect=ValueError("VAD Fail")):
            res = engine._apply_vad(np.zeros(16000))
            assert len(res) == 16000

        # _prepare_gen_config fallback
        engine.pipeline.get_generation_config.side_effect = AttributeError()
        cfg = engine._prepare_gen_config("en", "transcribe")
        assert cfg.task == "transcribe"

        # Language resolution fallbacks
        cfg.lang_to_id = {}
        assert engine._resolve_language("en", cfg) is None

        cfg.lang_to_id = {"English": 1}
        assert engine._resolve_language("en", cfg) == "English"

        # Sanitize audio branches
        res = engine._sanitize_audio([0.1, 0.2])
        assert res.dtype == np.float32

        # Unload
        engine.unload()
        assert engine.pipeline is None


def test_intel_detect_language_branches():
    """Cover detect_language in Intel Engine."""
    with mock.patch("importlib.import_module"):
        engine = intel_engine.IntelWhisperEngine("path", "CPU")
        engine.pipeline = mock.MagicMock()

        # Success
        from argparse import Namespace
        engine.pipeline.generate.return_value = Namespace(language="fr")
        lang, prob, _ = engine.detect_language(np.zeros(16000))
        assert lang == "fr"

        # Failure
        engine.pipeline.generate.side_effect = RuntimeError("Detect Fail")
        lang, prob, _ = engine.detect_language(np.zeros(16000))
        assert lang == "en"
