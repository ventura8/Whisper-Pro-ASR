"""Tests for modules/intel_engine.py using mocks."""
# pylint: disable=protected-access, import-error, no-member, reimported, wrong-import-order, wrong-import-position
import sys
from unittest import mock
import numpy as np
import pytest

# Mock openvino_genai BEFORE importing intel_engine
mock_genai = mock.MagicMock()
sys.modules['openvino_genai'] = mock_genai

# Now import intel_engine
from modules import intel_engine


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
        engine = intel_engine.IntelWhisperEngine(
            "/path/to/model", device="CPU")
        assert engine.pipeline is not None
        mock_genai.WhisperPipeline.assert_called_once_with(
            "/path/to/model", "CPU")

    def test_init_failure(self):
        """Test initialization failure logs error."""
        mock_genai.WhisperPipeline.side_effect = RuntimeError("OpenVINO Error")
        with mock.patch("os.path.exists", return_value=True):
            with mock.patch("os.listdir", return_value=["file1", "file2"]):
                with pytest.raises(RuntimeError):
                    intel_engine.IntelWhisperEngine("/path/to/model")

    def test_transcribe_uninitialized(self):
        """Test transcribe raises if pipeline not initialized."""
        # Create engine but manually set pipeline to None to simulate uninitialized state
        # (Since __init__ will succeed due to reset_mocks)
        engine = intel_engine.IntelWhisperEngine("/path/to/model")
        engine.pipeline = None
        with pytest.raises(RuntimeError, match="not initialized"):
            engine.transcribe(np.zeros(10))

    def test_transcribe_with_path(self):
        """Test transcribe converts path to audio."""
        engine = intel_engine.IntelWhisperEngine("/path/to/model")

        with mock.patch("modules.vad.decode_audio") as mock_decode:
            mock_decode.return_value = np.zeros(16000, dtype=np.float32)

            # Configure pipeline logic
            mock_result = mock.MagicMock()
            # Ensure 'texts' attribute doesn't exist or is empty so it proceeds to 'text'
            del mock_result.texts
            mock_result.text = "Hello"
            mock_result.chunks = []
            engine.pipeline.generate.return_value = mock_result

            res = engine.transcribe("/path/audio.wav")
            mock_decode.assert_called_once_with("/path/audio.wav")
            assert res['text'] == "Hello"

    def test_transcribe_vad_suppression(self):
        """Test VAD suppression logic."""
        engine = intel_engine.IntelWhisperEngine("/path/to/model")
        audio = np.ones(16000, dtype=np.float32)  # All 1s

        msg = {'start': 0.0, 'end': 0.5}  # Speech in first half

        with mock.patch("modules.vad.get_speech_timestamps",
                        return_value=[msg]) as mock_get_timestamps:
            mock_result = mock.MagicMock()
            del mock_result.texts
            mock_result.text = "Speech"
            engine.pipeline.generate.return_value = mock_result

            engine.transcribe(audio, vad_filter=True, vad_threshold=0.5)

            mock_get_timestamps.assert_called_once()
            engine.pipeline.generate.assert_called_once()
            assert np.all(audio[8000:] == 0.0)
            assert np.all(audio[:8000] == 1.0)

    def test_transcribe_vad_exception(self):
        """Test VAD suppression handles exception."""
        engine = intel_engine.IntelWhisperEngine("/path/to/model")

        with mock.patch("modules.vad.get_speech_timestamps", side_effect=Exception("VAD Fail")):
            # Should continue without VAD
            engine.transcribe(np.zeros(16000), vad_filter=True)
            # Should log warning but not crash

    def test_transcribe_vad_no_speech(self):
        """Test early return when VAD finds no speech."""
        engine = intel_engine.IntelWhisperEngine("/path/to/model")
        with mock.patch("modules.vad.get_speech_timestamps", return_value=[]):
            result = engine.transcribe(np.zeros(16000), vad_filter=True)
            assert result['text'] == ""
            assert not result['chunks']
            engine.pipeline.generate.assert_not_called()

    def test_transcribe_task_resolution(self):
        """Test task defaulting."""
        engine = intel_engine.IntelWhisperEngine("/path/to/model")

        mock_config = mock.MagicMock()
        mock_config.task_to_id = {'transcribe': 1}
        engine.pipeline.get_generation_config.return_value = mock_config

        mock_result = mock.MagicMock()
        del mock_result.texts
        mock_result.text = "OK"
        engine.pipeline.generate.return_value = mock_result

        engine.transcribe(np.zeros(10), task='translate')
        assert mock_config.task == 'transcribe'

    def test_transcribe_language_resolution(self):
        """Test language resolution."""
        engine = intel_engine.IntelWhisperEngine("/path/to/model")
        mock_config = mock.MagicMock()
        mock_config.lang_to_id = {'<|en|>': 1, '<|fr|>': 2}
        engine.pipeline.get_generation_config.return_value = mock_config

        mock_result = mock.MagicMock()
        engine.pipeline.generate.return_value = mock_result

        # Exact match
        engine.transcribe(np.zeros(10), language='fr')
        # We need to check if language was set.
        # The code does: if token: gen_config.language = token
        # We can't easily check gen_config.language unless we mock the object returned strictly
        assert mock_config.language == '<|fr|>'

    def test_transcribe_language_fallback(self):
        """Test language resolution fallback."""
        engine = intel_engine.IntelWhisperEngine("/path/to/model")
        mock_config = mock.MagicMock()
        # Mapping doesn't match 'es' directly
        mock_config.lang_to_id = {'<|spanish|>': 1}
        engine.pipeline.get_generation_config.return_value = mock_config

        mock_result = mock.MagicMock()
        engine.pipeline.generate.return_value = mock_result

        engine.transcribe(np.zeros(10), language='spanish')
        assert mock_config.language == '<|spanish|>'

    def test_transcribe_language_substring_match(self):
        """Test language resolution via substring match."""
        engine = intel_engine.IntelWhisperEngine("/path/to/model")
        mock_config = mock.MagicMock()
        mock_config.lang_to_id = {'<|french_canadian|>': 1}
        engine.pipeline.get_generation_config.return_value = mock_config

        mock_result = mock.MagicMock()
        engine.pipeline.generate.return_value = mock_result

        engine.transcribe(np.zeros(10), language='french')
        assert mock_config.language == '<|french_canadian|>'

    def test_transcribe_language_not_found(self):
        """Test language resolution when not found in map."""
        engine = intel_engine.IntelWhisperEngine("/path/to/model")
        mock_config = mock.MagicMock()
        mock_config.lang_to_id = {'<|english|>': 1}
        engine.pipeline.get_generation_config.return_value = mock_config

        mock_result = mock.MagicMock()
        engine.pipeline.generate.return_value = mock_result

        # Should log warning but continue without setting language
        engine.transcribe(np.zeros(10), language='swahili')

    def test_transcribe_gen_config_fallback(self):
        """Test generation config fallback on exception."""
        engine = intel_engine.IntelWhisperEngine("/path/to/model")
        engine.pipeline.get_generation_config.side_effect = Exception(
            "Config error")

        with mock.patch("openvino_genai.WhisperGenerationConfig") as mock_gen_config_cls:
            mock_conf = mock_gen_config_cls.return_value # pylint: disable=unused-variable
            mock_result = mock.MagicMock()
            engine.pipeline.generate.return_value = mock_result

            engine.transcribe(np.zeros(10))
            mock_gen_config_cls.assert_called_once()

    def test_transcribe_tensor_sanitization(self):
        """Test tensor sanitization (flatten/contiguous)."""
        engine = intel_engine.IntelWhisperEngine("/path/to/model")
        # 2D non-contiguous float64 array
        audio = np.ones((2, 5000), dtype=np.float64).T
        # T (transpose) often makes it non-contiguous

        mock_result = mock.MagicMock()
        del mock_result.texts
        mock_result.text = "OK"
        engine.pipeline.generate.return_value = mock_result

        engine.transcribe(audio)

        # Verify generate was called with sanitized array
        call_args = engine.pipeline.generate.call_args[0]
        sanitized_audio = call_args[0]
        assert sanitized_audio.ndim == 1
        assert sanitized_audio.dtype == np.float32
        assert sanitized_audio.flags.c_contiguous

    def test_transcribe_fallback_text_from_chunks(self):
        """Test text generation from chunks if main text is empty."""
        engine = intel_engine.IntelWhisperEngine("/path/to/model")

        mock_chunk = mock.MagicMock()
        mock_chunk.text = "Hello"
        mock_chunk.start_ts = 0.0
        mock_chunk.end_ts = 1.0

        mock_result = mock.MagicMock()
        del mock_result.texts
        mock_result.text = ""  # Empty main text
        mock_result.chunks = [mock_chunk]

        engine.pipeline.generate.return_value = mock_result

        res = engine.transcribe(np.zeros(1000))
        assert res['text'] == "Hello"

    def test_transcribe_inference_exception(self):
        """Test exception during generate phase."""
        engine = intel_engine.IntelWhisperEngine("/path/to/model")
        engine.pipeline.generate.side_effect = RuntimeError("Inference fail")

        with pytest.raises(RuntimeError, match="Inference fail"):
            engine.transcribe(np.zeros(1000))

    def test_transcribe_initial_prompt_exception(self):
        """Test initial_prompt setting failure."""
        engine = intel_engine.IntelWhisperEngine("/path/to/model")
        mock_config = mock.MagicMock()
        # Mocking an attribute that raises error when set
        type(mock_config).initial_prompt = mock.PropertyMock(
            side_effect=AttributeError("No prompt support"))
        engine.pipeline.get_generation_config.return_value = mock_config

        mock_result = mock.MagicMock()
        engine.pipeline.generate.return_value = mock_result

        # Should not crash
        engine.transcribe(np.zeros(10), initial_prompt="Hello")

    def test_transcribe_result_parsing(self):
        """Test full result parsing with chunks."""
        engine = intel_engine.IntelWhisperEngine("/path/to/model")

        mock_chunk = mock.MagicMock()
        mock_chunk.start_ts = 0.5
        mock_chunk.end_ts = 1.5
        mock_chunk.text = " Chunk"

        mock_result = mock.MagicMock()
        mock_result.chunks = [mock_chunk]
        mock_result.texts = ["Full text"]

        engine.pipeline.generate.return_value = mock_result

        res = engine.transcribe(np.zeros(10))

        assert res['text'] == "Full text"
        assert len(res['chunks']) == 1
        assert res['chunks'][0]['text'] == " Chunk"
        assert res['chunks'][0]['start'] == 0.5
