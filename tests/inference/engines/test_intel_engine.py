"""Tests for modules/inference/intel_engine.py using mocks."""

import importlib
import sys
from argparse import Namespace
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

mock_genai = mock.MagicMock()
intel_engine = None


@pytest.fixture(autouse=True)
def reset_mocks(monkeypatch):
    """Reset and inject OpenVINO mocks before importing the module under test."""
    global mock_genai, intel_engine
    mock_genai = mock.MagicMock()
    monkeypatch.setitem(sys.modules, "openvino_genai", mock_genai)
    intel_engine = importlib.reload(importlib.import_module("modules.inference.engines.intel_engine"))
    mock_genai.reset_mock()
    mock_genai.WhisperPipeline.side_effect = None
    mock_genai.WhisperPipeline.return_value = mock.MagicMock()
    yield


def test_find_split_points():
    """Test find_split_points helper."""
    # Test empty speech timestamps
    res = intel_engine.find_split_points(600.0, [], target_chunk_len=300.0)

    # Test speech timestamps with gaps
    speech_ts = [
        {"start": 10.0, "end": 20.0},
        {"start": 285.0, "end": 295.0},  # gap around 295-310
        {"start": 310.0, "end": 320.0},
        {"start": 590.0, "end": 595.0},
    ]
    res = intel_engine.find_split_points(600.0, speech_ts, target_chunk_len=300.0)
    assert res == [0.0, 300.0, 600.0] or (len(res) == 3 and res[0] == 0.0 and 295.0 <= res[1] <= 310.0 and res[2] == 600.0)


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

        with mock.patch("modules.inference.pipeline.vad.decode_audio") as mock_decode:
            mock_decode.return_value = np.zeros(16000, dtype=np.float32)

            mock_result = mock.MagicMock()
            mock_result.text = "Hello"
            mock_result.chunks = []
            mock_result.language = "en"  # Set language on the mock result
            engine.pipeline.generate.return_value = mock_result

            segments, info = engine.transcribe("/path/audio.wav")
            list(segments)
            mock_decode.assert_called_once_with("/path/audio.wav")
            assert info.language == "en"

    def test_transcribe_vad_suppression(self):
        """Test VAD suppression logic."""
        engine = intel_engine.IntelWhisperEngine("/path/to/model")
        audio = np.ones(16000, dtype=np.float32)

        msg = {"start": 0.0, "end": 0.5}

        with mock.patch("modules.inference.pipeline.vad.get_speech_timestamps", return_value=[msg]) as mock_get_timestamps:
            mock_result = mock.MagicMock()
            mock_result.chunks = []
            engine.pipeline.generate.return_value = mock_result

            segments, _ = engine.transcribe(audio, vad_filter=True, vad_threshold=0.5)
            list(segments)

            mock_get_timestamps.assert_called_once()
            # Construction runs one warmup generate() call (see _verify_device_executes),
            # so this is the second call, not the only one.
            assert engine.pipeline.generate.call_count == 2
            generated_audio = engine.pipeline.generate.call_args[0][0]
            # Verify non-suppressed region remains unchanged.
            assert np.all(generated_audio[:8000] == 1.0)
            # Verify masked audio is used for inference (8000 samples @ 16kHz = 0.5s)
            assert np.all(generated_audio[8000:] == 0.0)
            # Original input should remain unchanged.
            assert np.all(audio == 1.0)

    def test_transcribe_vad_no_speech(self):
        """Test early return when VAD finds no speech."""
        engine = intel_engine.IntelWhisperEngine("/path/to/model")
        with mock.patch("modules.inference.pipeline.vad.get_speech_timestamps", return_value=[]):
            segments, info = engine.transcribe(np.zeros(16000), vad_filter=True)
            assert not list(segments)
            assert info.language == "en"

    def test_transcribe_language_resolution(self):
        """Test language resolution."""
        engine = intel_engine.IntelWhisperEngine("/path/to/model")
        mock_config = mock.MagicMock()
        mock_config.lang_to_id = {"<|en|>": 1, "<|fr|>": 2}
        engine.pipeline.get_generation_config.return_value = mock_config

        mock_result = mock.MagicMock()
        mock_result.chunks = []
        engine.pipeline.generate.return_value = mock_result

        segments, _ = engine.transcribe(np.zeros(10), language="fr")
        list(segments)
        assert mock_config.language == "<|fr|>"

    def test_transcribe_tensor_sanitization(self):
        """Test tensor sanitization."""
        engine = intel_engine.IntelWhisperEngine("/path/to/model")
        audio = np.ones((2, 5000), dtype=np.float64).T

        mock_result = mock.MagicMock()
        mock_result.chunks = []
        engine.pipeline.generate.return_value = mock_result

        segments, _ = engine.transcribe(audio)
        list(segments)

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

    def test_apply_vad_mask_disjoint_segments(self):
        """_apply_vad_mask should preserve only disjoint speech windows."""
        engine = intel_engine.IntelWhisperEngine("/path/to/model")
        audio = np.ones(32000, dtype=np.float32)
        speech_ts = [{"start": 0.10, "end": 0.20}, {"start": 1.00, "end": 1.10}]

        masked = engine._apply_vad_mask(audio, speech_ts)

        assert [
            np.all(masked[:1600] == 0.0),
            np.all(masked[1600:3200] == 1.0),
            np.all(masked[3200:16000] == 0.0),
            np.all(masked[16000:17600] == 1.0),
            np.all(masked[17600:] == 0.0),
        ] == [True, True, True, True, True]

    def test_apply_vad_mask_boundary_timestamps(self):
        """_apply_vad_mask should handle start/end timestamps at array boundaries."""
        engine = intel_engine.IntelWhisperEngine("/path/to/model")
        audio = np.ones(16000, dtype=np.float32)
        speech_ts = [{"start": 0.0, "end": 0.25}, {"start": 0.75, "end": 1.0}]

        masked = engine._apply_vad_mask(audio, speech_ts)

        assert np.all(masked[:4000] == 1.0)
        assert np.all(masked[4000:12000] == 0.0)
        assert np.all(masked[12000:] == 1.0)

    def test_apply_vad_mask_empty_audio(self):
        """_apply_vad_mask should return an empty array for empty audio input."""
        engine = intel_engine.IntelWhisperEngine("/path/to/model")
        audio = np.array([], dtype=np.float32)

        masked = engine._apply_vad_mask(audio, [{"start": 0.0, "end": 0.5}])

        assert masked.size == 0

    def test_apply_vad_mask_preserves_original_audio(self):
        """_apply_vad_mask should not mutate the source audio buffer."""
        engine = intel_engine.IntelWhisperEngine("/path/to/model")
        audio = np.ones(16000, dtype=np.float32)
        original = audio.copy()

        masked = engine._apply_vad_mask(audio, [{"start": 0.0, "end": 0.5}])

        assert np.all(audio == original)
        assert np.all(masked[8000:] == 0.0)


def _make_intel_engine():
    """Create an IntelWhisperEngine instance backed by mocked OpenVINO imports."""
    with mock.patch("importlib.import_module") as mock_imp:
        mock_imp.return_value = mock.MagicMock()
        return intel_engine.IntelWhisperEngine("path", "CPU")


def test_intel_engine_transcribe_failure_raises():
    """Transcribe iteration should surface pipeline generation failures."""
    engine = _make_intel_engine()
    engine.pipeline.generate.side_effect = RuntimeError("Infer Fail")
    segments, _ = engine.transcribe(np.zeros(16000))

    with pytest.raises(RuntimeError):
        list(segments)


def test_intel_engine_apply_vad_failure_returns_full_audio():
    """VAD failures should fall back to returning the original audio length."""
    engine = _make_intel_engine()
    with mock.patch("modules.inference.pipeline.vad.get_speech_timestamps", side_effect=ValueError("VAD Fail")):
        res = engine.apply_vad(np.zeros(16000))

    assert len(res) == 16000


def test_intel_engine_prepare_gen_config_falls_back_to_transcribe():
    """prepare_gen_config should keep the transcribe task on missing OpenVINO config."""
    engine = _make_intel_engine()
    engine.pipeline.get_generation_config.side_effect = AttributeError()

    cfg = engine.prepare_gen_config("en", "transcribe")
    assert cfg.task == "transcribe"


def test_intel_engine_resolve_language_fallbacks():
    """resolve_language should fall back across missing and translated mappings."""
    engine = _make_intel_engine()
    cfg = Namespace(lang_to_id={})

    assert engine.resolve_language("en", cfg) is None

    cfg.lang_to_id = {"English": 1}
    assert engine.resolve_language("en", cfg) == "English"


def test_intel_engine_sanitize_audio_and_unload():
    """sanitize_audio should coerce dtype and unload should clear the pipeline."""
    engine = _make_intel_engine()

    res = engine.sanitize_audio([0.1, 0.2])
    engine.unload()

    assert res.dtype == np.float32 and engine.pipeline is None


def test_intel_detect_language_branches():
    """Cover detect_language in Intel Engine."""
    with mock.patch("importlib.import_module"):
        engine = intel_engine.IntelWhisperEngine("path", "CPU")
        engine.pipeline = mock.MagicMock()

        # Success
        engine.pipeline.generate.return_value = Namespace(language="fr")
        lang, _, _ = engine.detect_language(np.zeros(16000))
        assert lang == "fr"

        # Failure
        engine.pipeline.generate.side_effect = RuntimeError("Detect Fail")
        lang, _, _ = engine.detect_language(np.zeros(16000))
        assert lang == "en"


class TestRepetitionPenaltyIsScopedToGreedy:
    """OpenVINO GenAI rejects repetition_penalty under beam search, as a hard error.

        'repetition_penalty' is not currently supported by beam search and should be
        1.0f, but got 1.15

    That is a RuntimeError on every request, not a warning. It hid on Intel GPU and NPU
    because those are already clamped to greedy, and surfaced only on the CPU path --
    including the NPU-cannot-execute fallback, which rewrites the device to CPU and so
    re-enables beam search. Found on the Intel NUC.
    """

    def _config_for(self, device, beam_size):
        with mock.patch("importlib.import_module") as mock_imp:
            mock_imp.return_value = mock.MagicMock()
            engine = intel_engine.IntelWhisperEngine("path", device)
        # A plain object, so an unset attribute is genuinely absent rather than a MagicMock
        # that would satisfy any assertion about it.
        engine.pipeline.get_generation_config.return_value = SimpleNamespace()
        return engine.prepare_gen_config("en", "transcribe", beam_size=beam_size)

    def test_cpu_beam_search_leaves_the_penalty_untouched(self):
        config = self._config_for("CPU", beam_size=5)

        assert config.num_beams == 5, "the CPU path keeps beam search"
        assert getattr(config, "repetition_penalty", 1.0) == 1.0, (
            "setting repetition_penalty alongside beam search is rejected by OpenVINO GenAI"
        )

    def test_greedy_still_gets_the_loop_guard(self):
        config = self._config_for("GPU", beam_size=5)

        assert config.num_beams == 1, "GPU is clamped to greedy"
        assert config.repetition_penalty == 1.15

    def test_cpu_with_an_explicitly_greedy_request_gets_the_guard(self):
        config = self._config_for("CPU", beam_size=1)

        assert config.num_beams == 1
        assert config.repetition_penalty == 1.15


class TestIntelInitializationBranches:
    """Warmup verification, CPU fallback, and the corrupted-model retry.

    The NPU builds a WhisperPipeline in about four seconds and then fails every generate()
    with a Level Zero error, because the shipped IR is dynamic-shaped and the NPU plugin
    needs static upper bounds. Without the warmup the service starts healthy, prints
    "ASR Runtime: OpenVINO (NPU)", and returns 500 for every request -- so these branches are
    the difference between a broken service and one that says what it is really running on.
    """

    @staticmethod
    def _engine(monkeypatch, *, generate=None, load=None, device="NPU"):
        """Build an IntelWhisperEngine with the pipeline construction stubbed out."""
        pipelines = []

        def fake_init(_model_path, dev):
            pipeline = mock.MagicMock(name=f"pipeline-{dev}")
            if generate is not None:
                pipeline.generate.side_effect = generate
            pipelines.append((dev, pipeline))
            return pipeline

        monkeypatch.setattr(intel_engine, "_init_intel_pipeline", load or fake_init)
        return intel_engine.IntelWhisperEngine("/models/ov", device=device), pipelines

    def test_a_successful_npu_warmup_leaves_the_device_alone(self, monkeypatch):
        engine, pipelines = self._engine(monkeypatch)

        assert engine.device == "NPU"
        assert pipelines[0][0] == "NPU"
        pipelines[0][1].generate.assert_called_once()

    def test_a_failed_warmup_moves_asr_to_the_cpu_and_rewrites_the_device(self, monkeypatch):
        """The device attribute is what /status reports, so it has to follow the fallback.

        CPU rather than the iGPU deliberately: this engine serves an NPU unit on a machine
        whose GPU unit already runs ASR, and sending both to one device would serialise them.
        """
        calls = {"n": 0}

        def generate(_audio):
            calls["n"] += 1
            raise RuntimeError("L0 pfnAppendGraphExecute result: ZE_RESULT_ERROR_UNKNOWN")

        engine, pipelines = self._engine(monkeypatch, generate=generate)

        assert engine.device == "CPU"
        assert [dev for dev, _ in pipelines] == ["NPU", "CPU"]
        # Only the NPU pipeline is warmed up; the CPU path is exercised constantly and is
        # not made to pay a warmup on every start.
        assert calls["n"] == 1

    def test_a_failed_warmup_whose_cpu_fallback_also_fails_raises(self, monkeypatch):
        """Serving 500s silently is the outcome this whole path exists to prevent."""

        def load(_model_path, dev):
            if dev == "CPU":
                raise RuntimeError("no CPU plugin either")
            pipeline = mock.MagicMock()
            pipeline.generate.side_effect = RuntimeError("ZE_RESULT_ERROR_UNKNOWN")
            return pipeline

        with pytest.raises(RuntimeError, match="cannot execute this model"):
            self._engine(monkeypatch, load=load)

    def test_a_corrupted_model_is_purged_reprovisioned_and_retried(self, monkeypatch):
        """Purging deletes the directory, so a retry without re-provisioning cannot succeed."""
        attempts = {"n": 0}

        def load(_model_path, dev):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("corrupted OpenVINO model")
            pipeline = mock.MagicMock(name=dev)
            return pipeline

        monkeypatch.setattr(intel_engine.os.path, "isdir", lambda _p: True)
        monkeypatch.setattr(intel_engine.os, "listdir", lambda _p: [])
        monkeypatch.setattr(intel_engine.model_integrity, "verify_openvino_model_dir", lambda _p: False)
        purge = mock.MagicMock(return_value=True)
        provision = mock.MagicMock(return_value=True)
        monkeypatch.setattr(intel_engine.model_integrity, "purge_corrupted_path", purge)
        monkeypatch.setattr(intel_engine.model_provisioning, "ensure_openvino_whisper", provision)

        engine, _ = self._engine(monkeypatch, load=load, device="GPU")

        assert engine.device == "GPU"
        assert attempts["n"] == 2
        purge.assert_called_once()
        provision.assert_called_once()

    def test_a_failed_reprovision_preserves_the_original_failure(self, monkeypatch):
        """A bounded retry means one attempt, and no attempt at all when it cannot help."""

        def load(_model_path, _dev):
            raise RuntimeError("corrupted OpenVINO model")

        monkeypatch.setattr(intel_engine.os.path, "isdir", lambda _p: True)
        monkeypatch.setattr(intel_engine.os, "listdir", lambda _p: [])
        monkeypatch.setattr(intel_engine.model_integrity, "verify_openvino_model_dir", lambda _p: False)
        monkeypatch.setattr(intel_engine.model_integrity, "purge_corrupted_path", lambda *a, **k: True)
        monkeypatch.setattr(intel_engine.model_provisioning, "ensure_openvino_whisper", lambda *a, **k: False)

        with pytest.raises(RuntimeError, match="corrupted OpenVINO model"):
            self._engine(monkeypatch, load=load, device="GPU")
