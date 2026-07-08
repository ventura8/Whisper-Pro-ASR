"""Tests for speaker diarization integration and WhisperX orchestration."""

import asyncio
import sys
from unittest import mock

import pytest

from modules.api.routes_asr import get_request_params
from modules.core import utils
from modules.inference import model_manager, scheduler

# Inject mock whisperx into sys.modules before importing modules
mock_whisperx = mock.MagicMock()
mock_whisperx.load_audio.return_value = "dummy_audio"
mock_whisperx.load_align_model.return_value = ("mock_align_model", "mock_metadata")
mock_whisperx.align.return_value = {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}]}
mock_diarize_pipeline = mock.MagicMock()
mock_whisperx.diarization.DiarizationPipeline.return_value = mock_diarize_pipeline
mock_diarize_pipeline.return_value = "mock_diarize_segments"
mock_whisperx.assign_word_speakers.return_value = {
    "segments": [{"start": 0.0, "end": 1.0, "text": "hello", "speaker": "SPEAKER_00"}]
}
sys.modules["whisperx"] = mock_whisperx


@pytest.fixture(autouse=True)
def reset_state():
    """Reset model_manager pools and scheduler states between tests."""
    model_manager.MODEL_POOL.clear()
    model_manager.PREPROCESSOR_POOL.clear()
    model_manager.DIARIZE_POOL.clear()
    model_manager.ALIGN_POOL.clear()

    with mock.patch("modules.core.config.HARDWARE_UNITS", [{"id": "CPU", "type": "CPU", "name": "CPU"}]):
        from modules.inference.scheduler import SchedulerState

        scheduler.STATE = SchedulerState()
        scheduler.STATE.engine_initialized = True

    # Reset thread context
    utils.THREAD_CONTEXT.is_priority = False
    if hasattr(utils.THREAD_CONTEXT, "assigned_unit"):
        utils.THREAD_CONTEXT.assigned_unit = None
    yield


def test_diarization_success():
    """Verify successful transcription, alignment, and diarization flow."""
    mock_model = mock.MagicMock()
    mock_info = mock.MagicMock(language="en", language_probability=0.95, duration=5.0)
    mock_segment = mock.MagicMock(start=0.0, end=1.0, text="hello")
    mock_model.transcribe.return_value = ([mock_segment], mock_info)

    model_manager.MODEL_POOL["CPU"] = mock_model

    # Reset mock calls
    mock_whisperx.load_audio.reset_mock()
    mock_whisperx.load_align_model.reset_mock()
    mock_whisperx.align.reset_mock()
    mock_whisperx.assign_word_speakers.reset_mock()
    mock_diarize_pipeline.reset_mock()

    result = model_manager.run_transcription(
        "test.wav",
        language="en",
        task="transcribe",
        diarize=True,
        min_speakers=1,
        max_speakers=2,
        hf_token="fake_token",
    )

    # Verify return result
    assert result["segments"][0]["speaker"] == "SPEAKER_00"
    assert result["segments"][0]["text"] == "hello"

    # Verify calls
    mock_whisperx.load_audio.assert_called_once_with("test.wav")
    mock_whisperx.load_align_model.assert_called_once_with(language_code="en", device="cpu")
    mock_whisperx.align.assert_called_once()
    mock_whisperx.diarization.DiarizationPipeline.assert_called_once_with(use_auth_token="fake_token", device="cpu")
    mock_diarize_pipeline.assert_called_once_with("dummy_audio", min_speakers=1, max_speakers=2)
    mock_whisperx.assign_word_speakers.assert_called_once()


def test_diarization_caching_and_unloading():
    """Verify align and diarize models are cached and properly unloaded."""
    mock_model = mock.MagicMock()
    mock_info = mock.MagicMock(language="en", language_probability=0.95, duration=5.0)
    mock_segment = mock.MagicMock(start=0.0, end=1.0, text="hello")
    mock_model.transcribe.return_value = ([mock_segment], mock_info)

    model_manager.MODEL_POOL["CPU"] = mock_model

    # Run twice
    for _ in range(2):
        model_manager.run_transcription(
            "test.wav", language="en", task="transcribe", diarize=True, hf_token="fake_token"
        )

    # Verify loading functions only called once
    assert len(model_manager.DIARIZE_POOL) == 1
    assert len(model_manager.ALIGN_POOL) == 1

    # Unload models
    with mock.patch("modules.inference.model_manager.utils.get_system_telemetry", return_value={}):
        model_manager.unload_models()

    # Verify pools are cleared
    assert len(model_manager.DIARIZE_POOL) == 0
    assert len(model_manager.ALIGN_POOL) == 0


def test_diarization_missing_token_fallback():
    """Verify fallback to non-diarized output when HF token is missing."""
    mock_model = mock.MagicMock()
    mock_info = mock.MagicMock(language="en", language_probability=0.95, duration=5.0)
    mock_segment = mock.MagicMock(start=0.0, end=1.0, text="hello")
    mock_model.transcribe.return_value = ([mock_segment], mock_info)

    model_manager.MODEL_POOL["CPU"] = mock_model

    # Ensure config token is empty
    with mock.patch("modules.core.config.HF_TOKEN", ""):
        result = model_manager.run_transcription(
            "test.wav", language="en", task="transcribe", diarize=True, hf_token=None
        )

    # Should fall back to standard results without speaker
    assert "speaker" not in result["segments"][0]
    assert result["segments"][0]["text"] == "hello"


def test_diarization_failure_fallback():
    """Verify fallback to non-diarized output when diarization pipeline fails."""
    mock_model = mock.MagicMock()
    mock_info = mock.MagicMock(language="en", language_probability=0.95, duration=5.0)
    mock_segment = mock.MagicMock(start=0.0, end=1.0, text="hello")
    mock_model.transcribe.return_value = ([mock_segment], mock_info)

    model_manager.MODEL_POOL["CPU"] = mock_model

    # Force an exception during alignment
    with mock.patch("whisperx.align", side_effect=RuntimeError("Align fail")):
        result = model_manager.run_transcription(
            "test.wav", language="en", task="transcribe", diarize=True, hf_token="fake_token"
        )

    # Should fall back and return standard segment
    assert "speaker" not in result["segments"][0]
    assert result["segments"][0]["text"] == "hello"


def test_routes_extract_diarize_params():
    """Verify that ASR endpoints parse and forward diarization params."""
    mock_req = mock.MagicMock()
    mock_req.headers = {}
    mock_req.query_params = {"diarize": "true", "min_speakers": "2", "max_speakers": "4", "hf_token": "test_tok"}
    mock_req.url.path = "/asr"
    params = asyncio.run(get_request_params(mock_req, {}))
    assert params["diarize"] is True
    assert params["min_speakers"] == 2
    assert params["max_speakers"] == 4
    assert params["hf_token"] == "test_tok"

    # Test default fallback values
    mock_req_default = mock.MagicMock()
    mock_req_default.headers = {}
    mock_req_default.query_params = {}
    mock_req_default.url.path = "/asr"
    params_default = asyncio.run(get_request_params(mock_req_default, {}))
    assert params_default["diarize"] is False
    assert params_default["min_speakers"] is None
    assert params_default["max_speakers"] is None
    assert params_default["hf_token"] is None

    # Test invalid int params fallback to None
    mock_req_invalid = mock.MagicMock()
    mock_req_invalid.headers = {}
    mock_req_invalid.query_params = {"min_speakers": "invalid", "max_speakers": "invalid"}
    mock_req_invalid.url.path = "/asr"
    params_invalid = asyncio.run(get_request_params(mock_req_invalid, {}))
    assert params_invalid["min_speakers"] is None
    assert params_invalid["max_speakers"] is None

    # Test X-HF-Token header fallback
    mock_req_header = mock.MagicMock()
    mock_req_header.query_params = {}
    mock_req_header.headers = {"X-HF-Token": "header_tok"}
    mock_req_header.url.path = "/asr"
    params_header = asyncio.run(get_request_params(mock_req_header, {}))
    assert params_header["hf_token"] == "header_tok"


def test_utils_speaker_formatting():
    """Verify that subtitle/text writers properly format speaker labels."""
    result = {
        "segments": [
            {"start": 0.0, "end": 2.5, "text": "Hello world", "speaker": "SPEAKER_00"},
            {"start": 3.0, "end": 4.5, "text": "Goodbye", "speaker": "SPEAKER_01"},
        ]
    }

    # Test SRT formatting
    srt_out = utils.generate_srt(result)
    assert "[SPEAKER_00]: Hello world" in srt_out
    assert "[SPEAKER_01]: Goodbye" in srt_out

    # Test VTT formatting
    vtt_out = utils.generate_vtt(result)
    assert "[SPEAKER_00]: Hello world" in vtt_out
    assert "[SPEAKER_01]: Goodbye" in vtt_out

    # Test TSV formatting
    tsv_out = utils.generate_tsv(result)
    assert "[SPEAKER_00]: Hello world" in tsv_out
    assert "[SPEAKER_01]: Goodbye" in tsv_out

    # Test TXT formatting
    txt_out = utils.generate_txt(result)
    assert "[SPEAKER_00]: Hello world" in txt_out
    assert "[SPEAKER_01]: Goodbye" in txt_out


def test_run_diarization_duration_skip(monkeypatch):
    """Verify that run_diarization skips when audio duration exceeds limit."""
    import modules.inference.diarization as diarization_mod
    from modules.inference.diarization import run_diarization

    # Set MAX_DIARIZE_DURATION_SEC
    monkeypatch.setattr(diarization_mod, "MAX_DIARIZE_DURATION_SEC", 10)

    info = mock.MagicMock(duration=15.0)
    raw_segments = [{"start": 0.0, "end": 1.0, "text": "hello"}]

    res = run_diarization(
        processed_path="dummy.wav",
        raw_segments=raw_segments,
        info=info,
        language="en",
        min_speakers=1,
        max_speakers=2,
        hf_token="token",
        unit_id="CPU",
    )
    assert len(res) == 1
    assert res[0]["text"] == "hello"
    assert "speaker" not in res[0]


def test_run_diarization_duration_warning(monkeypatch):
    """Verify that run_diarization logs a warning for long files."""
    import modules.inference.diarization as diarization_mod
    from modules.inference.diarization import run_diarization

    monkeypatch.setattr(diarization_mod, "_DIARIZE_WARN_THRESHOLD_SEC", 10)

    info = mock.MagicMock(duration=15.0, language="en")
    raw_segments = [{"start": 0.0, "end": 1.0, "text": "hello"}]

    with mock.patch("modules.inference.diarization.logger") as mock_logger:
        with mock.patch("modules.inference.diarization._get_align_model", return_value=(mock.MagicMock(), {})):
            with mock.patch("modules.inference.diarization._get_diarize_pipeline") as mock_dp:
                mock_dp.return_value = mock.MagicMock()
                mock_whisperx.assign_word_speakers.return_value = {
                    "segments": [{"start": 0.0, "end": 1.0, "text": "hello", "speaker": "SPEAKER_00"}]
                }

                run_diarization(
                    processed_path="dummy.wav",
                    raw_segments=raw_segments,
                    info=info,
                    language="en",
                    min_speakers=1,
                    max_speakers=2,
                    hf_token="token",
                    unit_id="CPU",
                )
                mock_logger.warning.assert_called()


def test_run_diarization_words_key():
    """Verify that run_diarization preserves 'words' in result segments."""
    from modules.inference.diarization import run_diarization

    info = mock.MagicMock(duration=5.0, language="en")
    raw_segments = [{"start": 0.0, "end": 1.0, "text": "hello"}]

    with mock.patch("modules.inference.diarization._get_align_model", return_value=(mock.MagicMock(), {})):
        with mock.patch("modules.inference.diarization._get_diarize_pipeline") as mock_dp:
            mock_dp.return_value = mock.MagicMock()

            mock_whisperx.assign_word_speakers.return_value = {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "text": "hello",
                        "speaker": "SPEAKER_00",
                        "words": [{"word": "hello", "start": 0.0, "end": 0.5}],
                    }
                ]
            }

            res = run_diarization(
                processed_path="dummy.wav",
                raw_segments=raw_segments,
                info=info,
                language="en",
                min_speakers=1,
                max_speakers=2,
                hf_token="token",
                unit_id="CPU",
            )
            assert res[0]["words"] == [{"word": "hello", "start": 0.0, "end": 0.5}]
