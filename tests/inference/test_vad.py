"""Tests for modules/inference/vad.py."""
from unittest import mock
import numpy as np
import pytest
from modules.inference import vad


@pytest.fixture
def mock_vad_components():
    """Mock faster-whisper components returned by _lazy_import_vad."""
    mock_get_ts = mock.MagicMock()
    mock_opts = mock.MagicMock()
    mock_decode = mock.MagicMock()

    with mock.patch("modules.inference.vad._lazy_import_vad", return_value=(mock_get_ts, mock_opts, mock_decode)):
        yield mock_get_ts, mock_opts, mock_decode


def test_decode_audio_simple(mock_vad_components):
    """Test standard audio decoding."""
    _, _, mock_decode = mock_vad_components
    mock_decode.return_value = np.zeros(16000)

    audio = vad.decode_audio("test.wav")
    assert len(audio) == 16000
    mock_decode.assert_called_once_with("test.wav", sampling_rate=16000)


def test_decode_audio_with_offset(mock_vad_components):
    """Test audio decoding with ffmpeg seeking."""
    _, _, mock_decode = mock_vad_components
    mock_decode.return_value = np.zeros(8000)

    with mock.patch("subprocess.run") as mock_run:
        audio = vad.decode_audio("test.wav", start_offset=10, duration=5)
        assert len(audio) == 8000
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "-ss" in cmd
        assert "10" in cmd


def test_get_speech_timestamps(mock_vad_components):
    """Test VAD timestamp extraction."""
    mock_get_ts, mock_opts_class, _ = mock_vad_components
    mock_get_ts.return_value = [{"start": 16000, "end": 32000}]

    audio = np.zeros(48000)
    results = vad.get_speech_timestamps(audio)

    assert len(results) == 1
    assert results[0]["start"] == 1.0
    assert results[0]["end"] == 2.0
    mock_opts_class.assert_called_once()


def test_get_speech_timestamps_from_path(mock_vad_components):
    """Test full VAD pipeline from file path."""
    mock_get_ts, _, mock_decode = mock_vad_components
    mock_decode.return_value = np.zeros(16000)
    mock_get_ts.return_value = [{"start": 0, "end": 16000}]

    # Mock subprocess.run for ffmpeg call when offset is used
    with mock.patch("subprocess.run"):
        results = vad.get_speech_timestamps_from_path("test.wav", start_offset=5.0)

        assert len(results) == 1
        assert results[0]["start"] == 5.0
        assert results[0]["end"] == 6.0


def test_vad_missing_dependencies():
    """Test behavior when faster-whisper is not installed."""
    with mock.patch("modules.inference.vad._lazy_import_vad", return_value=(None, None, None)):
        with pytest.raises(ImportError):
            vad.decode_audio("test.wav")


def test_vad_exception_handling(mock_vad_components):
    """Test that VAD handles internal errors gracefully."""
    mock_get_ts, _, _ = mock_vad_components
    mock_get_ts.side_effect = Exception("VAD error")

    results = vad.get_speech_timestamps(np.zeros(16000))
    assert results == []
