"""Comprehensive coverage for VAD functions."""
# pylint: disable=no-member, protected-access, import-error, redefined-outer-name
from unittest import mock
import pytest
from modules import vad

# Mock objects for faster_whisper


@pytest.fixture
def mock_fw_vad(monkeypatch):
    """Mock faster_whisper VAD functions."""
    mock_get_ts = mock.MagicMock()
    mock_decode = mock.MagicMock()
    mock_options = mock.MagicMock()

    monkeypatch.setattr(vad, 'fw_get_vad_ts', mock_get_ts)
    monkeypatch.setattr(vad, 'fw_decode_audio', mock_decode)
    monkeypatch.setattr(vad, 'VadOptions', mock_options)

    return mock_get_ts, mock_decode, mock_options


def test_decode_audio_success(mock_fw_vad):
    """Test decode_audio calls faster-whisper decode."""
    _, mock_decode, _ = mock_fw_vad
    mock_decode.return_value = "fake_audio_array"

    result = vad.decode_audio("test.wav")

    assert result == "fake_audio_array"
    mock_decode.assert_called_once_with("test.wav", sampling_rate=16000)


def test_decode_audio_missing_backend(monkeypatch):
    """Test decode_audio raises ImportError if backend is missing."""
    monkeypatch.setattr(vad, 'fw_decode_audio', None)

    with pytest.raises(ImportError, match="faster-whisper audio utilities not found"):
        vad.decode_audio("test.wav")


def test_get_speech_timestamps_flow(mock_fw_vad):
    """Test get_speech_timestamps converts timestamps correctly."""
    mock_get_ts, _, mock_options = mock_fw_vad

    # Mock return value from faster_whisper: list of dicts with 'start', 'end' in SAMPLES
    # vad.py assumes input audio is 16kHz
    # So 16000 samples = 1 second
    mock_get_ts.return_value = [
        {'start': 0, 'end': 16000},       # 0.0s -> 1.0s
        {'start': 32000, 'end': 48000}    # 2.0s -> 3.0s
    ]

    audio_data = "fake_numpy_array"
    results = vad.get_speech_timestamps(audio_data)

    assert len(results) == 2
    assert results[0] == {'start': 0.0, 'end': 1.0}
    assert results[1] == {'start': 2.0, 'end': 3.0}

    mock_options.assert_called_once()
    mock_get_ts.assert_called_once()


def test_get_speech_timestamps_missing_backend(monkeypatch):
    """Test get_speech_timestamps handles missing backend gracefully."""
    monkeypatch.setattr(vad, 'fw_get_vad_ts', None)

    results = vad.get_speech_timestamps("audio")
    assert results == []


def test_get_speech_timestamps_error_handling(mock_fw_vad):
    """Test get_speech_timestamps handles internal errors."""
    mock_get_ts, _, _ = mock_fw_vad
    mock_get_ts.side_effect = RuntimeError("VAD crash")

    results = vad.get_speech_timestamps("audio")
    assert results == []


def test_get_speech_timestamps_from_path_success(mock_fw_vad):
    """Test get_speech_timestamps_from_path integration."""
    mock_get_ts, mock_decode, _ = mock_fw_vad

    mock_decode.return_value = "decoded_audio"
    mock_get_ts.return_value = [{'start': 16000, 'end': 32000}]

    results = vad.get_speech_timestamps_from_path("test.wav")

    mock_decode.assert_called_once_with("test.wav", sampling_rate=16000)
    mock_get_ts.assert_called_once_with("decoded_audio", vad_options=mock.ANY)
    assert results == [{'start': 1.0, 'end': 2.0}]


def test_get_speech_timestamps_from_path_decode_error(mock_fw_vad):
    """Test get_speech_timestamps_from_path handles decode errors."""
    _, mock_decode, _ = mock_fw_vad
    mock_decode.side_effect = Exception("Decode failed")

    results = vad.get_speech_timestamps_from_path("test.wav")
    assert results == []
