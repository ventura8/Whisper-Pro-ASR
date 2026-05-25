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


def test_lazy_import_vad_monkeypatching():
    """Test that _lazy_import_vad properly monkeypatches get_speech_timestamps."""
    # Reset VAD state to simulate first load
    vad._VAD_STATE["wrapped"] = False
    vad._VAD_STATE["wrapped_func"] = None

    # Mock fw_get_ts
    mock_orig_get_ts = mock.MagicMock()
    mock_orig_get_ts.return_value = [{"start": 16000, "end": 32000}]

    # Backup original fw_get_ts
    orig_fw_get_ts = vad.fw_get_ts
    vad.fw_get_ts = mock_orig_get_ts

    try:
        # Run _lazy_import_vad
        fw_get_ts_wrapped, _, _ = vad._lazy_import_vad()

        assert vad._VAD_STATE["wrapped"] is True
        assert vad._VAD_STATE["wrapped_func"] is not None
        assert fw_get_ts_wrapped is not mock_orig_get_ts

        # Test calling the wrapped function with mock logger
        audio = np.zeros(48000)
        with mock.patch("modules.inference.vad.logger") as mock_logger:
            res = fw_get_ts_wrapped(audio)

            # Verify the original get_speech_timestamps was called
            mock_orig_get_ts.assert_called_once_with(audio)
            assert res == [{"start": 16000, "end": 32000}]
            mock_logger.info.assert_called_once()
            log_arg = mock_logger.info.call_args[0][0]
            assert "[VAD] Speech detection complete" in log_arg

    finally:
        # Restore original state
        vad.fw_get_ts = orig_fw_get_ts
        vad._VAD_STATE["wrapped"] = False
        vad._VAD_STATE["wrapped_func"] = None


def test_lazy_import_vad_sys_modules_patching():
    """Test that _lazy_import_vad patches sys.modules['faster_whisper.vad']."""
    vad._VAD_STATE["wrapped"] = False
    vad._VAD_STATE["wrapped_func"] = None

    mock_orig_get_ts = mock.MagicMock()
    orig_fw_get_ts = vad.fw_get_ts
    vad.fw_get_ts = mock_orig_get_ts

    mock_module = mock.MagicMock()

    with mock.patch.dict("sys.modules", {"faster_whisper.vad": mock_module}):
        try:
            vad._lazy_import_vad()
            assert mock_module.get_speech_timestamps == vad._VAD_STATE["wrapped_func"]
        finally:
            vad.fw_get_ts = orig_fw_get_ts
            vad._VAD_STATE["wrapped"] = False
            vad._VAD_STATE["wrapped_func"] = None


def test_lazy_import_vad_none():
    """Test _lazy_import_vad behavior when fw_get_ts is None."""
    vad._VAD_STATE["wrapped"] = False
    vad._VAD_STATE["wrapped_func"] = None

    orig_fw_get_ts = vad.fw_get_ts
    vad.fw_get_ts = None

    try:
        fw_get_ts_ret, _, _ = vad._lazy_import_vad()
        assert fw_get_ts_ret is None
        assert vad._VAD_STATE["wrapped"] is False
    finally:
        vad.fw_get_ts = orig_fw_get_ts


def test_get_speech_timestamps_wrapped_exceptions():
    """Test that get_speech_timestamps_wrapped handles exceptions gracefully."""
    vad._VAD_STATE["wrapped"] = False
    vad._VAD_STATE["wrapped_func"] = None

    # Mock fw_get_ts to return non-iterable to trigger exception in speech_sec sum
    mock_orig_get_ts = mock.MagicMock()
    mock_orig_get_ts.return_value = 123  # Non-iterable

    orig_fw_get_ts = vad.fw_get_ts
    vad.fw_get_ts = mock_orig_get_ts

    try:
        fw_get_ts_wrapped, _, _ = vad._lazy_import_vad()

        # Call with mock logger
        audio = np.zeros(16000)
        with mock.patch("modules.inference.vad.logger") as mock_logger:
            res = fw_get_ts_wrapped(audio)
            assert res == 123
            # Verify no warning/error crashed the function, and it logged debug info
            mock_logger.debug.assert_not_called()  # since it returned 123, which is not list/tuple, it skipped sum
    finally:
        vad.fw_get_ts = orig_fw_get_ts
        vad._VAD_STATE["wrapped"] = False
        vad._VAD_STATE["wrapped_func"] = None
