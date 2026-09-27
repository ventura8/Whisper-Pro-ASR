"""Tests for modules/inference/vad.py."""

from unittest import mock

import numpy as np
import pytest

from modules.inference.pipeline import vad


@pytest.fixture
def mock_vad_components():
    """Mock faster-whisper components returned by lazy_import_vad."""
    mock_get_ts = mock.MagicMock()
    mock_opts = mock.MagicMock()
    mock_decode = mock.MagicMock()

    with mock.patch("modules.inference.pipeline.vad.lazy_import_vad", return_value=(mock_get_ts, mock_opts, mock_decode)):
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

    with mock.patch("modules.core.process_exec.run_capture") as mock_run:
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

    # Mock process execution for ffmpeg call when offset is used
    with mock.patch("modules.core.process_exec.run_capture"):
        results = vad.get_speech_timestamps_from_path("test.wav", start_offset=5.0)

        assert len(results) == 1
        assert results[0]["start"] == 5.0
        assert results[0]["end"] == 6.0


def test_vad_missing_dependencies():
    """Test behavior when faster-whisper is not installed."""
    with mock.patch("modules.inference.pipeline.vad.lazy_import_vad", return_value=(None, None, None)):
        with pytest.raises(ImportError):
            vad.decode_audio("test.wav")


def test_vad_exception_handling(mock_vad_components):
    """Test that VAD handles internal errors gracefully."""
    mock_get_ts, _, _ = mock_vad_components
    mock_get_ts.side_effect = Exception("VAD error")

    results = vad.get_speech_timestamps(np.zeros(16000))
    assert results == []


@pytest.fixture
def fresh_vad_state():
    """Reset the VAD wrap state to simulate a first load, and reset it again afterwards.

    The state is reset to the unwrapped defaults (rather than restored) on teardown so the
    next real ``lazy_import_vad`` call re-wraps the genuine ``fw_get_ts``.
    """
    vad._VAD_STATE["wrapped"] = False
    vad._VAD_STATE["wrapped_func"] = None
    yield
    vad._VAD_STATE["wrapped"] = False
    vad._VAD_STATE["wrapped_func"] = None


@pytest.mark.usefixtures("fresh_vad_state")
def testlazy_import_vad_monkeypatching(monkeypatch):
    """Test that lazy_import_vad properly monkeypatches get_speech_timestamps."""
    # Mock fw_get_ts
    mock_orig_get_ts = mock.MagicMock()
    mock_orig_get_ts.return_value = [{"start": 16000, "end": 32000}]

    # monkeypatch restores the original fw_get_ts on teardown
    monkeypatch.setattr(vad, "fw_get_ts", mock_orig_get_ts)

    # Run lazy_import_vad
    fw_get_ts_wrapped, _, _ = vad.lazy_import_vad()

    # Test calling the wrapped function with mock logger
    audio = np.zeros(48000)
    with mock.patch("modules.inference.pipeline.vad.logger") as mock_logger:
        res = fw_get_ts_wrapped(audio)
        log_arg = mock_logger.info.call_args[0][0]
        assert all(
            [
                vad._VAD_STATE["wrapped"] is True,
                vad._VAD_STATE["wrapped_func"] is not None,
                fw_get_ts_wrapped is not mock_orig_get_ts,
                res == [{"start": 16000, "end": 32000}],
                mock_orig_get_ts.call_args == mock.call(audio),
                mock_logger.info.call_count == 1,
                "[VAD] Speech detection complete" in log_arg,
            ]
        )


@pytest.mark.usefixtures("fresh_vad_state")
def testlazy_import_vad_sys_modules_patching(monkeypatch):
    """Test that lazy_import_vad patches sys.modules['faster_whisper.vad']."""
    monkeypatch.setattr(vad, "fw_get_ts", mock.MagicMock())

    mock_module = mock.MagicMock()

    with mock.patch.dict("sys.modules", {"faster_whisper.vad": mock_module}):
        vad.lazy_import_vad()
        assert mock_module.get_speech_timestamps == vad._VAD_STATE["wrapped_func"]


@pytest.mark.usefixtures("fresh_vad_state")
def testlazy_import_vad_none(monkeypatch):
    """Test lazy_import_vad behavior when fw_get_ts is None."""
    monkeypatch.setattr(vad, "fw_get_ts", None)

    fw_get_ts_ret, _, _ = vad.lazy_import_vad()
    assert fw_get_ts_ret is None
    assert vad._VAD_STATE["wrapped"] is False


@pytest.mark.usefixtures("fresh_vad_state")
def test_get_speech_timestamps_wrapped_exceptions(monkeypatch):
    """Test that get_speech_timestamps_wrapped handles exceptions gracefully."""
    # Mock fw_get_ts to return non-iterable to trigger exception in speech_sec sum
    mock_orig_get_ts = mock.MagicMock()
    mock_orig_get_ts.return_value = 123  # Non-iterable
    monkeypatch.setattr(vad, "fw_get_ts", mock_orig_get_ts)

    fw_get_ts_wrapped, _, _ = vad.lazy_import_vad()

    # Call with mock logger
    audio = np.zeros(16000)
    with mock.patch("modules.inference.pipeline.vad.logger") as mock_logger:
        res = fw_get_ts_wrapped(audio)
        assert res == 123
        # Verify no warning/error crashed the function, and it logged debug info
        mock_logger.debug.assert_not_called()  # since it returned 123, which is not list/tuple, it skipped sum


@pytest.mark.usefixtures("fresh_vad_state")
def test_get_speech_timestamps_wrapped_exception(monkeypatch):
    """Test exception path inside get_speech_timestamps_wrapped."""
    # Return list of dicts that misses 'end'/'start' keys to raise KeyError during sum
    mock_orig_get_ts = mock.MagicMock()
    mock_orig_get_ts.return_value = [{"invalid_key": 100}]
    monkeypatch.setattr(vad, "fw_get_ts", mock_orig_get_ts)

    fw_get_ts_wrapped, _, _ = vad.lazy_import_vad()
    audio = np.zeros(16000)

    with mock.patch("modules.inference.pipeline.vad.logger") as mock_logger:
        res = fw_get_ts_wrapped(audio)
        # Should catch exception and log debug message
        mock_logger.debug.assert_called_once()
        assert res == [{"invalid_key": 100}]


@pytest.mark.usefixtures("fresh_vad_state")
def testlazy_import_vad_sys_modules_exception(monkeypatch):
    """Test exception path inside lazy_import_vad's sys.modules loop."""
    monkeypatch.setattr(vad, "fw_get_ts", mock.MagicMock())

    # Mock sys.modules.items to raise an exception
    with mock.patch("sys.modules", mock.MagicMock(items=mock.Mock(side_effect=RuntimeError("sys.modules mock error")))):
        # Should catch exception gracefully and not crash
        vad.lazy_import_vad()


def test_get_speech_timestamps_from_path_exception():
    """Test get_speech_timestamps_from_path handles exceptions gracefully."""
    with mock.patch("modules.inference.pipeline.vad.decode_audio", side_effect=RuntimeError("Decoding mock error")):
        with mock.patch("modules.inference.pipeline.vad.logger") as mock_logger:
            res = vad.get_speech_timestamps_from_path("dummy.wav")
            assert res == []
            mock_logger.exception.assert_called_once()


def test_get_speech_timestamps_missing_dependencies():
    """Test get_speech_timestamps returns [] when VAD components are missing."""
    with mock.patch("modules.inference.pipeline.vad.lazy_import_vad", return_value=(None, None, None)):
        res = vad.get_speech_timestamps(np.zeros(16000))
        assert res == []
