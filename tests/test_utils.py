"""Comprehensive coverage for utility functions."""
import subprocess
from unittest import mock
from modules import utils


def test_convert_to_wav_success():
    """Test successful conversion to WAV."""
    # Create a proper mock for Popen with correct returncode
    mock_process = mock.MagicMock()
    mock_process.__enter__.return_value = mock_process
    # Configure stdout.readline to return lines then empty string
    mock_process.stdout.readline.side_effect = ["out_time_ms=1000000", ""]
    mock_process.communicate.return_value = (None, "")
    # Ensure it's a real int to avoid CalledProcessError __str__ issues
    mock_process.returncode = 0

    with mock.patch("modules.utils.subprocess.Popen", return_value=mock_process):
        with mock.patch("os.path.exists", return_value=True):
            with mock.patch("os.path.getsize", return_value=1024):
                # Mock get_audio_duration at module level
                original_get_audio_duration = utils.get_audio_duration
                utils.get_audio_duration = mock.MagicMock(return_value=120.0)
                try:
                    with mock.patch("tempfile.NamedTemporaryFile") as mock_temp:
                        mock_temp.return_value.__enter__.return_value.name = "temp.wav"
                        res = utils.convert_to_wav("input.mp3")
                        assert res == "temp.wav"
                finally:
                    utils.get_audio_duration = original_get_audio_duration


def test_convert_to_wav_subprocess_error():
    """Test convert_to_wav handles subprocess errors."""
    with mock.patch("modules.utils.subprocess.Popen") as mock_popen:
        mock_process = mock.MagicMock()
        mock_process.__enter__.return_value = mock_process
        mock_process.stdout.readline.return_value = ""
        mock_process.returncode = 1
        mock_popen.return_value = mock_process

        with mock.patch("os.path.exists", return_value=True):
            with mock.patch("os.path.getsize", return_value=123):
                with mock.patch("os.remove") as mock_remove:
                    with mock.patch("modules.config.get_temp_dir", return_value="/tmp"):
                        res = utils.convert_to_wav("input.mp3")
                        assert res is None
                        mock_remove.assert_called()


def test_convert_to_wav_generic_error():
    """Test convert_to_wav handles generic errors."""
    with mock.patch("modules.utils.subprocess.Popen", side_effect=RuntimeError("Generic error")):
        with mock.patch("os.path.exists", return_value=True):
            with mock.patch("os.path.getsize", return_value=123):
                with mock.patch("os.remove") as mock_remove:
                    with mock.patch("modules.config.get_temp_dir", return_value="/tmp"):
                        res = utils.convert_to_wav("input.mp3")
                        assert res is None
                        mock_remove.assert_called()


def test_get_audio_duration_success():
    """Test successful duration retrieval."""
    # Mock subprocess.check_output
    original_check_output = subprocess.check_output

    def mock_check_output(cmd, **kwargs):
        if "ffprobe" in cmd[0]:
            return b"123.45\n"
        return original_check_output(cmd, **kwargs)

    subprocess.check_output = mock_check_output
    try:
        result = utils.get_audio_duration("test.wav")
        assert result == 123.45
    finally:
        subprocess.check_output = original_check_output


def test_get_audio_duration_error():
    """Test duration retrieval error fallback."""
    with mock.patch("subprocess.check_output", side_effect=Exception("fail")):
        assert utils.get_audio_duration("test.wav") == 0.0


def test_format_duration_variants():
    """Test duration formatting for different values."""
    assert utils.format_duration(0) == "00:00:00"
    assert utils.format_duration(3661) == "01:01:01"
    assert utils.format_duration(59) == "00:00:59"


def test_generate_srt_edge_cases():
    """Test SRT generation with various input structures."""
    # 1. Standard
    res = {
        "text": "Hello world",
        "segments": [{"timestamp": (0.0, 1.0), "text": "Hello world"}]
    }
    srt = utils.generate_srt(res)
    assert "00:00:00,000 --> 00:00:01,000" in srt
    assert "Hello world" in srt

    # 2. Results without segments but with text
    res = {"text": "Simple text"}
    srt = utils.generate_srt(res)
    assert "Simple text" in srt
    assert "00:00:00,000 --> 00:00:05,000" in srt

    # 3. Completely empty results
    srt = utils.generate_srt({})
    assert "[No dialogue detected]" in srt

    # 4. Segment with None timestamp
    res = {"segments": [{"timestamp": (None, None), "text": "Missing"}]}
    srt = utils.generate_srt(res)
    assert "00:00:00,000 --> 00:00:05,000" in srt

    # 5. Segment with invalid timestamp format
    res = {"segments": [{"timestamp": "invalid", "text": "Bad"}]}
    srt = utils.generate_srt(res)
    assert "00:00:00,000 --> 00:00:05,000" in srt


def test_generate_vtt():
    """Test VTT generation."""
    res = {
        "segments": [{"timestamp": (0.0, 1.5), "text": "Testing VTT"}]
    }
    vtt = utils.generate_vtt(res)
    assert "WEBVTT" in vtt
    assert "00:00:00.000 --> 00:00:01.500" in vtt
    assert "Testing VTT" in vtt


def test_generate_vtt_empty():
    """Test VTT with empty result."""
    assert "WEBVTT" in utils.generate_vtt({})
    assert "[No dialogue detected]" in utils.generate_vtt({})


def test_generate_vtt_fallback():
    """Test VTT fallback with no segments."""
    res = {"text": "Fallback text"}
    vtt = utils.generate_vtt(res)
    assert "WEBVTT" in vtt
    assert "Fallback text" in vtt


def test_generate_txt():
    """Test standard text generation."""
    res = {"text": "Just pure text."}
    assert utils.generate_txt(res) == "Just pure text."
    assert utils.generate_txt({}) == ""


def test_generate_tsv():
    """Test TSV generation."""
    res = {
        "segments": [{"timestamp": (0.0, 1.0), "text": "Tab Separated"}]
    }
    tsv = utils.generate_tsv(res)
    assert "start\tend\ttext" in tsv
    assert "0\t1000\tTab Separated" in tsv


def test_generate_tsv_empty():
    """Test TSV with empty result."""
    assert utils.generate_tsv({}) == "start\tend\ttext"


def test_convert_to_wav_missing_file():
    """Test convert_to_wav returns None if file missing."""
    with mock.patch("os.path.exists", return_value=False):
        assert utils.convert_to_wav("missing.mp3") is None


def test_convert_to_wav_empty_file():
    """Test convert_to_wav returns None if file empty."""
    with mock.patch("os.path.exists", return_value=True):
        with mock.patch("os.path.getsize", return_value=0):
            assert utils.convert_to_wav("empty.mp3") is None


def test_convert_to_wav_logging_coverage():
    """Test progress logging logic inside convert_to_wav loop."""
    mock_process = mock.MagicMock()
    mock_process.__enter__.return_value = mock_process
    # Simulate multiple progress updates
    mock_process.stdout.readline.side_effect = [
        "out_time_ms=1000000",  # 1s
        "out_time_ms=5000000",  # 5s
        "out_time_ms=10000000",  # 10s
        ""  # End
    ]
    mock_process.communicate.return_value = (None, "")
    mock_process.returncode = 0

    with mock.patch("modules.utils.subprocess.Popen", return_value=mock_process):
        with mock.patch("os.path.exists", return_value=True):
            with mock.patch("os.path.getsize", return_value=123):
                with mock.patch("modules.utils.get_audio_duration", return_value=10.0):
                    with mock.patch("tempfile.NamedTemporaryFile"):
                        # Just ensure it runs without error and exercises the logging block
                        utils.convert_to_wav("input.mp3")


class TestUtilsEdgeCases:
    """Uncovered edge cases in utils.py."""

    def test_standardization_filters_logic(self):
        """Test that the filters can be accessed."""
        # Just check the current value to avoid reload-induced hangs
        assert "norm" in utils.STANDARD_NORMALIZATION_FILTERS

    def test_convert_to_wav_cleanup_error(self):
        """Test exception handling when os.remove fails during cleanup."""
        mock_process = mock.MagicMock()
        mock_process.__enter__.return_value = mock_process
        mock_process.stdout.readline.return_value = ""
        mock_process.returncode = 1  # Force failure

        with mock.patch("modules.utils.get_audio_duration", return_value=10.0):
            with mock.patch("modules.utils.subprocess.Popen", return_value=mock_process):
                with mock.patch("os.path.exists", return_value=True):
                    with mock.patch("os.path.getsize", return_value=123):
                        with mock.patch("os.remove", side_effect=Exception("Perm error")):
                            with mock.patch("modules.config.get_temp_dir", return_value="/tmp"):
                                # Should catch the exception on remove and return None
                                assert utils.convert_to_wav("in.mp3") is None

    def test_parse_ffmpeg_progress_invalid_line(self):
        """Test _parse_ffmpeg_progress handles malformed status lines."""
        mock_proc = mock.MagicMock()
        mock_proc.stdout.readline.side_effect = ["out_time_ms=invalid", ""]
        # Should not raise
        utils._parse_ffmpeg_progress(mock_proc, 10.0)  # pylint: disable=protected-access

    def test_generate_vtt_edge_cases(self):
        """Test generate_vtt with no segments and error handling."""
        # Text only
        res = {"text": "Hello"}
        assert "00:00:00.000" in utils.generate_vtt(res)

        # Empty text
        assert "[No dialogue detected]" in utils.generate_vtt({"text": ""})

        # None timestamp
        res = {"segments": [{"timestamp": (None, None), "text": "X"}]}
        assert "00:00:00.000" in utils.generate_vtt(res)

    def test_generate_tsv_exception_in_loop(self):
        """Test generate_tsv handles exceptions inside the segment loop."""
        res = {"segments": [None]}
        tsv = utils.generate_tsv(res)
        assert "start\tend\ttext" in tsv  # Header only


def test_convert_to_wav_uses_temp_dir():
    """Test that convert_to_wav uses config.get_temp_dir."""
    mock_process = mock.MagicMock()
    mock_process.__enter__.return_value = mock_process
    mock_process.stdout.readline.return_value = ""
    mock_process.returncode = 0

    with mock.patch("modules.utils.subprocess.Popen", return_value=mock_process):
        with mock.patch("os.path.exists", return_value=True):
            with mock.patch("os.path.getsize", return_value=1024):
                with mock.patch("modules.utils.get_audio_duration", return_value=10.0):
                    with mock.patch("modules.config.get_temp_dir",
                                    return_value="/tmp/whisper") as mock_get_temp:
                        with mock.patch("tempfile.NamedTemporaryFile") as mock_temp:
                            mock_temp.return_value.__enter__.return_value.name = "temp.wav"
                            utils.convert_to_wav("input.mp3")
                            # Verify get_temp_dir was called with estimated size
                            # 10s * 16000 * 2 = 320000
                            mock_get_temp.assert_called_once_with(required_bytes=320000)
                            # Verify NamedTemporaryFile used that dir
                            mock_temp.assert_called_once()
                            assert mock_temp.call_args[1]["dir"] == "/tmp/whisper"
