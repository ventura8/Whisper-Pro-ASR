"""Comprehensive coverage for utility functions."""
import time
from unittest import mock
from modules import utils


def test_convert_to_wav_success():
    """Test successful conversion to WAV."""
    mock_process = mock.MagicMock()
    mock_process.__enter__.return_value = mock_process
    mock_process.stdout.readline.side_effect = ["out_time_ms=1000000", ""]
    mock_process.communicate.return_value = (None, "")
    mock_process.returncode = 0

    with mock.patch("modules.utils.subprocess.Popen", return_value=mock_process):
        with mock.patch("os.path.exists", return_value=True):
            with mock.patch("os.path.getsize", return_value=1024):
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


def test_get_audio_duration_success():
    """Test successful duration retrieval."""
    with mock.patch("subprocess.check_output", return_value=b"123.45\n"):
        result = utils.get_audio_duration("test.wav")
        assert result == 123.45


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
        "segments": [{"start": 0.0, "end": 1.0, "text": "Hello world"}]
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


def test_generate_vtt():
    """Test VTT generation."""
    res = {
        "segments": [{"start": 0.0, "end": 1.5, "text": "Testing VTT"}]
    }
    vtt = utils.generate_vtt(res)
    assert "WEBVTT" in vtt
    assert "00:00:00.000 --> 00:00:01.500" in vtt
    assert "Testing VTT" in vtt


def test_generate_tsv():
    """Test TSV generation with start/end keys."""
    res = {
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "Tab\tSeparated\nLines"},
            {"timestamp": (1.0, 2.0), "text": "Tuple"}
        ]
    }
    tsv = utils.generate_tsv(res)
    assert "start\tend\ttext" in tsv
    assert "0\t1000\tTab Separated Lines" in tsv
    assert "1000\t2000\tTuple" in tsv


def test_get_system_telemetry():
    """Test gathering system telemetry via psutil mocks."""
    with mock.patch("modules.utils.psutil.cpu_percent", return_value=10.0):
        with mock.patch("modules.utils.psutil.virtual_memory") as mock_mem:
            mock_mem.return_value.percent = 50.0
            mock_mem.return_value.used = 8 * (1024**3)
            mock_mem.return_value.total = 16 * (1024**3)
            with mock.patch("modules.utils._PROCESS_OBJ") as mock_proc:
                mock_proc.cpu_percent.return_value = 5.0
                mock_proc.memory_info.return_value.rss = 1024 * 1024 * 1024  # 1GB

                telemetry = utils.get_system_telemetry()
                assert telemetry["cpu_percent"] == 10.0
                assert telemetry["memory_percent"] == 50.0
                assert telemetry["app_cpu_percent"] == 5.0
                assert telemetry["app_memory_gb"] == 1.0


def test_get_pretty_model_name():
    """Test model name formatting."""
    assert utils.get_pretty_model_name("distil-whisper/distil-large-v3") == "Distil Large v3"
    assert utils.get_pretty_model_name("openai/whisper-tiny") == "Whisper Tiny"
    assert utils.get_pretty_model_name("unknown-model") == "Unknown Model"


def test_cleanup_old_files(tmp_path):
    """Test cleaning up old files based on retention."""
    test_dir = tmp_path / "cleanup_test"
    test_dir.mkdir()
    old_file = test_dir / "old.wav"
    old_file.write_text("old")

    now = time.time()
    with mock.patch("modules.utils.os.path.getmtime", return_value=now - (10 * 86400)):
        # Test success
        utils.cleanup_old_files(str(test_dir), days=5)
        assert not old_file.exists()

        # Test exception path
        fail_file = test_dir / "fail.wav"
        fail_file.write_text("fail")
        with mock.patch("modules.utils.os.remove", side_effect=Exception("Cleanup Fail")):
            utils.cleanup_old_files(str(test_dir), days=5)
            assert fail_file.exists()  # Should still exist if remove fails


def test_validate_audio():
    """Test audio file validation."""
    with mock.patch("os.path.exists", return_value=True):
        with mock.patch("os.path.getsize", return_value=1024):
            assert utils.validate_audio("test.wav") is True

    with mock.patch("os.path.exists", return_value=False):
        assert utils.validate_audio("test.wav") is False

    with mock.patch("os.path.exists", return_value=True):
        with mock.patch("os.path.getsize", return_value=0):
            assert utils.validate_audio("test.wav") is False


def test_clear_gpu_cache():
    """Test GPU cache clearing."""
    with mock.patch("modules.utils.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = True
        utils.clear_gpu_cache()
        mock_torch.cuda.empty_cache.assert_called_once()
