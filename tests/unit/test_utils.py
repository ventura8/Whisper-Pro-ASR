"""Comprehensive coverage for utility functions."""

import importlib
import os
import sys
import time
from unittest import mock

import pytest

from modules.core import utils


def test_convert_to_wav_success():
    """Test successful conversion to WAV."""
    mock_process = mock.MagicMock()
    mock_process.__enter__.return_value = mock_process
    mock_process.stdout.readline.side_effect = ["out_time_ms=1000000", ""]
    mock_process.communicate.return_value = (None, "")
    mock_process.returncode = 0

    with mock.patch("modules.core.utils.subprocess.Popen", return_value=mock_process):
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
    with mock.patch("modules.core.utils.subprocess.Popen") as mock_popen:
        mock_process = mock.MagicMock()
        mock_process.__enter__.return_value = mock_process
        mock_process.stdout.readline.return_value = ""
        mock_process.returncode = 1
        mock_popen.return_value = mock_process

        with mock.patch("os.path.exists", return_value=True):
            with mock.patch("os.path.getsize", return_value=123):
                with mock.patch("os.remove") as mock_remove:
                    with mock.patch("modules.core.config.get_temp_dir", return_value="/tmp"):
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
    res = {"text": "Hello world", "segments": [{"start": 0.0, "end": 1.0, "text": "Hello world"}]}
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
    res = {"segments": [{"start": 0.0, "end": 1.5, "text": "Testing VTT"}]}
    vtt = utils.generate_vtt(res)
    assert "WEBVTT" in vtt
    assert "00:00:00.000 --> 00:00:01.500" in vtt
    assert "Testing VTT" in vtt


def test_generate_tsv():
    """Test TSV generation with start/end keys."""
    res = {
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "Tab\tSeparated\nLines"},
            {"timestamp": (1.0, 2.0), "text": "Tuple"},
        ]
    }
    tsv = utils.generate_tsv(res)
    assert "start\tend\ttext" in tsv
    assert "0\t1000\tTab Separated Lines" in tsv
    assert "1000\t2000\tTuple" in tsv


def test_get_system_telemetry():
    """Test gathering system telemetry via psutil mocks."""
    with mock.patch("modules.core.utils.psutil.cpu_percent", return_value=10.0):
        with mock.patch("modules.core.utils.psutil.virtual_memory") as mock_mem:
            mock_mem.return_value.percent = 50.0
            mock_mem.return_value.used = 8 * (1024**3)
            mock_mem.return_value.total = 16 * (1024**3)
            with mock.patch("modules.core.utils._PROCESS_OBJ") as mock_proc:
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
    with mock.patch("modules.core.utils.os.path.getmtime", return_value=now - (10 * 86400)):
        # Test success
        utils.cleanup_old_files(str(test_dir), days=5)
        assert not old_file.exists()

        # Test exception path
        fail_file = test_dir / "fail.wav"
        fail_file.write_text("fail")
        with mock.patch("modules.core.utils.os.remove", side_effect=Exception("Cleanup Fail")):
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
    with mock.patch("modules.core.utils.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = True
        utils.clear_gpu_cache()
        mock_torch.cuda.empty_cache.assert_called_once()


def test_subtitle_wrapping_logic():
    """Verify text wrapping utilities and layout constraints in SRT/VTT writers."""
    # Test wrap_text directly
    long_text = "This is a very long text that we want to wrap to a maximum width of characters."
    wrapped = utils.wrap_text(long_text, max_line_width=20)
    lines = wrapped.split("\n")
    for line in lines:
        assert len(line) <= 20

    # Test max_line_count limit
    limited = utils.wrap_text(long_text, max_line_width=20, max_line_count=2)
    assert len(limited.split("\n")) == 2

    # Test SRT/VTT formatters wrapping
    result = {"segments": [{"start": 0.0, "end": 5.0, "text": long_text}]}

    srt_out = utils.generate_srt(result, max_line_width=20, max_line_count=2)
    # The segment text block should contain wrapped lines and be capped at 2 lines
    assert "wrap to a maximum" not in srt_out
    assert "This is a very long\ntext that we want to" in srt_out

    vtt_out = utils.generate_vtt(result, max_line_width=20, max_line_count=2)
    assert "wrap to a maximum" not in vtt_out
    assert "This is a very long\ntext that we want to" in vtt_out


def test_generate_srt_vtt_highlight_words():
    """Verify subtitle generators support highlight_words parameter."""
    res = {
        "segments": [
            {
                "start": 0.5,
                "end": 2.0,
                "text": "Hello world",
                "words": [{"word": " Hello", "start": 0.5, "end": 1.0}, {"word": " world", "start": 1.0, "end": 2.0}],
            }
        ]
    }

    # 1. Test SRT highlighting (generates individual sub-blocks for each word)
    srt = utils.generate_srt(res, highlight_words=True)
    assert "1\n00:00:00,500 --> 00:00:01,000" in srt
    assert '<font color="#E0E0E0">Hello</font> world' in srt
    assert "2\n00:00:01,000 --> 00:00:02,000" in srt
    assert 'Hello <font color="#E0E0E0">world</font>' in srt

    # 2. Test VTT highlighting (generates intra-cue karaoke timestamps)
    vtt = utils.generate_vtt(res, highlight_words=True)
    assert "WEBVTT" in vtt
    assert "00:00:00.500 --> 00:00:02.000" in vtt
    assert "<00:00:00.500>Hello <00:00:01.000>world" in vtt


def test_generate_vtt_highlight_words_handles_none_word_start():
    """VTT karaoke formatting should fall back to segment start when word start is None."""
    res = {
        "segments": [
            {
                "start": 0.5,
                "end": 2.0,
                "text": "Hello world",
                "words": [{"word": " Hello", "start": None}, {"word": " world", "start": 1.0}],
            }
        ]
    }

    vtt = utils.generate_vtt(res, highlight_words=True)
    assert "<00:00:00.500>Hello <00:00:01.000>world" in vtt


def test_generate_srt_highlight_words_handles_none_word_timestamps():
    """SRT word highlighting should use segment timestamps when word start/end are None."""
    res = {
        "segments": [
            {
                "start": 0.5,
                "end": 2.0,
                "text": "Hello world",
                "words": [
                    {"word": " Hello", "start": None, "end": None},
                    {"word": " world", "start": 1.0, "end": 2.0},
                ],
            }
        ]
    }

    srt = utils.generate_srt(res, highlight_words=True)
    assert "1\n00:00:00,500 --> 00:00:02,000" in srt
    assert "2\n00:00:01,000 --> 00:00:02,000" in srt


def test_thread_context_reset():
    """Verify ContextVarProxy reset behavior."""
    utils.THREAD_CONTEXT.filename = "initial_file.mp3"
    assert utils.THREAD_CONTEXT.filename == "initial_file.mp3"

    utils.THREAD_CONTEXT.reset()
    # Check that accessing filename now raises AttributeError since context was cleared
    with pytest.raises(AttributeError):
        _ = utils.THREAD_CONTEXT.filename


def test_parse_ffmpeg_progress_updates_dashboard_stage_with_percentage():
    """FFmpeg progress parsing should publish percentage stage updates for dashboard visibility."""
    process = mock.MagicMock()
    process.stdout.readline.side_effect = [
        "out_time_ms=1000000\n",
        "out_time_ms=6000000\n",
        "speed= 1.50x\n",
        "",
    ]

    with mock.patch("modules.inference.scheduler.update_task_progress") as mock_update_stage:
        speed = utils.parse_ffmpeg_progress(process, 10.0)

    assert speed == "1.50x"
    stage_updates = [call.args[1] for call in mock_update_stage.call_args_list if len(call.args) >= 2]
    assert "FFmpeg (10%)" in stage_updates
    assert "FFmpeg (60%)" in stage_updates


def test_parse_ffmpeg_progress_invokes_yield_callback_on_stage_updates():
    """FFmpeg progress parsing should trigger cooperative preemption callbacks."""
    process = mock.MagicMock()
    process.stdout.readline.side_effect = [
        "out_time_ms=1000000\n",
        "out_time_ms=2000000\n",
        "out_time_ms=7000000\n",
        "",
    ]

    yield_calls = []

    def _yield_cb():
        yield_calls.append("yield")

    utils.parse_ffmpeg_progress(process, 10.0, yield_cb=_yield_cb)

    # 10%, 20%, 70% each advance at least 5% and should yield.
    assert len(yield_calls) == 3


def test_torch_import_none():
    """Verify torch import failure handling."""
    with mock.patch.dict(sys.modules, {"torch": None}):
        importlib.reload(utils)
        assert utils.torch is None
    importlib.reload(utils)


def test_context_var_proxy_edge_cases():
    """Verify ContextVarProxy edge attributes and deletions."""
    # Attribute missing source_path
    with pytest.raises(AttributeError):
        _ = utils.THREAD_CONTEXT.source_path

    # Setters and getters
    utils.THREAD_CONTEXT.tracked_files = ["/tmp/fake.wav"]
    assert utils.THREAD_CONTEXT.tracked_files == ["/tmp/fake.wav"]

    utils.THREAD_CONTEXT.source_path = "/tmp/fake.wav"
    assert utils.THREAD_CONTEXT.source_path == "/tmp/fake.wav"

    # Deletions
    del utils.THREAD_CONTEXT.tracked_files
    assert not utils.THREAD_CONTEXT.tracked_files

    del utils.THREAD_CONTEXT.source_path
    with pytest.raises(AttributeError):
        _ = utils.THREAD_CONTEXT.source_path

    del utils.THREAD_CONTEXT.filename
    with pytest.raises(AttributeError):
        _ = utils.THREAD_CONTEXT.filename

    # Dynamic attribute deletion
    utils.THREAD_CONTEXT.custom_val = "custom"
    assert utils.THREAD_CONTEXT.custom_val == "custom"
    del utils.THREAD_CONTEXT.custom_val
    with pytest.raises(AttributeError):
        _ = utils.THREAD_CONTEXT.custom_val

    # setattr / hasattr fallback path
    original_getattr = utils.ContextVarProxy.__getattr__
    raise_err = True

    def mock_getattr(self, name):
        nonlocal raise_err
        if name == "tracked_files" and raise_err:
            raise_err = False
            raise AttributeError("mocked")
        return original_getattr(self, name)

    with mock.patch.object(utils.ContextVarProxy, "__getattr__", mock_getattr):
        # Clear/delete tracked_files context first
        del utils.THREAD_CONTEXT.tracked_files
        files = utils.get_tracked_files()
        assert not files


def test_convert_base_edge_cases(tmp_path):
    """Test standard conversion filter config, prepare_for_uvr, and conversion failures."""
    # loudnorm filter configuration path
    with mock.patch("modules.core.config.FFMPEG_FILTER", "loudnorm"):
        importlib.reload(utils)
        assert utils.STANDARD_NORMALIZATION_FILTERS == "loudnorm=I=-16:TP=-1.5:LRA=11"
    importlib.reload(utils)

    # prepare_for_uvr
    test_file = tmp_path / "hq_test.wav"
    test_file.write_text("audio")
    # prepare_for_uvr (which calls conversion since extension skip is removed)
    with (
        mock.patch("modules.core.utils._convert_base", return_value="converted.wav") as mock_conv,
        mock.patch("modules.core.utils.track_file", return_value="converted.wav"),
    ):
        assert utils.prepare_for_uvr(str(test_file)) == "converted.wav"
        mock_conv.assert_called_once()

    # prepare_for_uvr with conversion
    test_mp3 = tmp_path / "hq_test.mp3"
    test_mp3.write_text("audio")

    _convert_base = utils.__dict__["_convert_base"]
    with (
        mock.patch("modules.core.utils._convert_base", return_value="converted.wav"),
        mock.patch("modules.core.utils.track_file", return_value="converted.wav"),
    ):
        assert utils.prepare_for_uvr(str(test_mp3)) == "converted.wav"

    # convert_base yield_cb and failures
    test_err_file = tmp_path / "err_test.mp3"
    test_err_file.write_text("bad data")

    yield_calls = 0

    def _yield():
        nonlocal yield_calls
        yield_calls += 1

    # Force conversion exception
    with (
        mock.patch("modules.core.utils._run_ffmpeg_standardization", side_effect=RuntimeError("convert error")),
        mock.patch("modules.core.utils.get_audio_duration", return_value=5.0),
        mock.patch("os.remove", side_effect=OSError("remove error")),
    ):
        res = _convert_base(str(test_err_file), utils.STANDARD_AUDIO_FLAGS, 16000, 1, yield_cb=_yield)
        assert res is None
        assert yield_calls == 1


def test_run_ffmpeg_standardization_edge_cases(tmp_path):
    """Verify hwaccel command injection, default flags, and watchdog timeout/termination."""
    output_wav = tmp_path / "out.wav"

    # default flags and hwaccel
    _run_ffmpeg_standardization = utils.__dict__["_run_ffmpeg_standardization"]
    with (
        mock.patch("modules.core.config.FFMPEG_HWACCEL", "cuvid"),
        mock.patch("modules.core.config.FFMPEG_THREADS", 2),
        mock.patch("subprocess.Popen") as mock_popen,
    ):
        mock_proc = mock.MagicMock()
        mock_proc.__enter__.return_value = mock_proc
        mock_proc.stdout.readline.return_value = ""
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc

        _run_ffmpeg_standardization("in.mp3", str(output_wav), 10.0, flags=None)
        args, _ = mock_popen.call_args
        command = args[0]
        assert "-hwaccel" in command
        assert "cuvid" in command

    # timeout logic trigger
    with (
        mock.patch("modules.core.config.FFMPEG_HWACCEL", "none"),
        mock.patch("subprocess.Popen") as mock_popen,
        mock.patch("threading.Timer") as mock_timer,
    ):
        mock_proc = mock.MagicMock()
        mock_proc.__enter__.return_value = mock_proc
        mock_proc.stdout.readline.return_value = ""
        mock_proc.poll.side_effect = [None, None]  # still active during kill
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc

        def timer_cb():
            pass

        def mock_timer_init(_timeout, callback, *_args, **_kwargs):
            nonlocal timer_cb
            timer_cb = callback
            return mock.MagicMock()

        mock_timer.side_effect = mock_timer_init

        _run_ffmpeg_standardization("in.mp3", str(output_wav), 0.01)
        assert timer_cb is not None

        # Trigger the kill_process callback
        timer_cb()
        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_called_once()


def test_parse_ffmpeg_progress_edge_cases():
    """Verify unknown duration yielding and exception parsing lines."""
    # Speed parse exception
    line_mock = mock.MagicMock()
    line_mock.__contains__.side_effect = lambda x: x == "speed="
    line_mock.split.side_effect = ValueError("split fail")

    process = mock.MagicMock()
    process.stdout.readline.side_effect = [
        line_mock,
        "out_time_ms=1000000\n",
        "",
    ]

    speed = utils.parse_ffmpeg_progress(process, 10.0)
    assert speed == "N/A"

    # Unknown duration yielding (periodic)
    process_unknown = mock.MagicMock()
    process_unknown.stdout.readline.side_effect = [
        "out_time_ms=1000000\n",
        "out_time_ms=2000000\n",
        "",
    ]

    yield_calls = 0

    def _yield():
        nonlocal yield_calls
        yield_calls += 1

    # Control time to trigger > 1s diff
    with mock.patch("time.time", side_effect=[0.0, 1.5, 3.0]):
        utils.parse_ffmpeg_progress(process_unknown, 0, yield_cb=_yield)

    assert yield_calls == 1


def test_cleanup_old_files_missing_directory():
    """Verify cleanup_old_files returns early when directory is missing."""
    with mock.patch("os.path.exists", return_value=False):
        # Should return without exception
        utils.cleanup_old_files("/nonexistent_dir_123")


def test_purge_temporary_assets_no_env_and_files(tmp_path):
    """Verify purge_temporary_assets without env var, and file purging behavior."""
    fake_temp = tmp_path / "fake_temp"
    fake_temp.mkdir()

    # Create matching and non-matching temp files
    target_file = fake_temp / "tmp_123.wav"
    target_file.write_text("data")

    non_target_file = fake_temp / "keep.txt"
    non_target_file.write_text("data")

    # Clear env to trigger fallback
    with (
        mock.patch.dict(os.environ, {"WHISPER_TEMP_DIR": ""}),
        mock.patch("tempfile.gettempdir", return_value=str(tmp_path)),
    ):
        # We also mock os.path.exists to return True for our fake_temp
        with (
            mock.patch("os.path.exists", return_value=True),
            mock.patch("os.listdir", return_value=["tmp_123.wav", "keep.txt"]),
            mock.patch("os.path.isfile", return_value=True),
            mock.patch("os.remove") as mock_remove,
        ):
            utils.purge_temporary_assets()
            # Verify only target_file was removed
            removed_paths = [call.args[0] for call in mock_remove.call_args_list]
            assert any("tmp_123.wav" in p for p in removed_paths)
            assert not any("keep.txt" in p for p in removed_paths)
