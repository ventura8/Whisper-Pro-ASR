"""Comprehensive coverage for utility functions."""

import importlib
import sys
import time
from unittest import mock

import pytest

from modules.core import subtitles, utils


def _get_stage_updates(mock_update_stage):
    return [call.args[1] for call in mock_update_stage.call_args_list if len(call.args) >= 2]


def test_convert_to_wav_success():
    """Test successful conversion to WAV."""
    with mock.patch("modules.core.utils._run_ffmpeg_standardization"):
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
    with mock.patch("modules.core.utils._run_ffmpeg_standardization", side_effect=RuntimeError("ffmpeg failed")):
        with mock.patch("os.path.exists", return_value=True):
            with mock.patch("os.path.getsize", return_value=123):
                with mock.patch("os.remove") as mock_remove:
                    with mock.patch("modules.core.config.get_temp_dir", return_value="/tmp"):
                        res = utils.convert_to_wav("input.mp3")
                        assert res is None
                        mock_remove.assert_called()


def test_get_audio_duration_success():
    """Test successful duration retrieval."""
    with mock.patch("modules.core.process_exec.check_output_text", return_value="123.45\n"):
        result = utils.get_audio_duration("test.wav")
        assert result == 123.45


def test_get_audio_duration_error():
    """Test duration retrieval error fallback."""
    with mock.patch("modules.core.process_exec.check_output_text", side_effect=Exception("fail")):
        assert utils.get_audio_duration("test.wav") == 0.0


def test_format_duration_variants():
    """Test duration formatting for different values."""
    assert utils.format_duration(0) == "00:00:00"
    assert utils.format_duration(3661) == "01:01:01"
    assert utils.format_duration(59) == "00:00:59"


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


def test_resolve_segment_timestamps_coalesces_end_to_default_end():
    """Segment end timestamp should coalesce to caller-provided default_end when missing/None."""
    resolve_func = getattr(subtitles, "_resolve_segment_timestamps")
    start_ts, end_ts = resolve_func({"start": 1.25, "end": None}, default_end=7.5)
    assert start_ts == 1.25
    assert end_ts == 7.5

    start_ts_tuple, end_ts_tuple = resolve_func({"timestamp": (2.0, None)}, default_end=9.0)
    assert start_ts_tuple == 2.0
    assert end_ts_tuple == 9.0


def test_get_system_telemetry():
    """Test gathering system telemetry via psutil mocks."""
    with mock.patch("modules.core.utils.psutil.cpu_percent", return_value=10.0):
        with mock.patch("modules.core.utils.psutil.virtual_memory") as mock_mem:
            mock_mem.return_value.percent = 50.0
            mock_mem.return_value.used = 8 * (1024**3)
            mock_mem.return_value.total = 16 * (1024**3)
            with mock.patch("modules.core.utils.psutil.cpu_count", return_value=4):
                with mock.patch("modules.core.utils._PROCESS_OBJ") as mock_proc:
                    mock_proc.cpu_percent.return_value = 5.0
                    mock_proc.memory_info.return_value.rss = 1024 * 1024 * 1024  # 1GB

                    telemetry = utils.get_system_telemetry()
                    assert telemetry["cpu_percent"] == 10.0
                    assert telemetry["memory_percent"] == 50.0
                    # Normalized by cpu_count=4 => 5.0/4 = 1.25 -> rounded to 1.2
                    assert telemetry["app_cpu_percent"] == 1.2
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
        with mock.patch("modules.core.utils.os.remove", side_effect=OSError("Cleanup Fail")):
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
        mock_torch.cuda.device_count.return_value = 1
        utils.clear_gpu_cache()
        mock_torch.cuda.empty_cache.assert_called_once()


def test_clear_gpu_cache_clears_all_visible_cuda_devices():
    """Multi-CUDA hosts should clear allocator caches on every visible device."""
    with mock.patch("modules.core.utils.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 3

        device_ctx = mock.MagicMock()
        mock_torch.cuda.device.return_value = device_ctx

        utils.clear_gpu_cache()

        assert mock_torch.cuda.device.call_count == 3
        assert device_ctx.__enter__.call_count == 3
        assert mock_torch.cuda.empty_cache.call_count == 3


def test_wrap_text_limits_line_length():
    """Verify wrap_text enforces the requested maximum width."""
    long_text = "This is a very long text that we want to wrap to a maximum width of characters."
    wrapped = utils.wrap_text(long_text, max_line_width=20)
    for line in wrapped.split("\n"):
        assert len(line) <= 20


def test_wrap_text_limits_line_count():
    """Verify wrap_text caps the number of lines."""
    long_text = "This is a very long text that we want to wrap to a maximum width of characters."
    limited = utils.wrap_text(long_text, max_line_width=20, max_line_count=2)
    assert len(limited.split("\n")) == 2


def test_subtitle_formatters_wrap_text():
    """Verify SRT and VTT formatters apply the same wrapping rules."""
    long_text = "This is a very long text that we want to wrap to a maximum width of characters."
    result = {"segments": [{"start": 0.0, "end": 5.0, "text": long_text}]}

    srt_out = utils.generate_srt(result, max_line_width=20, max_line_count=2)
    assert "wrap to a maximum" not in srt_out
    assert "This is a very long\ntext that we want to" in srt_out

    vtt_out = utils.generate_vtt(result, max_line_width=20, max_line_count=2)
    assert "wrap to a maximum" not in vtt_out
    assert "This is a very long\ntext that we want to" in vtt_out


def test_generate_srt_highlight_words():
    """Verify SRT highlighting splits words into separate cues."""
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

    with mock.patch("modules.core.config.SUBTITLE_PROMO_ENABLED", False):
        srt = utils.generate_srt(res, highlight_words=True)
    assert "1\n00:00:00,500 --> 00:00:01,000" in srt
    assert '<font color="#E0E0E0">Hello</font> world' in srt
    assert "2\n00:00:01,000 --> 00:00:02,000" in srt
    assert 'Hello <font color="#E0E0E0">world</font>' in srt


def test_generate_vtt_highlight_words():
    """Verify VTT highlighting emits karaoke timestamps."""
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

    with mock.patch("modules.core.config.SUBTITLE_PROMO_ENABLED", False):
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

    with mock.patch("modules.core.config.SUBTITLE_PROMO_ENABLED", False):
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

    with mock.patch("modules.core.config.SUBTITLE_PROMO_ENABLED", False):
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
    stage_updates = _get_stage_updates(mock_update_stage)
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


def test_context_var_proxy_missing_source_path_raises():
    """Accessing source_path before it is set should raise AttributeError."""
    with pytest.raises(AttributeError):
        _ = utils.THREAD_CONTEXT.source_path


def test_context_var_proxy_tracked_files_round_trip():
    """Tracked files should support set and delete operations."""
    utils.THREAD_CONTEXT.tracked_files = ["/tmp/fake.wav"]
    assert utils.THREAD_CONTEXT.tracked_files == ["/tmp/fake.wav"]

    del utils.THREAD_CONTEXT.tracked_files
    assert not utils.THREAD_CONTEXT.tracked_files


def test_context_var_proxy_source_path_round_trip():
    """Source path should support set and delete operations."""
    utils.THREAD_CONTEXT.source_path = "/tmp/fake.wav"
    assert utils.THREAD_CONTEXT.source_path == "/tmp/fake.wav"

    del utils.THREAD_CONTEXT.source_path
    with pytest.raises(AttributeError):
        _ = utils.THREAD_CONTEXT.source_path


def test_context_var_proxy_filename_delete_raises():
    """Deleting filename should reset the attribute to missing."""
    utils.THREAD_CONTEXT.filename = "file.mp3"
    del utils.THREAD_CONTEXT.filename
    with pytest.raises(AttributeError):
        _ = utils.THREAD_CONTEXT.filename


def test_context_var_proxy_dynamic_attribute_delete():
    """Dynamic attributes should be removed from the proxy dictionary."""
    utils.THREAD_CONTEXT.custom_val = "custom"
    assert utils.THREAD_CONTEXT.custom_val == "custom"
    del utils.THREAD_CONTEXT.custom_val
    with pytest.raises(AttributeError):
        _ = utils.THREAD_CONTEXT.custom_val


def test_get_tracked_files_falls_back_after_attribute_error():
    """get_tracked_files should recover when tracked_files access initially fails."""
    original_getattr = utils.ContextVarProxy.__getattr__
    raise_err = True

    def mock_getattr(self, name):
        nonlocal raise_err
        if name == "tracked_files" and raise_err:
            raise_err = False
            raise AttributeError("mocked")
        return original_getattr(self, name)

    with mock.patch.object(utils.ContextVarProxy, "__getattr__", mock_getattr):
        del utils.THREAD_CONTEXT.tracked_files
        files = utils.get_tracked_files()
        assert not files


def test_convert_base_uses_loudnorm_filter_when_configured():
    """The loudnorm filter configuration should survive module reloads."""
    with mock.patch("modules.core.config.FFMPEG_FILTER", "loudnorm"):
        importlib.reload(utils)
        assert utils.STANDARD_NORMALIZATION_FILTERS == "loudnorm=I=-16:TP=-1.5:LRA=11"
    importlib.reload(utils)


def test_prepare_for_uvr_reuses_conversion_output(tmp_path):
    """prepare_for_uvr should return the converted output when conversion succeeds."""
    test_file = tmp_path / "hq_test.wav"
    test_file.write_text("audio")
    with (
        mock.patch("modules.core.utils._convert_base", return_value="converted.wav") as mock_conv,
        mock.patch("modules.core.utils.track_file", return_value="converted.wav"),
    ):
        assert utils.prepare_for_uvr(str(test_file)) == "converted.wav"
        mock_conv.assert_called_once()


def test_prepare_for_uvr_converts_non_wav_inputs(tmp_path):
    """prepare_for_uvr should invoke conversion for non-WAV inputs."""
    test_mp3 = tmp_path / "hq_test.mp3"
    test_mp3.write_text("audio")

    with (
        mock.patch("modules.core.utils._convert_base", return_value="converted.wav"),
        mock.patch("modules.core.utils.track_file", return_value="converted.wav"),
    ):
        assert utils.prepare_for_uvr(str(test_mp3)) == "converted.wav"


def test_convert_base_yield_callback_runs_on_failure(tmp_path):
    """Conversion failures should trigger the cooperative yield callback."""
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
        res = utils.__dict__["_convert_base"](
            str(test_err_file),
            utils.STANDARD_AUDIO_FLAGS,
            16000,
            1,
            yield_cb=_yield,
        )
        assert res is None
        assert yield_calls == 1


def test_run_ffmpeg_standardization_edge_cases(tmp_path):
    """Verify hwaccel command injection and timeout handling."""
    output_wav = tmp_path / "out.wav"

    # default flags and hwaccel
    _run_ffmpeg_standardization = utils.__dict__["_run_ffmpeg_standardization"]
    with (
        mock.patch("modules.core.config.FFMPEG_HWACCEL", "cuvid"),
        mock.patch("modules.core.config.FFMPEG_THREADS", 2),
        mock.patch("modules.core.process_exec.run_stream") as mock_run_stream,
    ):
        _run_ffmpeg_standardization("in.mp3", str(output_wav), 10.0, flags=None)
        command = mock_run_stream.call_args.args[0]
        assert "-hwaccel" in command
        assert "cuvid" in command

    # timeout logic trigger
    with (
        mock.patch("modules.core.config.FFMPEG_HWACCEL", "none"),
        mock.patch(
            "modules.core.process_exec.run_stream",
            side_effect=utils.process_exec.CommandTimeoutError("timeout"),
        ),
    ):
        with pytest.raises(RuntimeError, match="timed out"):
            _run_ffmpeg_standardization("in.mp3", str(output_wav), 0.01)


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
    """Verify purge_temporary_assets uses config temp dir and purges only matching files."""
    fake_temp = tmp_path / "fake_temp"
    fake_temp.mkdir()

    # Create matching and non-matching temp files
    target_file = fake_temp / "tmp_123.wav"
    target_file.write_text("data")

    non_target_file = fake_temp / "keep.txt"
    non_target_file.write_text("data")

    with mock.patch("modules.core.utils_helpers.config.get_temp_dir", return_value=str(fake_temp)):
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
