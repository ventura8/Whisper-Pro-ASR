"""Unit tests for audio standardization helpers exposed via request_utils."""

from unittest import mock

from modules.api.support import request_utils as routes_utils


def test_get_clean_wav_or_error_corrupt():
    """Verify get_clean_wav_or_error handles corrupted audio files."""
    file_content = b"\x00" * 1024
    mock_open_obj = mock.mock_open(read_data=file_content)

    with mock.patch("os.path.exists", return_value=True):
        with mock.patch("os.path.getsize", return_value=len(file_content)):
            with mock.patch("builtins.open", mock_open_obj):
                with mock.patch("modules.api.support.audio_standardization.model_manager"):
                    res, err = routes_utils.get_clean_wav_or_error("corrupt.wav")
                    assert res is None
                    assert "corrupt" in err[0]


def test_get_clean_wav_or_error_silent_prefix_pcm_not_corruption(tmp_path):
    """Silent-prefix raw PCM must not be treated as corruption, and must still be
    run through FFmpeg with the raw-PCM input_flags (not returned unmodified) --
    faster-whisper's decode_audio() cannot identify headerless PCM without them."""
    pcm_path = tmp_path / "silent_prefix.pcm"
    pcm_path.write_bytes(b"\x00" * 512 + b"\x01" * 512)
    input_flags = ["-f", "s16le", "-ar", "16000", "-ac", "1"]

    with (
        mock.patch("modules.api.support.audio_standardization.model_manager"),
        mock.patch("modules.core.utils.convert_to_wav", return_value="clean.wav") as mock_convert,
    ):
        res, err = routes_utils.get_clean_wav_or_error(str(pcm_path), input_flags=input_flags)
        assert res == "clean.wav"
        assert err is None
        mock_convert.assert_called_once_with(str(pcm_path), input_flags=input_flags, stream_index=None, delay_filter=None)


def test_get_clean_wav_or_error_apply_stream_alignment_probes_and_forwards_directives(tmp_path):
    """apply_stream_alignment=True must probe and forward the resulting directives into
    convert_to_wav, using the given language as the probe's target."""
    source = tmp_path / "movie.mkv"
    source.write_bytes(b"data")

    with (
        mock.patch("modules.api.support.audio_standardization.model_manager"),
        mock.patch("modules.core.utils.convert_to_wav", return_value="clean.wav") as mock_convert,
        mock.patch("modules.core.utils.get_stream_alignment_directives", return_value=(1, "adelay=45:all=1")) as mock_alignment,
    ):
        res, err = routes_utils.get_clean_wav_or_error(str(source), language="en", apply_stream_alignment=True)
        assert res == "clean.wav"
        assert err is None
        mock_alignment.assert_called_once_with(str(source), "en")
        mock_convert.assert_called_once_with(str(source), input_flags=[], stream_index=1, delay_filter="adelay=45:all=1")


def test_get_clean_wav_or_error_default_never_probes_stream_alignment(tmp_path):
    """apply_stream_alignment defaults to False -- must never probe (uploaded-audio path)."""
    source = tmp_path / "movie.mkv"
    source.write_bytes(b"data")

    with (
        mock.patch("modules.api.support.audio_standardization.model_manager"),
        mock.patch("modules.core.utils.convert_to_wav", return_value="clean.wav") as mock_convert,
        mock.patch("modules.core.utils.get_stream_alignment_directives") as mock_alignment,
    ):
        res, err = routes_utils.get_clean_wav_or_error(str(source), language="en")
        assert res == "clean.wav"
        assert err is None
        mock_alignment.assert_not_called()
        mock_convert.assert_called_once_with(str(source), input_flags=[], stream_index=None, delay_filter=None)


def test_get_clean_wav_or_error_read_exception():
    """Verify get_clean_wav_or_error falls back to transcoding on read failure."""
    with mock.patch("os.path.exists", return_value=True):
        with mock.patch("builtins.open", side_effect=PermissionError("no read")):
            with mock.patch("modules.api.support.audio_standardization.model_manager"):
                with mock.patch("modules.core.utils.convert_to_wav", return_value="clean.wav"):
                    res, err = routes_utils.get_clean_wav_or_error("test.wav")
                    assert res == "clean.wav"
                    assert err is None


def test_get_clean_wav_or_error_ffmpeg_failure():
    """FFmpeg conversion failures should return a 400 error payload."""
    with (
        mock.patch("modules.api.support.audio_standardization.model_manager"),
        mock.patch("modules.core.utils.convert_to_wav", side_effect=RuntimeError("ffmpeg down")),
    ):
        res, err = routes_utils.get_clean_wav_or_error("broken.wav")
        assert res is None
        assert err[1] == 400


def test_get_clean_wav_or_error_warns_on_truncated_output(tmp_path):
    """Large duration drop between source and WAV should emit a truncation warning."""
    source = tmp_path / "source.wav"
    clean = tmp_path / "clean.wav"
    source.write_bytes(b"data")
    clean.write_bytes(b"data")

    with (
        mock.patch("modules.api.support.audio_standardization.model_manager"),
        mock.patch("modules.core.utils.convert_to_wav", return_value=str(clean)),
        mock.patch("modules.core.utils.get_audio_duration", side_effect=[100.0, 10.0]),
        mock.patch("modules.api.support.audio_standardization.logger.warning") as warn,
    ):
        res, err = routes_utils.get_clean_wav_or_error(str(source))
        assert res == str(clean)
        assert err is None
        warn.assert_called_once()


def test_get_clean_wav_or_error_skips_truncation_warning_for_invalid_duration(tmp_path):
    """Invalid duration probes should not emit truncation warnings."""
    source = tmp_path / "source.wav"
    clean = tmp_path / "clean.wav"
    source.write_bytes(b"data")
    clean.write_bytes(b"data")

    with (
        mock.patch("modules.api.support.audio_standardization.model_manager"),
        mock.patch("modules.core.utils.convert_to_wav", return_value=str(clean)),
        mock.patch("modules.core.utils.get_audio_duration", return_value="bad"),
        mock.patch("modules.api.support.audio_standardization.logger.warning") as warn,
    ):
        res, err = routes_utils.get_clean_wav_or_error(str(source))
        assert res == str(clean)
        assert err is None
        warn.assert_not_called()
