"""Tests for modules/api/routes_utils.py."""

import io
import os
from contextlib import contextmanager
from unittest import mock

import pytest
from fastapi import UploadFile

from modules.api.routes.asr import _apply_prompt_and_format_flags
from modules.api.support import request_utils as routes_utils
from modules.core import config


@contextmanager
def _temporary_approved_roots(root_path):
    original_roots = config.APPROVED_ROOTS
    config.APPROVED_ROOTS = [str(root_path)]
    try:
        yield
    finally:
        config.APPROVED_ROOTS = original_roots


def _count_optimization_logs(call_args_list):
    return sum(1 for call in call_args_list if "Optimization: Using Local Path" in str(call))


def test_prepare_source_path_upload():
    """Verify prepare_source_path when uploading a file."""
    # Mock handle_upload to return valid paths
    with mock.patch("modules.api.support.request_utils.handle_upload", return_value=("tmp", "temp", "orig")):
        res = routes_utils.prepare_source_path(audio_file="dummy")
        assert res == ("tmp", "temp", "orig")


def test_handle_upload_none():
    """Verify handle_upload with None input returns empty paths."""
    assert routes_utils.handle_upload(None) == (None, None, None)


def test_handle_upload_long_extension():
    """Verify handle_upload correctly maps files with long/custom extensions."""
    mock_file = mock.MagicMock()
    mock_file.filename = "test.abcdefgh"
    mock_file.file.read.side_effect = [b"dummy data", b""]
    # Mock open and other operations
    with mock.patch("builtins.open", mock.mock_open()):
        with mock.patch("os.path.getsize", return_value=10):
            with mock.patch("modules.api.support.request_utils.uuid") as mock_uuid:
                mock_uuid.uuid4.return_value.hex = "1234"
                res = routes_utils.handle_upload(mock_file)
                # Verify that it used .tmp extension due to length
                assert "upload_1234.tmp" in res[0]


def test_handle_upload_seek_exception():
    """Verify handle_upload recovery when seek operation fails."""
    mock_file = mock.MagicMock()
    mock_file.filename = "test.wav"
    mock_file.file.seek.side_effect = Exception("Seek error")

    with mock.patch("builtins.open", mock.mock_open()):
        with mock.patch("os.path.getsize", return_value=10):
            with mock.patch("modules.api.support.request_utils.shutil_copy_file_in_chunks"):
                res = routes_utils.handle_upload(mock_file)
                assert res[2] == "test.wav"


def test_handle_upload_fallback_read():
    """Verify handle_upload when uploaded object only has read method."""
    mock_file = mock.MagicMock(spec=["read"])
    # mock_file has read but no file attribute
    mock_file.read.return_value = b"some data"

    with mock.patch("builtins.open", mock.mock_open()):
        with mock.patch("os.path.getsize", return_value=9):
            res = routes_utils.handle_upload(mock_file)
            assert res[2] == "uploaded_file"


def test_get_display_name_early_from_local_path():
    """A local path should map to its basename."""
    assert routes_utils.get_display_name_early(local_path="/path/to/my_file.mp3") == "my_file.mp3"
    assert routes_utils.get_display_name_early(local_path='"clean_path.wav"') == "clean_path.wav"


def test_get_display_name_early_from_uploaded_filename():
    """Uploaded filenames should be preserved when they are meaningful."""
    mock_file = mock.MagicMock()
    mock_file.filename = "presentation.mp4"
    assert routes_utils.get_display_name_early(audio_file=mock_file) == "presentation.mp4"


def test_get_display_name_early_generic_uploaded_filename_variants():
    """Generic uploaded names should pass through unchanged."""
    mock_file = mock.MagicMock()
    mock_file.filename = "audio_file"
    assert routes_utils.get_display_name_early(audio_file=mock_file) == "audio_file"

    mock_file.filename = "file.mp3"
    assert routes_utils.get_display_name_early(audio_file=mock_file) == "file.mp3"

    mock_file.filename = "blob"
    assert routes_utils.get_display_name_early(audio_file=mock_file) == "blob"


def test_get_display_name_early_missing_filename_returns_unknown_media():
    """Missing filenames should fall back to Unknown Media."""
    mock_file = mock.MagicMock()
    mock_file.filename = None
    assert routes_utils.get_display_name_early(audio_file=mock_file) == "Unknown Media"


def test_prepare_source_path_fallback_preserves_local_path():
    """Verify fallback paths extraction logic preserves original local path basename."""
    # local_path provided but does not exist
    # audio_file provided and handles upload
    with mock.patch("modules.api.support.request_utils.resolve_local_path", return_value=None):
        with mock.patch("modules.api.support.request_utils.handle_upload", return_value=("tmp", "temp", "audio_file")):
            res = routes_utils.prepare_source_path(local_path="/home/user/music/my_real_song.mp3", audio_file="dummy")
            # Should use the basename of local_path instead of the upload name "audio_file"
            assert res == ("tmp", "temp", "my_real_song.mp3")


def test_handle_upload_empty_stream_filenotfound(tmp_path):
    """Verify handle_upload exception triggers on 0-byte empty stream upload."""
    mock_file = mock.MagicMock()
    mock_file.filename = "test.wav"
    mock_file.file.read.side_effect = [b"", b""]

    with (
        mock.patch("modules.core.config.get_temp_dir", return_value=str(tmp_path)),
        mock.patch("builtins.open", mock.mock_open()),
        mock.patch("os.path.getsize", return_value=0),
        mock.patch("os.remove", side_effect=FileNotFoundError()),
    ):
        with pytest.raises(ValueError, match="Remote data stream is empty"):
            routes_utils.handle_upload(mock_file)


def test_handle_upload_general_exception_filenotfound():
    """Verify handle_upload cleanup paths when a write error happens."""
    mock_file = mock.MagicMock()
    mock_file.filename = "test.wav"
    mock_file.file.read.side_effect = RuntimeError("General write error")

    # os.remove raises FileNotFoundError during cleanup
    with mock.patch("os.remove", side_effect=FileNotFoundError()):
        with pytest.raises(RuntimeError, match="General write error"):
            routes_utils.handle_upload(mock_file)


def test_handle_error_filenotfound():
    """Verify error status mapping for FileNotFoundError."""
    msg, code = routes_utils.handle_error(FileNotFoundError("not found"))
    assert code == 404
    assert "not found" in msg


def test_get_clean_wav_or_error_corrupt():
    """Verify get_clean_wav_or_error handles corrupted audio files."""
    # Make get_clean_wav_or_error read null bytes
    file_content = b"\x00" * 1024
    mock_open_obj = mock.mock_open(read_data=file_content)

    with mock.patch("os.path.exists", return_value=True):
        with mock.patch("builtins.open", mock_open_obj):
            with mock.patch("modules.api.support.request_utils.model_manager"):
                res, err = routes_utils.get_clean_wav_or_error("corrupt.wav")
                assert res is None
                assert "corrupt" in err[0]


def test_get_clean_wav_or_error_read_exception():
    """Verify get_clean_wav_or_error falls back to transcoding on read failure."""
    # Make reading raise an exception, should proceed to convert_to_wav
    with mock.patch("os.path.exists", return_value=True):
        with mock.patch("builtins.open", side_effect=PermissionError("no read")):
            with mock.patch("modules.api.support.request_utils.model_manager"):
                with mock.patch("modules.core.utils.convert_to_wav", return_value="clean.wav"):
                    res, err = routes_utils.get_clean_wav_or_error("test.wav")
                    assert res == "clean.wav"
                    assert err is None


def test_extract_local_path_from_form_data():
    """Form data keys should be checked in priority order."""
    mock_req = mock.MagicMock()
    mock_req.query_params = {}

    assert routes_utils.extract_local_path(None, {"local_path": "/path/to/audio.mp3"}, mock_req) == "/path/to/audio.mp3"
    assert routes_utils.extract_local_path(None, {"video_file": "/path/to/video.mp4"}, mock_req) == "/path/to/video.mp4"
    assert routes_utils.extract_local_path(None, {"file": "/path/to/file.wav"}, mock_req) == "/path/to/file.wav"
    assert routes_utils.extract_local_path(None, {"audio_file": "/path/to/audio.wav"}, mock_req) == "/path/to/audio.wav"


def test_extract_local_path_from_query_params():
    """Query parameters should be used when form data is absent."""
    mock_req = mock.MagicMock()
    mock_req.query_params = {"file": "/query/file.mp3"}
    assert routes_utils.extract_local_path(None, {}, mock_req) == "/query/file.mp3"


def test_extract_local_path_ignores_non_strings():
    """Non-string payloads should be ignored when selecting the local path."""
    mock_req = mock.MagicMock()
    mock_req.query_params = {"file": "/query/file.mp3"}
    dummy_file = UploadFile(file=io.BytesIO(b""), filename="test.wav")
    assert routes_utils.extract_local_path(None, {"file": dummy_file}, mock_req) == "/query/file.mp3"


def test_extract_uploaded_file():
    """Verify extract_uploaded_file resolves files from different parameter names."""
    dummy_file = UploadFile(file=io.BytesIO(b""), filename="test.wav")

    # Explicit parameters
    assert routes_utils.extract_uploaded_file(dummy_file, None, {}) == dummy_file

    # Form data extraction
    assert routes_utils.extract_uploaded_file(None, None, {"video_file": dummy_file}) == dummy_file

    # Fallback to any UploadFile in items
    assert routes_utils.extract_uploaded_file(None, None, {"some_random_key": dummy_file}) == dummy_file


def test_resolve_local_path_dynamic_approved_roots(tmp_path):
    """Verify resolve_local_path with configuration of APPROVED_ROOTS environment variable."""
    # 1. Path outside approved roots returns None (graceful fallback to upload)
    outside_path = "/some/random/unapproved/path/outside_file.wav"
    assert routes_utils.resolve_local_path(outside_path) is None

    # 2. Path inside dynamic approved roots is verified successfully
    test_file = tmp_path / "test_file.wav"
    test_file.write_text("audio data")

    original_roots = config.APPROVED_ROOTS
    config.APPROVED_ROOTS = [str(tmp_path)]
    try:
        res = routes_utils.resolve_local_path(str(test_file))
        assert res == os.path.realpath(str(test_file))
    finally:
        config.APPROVED_ROOTS = original_roots


def test_resolve_local_path_logs_once_per_request(tmp_path):
    """The optimization log should appear on the first lookup."""
    test_file = tmp_path / "mapped_movie.mkv"
    test_file.write_text("media")

    with _temporary_approved_roots(tmp_path):
        routes_utils.utils.THREAD_CONTEXT.optimized_local_path_logged = None
        with mock.patch("modules.api.support.request_utils.logger.info") as info_mock:
            first = routes_utils.resolve_local_path(str(test_file))

            assert first == os.path.realpath(str(test_file))
            assert _count_optimization_logs(info_mock.call_args_list) == 1
        routes_utils.utils.THREAD_CONTEXT.optimized_local_path_logged = None


def test_resolve_local_path_suppresses_duplicate_optimization_log(tmp_path):
    """A second lookup in the same request should not log the optimization again."""
    test_file = tmp_path / "mapped_movie.mkv"
    test_file.write_text("media")

    with _temporary_approved_roots(tmp_path):
        routes_utils.utils.THREAD_CONTEXT.optimized_local_path_logged = None
        with mock.patch("modules.api.support.request_utils.logger.info") as info_mock:
            first = routes_utils.resolve_local_path(str(test_file))
            second = routes_utils.resolve_local_path(str(test_file))

            assert first == os.path.realpath(str(test_file))
            assert second == os.path.realpath(str(test_file))
            assert _count_optimization_logs(info_mock.call_args_list) == 1
        routes_utils.utils.THREAD_CONTEXT.optimized_local_path_logged = None


@pytest.mark.anyio
async def test_materialize_upload_file_valid_preserves_filename():
    """Valid uploads should preserve the source filename."""
    valid_file = UploadFile(file=io.BytesIO(b"valid audio content"), filename="valid.wav")
    path, filename = await routes_utils.materialize_upload_file(valid_file)
    assert path is not None
    assert filename == "valid.wav"


@pytest.mark.anyio
async def test_materialize_upload_file_valid_writes_content():
    """Valid uploads should be written to disk with the same content."""
    valid_file = UploadFile(file=io.BytesIO(b"valid audio content"), filename="valid.wav")
    path, _ = await routes_utils.materialize_upload_file(valid_file)
    assert os.path.exists(path)
    with open(path, "rb") as file_handle:
        assert file_handle.read() == b"valid audio content"
    if os.path.exists(path):
        os.remove(path)


@pytest.mark.anyio
async def test_materialize_upload_file_empty():
    """Verify materialize_upload_file raises ValueError for empty uploads."""
    empty_file = UploadFile(file=io.BytesIO(b""), filename="empty.mp3")
    with pytest.raises(ValueError, match="Remote data stream is empty"):
        await routes_utils.materialize_upload_file(empty_file)


@pytest.mark.anyio
async def test_materialize_upload_file_corrupt():
    """Verify materialize_upload_file materializes non-empty audio files cleanly."""
    valid_file = UploadFile(file=io.BytesIO(b"audio content"), filename="audio.mp3")
    path, filename = await routes_utils.materialize_upload_file(valid_file)
    assert path is not None
    assert filename == "audio.mp3"
    if os.path.exists(path):
        os.remove(path)


@pytest.mark.anyio
async def test_materialize_upload_file_corrupt_bypassed():
    """Verify materialize_upload_file works cleanly when is_raw_pcm is True."""
    pcm_file = UploadFile(file=io.BytesIO(b"pcm audio content"), filename="audio.pcm")
    path, filename = await routes_utils.materialize_upload_file(pcm_file, is_raw_pcm=True)
    assert path is not None
    assert filename == "audio.pcm"
    if os.path.exists(path):
        os.remove(path)


@pytest.mark.anyio
async def test_materialize_upload_file_sync_fallback_reads_chunks():
    """The sync fallback should materialize chunk data when async reads fail."""
    mock_file = mock.MagicMock(spec=UploadFile)
    mock_file.filename = "test.wav"
    mock_file.read.side_effect = TypeError("Async read failed")
    mock_file.file = mock.MagicMock()
    mock_file.file.read.side_effect = [b"chunk data", b""]

    def _fake_copy(_src, dst):
        dst.write(b"chunk data")

    with mock.patch("modules.api.support.request_utils.shutil_copy_file_in_chunks", side_effect=_fake_copy):
        path, filename = await routes_utils.materialize_upload_file(mock_file)
        assert path is not None
        assert filename == "test.wav"
        if os.path.exists(path):
            os.remove(path)


@pytest.mark.anyio
async def test_materialize_upload_file_sync_fallback_handles_exception():
    """The sync fallback should return None when sync reading fails."""
    mock_file = mock.MagicMock(spec=UploadFile)
    mock_file.filename = "test.wav"
    mock_file.read.side_effect = TypeError("Async read failed")
    mock_file.file = mock.MagicMock()
    mock_file.file.seek.side_effect = OSError("Sync seek failed")
    path, name = await routes_utils.materialize_upload_file(mock_file)
    assert path is None
    assert name is None


@pytest.mark.anyio
async def test_materialize_upload_file_empty_cleanup_error_raises():
    """Empty uploads should still raise when cleanup fails during validation."""
    empty_file = UploadFile(file=io.BytesIO(b""), filename="empty.mp3")
    with mock.patch(
        "modules.api.support.request_utils._ensure_non_empty_file",
        side_effect=ValueError("Remote data stream is empty (0 bytes received)."),
    ):
        with pytest.raises(ValueError, match="Remote data stream is empty"):
            await routes_utils.materialize_upload_file(empty_file)


@pytest.mark.anyio
async def test_materialize_upload_file_sync_fallback_edge_cases(tmp_path):
    """Cover sync fallback branches for missing/empty temp output and invalid upload types."""
    # Invalid upload type should short-circuit immediately.
    assert await routes_utils.materialize_upload_file("not-an-upload") == (None, None)

    mock_file = mock.MagicMock(spec=UploadFile)
    mock_file.filename = "test.wav"
    mock_file.read.side_effect = TypeError("Async read failed")
    mock_file.file = mock.MagicMock()
    mock_file.file.read.side_effect = [b"sync data", b""]

    with mock.patch("modules.core.config.get_temp_dir", return_value=str(tmp_path)):
        # Temp file does not exist after sync fallback copy.
        with (
            mock.patch("modules.api.support.request_utils.os.path.exists", return_value=False),
            mock.patch("modules.api.support.request_utils.shutil_copy_file_in_chunks"),
        ):
            path, name = await routes_utils.materialize_upload_file(mock_file)
            assert path is None
            assert name is None

        # Temp file exists but is empty after sync fallback copy.
        with (
            mock.patch("modules.api.support.request_utils.os.path.exists", return_value=True),
            mock.patch("modules.api.support.request_utils.os.path.getsize", return_value=0),
            mock.patch("modules.api.support.request_utils.shutil_copy_file_in_chunks"),
            mock.patch("modules.api.support.request_utils.os.remove"),
        ):
            with pytest.raises(ValueError, match="Remote data stream is empty"):
                await routes_utils.materialize_upload_file(mock_file)


def test_prepare_source_path_string_path(tmp_path):
    """Verify prepare_source_path uses string audio_file directly."""
    test_file = tmp_path / "valid.wav"
    test_file.write_text("some audio content")
    res_path, res_temp, res_name = routes_utils.prepare_source_path(audio_file=str(test_file))
    assert res_path == str(test_file)
    assert res_temp == str(test_file)
    assert res_name == "valid.wav"


@pytest.mark.anyio
async def test_resolve_and_materialize_upload_skips_materialization_for_resolved_path():
    """Resolved local paths should bypass upload materialization."""
    mock_req = mock.MagicMock()
    mock_req.query_params = {}
    dummy_file = UploadFile(file=io.BytesIO(b"audio"), filename="test.wav")

    with (
        mock.patch("modules.api.support.request_utils.resolve_local_path", return_value="/mapped/local/file.mkv"),
        mock.patch("modules.api.support.request_utils.extract_uploaded_file", return_value=dummy_file),
        mock.patch("modules.api.support.request_utils.materialize_upload_file") as materialize_mock,
    ):
        path, upload = await routes_utils.resolve_and_materialize_upload("/local/path", dummy_file, None, {}, mock_req)
        assert path == "/mapped/local/file.mkv"
        assert upload is None
        materialize_mock.assert_not_called()


@pytest.mark.anyio
async def test_resolve_and_materialize_upload_materializes_when_needed():
    """Missing local paths should materialize uploaded files."""
    mock_req = mock.MagicMock()
    mock_req.query_params = {}
    dummy_file = UploadFile(file=io.BytesIO(b"audio"), filename="test.wav")

    with (
        mock.patch("modules.api.support.request_utils.resolve_local_path", return_value=None),
        mock.patch("modules.api.support.request_utils.extract_uploaded_file", return_value=dummy_file),
        mock.patch("modules.api.support.request_utils.materialize_upload_file", return_value=("/materialized/path", "test.wav")),
    ):
        path, upload = await routes_utils.resolve_and_materialize_upload("/missing/path.mkv", dummy_file, None, {}, mock_req)
        assert path == "/missing/path.mkv"
        assert upload == "/materialized/path"


@pytest.mark.anyio
async def test_resolve_and_materialize_upload_sets_raw_pcm_flags_when_raw_pcm_true():
    """raw_pcm=true should force raw PCM input flags if materialization fails."""
    mock_req = mock.MagicMock()
    mock_req.query_params = {"raw_pcm": "true"}
    dummy_file = UploadFile(file=io.BytesIO(b"audio"), filename="test.wav")

    with (
        mock.patch("modules.api.support.request_utils.resolve_local_path", return_value=None),
        mock.patch("modules.api.support.request_utils.extract_uploaded_file", return_value=dummy_file),
        mock.patch("modules.api.support.request_utils.materialize_upload_file", return_value=(None, None)) as mat_mock,
    ):
        path, upload = await routes_utils.resolve_and_materialize_upload("/missing/path.mkv", dummy_file, None, {}, mock_req)
        assert path == "/missing/path.mkv"
        assert upload is None
        mat_mock.assert_called_once()
        assert routes_utils.utils.THREAD_CONTEXT.input_flags == ["-f", "s16le", "-ar", "16000", "-ac", "1"]


@pytest.mark.anyio
async def test_resolve_and_materialize_upload_sets_raw_pcm_flags_when_encode_false():
    """encode=false (from Bazarr) sets raw s16le PCM input flags."""
    routes_utils.utils.THREAD_CONTEXT.input_flags = None
    mock_req = mock.MagicMock()
    mock_req.query_params = {"encode": "false"}
    dummy_file = UploadFile(file=io.BytesIO(b"audio"), filename="test.wav")

    with (
        mock.patch("modules.api.support.request_utils.resolve_local_path", return_value=None),
        mock.patch("modules.api.support.request_utils.extract_uploaded_file", return_value=dummy_file),
        mock.patch("modules.api.support.request_utils.materialize_upload_file", return_value=(None, None)) as mat_mock,
    ):
        path, upload = await routes_utils.resolve_and_materialize_upload("/missing/path.mkv", dummy_file, None, {}, mock_req)
        assert path == "/missing/path.mkv"
        assert upload is None
        mat_mock.assert_called_once()
        assert routes_utils.utils.THREAD_CONTEXT.input_flags == ["-f", "s16le", "-ar", "16000", "-ac", "1"]


def test_prepare_source_path_local_missing_raises():
    """Verify local path only input raises a clear accessibility error."""
    with mock.patch("modules.api.support.request_utils.resolve_local_path", return_value=None):
        with pytest.raises(ValueError, match="Path not accessible"):
            routes_utils.prepare_source_path(local_path="/not/mounted/movie.mkv", audio_file=None)


def test_cleanup_files_remove_exception_is_swallowed(tmp_path):
    """Cleanup should ignore remove failures and clear tracked state."""
    f = tmp_path / "tmp.wav"
    f.write_text("x")
    with (
        mock.patch("modules.api.support.request_utils.utils.get_tracked_files", return_value={str(f)}),
        mock.patch("modules.api.support.request_utils.os.remove", side_effect=PermissionError("deny")),
        mock.patch("modules.api.support.request_utils.os.path.exists", return_value=True),
    ):
        routes_utils.cleanup_files(str(f))


@pytest.mark.anyio
async def test_parse_form_data_json_and_form_exceptions():
    """Verify parse_form_data handles JSON and form failures gracefully."""
    req = mock.MagicMock()
    req.headers = {"content-type": "application/json"}
    req.json = mock.AsyncMock(return_value={"local_path": "/x"})
    assert await routes_utils.parse_form_data(req) == {"local_path": "/x"}

    req.json = mock.AsyncMock(side_effect=RuntimeError("bad json"))
    assert await routes_utils.parse_form_data(req) == {}

    req.headers = {"content-type": "multipart/form-data"}
    req.form = mock.AsyncMock(return_value={"file_path": "/y"})
    assert await routes_utils.parse_form_data(req) == {"file_path": "/y"}

    req.form = mock.AsyncMock(side_effect=RuntimeError("bad form"))
    assert await routes_utils.parse_form_data(req) == {}


def test_extract_uploaded_file_rejects_non_upload_values():
    """Verify invalid non-upload values are rejected before fallback scan."""
    dummy = UploadFile(file=io.BytesIO(b""), filename="a.wav")
    assert routes_utils.extract_uploaded_file("bad", None, {}) is None
    assert routes_utils.extract_uploaded_file(None, None, {"audio_file": "bad", "x": dummy}) == dummy


# --- ASR route: clean_audio fallback chain ---


def _call_apply_prompt_flags(query_params: dict, form_data: dict) -> dict:
    """Helper: invoke _apply_prompt_and_format_flags and return the populated params dict."""
    params: dict = {}
    _apply_prompt_and_format_flags(params, query_params, form_data)
    return params


def test_clean_audio_precedence_over_vocal_separation():
    """clean_audio takes precedence over vocal_separation and enable_vocal_separation."""
    params = _call_apply_prompt_flags(
        {"clean_audio": "true", "vocal_separation": "false", "enable_vocal_separation": "false"},
        {},
    )
    assert params["clean_audio"] is True


def test_vocal_separation_fallback_when_clean_audio_absent():
    """vocal_separation is used when clean_audio is not provided."""
    params = _call_apply_prompt_flags(
        {"vocal_separation": "true", "enable_vocal_separation": "false"},
        {},
    )
    assert params["clean_audio"] is True


def test_enable_vocal_separation_fallback_when_both_absent():
    """enable_vocal_separation is used when both clean_audio and vocal_separation are absent."""
    params = _call_apply_prompt_flags(
        {"enable_vocal_separation": "true"},
        {},
    )
    assert params["clean_audio"] is True


def test_clean_audio_is_none_when_no_param_provided():
    """params["clean_audio"] is None when none of the three params are present."""
    params = _call_apply_prompt_flags({}, {})
    assert params["clean_audio"] is None
