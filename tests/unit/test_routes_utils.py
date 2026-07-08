"""Tests for modules/api/routes_utils.py."""

import io
import os
from unittest import mock

import pytest
from fastapi import UploadFile

from modules.api import routes_utils
from modules.core import config


def test_prepare_source_path_upload():
    """Verify prepare_source_path when uploading a file."""
    # Mock handle_upload to return valid paths
    with mock.patch("modules.api.routes_utils.handle_upload", return_value=("tmp", "temp", "orig")):
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
            with mock.patch("modules.api.routes_utils.uuid") as mock_uuid:
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
            with mock.patch("modules.api.routes_utils.shutil_copy_file_in_chunks"):
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


def test_get_display_name_early():
    """Verify get_display_name_early extraction logic under different scenarios."""
    # Test local path
    assert routes_utils.get_display_name_early(local_path="/path/to/my_file.mp3") == "my_file.mp3"
    assert routes_utils.get_display_name_early(local_path='"clean_path.wav"') == "clean_path.wav"

    # Test real uploaded filename
    mock_file = mock.MagicMock()
    mock_file.filename = "presentation.mp4"
    assert routes_utils.get_display_name_early(audio_file=mock_file) == "presentation.mp4"

    # Test generic uploaded filename
    mock_file.filename = "audio_file"
    assert routes_utils.get_display_name_early(audio_file=mock_file) == "audio_file"

    mock_file.filename = "file.mp3"
    assert routes_utils.get_display_name_early(audio_file=mock_file) == "file.mp3"

    mock_file.filename = "blob"
    assert routes_utils.get_display_name_early(audio_file=mock_file) == "blob"

    # Test no filename or None
    mock_file.filename = None
    assert routes_utils.get_display_name_early(audio_file=mock_file) == "Unknown Media"


def test_prepare_source_path_fallback_preserves_local_path():
    """Verify fallback paths extraction logic preserves original local path basename."""
    # local_path provided but does not exist
    # audio_file provided and handles upload
    with mock.patch("modules.api.routes_utils.resolve_local_path", return_value=None):
        with mock.patch("modules.api.routes_utils.handle_upload", return_value=("tmp", "temp", "audio_file")):
            res = routes_utils.prepare_source_path(local_path="/home/user/music/my_real_song.mp3", audio_file="dummy")
            # Should use the basename of local_path instead of the upload name "audio_file"
            assert res == ("tmp", "temp", "my_real_song.mp3")


def test_handle_upload_corrupt_null_bytes_filenotfound():
    """Verify handle_upload exception triggers on empty/corrupted file upload."""
    mock_file = mock.MagicMock()
    mock_file.filename = "test.wav"
    mock_file.file.read.side_effect = [b"\x00" * 1025, b""]

    # 1025 null bytes
    file_content = b"\x00" * 1025
    mock_open_obj = mock.mock_open(read_data=file_content)

    with mock.patch("builtins.open", mock_open_obj):
        with mock.patch("os.path.getsize", return_value=1025):
            # os.remove raises FileNotFoundError
            with mock.patch("os.remove", side_effect=FileNotFoundError()):
                with pytest.raises(ValueError, match="Input file is corrupted"):
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
            with mock.patch("modules.api.routes_utils.model_manager"):
                res, err = routes_utils.get_clean_wav_or_error("corrupt.wav")
                assert res is None
                assert "corrupt" in err[0]


def test_get_clean_wav_or_error_read_exception():
    """Verify get_clean_wav_or_error falls back to transcoding on read failure."""
    # Make reading raise an exception, should proceed to convert_to_wav
    with mock.patch("os.path.exists", return_value=True):
        with mock.patch("builtins.open", side_effect=PermissionError("no read")):
            with mock.patch("modules.api.routes_utils.model_manager"):
                with mock.patch("modules.core.utils.convert_to_wav", return_value="clean.wav"):
                    res, err = routes_utils.get_clean_wav_or_error("test.wav")
                    assert res == "clean.wav"
                    assert err is None


def test_extract_local_path():
    """Verify extract_local_path extracts local paths from different keys/sources."""
    mock_req = mock.MagicMock()
    mock_req.query_params = {}

    # Test form data / JSON body extraction
    assert routes_utils.extract_local_path(None, {"local_path": "/path/to/audio.mp3"}, mock_req) == "/path/to/audio.mp3"
    assert routes_utils.extract_local_path(None, {"video_file": "/path/to/video.mp4"}, mock_req) == "/path/to/video.mp4"
    assert routes_utils.extract_local_path(None, {"file": "/path/to/file.wav"}, mock_req) == "/path/to/file.wav"
    assert routes_utils.extract_local_path(None, {"audio_file": "/path/to/audio.wav"}, mock_req) == "/path/to/audio.wav"

    # Test query param extraction
    mock_req.query_params = {"file": "/query/file.mp3"}
    assert routes_utils.extract_local_path(None, {}, mock_req) == "/query/file.mp3"

    # Test ignore non-strings
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
    """Verify optimization local-path log is emitted once per request context."""
    test_file = tmp_path / "mapped_movie.mkv"
    test_file.write_text("media")

    original_roots = config.APPROVED_ROOTS
    config.APPROVED_ROOTS = [str(tmp_path)]
    try:
        routes_utils.utils.THREAD_CONTEXT.optimized_local_path_logged = None
        with mock.patch("modules.api.routes_utils.logger.info") as info_mock:
            first = routes_utils.resolve_local_path(str(test_file))
            second = routes_utils.resolve_local_path(str(test_file))

            assert first == os.path.realpath(str(test_file))
            assert second == os.path.realpath(str(test_file))
            optimization_calls = [c for c in info_mock.call_args_list if "Optimization: Using Local Path" in str(c)]
            assert len(optimization_calls) == 1
    finally:
        routes_utils.utils.THREAD_CONTEXT.optimized_local_path_logged = None
        config.APPROVED_ROOTS = original_roots


@pytest.mark.anyio
async def test_materialize_upload_file_valid():
    """Verify materialize_upload_file correctly writes valid uploads to disk."""
    valid_file = UploadFile(file=io.BytesIO(b"valid audio content"), filename="valid.wav")
    path, filename = await routes_utils.materialize_upload_file(valid_file)
    assert path is not None
    assert filename == "valid.wav"
    assert os.path.exists(path)
    with open(path, "rb") as f:
        assert f.read() == b"valid audio content"
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
    """Verify materialize_upload_file raises ValueError for null-filled corrupt uploads."""
    corrupt_file = UploadFile(file=io.BytesIO(b"\x00" * 2000), filename="corrupt.mp3")
    with pytest.raises(ValueError, match="Input file is corrupted"):
        await routes_utils.materialize_upload_file(corrupt_file)


@pytest.mark.anyio
async def test_materialize_upload_file_corrupt_bypassed():
    """Verify materialize_upload_file bypasses corruption check if is_raw_pcm is True."""
    corrupt_file = UploadFile(file=io.BytesIO(b"\x00" * 2000), filename="corrupt.mp3")
    path, filename = await routes_utils.materialize_upload_file(corrupt_file, is_raw_pcm=True)
    assert path is not None
    assert filename == "corrupt.mp3"
    if os.path.exists(path):
        os.remove(path)


@pytest.mark.anyio
async def test_materialize_upload_file_fallbacks():
    """Verify materialize_upload_file handles read fallbacks and OSErrors during cleanup."""
    # 1. Fallback to sync read when await upload_file.read fails
    mock_file = mock.MagicMock(spec=UploadFile)
    mock_file.filename = "test.wav"
    mock_file.read.side_effect = TypeError("Async read failed")
    mock_file.file = mock.MagicMock()
    mock_file.file.read.side_effect = [b"chunk data", b""]

    with mock.patch("modules.core.config.get_temp_dir", return_value="/tmp"):
        with mock.patch("os.path.getsize", return_value=10):
            path, name = await routes_utils.materialize_upload_file(mock_file)
            assert path is not None
            assert name == "test.wav"
            if os.path.exists(path):
                os.remove(path)

    # 2. Fallback raised when sync read also fails
    mock_file = mock.MagicMock(spec=UploadFile)
    mock_file.filename = "test.wav"
    mock_file.read.side_effect = TypeError("Async read failed")
    mock_file.file = mock.MagicMock()
    mock_file.file.seek.side_effect = OSError("Sync seek failed")
    path, name = await routes_utils.materialize_upload_file(mock_file)
    assert path is None
    assert name is None

    # 3. OSError in os.remove when corrupt file check fails
    corrupt_file = UploadFile(file=io.BytesIO(b"\x00" * 2000), filename="corrupt.mp3")
    with mock.patch("os.path.getsize", return_value=2000):
        with mock.patch("builtins.open", mock.mock_open(read_data=b"\x00" * 2000)):
            with mock.patch("os.remove", side_effect=OSError("Permission denied")):
                with pytest.raises(ValueError, match="Input file is corrupted"):
                    await routes_utils.materialize_upload_file(corrupt_file)


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
            mock.patch("modules.api.routes_utils.os.path.exists", return_value=False),
            mock.patch("modules.api.routes_utils.shutil_copy_file_in_chunks"),
        ):
            path, name = await routes_utils.materialize_upload_file(mock_file)
            assert path is None
            assert name is None

        # Temp file exists but is empty after sync fallback copy.
        with (
            mock.patch("modules.api.routes_utils.os.path.exists", return_value=True),
            mock.patch("modules.api.routes_utils.os.path.getsize", return_value=0),
            mock.patch("modules.api.routes_utils.shutil_copy_file_in_chunks"),
            mock.patch("modules.api.routes_utils.os.remove"),
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
async def test_resolve_and_materialize_upload_branches():
    """Verify resolve_and_materialize_upload branches when path or upload exists."""
    mock_req = mock.MagicMock()
    mock_req.query_params = {}

    # 1. Local path is resolved successfully and upload materialization is skipped.
    dummy_file = UploadFile(file=io.BytesIO(b"audio"), filename="test.wav")
    with (
        mock.patch("modules.api.routes_utils.resolve_local_path", return_value="/mapped/local/file.mkv"),
        mock.patch("modules.api.routes_utils.extract_uploaded_file", return_value=dummy_file),
        mock.patch("modules.api.routes_utils.materialize_upload_file") as materialize_mock,
    ):
        path, upload = await routes_utils.resolve_and_materialize_upload("/local/path", dummy_file, None, {}, mock_req)
        assert path == "/mapped/local/file.mkv"
        assert upload is None
        materialize_mock.assert_not_called()

    # 2. Local path is unavailable, uploaded file is materialized.
    dummy_file = UploadFile(file=io.BytesIO(b"audio"), filename="test.wav")
    with (
        mock.patch("modules.api.routes_utils.resolve_local_path", return_value=None),
        mock.patch("modules.api.routes_utils.extract_uploaded_file", return_value=dummy_file),
        mock.patch("modules.api.routes_utils.materialize_upload_file", return_value=("/materialized/path", "test.wav")),
    ):
        path, upload = await routes_utils.resolve_and_materialize_upload(
            "/missing/path.mkv", dummy_file, None, {}, mock_req
        )
        assert path == "/missing/path.mkv"
        assert upload == "/materialized/path"

    # 3. Raw PCM mode from encode=false with failed materialization clears upload.
    dummy_file = UploadFile(file=io.BytesIO(b"audio"), filename="test.wav")
    with (
        mock.patch("modules.api.routes_utils.resolve_local_path", return_value=None),
        mock.patch("modules.api.routes_utils.extract_uploaded_file", return_value=dummy_file),
        mock.patch("modules.api.routes_utils.materialize_upload_file", return_value=(None, None)) as mat_mock,
    ):
        mock_req.query_params = {"encode": "false"}
        path, upload = await routes_utils.resolve_and_materialize_upload(
            "/missing/path.mkv", dummy_file, None, {}, mock_req
        )
        assert path == "/missing/path.mkv"
        assert upload is None
        mat_mock.assert_called_once()
        assert routes_utils.utils.THREAD_CONTEXT.input_flags == ["-f", "s16le", "-ar", "16000", "-ac", "1"]


def test_prepare_source_path_local_missing_raises():
    """Verify local path only input raises a clear accessibility error."""
    with mock.patch("modules.api.routes_utils.resolve_local_path", return_value=None):
        with pytest.raises(ValueError, match="Path not accessible"):
            routes_utils.prepare_source_path(local_path="/not/mounted/movie.mkv", audio_file=None)


def test_cleanup_files_remove_exception_is_swallowed(tmp_path):
    """Cleanup should ignore remove failures and clear tracked state."""
    f = tmp_path / "tmp.wav"
    f.write_text("x")
    with (
        mock.patch("modules.api.routes_utils.utils.get_tracked_files", return_value={str(f)}),
        mock.patch("modules.api.routes_utils.os.remove", side_effect=PermissionError("deny")),
        mock.patch("modules.api.routes_utils.os.path.exists", return_value=True),
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
