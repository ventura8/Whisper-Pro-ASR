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
    with mock.patch("modules.api.support.source_resolution.handle_upload", return_value=("tmp", "temp", "orig")):
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
            with mock.patch("modules.api.support.source_resolution.uuid") as mock_uuid:
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
            with mock.patch("modules.api.support.source_resolution.shutil_copy_file_in_chunks"):
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


def test_get_display_name_early_normalizes_windows_path_separators():
    """Regression: Bazarr can run on Windows and send a Windows-style local_path
    (e.g. C:\\media\\episode.avi) even against a Linux server -- os.path.basename
    alone only splits on '/', so without normalizing '\\\\' first the dashboard
    would show the full Windows path instead of just the filename."""
    assert routes_utils.get_display_name_early(local_path=r"C:\media\episode.avi") == "episode.avi"
    assert routes_utils.get_display_name_early(video_file=r"C:\media\episode.avi") == "episode.avi"


def test_get_display_name_early_from_uploaded_filename():
    """Uploaded filenames should be preserved when they are meaningful."""
    mock_file = mock.MagicMock()
    mock_file.filename = "presentation.mp4"
    assert routes_utils.get_display_name_early(audio_file=mock_file) == "presentation.mp4"


def test_get_display_name_early_generic_uploaded_filename_variants():
    """Generic uploaded names should not become dashboard titles."""
    mock_file = mock.MagicMock()
    mock_file.filename = "audio_file"
    assert routes_utils.get_display_name_early(audio_file=mock_file) == "Unknown Media"

    mock_file.filename = "file.mp3"
    assert routes_utils.get_display_name_early(audio_file=mock_file) == "file.mp3"

    mock_file.filename = "blob"
    assert routes_utils.get_display_name_early(audio_file=mock_file) == "Unknown Media"


def test_get_display_name_early_prefers_video_file_over_generic_upload_name(tmp_path):
    """Regression: real Bazarr uploads a generic-named file (its own field name echoed
    back as filename='audio_file') and separately sends `video_file` as caller metadata.
    Once the upload is materialized to a tagged temp path, get_display_name_early must
    prefer video_file over showing the literal generic upload name."""
    tagged_path = routes_utils.MaterializedUploadPath(str(tmp_path / "upload_deadbeef.raw"))
    tagged_path.original_filename = "audio_file"
    video_file = "/tv/SpongeBob SquarePants/Season 4/SpongeBob SquarePants - S04E24 - Bummer Vacation SDTV.avi"
    assert (
        routes_utils.get_display_name_early(audio_file=tagged_path, video_file=video_file)
        == "SpongeBob SquarePants - S04E24 - Bummer Vacation SDTV.avi"
    )


def test_get_display_name_early_generic_upload_name_without_video_file_falls_back(tmp_path):
    """Without a usable video_file, a generic materialized-upload name must still fall
    back to Unknown Media rather than leaking the literal generic name."""
    tagged_path = routes_utils.MaterializedUploadPath(str(tmp_path / "upload_deadbeef.raw"))
    tagged_path.original_filename = "audio_file"
    assert routes_utils.get_display_name_early(audio_file=tagged_path) == "Unknown Media"


def test_get_display_name_early_bazarr_style_path_with_parens():
    """Bazarr/Sonarr local paths with spaces and parentheses should resolve to basename."""
    path = "/tv/Doc - In Your Hands/Season 3/Doc (IT) - S03E01 - Awakenings WEBDL-1080p.mkv"
    assert routes_utils.get_display_name_early(local_path=path) == "Doc (IT) - S03E01 - Awakenings WEBDL-1080p.mkv"


def test_get_display_name_early_missing_filename_returns_unknown_media():
    """Missing filenames should fall back to Unknown Media."""
    mock_file = mock.MagicMock()
    mock_file.filename = None
    assert routes_utils.get_display_name_early(audio_file=mock_file) == "Unknown Media"


def test_prepare_source_path_fallback_preserves_local_path():
    """Verify fallback paths extraction logic preserves original local path basename."""
    # local_path provided but does not exist
    # audio_file provided and handles upload
    with mock.patch("modules.api.support.source_resolution.resolve_local_path", return_value=None):
        with mock.patch("modules.api.support.source_resolution.handle_upload", return_value=("tmp", "temp", "audio_file")):
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


def test_extract_local_path_from_form_data():
    """Form data keys should be checked in priority order."""
    mock_req = mock.MagicMock()
    mock_req.query_params = {}

    assert routes_utils.extract_local_path(None, {"local_path": "/path/to/audio.mp3"}, mock_req) == "/path/to/audio.mp3"
    assert routes_utils.extract_local_path(None, {"file": "/path/to/file.wav"}, mock_req) == "/path/to/file.wav"
    assert routes_utils.extract_local_path(None, {"audio_file": "/path/to/audio.wav"}, mock_req) == "/path/to/audio.wav"


def test_extract_local_path_ignores_video_file():
    """video_file is Bazarr caller metadata (logging/display only, see whisperai.py's
    pass_video_name option) and must never be resolved as a local filesystem path."""
    mock_req = mock.MagicMock()
    mock_req.query_params = {}

    assert routes_utils.extract_local_path(None, {"video_file": "/path/to/video.mp4"}, mock_req) is None
    mock_req.query_params = {"video_file": "/path/to/video.mp4"}
    assert routes_utils.extract_local_path(None, {}, mock_req) is None


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
        with mock.patch("modules.api.support.source_resolution.logger.info") as info_mock:
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
        with mock.patch("modules.api.support.source_resolution.logger.info") as info_mock:
            first = routes_utils.resolve_local_path(str(test_file))
            second = routes_utils.resolve_local_path(str(test_file))

            assert first == os.path.realpath(str(test_file))
            assert second == os.path.realpath(str(test_file))
            assert _count_optimization_logs(info_mock.call_args_list) == 1
        routes_utils.utils.THREAD_CONTEXT.optimized_local_path_logged = None


def test_prepare_source_path_local_missing_raises():
    """Verify local path only input raises a clear accessibility error."""
    with mock.patch("modules.api.support.source_resolution.resolve_local_path", return_value=None):
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


def test_clean_audio_false_precedence_over_vocal_separation_true():
    """Explicit clean_audio=false overrides vocal_separation=true fallback flags."""
    params = _call_apply_prompt_flags(
        {"clean_audio": "false", "vocal_separation": "true", "enable_vocal_separation": "true"},
        {},
    )
    assert params["clean_audio"] is False


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


def test_extract_ext_extensionless_original_with_local_path():
    """_extract_ext uses local_path extension when original_filename is extensionless."""
    extract_ext = routes_utils.extract_ext
    assert extract_ext("audio_file_no_ext", "/mnt/media/local_audio.mp3") == ".mp3"
    assert extract_ext("", "/mnt/media/local_audio.flac") == ".flac"
    assert extract_ext("audio.wav", "/mnt/media/local_audio.mp3") == ".wav" and extract_ext("", "") == ".tmp"


def test_extract_ext_input_flags_and_candidates():
    """_extract_ext honors THREAD_CONTEXT.input_flags and validates candidate extensions."""
    extract_ext = routes_utils.extract_ext

    # Active THREAD_CONTEXT.input_flags forces .raw regardless of original filename extension
    with mock.patch.object(routes_utils.utils.THREAD_CONTEXT, "input_flags", ["-f", "s16le"]):
        assert extract_ext("audio.wav", "/mnt/media/file.mp3") == ".raw"
        assert extract_ext("recording.flac", None) == ".raw"

    # Valid candidate extensions
    assert extract_ext("audio.m4a", None) == ".m4a"
    assert extract_ext("audio.opus", None) == ".opus"
    assert extract_ext('"audio.mp3"', None) == ".mp3"

    # Missing / empty extensions fall back to local_path or .tmp
    assert extract_ext("audio_no_ext", None) == ".tmp"
    assert extract_ext("", None) == ".tmp"
    assert extract_ext(None, None) == ".tmp"

    # Overlength candidate extension (> 6 chars) rejected and falls back
    assert extract_ext("file.toolongextension", None) == ".tmp"
    assert extract_ext("file.toolongextension", "file.wav") == ".wav"
