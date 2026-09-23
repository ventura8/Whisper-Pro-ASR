"""Tests for the upload-materialization path in modules/api/support/request_utils.py.

Split from test_routes_utils.py to stay under the file size limit.
"""

import asyncio
import io
import os
from unittest import mock

import pytest
from fastapi import UploadFile

from modules.api.support import request_utils as routes_utils


@pytest.fixture(autouse=True)
def _reset_input_flags_thread_context():
    """Several tests below mutate THREAD_CONTEXT.input_flags (encode=false/raw_pcm
    handling); reset it before and after each test so mutations don't leak into
    other tests in this module (or other modules importing the same THREAD_CONTEXT)."""
    routes_utils.utils.THREAD_CONTEXT.input_flags = None
    yield
    routes_utils.utils.THREAD_CONTEXT.input_flags = None


@pytest.mark.anyio
async def test_materialize_upload_file_valid_preserves_filename():
    """Valid uploads should preserve the source filename."""
    valid_file = UploadFile(file=io.BytesIO(b"valid audio content"), filename="valid.wav")
    path, filename = await routes_utils.materialize_upload_file(valid_file)
    try:
        assert path is not None
        assert filename == "valid.wav"
    finally:
        if path and os.path.exists(path):
            os.remove(path)


@pytest.mark.anyio
async def test_materialize_upload_file_valid_writes_content():
    """Valid uploads should be written to disk with the same content."""
    valid_file = UploadFile(file=io.BytesIO(b"valid audio content"), filename="valid.wav")
    path, _ = await routes_utils.materialize_upload_file(valid_file)
    try:
        assert os.path.exists(path)
        with open(path, "rb") as file_handle:
            assert file_handle.read() == b"valid audio content"
    finally:
        if path and os.path.exists(path):
            os.remove(path)


@pytest.mark.anyio
async def test_materialize_upload_file_empty():
    """Verify materialize_upload_file raises ValueError for empty uploads."""
    empty_file = UploadFile(file=io.BytesIO(b""), filename="empty.mp3")
    with pytest.raises(ValueError, match="Remote data stream is empty"):
        await routes_utils.materialize_upload_file(empty_file)


@pytest.mark.anyio
async def test_materialize_upload_file_mp3():
    """Verify materialize_upload_file materializes non-empty audio files cleanly."""
    valid_file = UploadFile(file=io.BytesIO(b"audio content"), filename="audio.mp3")
    path, filename = await routes_utils.materialize_upload_file(valid_file)
    try:
        assert path is not None
        assert filename == "audio.mp3"
    finally:
        if path and os.path.exists(path):
            os.remove(path)


@pytest.mark.anyio
async def test_materialize_upload_file_pcm():
    """Verify materialize_upload_file works cleanly for raw PCM uploads."""
    pcm_file = UploadFile(file=io.BytesIO(b"pcm audio content"), filename="audio.pcm")
    path, filename = await routes_utils.materialize_upload_file(pcm_file)
    try:
        assert path is not None
        assert filename == "audio.pcm"
    finally:
        if path and os.path.exists(path):
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
    try:
        assert path is not None
        assert filename == "test.wav"
    finally:
        if path and os.path.exists(path):
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
        temp_root = os.path.realpath(str(tmp_path))

        def _under_temp(path: str) -> bool:
            real = os.path.realpath(path)
            return real == temp_root or real.startswith(temp_root + os.sep)

        # Temp file does not exist after sync fallback copy.
        with (
            mock.patch(
                "modules.api.support.request_utils.os.path.exists",
                side_effect=lambda path: False if _under_temp(path) else os.path.exists(path),
            ),
            mock.patch("modules.api.support.request_utils.shutil_copy_file_in_chunks"),
        ):
            path, name = await routes_utils.materialize_upload_file(mock_file)
            assert path is None
            assert name is None

        # Temp file exists but is empty after sync fallback copy.
        with (
            mock.patch(
                "modules.api.support.request_utils.os.path.exists",
                side_effect=lambda path: True if _under_temp(path) else os.path.exists(path),
            ),
            mock.patch(
                "modules.api.support.request_utils.os.path.getsize",
                side_effect=lambda path: 0 if _under_temp(path) else os.path.getsize(path),
            ),
            mock.patch("modules.api.support.request_utils.shutil_copy_file_in_chunks"),
            mock.patch(
                "modules.api.support.request_utils.os.remove",
                side_effect=lambda path: None if _under_temp(path) else os.remove(path),
            ),
        ):
            with pytest.raises(ValueError, match="Remote data stream is empty"):
                await routes_utils.materialize_upload_file(mock_file)


def test_prepare_source_path_pre_materialized_upload_uses_original_filename(tmp_path):
    """Regression: a pre-materialized upload (random `upload_<uuid>.ext` temp name) must
    resolve its display name from the tagged original filename, not the temp basename."""
    test_file = tmp_path / "upload_deadbeef.wav"
    test_file.write_text("some audio content")
    tagged_path = routes_utils.MaterializedUploadPath(str(test_file))
    tagged_path.original_filename = "clip.wav"
    res_path, res_temp, res_name = routes_utils.prepare_source_path(audio_file=tagged_path)
    assert res_path == str(test_file)
    assert res_temp == str(test_file)
    assert res_name == "clip.wav"


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
async def test_resolve_and_materialize_upload_preserves_original_filename_for_display(tmp_path):
    """Regression: materializing an upload must not lose the client's original filename.

    _build_upload_tmp_path always names the temp file `upload_<uuid>.ext` -- without
    tagging the returned path with the original filename, get_display_name_early would
    show that random temp basename in the dashboard/history instead of the file the
    client actually sent (e.g. 'clip.wav')."""
    mock_req = mock.MagicMock()
    mock_req.query_params = {}
    dummy_file = UploadFile(file=io.BytesIO(b"audio"), filename="clip.wav")
    materialized_path = str(tmp_path / "upload_deadbeef.wav")

    with (
        mock.patch("modules.api.support.request_utils.resolve_local_path", return_value=None),
        mock.patch("modules.api.support.request_utils.extract_uploaded_file", return_value=dummy_file),
        mock.patch(
            "modules.api.support.request_utils.materialize_upload_file",
            return_value=(materialized_path, "clip.wav"),
        ),
    ):
        _path, upload = await routes_utils.resolve_and_materialize_upload(None, dummy_file, None, {}, mock_req)
        assert upload == materialized_path
        assert routes_utils.get_display_name_early(None, upload) == "clip.wav"


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


@pytest.mark.anyio
async def test_write_upload_to_disk_async_closes_handle_when_cancelled_mid_write(tmp_path):
    """A cancellation during the copy must still close the handle and clear the partial file.

    The write side runs in a worker thread, so a naive implementation loses the handle when
    the awaiting task is cancelled: the file stays open and the partial upload stays on
    disk, one descriptor and one file per aborted request.
    """
    target = str(tmp_path / "cancelled.wav")
    opened = []
    real_open = open

    def _tracking_open(*args, **kwargs):
        handle = real_open(*args, **kwargs)
        opened.append(handle)
        return handle

    class _CancellingUpload:
        async def seek(self, _offset):
            return None

        async def read(self, _size):
            raise asyncio.CancelledError

    with mock.patch("builtins.open", _tracking_open):
        with pytest.raises(asyncio.CancelledError):
            await routes_utils._write_upload_to_disk_async(_CancellingUpload(), target)

    assert opened, "the handle was never opened, so this asserts nothing about closing it"
    assert all(handle.closed for handle in opened), "a cancelled upload left its handle open"
    assert not os.path.exists(target), "a cancelled upload left its partial file behind"


@pytest.mark.anyio
async def test_write_upload_to_disk_async_cleans_up_when_cancelled_before_any_chunk(tmp_path):
    """Cancellation between opening the file and the first chunk is the same contract."""
    target = str(tmp_path / "cancelled_early.wav")

    class _ImmediatelyCancellingUpload:
        async def seek(self, _offset):
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await routes_utils._write_upload_to_disk_async(_ImmediatelyCancellingUpload(), target)

    assert not os.path.exists(target)


@pytest.mark.anyio
async def test_write_upload_to_disk_async_writes_every_chunk(tmp_path):
    """The threaded copy must still reproduce the upload byte for byte."""
    target = str(tmp_path / "copied.wav")
    payload = b"chunk-one" * 2048

    upload = UploadFile(file=io.BytesIO(payload), filename="copied.wav")
    assert await routes_utils._write_upload_to_disk_async(upload, target) is True

    with open(target, "rb") as written:
        assert written.read() == payload
