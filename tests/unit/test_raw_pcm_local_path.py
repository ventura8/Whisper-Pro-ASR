"""Raw PCM local-path flag tests for request_utils."""

import io
from unittest import mock

import pytest
from fastapi import UploadFile

from modules.api.support import request_utils as routes_utils


@pytest.mark.anyio
async def test_resolve_and_materialize_upload_sets_raw_pcm_flags_for_local_path():
    """raw_pcm=true must set input flags before the zero-copy local-path return."""
    original_flags = getattr(routes_utils.utils.THREAD_CONTEXT, "input_flags", None)
    mock_req = mock.MagicMock()
    mock_req.query_params = {"raw_pcm": "true"}
    dummy_file = UploadFile(file=io.BytesIO(b"audio"), filename="raw.pcm")
    try:
        routes_utils.utils.THREAD_CONTEXT.input_flags = None
        with (
            mock.patch("modules.api.support.request_utils.resolve_local_path", return_value="/mapped/raw.pcm"),
            mock.patch("modules.api.support.request_utils.extract_uploaded_file", return_value=dummy_file),
            mock.patch("modules.api.support.request_utils.materialize_upload_file") as materialize_mock,
        ):
            path, upload = await routes_utils.resolve_and_materialize_upload("/raw.pcm", dummy_file, None, {}, mock_req)
        assert path == "/mapped/raw.pcm" and upload is None
        materialize_mock.assert_not_called()
        assert routes_utils.utils.THREAD_CONTEXT.input_flags == ["-f", "s16le", "-ar", "16000", "-ac", "1"]
    finally:
        routes_utils.utils.THREAD_CONTEXT.input_flags = original_flags
