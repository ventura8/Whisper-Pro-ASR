"""Raw PCM local-path flag tests for request_utils."""

import io
from unittest import mock

import pytest
from fastapi import UploadFile

from modules.api.support import audio_standardization, request_utils, source_resolution
from modules.core import utils


@pytest.mark.anyio
async def test_resolve_and_materialize_upload_clears_raw_pcm_flags_for_local_path():
    """Mapped media must not inherit raw-upload flags before its zero-copy return."""
    original_flags = getattr(utils.THREAD_CONTEXT, "input_flags", None)
    mock_req = mock.MagicMock()
    mock_req.query_params = {"raw_pcm": "true"}
    dummy_file = UploadFile(file=io.BytesIO(b"audio"), filename="raw.pcm")
    try:
        utils.THREAD_CONTEXT.input_flags = None
        with (
            mock.patch("modules.api.support.request_utils.resolve_local_path", return_value="/mapped/raw.pcm"),
            mock.patch("modules.api.support.request_utils.extract_uploaded_file", return_value=dummy_file),
            mock.patch("modules.api.support.request_utils.materialize_upload_file") as materialize_mock,
        ):
            path, upload = await request_utils.resolve_and_materialize_upload("/raw.pcm", dummy_file, None, {}, mock_req)
        assert path == "/mapped/raw.pcm" and upload is None
        materialize_mock.assert_not_called()
        assert utils.THREAD_CONTEXT.input_flags is None
    finally:
        utils.THREAD_CONTEXT.input_flags = original_flags


def test_prepare_source_path_clears_input_flags_and_probes_natively():
    """Setting INPUT_FLAGS_VAR before resolving local path must clear flags and probe native container."""
    captured_cmds = []

    def _capture(cmd, **_kw):
        captured_cmds.append(list(cmd))
        return "120.0"

    token = utils.INPUT_FLAGS_VAR.set(["-f", "s16le", "-ar", "16000", "-ac", "1"])
    try:
        with mock.patch("modules.api.support.source_resolution.resolve_local_path", return_value="/mapped/movie.mkv"):
            path, _, _ = source_resolution.prepare_source_path(local_path="/media/movie.mkv")
            assert path == "/mapped/movie.mkv"
            assert utils.THREAD_CONTEXT.input_flags is None

            with mock.patch("modules.core.process_exec.check_output_text", side_effect=_capture):
                duration = utils.get_audio_duration(path)
            assert duration == 120.0
            assert len(captured_cmds) == 1
            assert "-f" not in captured_cmds[0]
            assert "s16le" not in captured_cmds[0]
    finally:
        utils.INPUT_FLAGS_VAR.reset(token)


def test_standardize_audio_clears_input_flags_and_probes_natively():
    """Audio standardization clears THREAD_CONTEXT.input_flags upon success and probes natively."""
    captured_cmds = []

    def _capture(cmd, **_kw):
        captured_cmds.append(list(cmd))
        return "45.0"

    token = utils.INPUT_FLAGS_VAR.set(["-f", "s16le", "-ar", "16000", "-ac", "1"])
    flags = ["-f", "s16le", "-ar", "16000", "-ac", "1"]
    try:
        with (
            mock.patch("modules.api.support.audio_standardization._corrupt_file_error", return_value=None),
            mock.patch("modules.api.support.audio_standardization._resolve_stream_alignment", return_value=(None, None)),
            mock.patch("modules.api.support.audio_standardization._run_convert_to_wav", return_value=("/tmp/clean.wav", None)),
            mock.patch("modules.api.support.audio_standardization._warn_on_truncated_standardization"),
            mock.patch("modules.api.support.audio_standardization.model_manager.update_task_progress"),
        ):
            clean_wav, err = audio_standardization.get_clean_wav_or_error("/tmp/raw.pcm", input_flags=flags)
            assert err is None
            assert clean_wav == "/tmp/clean.wav"
            assert utils.THREAD_CONTEXT.input_flags is None

            with mock.patch("modules.core.process_exec.check_output_text", side_effect=_capture):
                duration = utils.get_audio_duration(clean_wav)
            assert duration == 45.0
            assert len(captured_cmds) == 1
            assert "-f" not in captured_cmds[0]
            assert "s16le" not in captured_cmds[0]
    finally:
        utils.INPUT_FLAGS_VAR.reset(token)
