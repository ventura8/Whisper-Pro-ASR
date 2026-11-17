"""E2E: Bazarr works both with and without media volume mounts.

Media mounts (/tv, /movies) are an optional zero-copy optimisation. Real Bazarr
always uploads pre-encoded raw PCM (encode=false) and may also send video_file as
caller metadata.

Two contracts must both hold:

1. Mounts absent / video_file unresolvable -- or TEMP_DIR present but not writable
   (Portainer stack missing tmpfs mode=1777) -- the upload path must still succeed.
   The NUC production outage of 2026-11-17 was exactly that: unwritable /tmp/whisper
   + no media mounts → every job "No audio source provided".
2. Mounts present / video_file readable under approved roots -- the mapped media
   path wins, uploads are not materialized, raw-PCM input flags are cleared, and
   stream alignment runs against the container (not the upload).
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from unittest import mock

import pytest

from modules.api.support import request_utils as routes_utils
from tests.conftest import FlaskCompatibleClient

_RAW_PCM = b"\x00\x01" * 1000
_DEFAULT_ASR_RESULT = {
    "text": "Hello from mapped media",
    "segments": [{"timestamp": (0.0, 1.0), "text": "Hello from mapped media"}],
}


def _bazarr_detect_post(client: FlaskCompatibleClient, *, video_file: str | None = "/tv/missing.mkv"):
    query = "/detect-language?encode=false"
    if video_file is not None:
        query = f"{query}&video_file={video_file}"
    return client.post(
        query,
        data={"audio_file": (io.BytesIO(_RAW_PCM), "audio.raw")},
        content_type="multipart/form-data",
    )


def _bazarr_asr_post(client: FlaskCompatibleClient, *, video_file: str | None = "/tv/missing.mkv"):
    query = "/asr?task=transcribe&language=en&output=srt&encode=false"
    if video_file is not None:
        query = f"{query}&video_file={video_file}"
    return client.post(
        query,
        data={"audio_file": (io.BytesIO(_RAW_PCM), "audio.raw")},
        content_type="multipart/form-data",
    )


@pytest.mark.parametrize("endpoint_helper", [_bazarr_detect_post, _bazarr_asr_post], ids=["detect-language", "asr"])
def test_bazarr_upload_succeeds_without_media_mounts(bazarr_client: FlaskCompatibleClient, bazarr_wav: str, endpoint_helper):
    """Unresolvable video_file must not block the uploaded audio_file."""
    with mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav):
        with mock.patch("modules.core.utils.get_audio_duration", return_value=90.0):
            resp = endpoint_helper(bazarr_client, video_file="/tv/not-mounted/show.mkv")
    assert resp.status_code == 200, resp.data
    if endpoint_helper is _bazarr_detect_post:
        body = json.loads(resp.data)
        assert body["language_code"] == "en"


def test_bazarr_detect_upload_survives_unwritable_temp_dir(bazarr_client: FlaskCompatibleClient, bazarr_wav: str, tmp_path: Path):
    """When TEMP_DIR is unwritable, materialization must use the persistent fallback.

    This is the NUC failure mode: /tmp/whisper existed as root:root 755, disk_usage
    succeeded, and every multipart upload returned 400 No audio source provided.
    """
    unwritable = tmp_path / "root_owned_whisper"
    unwritable.mkdir()
    persistent = tmp_path / "persistent_temp"
    persistent.mkdir()
    materialized: list[str] = []

    def _writable(path: str) -> bool:
        return os.path.abspath(path) == os.path.abspath(str(persistent))

    original_materialize = __import__("modules.api.support.request_utils", fromlist=["materialize_upload_file"]).materialize_upload_file

    async def _capture_materialize(upload_file, local_path=None):
        path, name = await original_materialize(upload_file, local_path=local_path)
        if path:
            materialized.append(path)
        return path, name

    ample = mock.MagicMock(free=10 * 1024 * 1024 * 1024)
    with (
        mock.patch("modules.core.config.TEMP_DIR", str(unwritable)),
        mock.patch("modules.core.config.PERSISTENT_TEMP_DIR", str(persistent)),
        mock.patch("modules.core.mount_helpers.is_path_writable", side_effect=_writable) as mock_writable,
        mock.patch("modules.core.mount_helpers.shutil.disk_usage", return_value=ample),
        mock.patch(
            "modules.api.support.request_utils.materialize_upload_file",
            side_effect=_capture_materialize,
        ),
        mock.patch("modules.core.utils.get_audio_duration", return_value=90.0),
        mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav),
    ):
        resp = _bazarr_detect_post(bazarr_client, video_file="/movies/unmounted.mkv")

    assert resp.status_code == 200, resp.data
    body = json.loads(resp.data)
    assert body["language_code"] == "en"
    assert materialized, "upload must be materialized"
    assert all(os.path.abspath(p).startswith(os.path.abspath(str(persistent))) for p in materialized)
    probed = {os.path.abspath(call.args[0]) for call in mock_writable.call_args_list if call.args}
    assert os.path.abspath(str(unwritable)) in probed


def test_bazarr_asr_unresolvable_video_file_uses_upload_not_mapped_path(bazarr_client: FlaskCompatibleClient, bazarr_wav: str):
    """Regression pin: upload wins when video_file cannot resolve; no stream alignment."""
    with (
        mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav) as mock_convert,
        mock.patch("modules.core.utils.get_stream_alignment_directives") as mock_alignment,
    ):
        resp = _bazarr_asr_post(bazarr_client, video_file="/etc/passwd")
    assert resp.status_code == 200
    mock_alignment.assert_not_called()
    assert mock_convert.called
    src = mock_convert.call_args[0][0]
    assert src != "/etc/passwd"
    assert "upload_" in os.path.basename(src)
    assert mock_convert.call_args.kwargs.get("input_flags") == ["-f", "s16le", "-ar", "16000", "-ac", "1"]


def _write_mapped_media(tmp_path: Path) -> Path:
    media = tmp_path / "tv" / "Show S01E01.mkv"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"REAL-MAPPED-MEDIA-CONTAINER-NOT-THE-UPLOAD")
    return media


def test_bazarr_asr_resolvable_video_file_uses_mapped_path_not_upload(
    bazarr_client: FlaskCompatibleClient, bazarr_wav: str, tmp_path: Path
):
    """Mounts present: resolvable video_file wins over multipart upload (zero-copy)."""
    mapped = _write_mapped_media(tmp_path)
    resolved = str(mapped.resolve())
    resolve_result: dict[str, object] = {}
    original = routes_utils.resolve_and_materialize_upload

    async def _capture(*args, **kwargs):
        result = await original(*args, **kwargs)
        resolve_result["local"] = result[0]
        resolve_result["upload"] = result[1]
        return result

    with (
        mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav) as mock_convert,
        mock.patch("modules.core.utils.get_stream_alignment_directives", return_value=(None, None)) as mock_alignment,
        mock.patch("modules.api.routes.asr.model_manager") as mock_mm,
        mock.patch("modules.api.routes.asr.routes_utils.resolve_and_materialize_upload", side_effect=_capture),
    ):
        mock_mm.is_engine_initialized.return_value = True
        mock_mm.increment_active_session.return_value = None
        mock_mm.decrement_active_session.return_value = None
        mock_mm.run_transcription.return_value = _DEFAULT_ASR_RESULT
        resp = _bazarr_asr_post(bazarr_client, video_file=str(mapped))

    assert resp.status_code == 200, resp.data
    mock_convert.assert_called_once_with(resolved, input_flags=[], stream_index=None, delay_filter=None)
    mock_alignment.assert_called_once_with(resolved, "en")
    assert resolve_result.get("local") == resolved
    assert resolve_result.get("upload") is None


def test_bazarr_detect_resolvable_video_file_uses_mapped_path_not_upload(bazarr_client: FlaskCompatibleClient, tmp_path: Path):
    """Mounts present: detect-language also prefers resolvable video_file over upload."""
    mapped = _write_mapped_media(tmp_path)
    resolved = str(mapped.resolve())
    materialized: list[str] = []
    original = routes_utils.materialize_upload_file

    async def _capture(upload_file, local_path=None):
        path, name = await original(upload_file, local_path=local_path)
        if path:
            materialized.append(path)
        return path, name

    with (
        mock.patch("modules.core.utils.get_audio_duration", return_value=90.0) as mock_duration,
        mock.patch("modules.api.support.request_utils.materialize_upload_file", side_effect=_capture),
    ):
        resp = _bazarr_detect_post(bazarr_client, video_file=str(mapped))

    assert resp.status_code == 200, resp.data
    body = json.loads(resp.data)
    assert body["language_code"] == "en"
    mock_duration.assert_called()
    assert mock_duration.call_args[0][0] == resolved
    assert not materialized, "resolvable mapped media must skip upload materialization"


def test_bazarr_local_path_with_mount_skips_upload_and_clears_raw_pcm_flags(
    bazarr_client: FlaskCompatibleClient, bazarr_wav: str, tmp_path: Path
):
    """Explicit local_path under approved roots remains the zero-copy ASR path."""
    mapped = _write_mapped_media(tmp_path)
    resolved = str(mapped.resolve())
    with (
        mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav) as mock_convert,
        mock.patch("modules.core.utils.get_stream_alignment_directives", return_value=(1, "adelay=10:all=1")) as mock_alignment,
    ):
        resp = bazarr_client.post(
            f"/asr?task=transcribe&language=en&output=srt&encode=false&local_path={mapped}",
            data={"audio_file": (io.BytesIO(_RAW_PCM), "audio.raw")},
            content_type="multipart/form-data",
        )
    assert resp.status_code == 200, resp.data
    mock_alignment.assert_called_once_with(resolved, "en")
    mock_convert.assert_called_once_with(resolved, input_flags=[], stream_index=1, delay_filter="adelay=10:all=1")
