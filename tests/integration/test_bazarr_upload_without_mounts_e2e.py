"""E2E: Bazarr multipart uploads must work without media volume mounts.

Media mounts (/tv, /movies) are an optional zero-copy optimisation. Real Bazarr
always uploads pre-encoded raw PCM (encode=false) and may also send video_file as
caller metadata. When video_file is unresolvable -- or TEMP_DIR is present but
not writable (Portainer stack missing tmpfs mode=1777) -- the upload path must
still succeed. The NUC production outage of 2026-11-17 was exactly this:
unwritable /tmp/whisper + no media mounts → every job "No audio source provided".
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from unittest import mock

import pytest

from tests.conftest import FlaskCompatibleClient

_RAW_PCM = b"\x00\x01" * 1000


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

    with (
        mock.patch("modules.core.config.TEMP_DIR", str(unwritable)),
        mock.patch("modules.core.config.PERSISTENT_TEMP_DIR", str(persistent)),
        mock.patch("modules.core.mount_helpers.is_path_writable", side_effect=_writable),
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
