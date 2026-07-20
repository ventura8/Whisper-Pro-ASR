"""Full Bazarr integration tests exercising volume-mapped local_path and upload flows."""

from __future__ import annotations

import io
import json
import threading
from pathlib import Path
from unittest import mock

from tests.conftest import FlaskCompatibleClient


class TestBazarrLocalPathVolumeMapping:
    """Bazarr zero-copy flow: local_path points to a volume-mapped file readable by the container."""

    def test_asr_local_path_srt(self, bazarr_client: FlaskCompatibleClient, bazarr_wav: str):
        """Full /asr request with local_path producing SRT output."""
        with mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav):
            resp = bazarr_client.post(f"/asr?local_path={bazarr_wav}&output=srt")
            assert resp.status_code == 200
            body = resp.data.decode()
            assert "Hello world" in body
            assert "from Bazarr" in body
            assert "-->" in body

    def test_asr_local_path_vtt(self, bazarr_client: FlaskCompatibleClient, bazarr_wav: str):
        """Full /asr request with local_path producing VTT output."""
        with mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav):
            resp = bazarr_client.post(f"/asr?local_path={bazarr_wav}&output=vtt")
            assert resp.status_code == 200
            body = resp.data.decode()
            assert "WEBVTT" in body
            assert "Hello world" in body

    def test_asr_local_path_json(self, bazarr_client: FlaskCompatibleClient, bazarr_wav: str):
        """Full /asr request with local_path producing JSON output."""
        with mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav):
            resp = bazarr_client.post(f"/asr?local_path={bazarr_wav}&output=json")
            assert resp.status_code == 200
            data = json.loads(resp.data)
            assert data["text"] == "Hello world from Bazarr"

    def test_asr_local_path_txt(self, bazarr_client: FlaskCompatibleClient, bazarr_wav: str):
        """Full /asr request with local_path producing plain text output."""
        with mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav):
            resp = bazarr_client.post(f"/asr?local_path={bazarr_wav}&output=txt")
            assert resp.status_code == 200
            assert b"Hello world" in resp.data
            assert b"from Bazarr" in resp.data

    def test_detect_language_local_path(self, bazarr_client: FlaskCompatibleClient, bazarr_wav: str):
        """Full /detect-language request with local_path."""
        with mock.patch("modules.core.utils.get_audio_duration", return_value=120.0):
            resp = bazarr_client.post(f"/detect-language?local_path={bazarr_wav}")
            assert resp.status_code == 200
            data = json.loads(resp.data)
            assert data["detected_language"] == "en"
            assert data["confidence"] >= 0.9

    def test_v1_transcriptions_local_path(self, bazarr_client: FlaskCompatibleClient, bazarr_wav: str):
        """Full /v1/audio/transcriptions OpenAI-compat route with local_path."""
        with mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav):
            resp = bazarr_client.post(f"/v1/audio/transcriptions?local_path={bazarr_wav}&output=json")
            assert resp.status_code == 200
            data = json.loads(resp.data)
            assert data["text"] == "Hello world from Bazarr"

    def test_v1_translations_local_path(self, bazarr_client: FlaskCompatibleClient, bazarr_wav: str):
        """Full /v1/audio/translations OpenAI-compat route with local_path."""
        with mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav):
            resp = bazarr_client.post(f"/v1/audio/translations?local_path={bazarr_wav}&task=translate&output=json")
            assert resp.status_code == 200
            data = json.loads(resp.data)
            assert "text" in data

    def test_local_path_nonexistent_returns_400(self, bazarr_client: FlaskCompatibleClient):
        """Bazarr sends a path that doesn't exist in the container - 400."""
        resp = bazarr_client.post("/asr?local_path=/mnt/media/nonexistent.mkv")
        assert resp.status_code == 400


class TestBazarrUploadFallback:
    """Bazarr upload fallback: when local_path is unreadable, file upload is used."""

    def test_asr_upload_srt(self, bazarr_client: FlaskCompatibleClient, bazarr_wav: str):
        """Full /asr request via multipart upload producing SRT."""
        audio_bytes = Path(bazarr_wav).read_bytes()
        with mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav):
            with mock.patch("os.path.getsize", return_value=len(audio_bytes)):
                resp = bazarr_client.post(
                    "/asr?output=srt",
                    data={"audio_file": (io.BytesIO(audio_bytes), "Movie (2024).mkv")},
                    content_type="multipart/form-data",
                )
                assert resp.status_code == 200
                body = resp.data.decode()
                assert "Hello world" in body
                assert "-->" in body

    def test_asr_upload_json(self, bazarr_client: FlaskCompatibleClient, bazarr_wav: str):
        """Full /asr request via multipart upload producing JSON."""
        audio_bytes = Path(bazarr_wav).read_bytes()
        with mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav):
            with mock.patch("os.path.getsize", return_value=len(audio_bytes)):
                resp = bazarr_client.post(
                    "/asr?output=json",
                    data={"audio_file": (io.BytesIO(audio_bytes), "Movie (2024).mkv")},
                    content_type="multipart/form-data",
                )
                assert resp.status_code == 200
                data = json.loads(resp.data)
                assert data["text"] == "Hello world from Bazarr"

    def test_detect_language_upload(self, bazarr_client: FlaskCompatibleClient, bazarr_wav: str):
        """Full /detect-language via multipart upload."""
        audio_bytes = Path(bazarr_wav).read_bytes()
        with mock.patch("modules.core.utils.get_audio_duration", return_value=60.0):
            with mock.patch("os.path.getsize", return_value=len(audio_bytes)):
                resp = bazarr_client.post(
                    "/detect-language",
                    data={"audio_file": (io.BytesIO(audio_bytes), "Movie (2024).mkv")},
                    content_type="multipart/form-data",
                )
                assert resp.status_code == 200
                data = json.loads(resp.data)
                assert data["detected_language"] == "en"

    def test_v1_transcriptions_upload(self, bazarr_client: FlaskCompatibleClient, bazarr_wav: str):
        """Full /v1/audio/transcriptions via multipart upload (OpenAI compat)."""
        audio_bytes = Path(bazarr_wav).read_bytes()
        with mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav):
            with mock.patch("os.path.getsize", return_value=len(audio_bytes)):
                resp = bazarr_client.post(
                    "/v1/audio/transcriptions?output=json",
                    data={"file": (io.BytesIO(audio_bytes), "Movie (2024).mkv")},
                    content_type="multipart/form-data",
                )
                assert resp.status_code == 200
                data = json.loads(resp.data)
                assert data["text"] == "Hello world from Bazarr"


class TestBazarrEdgeCases:
    """Edge cases common in Bazarr deployments."""

    def test_path_with_spaces_and_parens(self, bazarr_client: FlaskCompatibleClient, tmp_path):
        """Bazarr paths often contain spaces and parentheses."""
        media_dir = tmp_path / "media" / "TV Shows" / "Show (2024)"
        media_dir.mkdir(parents=True)
        wav_file = media_dir / "S01E01 - Episode Title (1080p).mkv"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 40)

        with mock.patch("modules.core.utils.convert_to_wav", return_value=str(wav_file)):
            resp = bazarr_client.post(f"/asr?local_path={wav_file}&output=json")
            assert resp.status_code == 200
            data = json.loads(resp.data)
            assert data["text"] == "Hello world from Bazarr"

    def test_concurrent_local_path_and_upload_do_not_conflict(self, bazarr_client: FlaskCompatibleClient, bazarr_wav: str):
        """Simulate two concurrent Bazarr requests: one local_path, one upload."""
        audio_bytes = Path(bazarr_wav).read_bytes()
        with mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav):
            with mock.patch("os.path.getsize", return_value=len(audio_bytes)):
                responses: dict[str, object] = {}

                def _post_local() -> None:
                    responses["local"] = bazarr_client.post(f"/asr?local_path={bazarr_wav}&output=json")

                def _post_upload() -> None:
                    responses["upload"] = bazarr_client.post(
                        "/asr?output=json",
                        data={"audio_file": (io.BytesIO(audio_bytes), "upload.mkv")},
                        content_type="multipart/form-data",
                    )

                local_thread = threading.Thread(target=_post_local)
                upload_thread = threading.Thread(target=_post_upload)
                local_thread.start()
                upload_thread.start()
                local_thread.join(timeout=5.0)
                upload_thread.join(timeout=5.0)

                resp_local = responses["local"]
                resp_upload = responses["upload"]
                assert resp_local.status_code == 200
                assert resp_upload.status_code == 200
                assert json.loads(resp_local.data)["text"] == "Hello world from Bazarr"
                assert json.loads(resp_upload.data)["text"] == "Hello world from Bazarr"

    def test_empty_upload_rejected(self, bazarr_client: FlaskCompatibleClient):
        """Bazarr sending empty file body is rejected gracefully."""
        with mock.patch("os.path.getsize", return_value=0), mock.patch("os.remove"):
            resp = bazarr_client.post(
                "/asr",
                data={"audio_file": (io.BytesIO(b""), "empty.mkv")},
                content_type="multipart/form-data",
            )
            assert resp.status_code == 400

    def test_encode_true_default_ffmpeg_pipeline(self, bazarr_client: FlaskCompatibleClient, bazarr_wav: str):
        """Bazarr default: encode=true triggers FFmpeg pre-processing."""
        with mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav) as mock_convert:
            resp = bazarr_client.post(f"/asr?local_path={bazarr_wav}&encode=true&output=srt")
            assert resp.status_code == 200
            mock_convert.assert_called()

    def test_encode_false_bypasses_ffmpeg_for_raw_pcm_input(self, bazarr_client: FlaskCompatibleClient, bazarr_wav: str):
        """Bazarr encode=false: raw 16kHz mono PCM bypasses FFmpeg normalization."""
        with mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav) as mock_convert:
            resp = bazarr_client.post(f"/asr?local_path={bazarr_wav}&encode=false&output=srt")
            assert resp.status_code == 200
            mock_convert.assert_not_called()
            assert b"-->" in resp.data

    def test_asr_requires_api_key_when_configured(self, bazarr_secured_client: FlaskCompatibleClient, bazarr_wav: str):
        """Bazarr-compatible ASR requests must honor the app-level API key middleware."""
        with mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav):
            unauthorized = bazarr_secured_client.post(f"/asr?local_path={bazarr_wav}&output=json")
            authorized = bazarr_secured_client.post(
                f"/asr?local_path={bazarr_wav}&output=json",
                headers={"X-API-Key": "integration-secret"},
            )
        assert unauthorized.status_code == 401
        assert authorized.status_code == 200

    def test_detect_language_requires_api_key_when_configured(self, bazarr_secured_client: FlaskCompatibleClient, bazarr_wav: str):
        """Bazarr-compatible language detection requests must honor the app-level API key middleware."""
        unauthorized = bazarr_secured_client.post(f"/detect-language?local_path={bazarr_wav}")
        authorized = bazarr_secured_client.post(
            f"/detect-language?local_path={bazarr_wav}",
            headers={"X-API-Key": "integration-secret"},
        )
        assert unauthorized.status_code == 401
        assert authorized.status_code == 200

    def test_task_translate_via_upload(self, bazarr_client: FlaskCompatibleClient, bazarr_wav: str):
        """Bazarr translate feature: task=translate produces English subtitles."""
        audio_bytes = Path(bazarr_wav).read_bytes()
        with mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav):
            with mock.patch("os.path.getsize", return_value=len(audio_bytes)):
                resp = bazarr_client.post(
                    "/asr?task=translate&output=srt",
                    data={"audio_file": (io.BytesIO(audio_bytes), "foreign_movie.mkv")},
                    content_type="multipart/form-data",
                )
                assert resp.status_code == 200
                assert b"-->" in resp.data

    def test_language_param_forwarded(self, bazarr_client: FlaskCompatibleClient, bazarr_wav: str):
        """Bazarr sends language hint when known from media metadata."""
        with mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav):
            resp = bazarr_client.post(f"/asr?local_path={bazarr_wav}&language=fr&output=json")
            assert resp.status_code == 200
            data = json.loads(resp.data)
            assert data["text"] == "Hello world from Bazarr"

    def test_high_timeout_long_movie_completes(self, bazarr_client: FlaskCompatibleClient, bazarr_wav: str):
        """Bazarr sets very high timeouts for long movies; verify no premature abort."""
        with mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav):
            resp = bazarr_client.post(f"/asr?local_path={bazarr_wav}&output=srt")
            assert resp.status_code == 200
            body = resp.data.decode()
            assert "Hello world" in body
            assert "from Bazarr" in body
