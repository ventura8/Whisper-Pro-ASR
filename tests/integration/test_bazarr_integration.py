"""Full Bazarr integration tests exercising volume-mapped local_path and upload flows."""

from __future__ import annotations

import io
import json
import threading
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from tests.conftest import FlaskCompatibleClient
from tests.integration.conftest import (
    _BAZARR_TRANSCRIPTION_RESULT,
    mock_asr_manager_for_transcription,
)


class TestBazarrLocalPathVolumeMapping:
    """local_path optimization: a caller-supplied path points to a volume-mapped
    file readable by the container, avoiding a re-upload.

    Note: real Bazarr's own client (subliminal_patch/providers/whisperai.py)
    never sends local_path — it always uploads pre-encoded audio via multipart.
    See TestRealBazarrWireFormat for tests matching Bazarr's actual wire format;
    this class covers the local_path optimization as a distinct, intentionally
    supported feature."""

    def test_asr_local_path_srt(
        self,
        bazarr_client: FlaskCompatibleClient,
        bazarr_wav: str,
    ):
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
        """local_path optimization, encode=true: triggers FFmpeg pre-processing.

        Note: real Bazarr never sends local_path (see TestRealBazarrWireFormat for its
        actual multipart-upload wire format) — this covers the local_path optimization
        this codebase separately supports."""
        with mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav) as mock_convert:
            resp = bazarr_client.post(f"/asr?local_path={bazarr_wav}&encode=true&output=srt")
            assert resp.status_code == 200
            mock_convert.assert_called()

    def test_encode_false_still_runs_ffmpeg_with_raw_pcm_input_flags(self, bazarr_client: FlaskCompatibleClient, bazarr_wav: str):
        """local_path optimization, encode=false: FFmpeg normalization is NOT bypassed --
        faster-whisper's decode_audio() (PyAV's av.open(), no format hint) cannot identify
        headerless raw PCM on its own, so FFmpeg must still run with the raw-PCM input
        flags (-f s16le -ar 16000 -ac 1) to produce a container faster-whisper can read.

        Note: real Bazarr never sends local_path — see TestRealBazarrWireFormat for its
        actual multipart-upload wire format with encode=false."""
        with mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav) as mock_convert:
            resp = bazarr_client.post(f"/asr?local_path={bazarr_wav}&encode=false&output=srt")
            assert resp.status_code == 200
            mock_convert.assert_called_once_with(
                bazarr_wav, input_flags=["-f", "s16le", "-ar", "16000", "-ac", "1"], stream_index=None, delay_filter=None
            )
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


class TestBazarrJsonSchemaContract:
    """6.3: JSON schema/contract stability for the /asr JSON transcript response.

    Bazarr's parser depends on specific field names/types in the JSON output.
    If a field is renamed, removed, or its type changes, this must fail loudly
    instead of Bazarr silently breaking.
    """

    EXPECTED_TOP_LEVEL_SCHEMA = {"text": str, "segments": list}
    EXPECTED_SEGMENT_SCHEMA = {"text": str, "timestamp": list}

    def _assert_matches_schema(self, obj: dict, schema: dict, context: str) -> None:
        for field, expected_type in schema.items():
            assert field in obj, f"{context}: expected field {field!r} is missing (contract broken)"
            type_ok = isinstance(obj[field], expected_type)
            type_msg = f"{context}: field {field!r} expected type {expected_type.__name__}, got {type(obj[field]).__name__}"
            assert type_ok, type_msg

    def _assert_all_segments_match_schema(self, segments: list[dict[str, Any]]) -> None:
        for seg in segments:
            assert isinstance(seg, dict), "each segment must serialize as a JSON object"
            self._assert_matches_schema(seg, self.EXPECTED_SEGMENT_SCHEMA, "segment")

    def _assert_segment_timing_matches_expected(self, actual_segments: list[dict[str, Any]]) -> None:
        """Assert each segment's timing and text (used by Bazarr for subtitle placement
        and content) survived the JSON round trip unchanged, not just present with the
        right type."""
        expected_segments = _BAZARR_TRANSCRIPTION_RESULT["segments"]
        for actual_segment, expected_segment in zip(actual_segments, expected_segments, strict=True):
            assert actual_segment["timestamp"] == list(expected_segment["timestamp"])
            assert actual_segment["text"] == expected_segment["text"]

    def test_asr_json_response_matches_expected_schema(self, bazarr_client: FlaskCompatibleClient, bazarr_wav: str):
        """Verify the /asr JSON response's top-level and segment fields match the expected schema."""
        with mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav):
            resp = bazarr_client.post(f"/asr?local_path={bazarr_wav}&output=json")
        assert resp.status_code == 200
        data = json.loads(resp.data)

        self._assert_matches_schema(data, self.EXPECTED_TOP_LEVEL_SCHEMA, "top-level JSON response")
        assert data["text"] == _BAZARR_TRANSCRIPTION_RESULT["text"]
        assert len(data["segments"]) == len(_BAZARR_TRANSCRIPTION_RESULT["segments"])
        self._assert_all_segments_match_schema(data["segments"])
        self._assert_segment_timing_matches_expected(data["segments"])

    def _assert_detect_language_field_matches_schema(self, data: dict, field: str, expected_type: type | tuple[type, ...]) -> None:
        assert field in data, f"detect-language response missing expected field {field!r}"
        value = data[field]
        type_ok = isinstance(value, expected_type) and not isinstance(value, bool)
        type_msg = f"detect-language field {field!r} expected {expected_type}, got {type(value).__name__!r} value {value!r}"
        assert type_ok, type_msg

    def test_detect_language_json_response_matches_expected_schema(self, bazarr_client: FlaskCompatibleClient, bazarr_wav: str):
        """Verify the /detect-language JSON response fields match the expected schema."""
        expected_schema = {
            "detected_language": str,
            "language": str,
            "language_code": str,
            "confidence": (int, float),
        }
        with mock.patch("modules.core.utils.get_audio_duration", return_value=120.0):
            resp = bazarr_client.post(f"/detect-language?local_path={bazarr_wav}")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        for field, expected_type in expected_schema.items():
            self._assert_detect_language_field_matches_schema(data, field, expected_type)


# 6.6: non-ASCII / multi-byte text survives the full HTTP round trip for SRT/VTT/JSON.
_NON_ASCII_MIXED_TEXT = "Café à côté — 你好世界 — مرحبا بالعالم — שלום עולם"


def _post_asr_with_mocked_non_ascii_result(bazarr_client: FlaskCompatibleClient, bazarr_wav: str, output: str) -> Any:
    """Shared setup: mock a transcription result containing the mixed non-ASCII text and POST /asr with the given output format."""
    mocked_result = {
        "text": _NON_ASCII_MIXED_TEXT,
        "segments": [{"timestamp": (0.0, 1.0), "text": _NON_ASCII_MIXED_TEXT}],
    }
    with (
        mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav),
        mock.patch("modules.api.routes.asr.model_manager") as mock_mm,
    ):
        mock_asr_manager_for_transcription(mock_mm, result=mocked_result)
        return bazarr_client.post(f"/asr?local_path={bazarr_wav}&output={output}")


def _assert_json_output_preserves_non_ascii_text(resp: Any) -> None:
    data = json.loads(resp.data)
    assert data["text"] == _NON_ASCII_MIXED_TEXT
    assert data["segments"][0]["text"] == _NON_ASCII_MIXED_TEXT


def _assert_text_output_preserves_non_ascii_text(resp: Any, *, expect_no_bom: bool) -> None:
    if expect_no_bom:
        assert not resp.data.startswith(b"\xef\xbb\xbf"), "SRT output must not have a UTF-8 BOM"
    assert _NON_ASCII_MIXED_TEXT in resp.data.decode("utf-8")


@pytest.mark.parametrize("output", ["json", "srt", "vtt"])
def test_asr_output_preserves_non_ascii_text(bazarr_client: FlaskCompatibleClient, bazarr_wav: str, output: str):
    """Verify non-ASCII/multi-byte text survives the round trip in the /asr output, for every format."""
    resp = _post_asr_with_mocked_non_ascii_result(bazarr_client, bazarr_wav, output)
    assert resp.status_code == 200

    if output == "json":
        _assert_json_output_preserves_non_ascii_text(resp)
    else:
        _assert_text_output_preserves_non_ascii_text(resp, expect_no_bom=output in ("srt", "vtt"))
