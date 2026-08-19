"""Tests for modules/routes.py"""

import asyncio
import concurrent.futures
import io
import json
from unittest import mock

from fastapi import FastAPI

from modules.api.routes import asr as routes_asr
from modules.api.routes import detect as routes_detect
from modules.api.routes.asr import get_request_params
from modules.api.support.request_utils import cleanup_files
from modules.core import config, utils
from tests.conftest import FlaskCompatibleClient


def test_root_get(routes_client):
    """Test GET / returns health status."""
    response = routes_client.get("/")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "healthy"


class TestStatusEndpoint:
    """Tests for the /status endpoint."""

    def test_status_model_loaded(self, routes_client):
        """Test /status when model is loaded."""
        with mock.patch("modules.api.routes.system.dashboard") as mock_db:
            mock_db.get_status_data.return_value = {"active_sessions": 0, "queued_sessions": 0, "hardware": []}
            response = routes_client.get("/status")
            assert response.status_code == 200
            data = json.loads(response.data)
            assert {"active_sessions", "asr_engine", "supported_asr_engines", "engines"}.issubset(data)
            assert data["engines"].get("selected") == data["asr_engine"]

    def test_status_model_not_loaded(self, routes_client):
        """Test /status when model stats fail."""
        with mock.patch("modules.api.routes.system.dashboard") as mock_db:
            mock_db.get_status_data.return_value = {
                "active_sessions": 0,
                "queued_sessions": 0,
                "hardware": [],
                "engines": {},
            }
            response = routes_client.get("/status")
            assert response.status_code == 200
            data = json.loads(response.data)
            assert "whisper" not in data["engines"]
            assert "uvr" not in data["engines"]


class TestDetectLanguageEndpoint:
    """Tests for /detect-language endpoint."""

    def test_detect_language_no_model(self, routes_client):
        """Test detect-language when model not loaded."""
        with mock.patch("modules.api.routes.detect.model_manager") as mock_mm:
            mock_mm.is_engine_initialized.return_value = False
            response = routes_client.post("/detect-language")
            assert response.status_code == 503

    def test_detect_language_no_input(self, routes_client):
        """Test detect-language with no input."""
        with mock.patch("modules.api.routes.detect.model_manager") as mock_mm:
            mock_mm.is_engine_initialized.return_value = True
            response = routes_client.post("/detect-language")
            assert response.status_code == 400
            assert b"No audio source provided" in response.data

    def test_detect_language_file_not_found(self, routes_client):
        """Test detect-language with non-existent file."""
        with mock.patch("modules.api.routes.detect.model_manager") as mock_mm:
            mock_mm.is_engine_initialized.return_value = True
            response = routes_client.post("/detect-language?local_path=/nonexistent/file.mp3")
            assert response.status_code == 400

    def test_detect_language_success(self, routes_client, tmp_path):
        """Test successful language detection."""
        test_file = tmp_path / "test.mp3"
        test_file.write_bytes(b"fake audio data")

        with mock.patch("modules.api.routes.detect.model_manager") as mock_mm:
            mock_mm.is_engine_initialized.return_value = True
            with mock.patch("modules.inference.pipeline.language_detection.run_voting_detection") as mock_ld:
                mock_ld.return_value = {
                    "detected_language": "en",
                    "language": "en",
                    "language_code": "en",
                    "confidence": 0.95,
                }
                response = routes_client.post(f"/detect-language?local_path={test_file}")
                assert response.status_code == 200
                data = json.loads(response.data)
                assert data["detected_language"] == "en"
                assert data["language"] == "en"
                assert data["language_code"] == "en"

    def test_detectlang_alias_success(self, routes_client, tmp_path):
        """Test legacy /detectlang alias maps to detect-language behavior."""
        test_file = tmp_path / "test.mp3"
        test_file.write_bytes(b"fake audio data")

        with mock.patch("modules.api.routes.detect.model_manager") as mock_mm:
            mock_mm.is_engine_initialized.return_value = True
            with mock.patch("modules.inference.pipeline.language_detection.run_voting_detection") as mock_ld:
                mock_ld.return_value = {
                    "detected_language": "en",
                    "language": "en",
                    "language_code": "en",
                    "confidence": 0.95,
                }
                response = routes_client.post(f"/detectlang?local_path={test_file}")
                assert response.status_code == 200
                data = json.loads(response.data)
                assert data["detected_language"] == "en"

    def test_detect_language_ffmpeg_fails(self, routes_client, tmp_path):
        """Test detect-language when the voting engine fails."""
        test_file = tmp_path / "test.mp3"
        test_file.write_bytes(b"fake audio data")

        with mock.patch("modules.api.routes.detect.model_manager") as mock_mm:
            mock_mm.is_engine_initialized.return_value = True
            with mock.patch("modules.inference.pipeline.language_detection.run_voting_detection") as mock_ld:
                mock_ld.side_effect = Exception("Detection error")
                response = routes_client.post(f"/detect-language?local_path={test_file}")
                assert response.status_code == 500
                assert b"Error" in response.data

    def test_detect_language_exception(self, routes_client, tmp_path):
        """Test detect-language handles exceptions."""
        test_file = tmp_path / "test.mp3"
        test_file.write_bytes(b"fake audio data")

        with mock.patch("modules.api.routes.detect.model_manager") as mock_mm:
            mock_mm.is_engine_initialized.return_value = True
            with mock.patch("modules.inference.pipeline.language_detection.run_voting_detection") as mock_ld:
                mock_ld.side_effect = Exception("Detection error")
                response = routes_client.post(f"/detect-language?local_path={test_file}")
                assert response.status_code == 500
                assert b"Error" in response.data

    def test_detect_language_uses_inflight_result_for_duplicate_local_path(self, routes_client, tmp_path):
        """Duplicate local-path requests should wait for in-flight result instead of spawning new work."""
        test_file = tmp_path / "test.mp3"
        test_file.write_bytes(b"fake audio data")

        _build_dedupe_key = routes_detect.__dict__["_build_dedupe_key"]
        inflight_detect_by_path = routes_detect.__dict__["_INFLIGHT_DETECT_BY_PATH"]

        with mock.patch("modules.api.routes.detect.model_manager") as mock_mm:
            mock_mm.is_engine_initialized.return_value = True
            future = concurrent.futures.Future()
            shared_result = {
                "detected_language": "en",
                "language": "en",
                "language_code": "en",
                "confidence": 0.99,
            }
            future.set_result((shared_result, None))
            dedupe_key = _build_dedupe_key(str(test_file), None)
            inflight_detect_by_path[dedupe_key] = future

            try:
                response = routes_client.post(f"/detect-language?local_path={test_file}")
            finally:
                inflight_detect_by_path.pop(dedupe_key, None)

            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["detected_language"] == "en"
            mock_mm.increment_active_session.assert_not_called()
            mock_mm.early_task_registration.assert_called_once()
            mock_mm.update_task_metadata.assert_any_call(
                stage="Coalesced Request (Waiting for Leader)",
                status="queued",
                coalesced=True,
                coalesced_key=dedupe_key,
            )

    def test_await_shared_result_returns_error_response_when_future_returns_error_tuple(self):
        """Follower requests should return error payloads from the leader result tuple."""
        shared_future = concurrent.futures.Future()
        shared_future.set_result((None, ("boom", 500)))

        _await_shared_result = routes_detect.__dict__["_await_shared_result"]
        response = asyncio.run(_await_shared_result(shared_future))

        assert response.status_code == 500
        body = json.loads(response.body)
        assert body["error"] == "boom"

    def test_run_leader_detection_handles_internal_exception(self):
        """Leader helper should publish exceptions to waiters and return a normalized error response."""
        shared_future = concurrent.futures.Future()
        dedupe_key = "local_path::/tmp/test.mp3"
        inflight_detect_by_path = routes_detect.__dict__["_INFLIGHT_DETECT_BY_PATH"]
        _run_leader_detection = routes_detect.__dict__["_run_leader_detection"]
        inflight_detect_by_path[dedupe_key] = shared_future

        with mock.patch("modules.api.routes.detect._run_detection_internal", side_effect=Exception("kaboom")):
            with mock.patch("modules.api.routes.detect.routes_utils.handle_error", return_value=("Error", 500)):
                response = asyncio.run(
                    _run_leader_detection(
                        shared_future,
                        dedupe_key,
                        {
                            "resolved_local_path": "/tmp/test.mp3",
                            "uploaded_file": None,
                            "filename": "test.mp3",
                            "start_time": 0.0,
                        },
                    )
                )

        assert response.status_code == 500
        assert shared_future.done()
        assert dedupe_key not in inflight_detect_by_path

    def test_detect_language_does_not_coalesce_when_config_disabled(self, routes_client, tmp_path):
        """When coalescing is disabled, requests must execute normally even if an in-flight future exists."""
        test_file = tmp_path / "test.mp3"
        test_file.write_bytes(b"fake audio data")

        _build_dedupe_key = routes_detect.__dict__["_build_dedupe_key"]
        inflight_detect_by_path = routes_detect.__dict__["_INFLIGHT_DETECT_BY_PATH"]

        stale_future = concurrent.futures.Future()
        stale_future.set_result(
            (
                {
                    "detected_language": "fr",
                    "language": "fr",
                    "language_code": "fr",
                    "confidence": 0.42,
                },
                None,
            )
        )
        dedupe_key = _build_dedupe_key(str(test_file), None)
        inflight_detect_by_path[dedupe_key] = stale_future

        try:
            with mock.patch("modules.api.routes.detect.config.ENABLE_LD_REQUEST_COALESCING", False):
                with mock.patch("modules.api.routes.detect.model_manager") as mock_mm:
                    mock_mm.is_engine_initialized.return_value = True
                    with mock.patch("modules.inference.pipeline.language_detection.run_voting_detection") as mock_ld:
                        mock_ld.return_value = {
                            "detected_language": "en",
                            "language": "en",
                            "language_code": "en",
                            "confidence": 0.95,
                        }
                        response = routes_client.post(f"/detect-language?local_path={test_file}")
        finally:
            inflight_detect_by_path.pop(dedupe_key, None)

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["detected_language"] == "en"


class TestASREndpoint:
    """Tests for /asr endpoint."""

    def test_asr_get_ready(self, routes_client):
        """Test GET /asr when model is ready."""
        with mock.patch("modules.api.routes.asr.model_manager") as mock_mm:
            mock_mm.is_engine_initialized.return_value = True
            response = routes_client.get("/asr")
            assert response.status_code == 200
            assert b"ready" in response.data

    def test_asr_get_not_ready(self, routes_client):
        """Test GET /asr when model not ready."""
        with mock.patch("modules.api.routes.asr.model_manager") as mock_mm:
            mock_mm.is_engine_initialized.return_value = False
            response = routes_client.get("/asr")
            assert response.status_code == 200
            assert b"not_ready" in response.data

    def test_asr_post_no_model(self):
        """Test POST /asr when model not loaded."""
        with mock.patch("modules.api.routes.asr.model_manager") as mock_mm:
            mock_mm.is_engine_initialized.return_value = False

            test_app = FastAPI(title="Test App")
            test_app.include_router(routes_asr.router)
            test_client = FlaskCompatibleClient(test_app)

            response = test_client.post("/asr")
            assert response.status_code == 503

    def test_audio_transcriptions_alias_no_model(self):
        """Test OpenAI-compatible transcriptions alias when model not loaded."""
        with mock.patch("modules.api.routes.asr.model_manager") as mock_mm:
            mock_mm.is_engine_initialized.return_value = False

            test_app = FastAPI(title="Test App")
            test_app.include_router(routes_asr.router)
            test_client = FlaskCompatibleClient(test_app)

            response = test_client.post("/v1/audio/transcriptions")
            assert response.status_code == 503

    def test_audio_translations_alias_forces_translate_task(self):
        """Test OpenAI-compatible translations alias forces task=translate."""
        mock_request = mock.MagicMock()
        mock_request.query_params = {}
        mock_request.url.path = "/v1/audio/translations"
        params = asyncio.run(get_request_params(mock_request, {}))
        assert params["task"] == "translate"

    def test_asr_post_no_input(self, routes_client):
        """Test POST /asr with no input."""
        with mock.patch("modules.api.routes.asr.model_manager") as mock_mm:
            mock_mm.is_engine_initialized.return_value = True
            response = routes_client.post("/asr")
            assert response.status_code == 400

    def test_asr_post_file_not_found(self, routes_client):
        """Test POST /asr with non-existent file."""
        with mock.patch("modules.api.routes.asr.model_manager") as mock_mm:
            mock_mm.is_engine_initialized.return_value = True
            response = routes_client.post("/asr?local_path=/nonexistent/file.mp3")
            assert response.status_code == 400

    def test_asr_post_success_srt(self, routes_client, tmp_path):
        """Test successful ASR with SRT output."""
        test_file = tmp_path / "test.mp3"
        test_file.write_bytes(b"fake audio data")

        with mock.patch("modules.api.routes.asr.model_manager") as mock_mm:
            mock_mm.is_engine_initialized.return_value = True
            mock_mm.run_transcription.return_value = {
                "text": "Hello",
                "segments": [{"timestamp": (0.0, 1.0), "text": "Hello"}],
            }
            with mock.patch("modules.core.utils.convert_to_wav") as mock_wav:
                mock_wav.return_value = str(test_file)
                response = routes_client.post(f"/asr?local_path={test_file}&output=srt")
                assert response.status_code == 200
                assert b"Hello" in response.data

    def test_asr_post_success_json(self, routes_client, tmp_path):
        """Test successful ASR with JSON output."""
        test_file = tmp_path / "test.mp3"
        test_file.write_bytes(b"fake audio data")

        with mock.patch("modules.api.routes.asr.model_manager") as mock_mm:
            mock_mm.is_engine_initialized.return_value = True
            mock_mm.run_transcription.return_value = {"text": "Hello", "segments": []}
            with mock.patch("modules.core.utils.convert_to_wav") as mock_wav:
                mock_wav.return_value = str(test_file)
                response = routes_client.post(f"/asr?local_path={test_file}&output=json")
                assert response.status_code == 200
                data = json.loads(response.data)
                assert data["text"] == "Hello"

    def test_asr_post_ffmpeg_fails(self, routes_client, tmp_path):
        """Test ASR when FFmpeg conversion fails."""
        test_file = tmp_path / "test.mp3"
        test_file.write_bytes(b"fake audio data")

        with mock.patch("modules.api.routes.asr.model_manager") as mock_mm:
            mock_mm.is_engine_initialized.return_value = True
            with mock.patch("modules.core.utils.convert_to_wav") as mock_wav:
                mock_wav.return_value = None
                response = routes_client.post(f"/asr?local_path={test_file}")
                assert response.status_code == 400
                assert b"FFmpeg" in response.data

    def test_asr_post_exception(self, routes_client, tmp_path):
        """Test ASR handles transcription exceptions."""
        test_file = tmp_path / "test.mp3"
        test_file.write_bytes(b"fake audio data")

        with mock.patch("modules.api.routes.asr.model_manager") as mock_mm:
            mock_mm.is_engine_initialized.return_value = True
            mock_mm.run_transcription.side_effect = Exception("Transcription error")
            with mock.patch("modules.core.utils.convert_to_wav") as mock_wav:
                mock_wav.return_value = str(test_file)
                response = routes_client.post(f"/asr?local_path={test_file}")
                assert response.status_code == 500
                assert b"Error" in response.data

    def test_asr_post_exception_generic(self, routes_client, tmp_path):
        """Test ASR handles generic exceptions."""
        test_file = tmp_path / "test.mp3"
        test_file.write_bytes(b"fake audio data")

        with mock.patch("modules.api.routes.asr.model_manager") as mock_mm:
            mock_mm.is_engine_initialized.return_value = True
            with mock.patch("modules.api.support.request_utils.prepare_source_path", return_value=(test_file, None, "test.mp3")):
                with mock.patch("os.path.exists", return_value=True):
                    with mock.patch("modules.core.utils.convert_to_wav", return_value="clean.wav"):
                        mock_mm.run_transcription.side_effect = Exception("Generic error")
                        response = routes_client.post("/asr?local_path=test.mp3")
                        assert response.status_code == 500

    def test_asr_various_outputs(self, routes_client, tmp_path):
        """Test ASR with various output formats."""
        test_file = tmp_path / "test.mp3"
        test_file.write_bytes(b"fake audio data")

        with mock.patch("modules.api.routes.asr.model_manager") as mock_mm:
            mock_mm.is_engine_initialized.return_value = True
            mock_mm.run_transcription.return_value = {
                "text": "Hello world",
                "segments": [{"timestamp": (0.0, 1.0), "text": "Hello world"}],
            }
            with mock.patch("modules.api.support.request_utils.prepare_source_path", return_value=(test_file, None, "test.mp3")):
                with mock.patch("os.path.exists", return_value=True):
                    with mock.patch("modules.core.utils.convert_to_wav", return_value="clean.wav"):
                        # VTT
                        resp = routes_client.post("/asr?local_path=test.mp3&output=vtt")
                        assert resp.status_code == 200

                        # TXT
                        resp = routes_client.post("/asr?local_path=test.mp3&output=txt")
                        assert resp.status_code == 200
                        assert b"Hello world" in resp.data

    def test_asr_upload_corrupt(self, routes_client):
        """Test ASR with uploaded corrupted file."""
        corrupt_data = b"\x00" * 2048
        with (
            mock.patch("os.path.join", return_value="fake.mp3"),
            mock.patch("os.path.getsize", return_value=2048),
            mock.patch("builtins.open", mock.mock_open(read_data=b"\x00" * 2048)),
            mock.patch("os.remove"),
        ):
            response = routes_client.post(
                "/asr",
                data={"audio_file": (io.BytesIO(corrupt_data), "corrupt.mp3")},
                content_type="multipart/form-data",
            )
            assert response.status_code == 400
            assert b"FFmpeg conversion failed" in response.data or b"corrupted" in response.data

    def test_asr_upload_empty(self, routes_client):
        """Test ASR with empty uploaded file."""
        with mock.patch("os.path.getsize", return_value=0), mock.patch("os.remove"):
            response = routes_client.post("/asr", data={"audio_file": (io.BytesIO(b""), "empty.mp3")}, content_type="multipart/form-data")
            assert response.status_code == 400
            assert b"empty" in response.data.lower()

    def test_cleanup_files_coverage(self):
        """Test cleanup_files helper coverage."""
        utils.get_tracked_files().clear()
        with mock.patch("os.path.exists", return_value=True):
            with mock.patch("os.remove") as mock_remove:
                cleanup_files("file1.wav", "file2.wav")
                assert mock_remove.call_count == 2

    def test_get_all_request_params_edge_cases(self):
        """Test get_request_params default fallbacks."""
        mock_request = mock.MagicMock()
        mock_request.query_params = {"batch_size": "not_an_int"}
        mock_request.url.path = "/asr"
        params = asyncio.run(get_request_params(mock_request, {}))
        assert params["batch_size"] == config.DEFAULT_BATCH_SIZE

        mock_request2 = mock.MagicMock()
        mock_request2.query_params = {"task": "translate"}
        mock_request2.url.path = "/asr"
        params2 = asyncio.run(get_request_params(mock_request2, {}))
        assert params2["task"] == "translate"
