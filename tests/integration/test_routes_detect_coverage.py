"""Targeted branch coverage tests for routes_detect helpers."""

import asyncio
import concurrent.futures
import contextlib
import io
import json
from unittest import mock

from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile as StarletteUploadFile

from modules.api.routes import detect as routes_detect
from modules.api.routes import detect_coalescing


def test_build_request_params_excludes_real_uploaded_file_objects():
    """Regression: `await request.form()` (Starlette) yields plain
    starlette.datastructures.UploadFile instances, never the fastapi.UploadFile
    subclass -- checking `isinstance(v, fastapi.UploadFile)` always failed for real
    uploads, so the file object's raw repr (e.g. "UploadFile(filename='audio_file', ...")
    leaked into the request_json audit payload instead of being skipped."""
    real_upload = StarletteUploadFile(file=io.BytesIO(b"pcm bytes"), filename="audio_file")
    request = mock.MagicMock()
    request.query_params = {}
    form_data = {"audio_file": real_upload, "encode": "false", "video_file": "/tv/show.avi"}

    _build_request_params = routes_detect.__dict__["_build_request_params"]
    params = _build_request_params(request, form_data)

    assert "audio_file" not in params
    assert params["encode"] == "false"
    assert params["video_file"] == "/tv/show.avi"


def _sample_worker_context() -> dict:
    return {
        "caller_info": {"ip": "127.0.0.1", "user_agent": "test"},
        "request_json": {"local_path": "/tmp/a.mp3"},
        "endpoint": "/detect-language",
        "input_flags": None,
    }


def test_await_shared_result_handles_wrap_future_exception():
    """Shared-result await should normalize exceptions from asyncio.wrap_future."""
    shared_future = concurrent.futures.Future()
    _await_shared_result = detect_coalescing.__dict__["_await_shared_result"]

    with (
        mock.patch("modules.api.routes.detect_coalescing.asyncio.wrap_future", side_effect=RuntimeError("future-error")),
        mock.patch("modules.api.routes.detect_coalescing.routes_utils.handle_error", return_value=("Error", 500)),
    ):
        response = asyncio.run(_await_shared_result(shared_future))

    assert response.status_code == 500
    body = json.loads(response.body)
    assert body["error"] == "Error"


def test_await_shared_result_returns_success_payload():
    """Shared-result await should return plain result payload on success."""
    shared_future = concurrent.futures.Future()
    payload = {"detected_language": "en"}
    shared_future.set_result((payload, None))

    _await_shared_result = detect_coalescing.__dict__["_await_shared_result"]
    response = asyncio.run(_await_shared_result(shared_future))
    assert response == payload


def test_detect_language_outer_exception_handler(routes_client):
    """Top-level detect-language should map unexpected exceptions via handle_error."""
    with mock.patch("modules.api.routes.detect.model_manager") as mock_mm:
        mock_mm.is_engine_initialized.return_value = True
        with (
            mock.patch("modules.api.routes.detect.routes_utils.parse_form_data", side_effect=RuntimeError("boom")),
            mock.patch("modules.api.routes.detect.routes_utils.handle_error", return_value=("mapped", 500)),
        ):
            response = routes_client.post("/detect-language")

    assert response.status_code == 500
    body = json.loads(response.data)
    assert body["error"] == "mapped"


def test_await_shared_result_with_dashboard_task_handles_future_exception():
    """Async coalesced follower path should map leader future exceptions."""
    shared_future = concurrent.futures.Future()
    shared_future.set_exception(RuntimeError("leader-failed"))
    _await_follower = detect_coalescing.__dict__["_await_shared_result_with_dashboard_task"]

    with (
        mock.patch("modules.api.routes.detect_coalescing.model_manager.early_task_registration", return_value=contextlib.nullcontext()),
        mock.patch("modules.api.routes.detect_coalescing.routes_utils.handle_error", return_value=("Error", 500)),
    ):
        response = asyncio.run(_await_follower(shared_future, "local_path::/tmp/a.mp3", "a.mp3", worker_context=_sample_worker_context()))

    assert response.status_code == 500
    body = json.loads(response.body)
    assert body["error"] == "Error"


def test_await_shared_result_with_dashboard_task_handles_error_tuple():
    """Async coalesced follower path should return err tuple payloads."""
    shared_future = concurrent.futures.Future()
    shared_future.set_result((None, ("boom", 500)))
    _await_follower = detect_coalescing.__dict__["_await_shared_result_with_dashboard_task"]

    with mock.patch("modules.api.routes.detect_coalescing.model_manager.early_task_registration", return_value=contextlib.nullcontext()):
        response = asyncio.run(_await_follower(shared_future, "local_path::/tmp/a.mp3", "a.mp3", worker_context=_sample_worker_context()))

    assert response.status_code == 500
    body = json.loads(response.body)
    assert body["error"] == "boom"


def test_await_shared_result_with_dashboard_task_marks_failed_for_json_error():
    """Async coalesced follower path should record JSON error metadata."""
    shared_future = concurrent.futures.Future()
    shared_future.set_result((JSONResponse(content={"error": "bad"}, status_code=500), None))
    _await_follower = detect_coalescing.__dict__["_await_shared_result_with_dashboard_task"]

    with (
        mock.patch("modules.api.routes.detect_coalescing.model_manager.early_task_registration", return_value=contextlib.nullcontext()),
        mock.patch("modules.api.routes.detect_coalescing.model_manager.record_task_failure") as mock_record,
    ):
        response = asyncio.run(_await_follower(shared_future, "local_path::/tmp/a.mp3", "a.mp3", worker_context=_sample_worker_context()))

    assert response.status_code == 500
    mock_record.assert_called_once_with("bad", 500, context="LD")


def test_log_detection_result_handles_invalid_candidate_list():
    """Logging helper should tolerate malformed list candidates without raising."""
    _log_detection_result = routes_detect.__dict__["_log_detection_result"]
    result = {
        "detected_language": "en",
        "confidence": 0.95,
        "voting_details": [("en", 0.95), ("bad",)],
        "segments_processed": 1,
        "performance": {},
    }

    with mock.patch("modules.api.routes.detect.logger.info"):
        _log_detection_result(result, 0.0)
