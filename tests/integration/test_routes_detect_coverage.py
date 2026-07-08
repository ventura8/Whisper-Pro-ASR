"""Targeted branch coverage tests for routes_detect helpers."""

import asyncio
import concurrent.futures
import contextlib
import json
from unittest import mock

from fastapi.responses import JSONResponse

from modules.api import routes_detect


def test_await_shared_result_handles_wrap_future_exception():
    """Shared-result await should normalize exceptions from asyncio.wrap_future."""
    shared_future = concurrent.futures.Future()
    _await_shared_result = routes_detect.__dict__["_await_shared_result"]

    with (
        mock.patch("modules.api.routes_detect.asyncio.wrap_future", side_effect=RuntimeError("future-error")),
        mock.patch("modules.api.routes_detect.routes_utils.handle_error", return_value=("Error", 500)),
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

    _await_shared_result = routes_detect.__dict__["_await_shared_result"]
    response = asyncio.run(_await_shared_result(shared_future))
    assert response == payload


def test_detect_language_outer_exception_handler(routes_client):
    """Top-level detect-language should map unexpected exceptions via handle_error."""
    with mock.patch("modules.api.routes_detect.model_manager") as mock_mm:
        mock_mm.is_engine_initialized.return_value = True
        with (
            mock.patch("modules.api.routes_detect.routes_utils.parse_form_data", side_effect=RuntimeError("boom")),
            mock.patch("modules.api.routes_detect.routes_utils.handle_error", return_value=("mapped", 500)),
        ):
            response = routes_client.post("/detect-language")

    assert response.status_code == 500
    body = json.loads(response.data)
    assert body["error"] == "mapped"


def test_await_shared_result_with_dashboard_sync_handles_future_exception():
    """Coalesced follower worker path should map leader future exceptions."""
    shared_future = concurrent.futures.Future()
    shared_future.set_exception(RuntimeError("leader-failed"))
    _sync_wait = routes_detect.__dict__["_await_shared_result_with_dashboard_task_sync"]

    with (
        mock.patch(
            "modules.api.routes_detect.model_manager.early_task_registration", return_value=contextlib.nullcontext()
        ),
        mock.patch("modules.api.routes_detect.routes_utils.handle_error", return_value=("Error", 500)),
    ):
        response = _sync_wait(shared_future, "local_path::/tmp/a.mp3", "a.mp3")

    assert response.status_code == 500
    body = json.loads(response.body)
    assert body["error"] == "Error"


def test_await_shared_result_with_dashboard_sync_handles_error_tuple():
    """Coalesced follower worker path should return err tuple payloads."""
    shared_future = concurrent.futures.Future()
    shared_future.set_result((None, ("boom", 500)))
    _sync_wait = routes_detect.__dict__["_await_shared_result_with_dashboard_task_sync"]

    with mock.patch(
        "modules.api.routes_detect.model_manager.early_task_registration", return_value=contextlib.nullcontext()
    ):
        response = _sync_wait(shared_future, "local_path::/tmp/a.mp3", "a.mp3")

    assert response.status_code == 500
    body = json.loads(response.body)
    assert body["error"] == "boom"


def test_await_shared_result_with_dashboard_sync_marks_failed_for_json_error():
    """Coalesced follower worker path should flag failed metadata for JSON error responses."""
    shared_future = concurrent.futures.Future()
    shared_future.set_result((JSONResponse(content={"error": "bad"}, status_code=500), None))
    _sync_wait = routes_detect.__dict__["_await_shared_result_with_dashboard_task_sync"]

    with (
        mock.patch(
            "modules.api.routes_detect.model_manager.early_task_registration", return_value=contextlib.nullcontext()
        ),
        mock.patch("modules.api.routes_detect.model_manager.update_task_metadata") as mock_update,
    ):
        response = _sync_wait(shared_future, "local_path::/tmp/a.mp3", "a.mp3")

    assert response.status_code == 500
    mock_update.assert_any_call(status="failed")


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

    with mock.patch("modules.api.routes_detect.logger.info"):
        _log_detection_result(result, 0.0)
