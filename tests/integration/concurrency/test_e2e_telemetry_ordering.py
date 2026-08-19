"""End-to-end tests for status polling, telemetry invariants, and deterministic task ordering during concurrency."""

from __future__ import annotations

import concurrent.futures
import threading
from typing import Any
from unittest import mock

from modules.inference import scheduler
from tests.conftest import FlaskCompatibleClient
from tests.integration.concurrency.concurrency_fixtures import (
    HW_TOPOLOGY_2_DUAL,
    HW_TOPOLOGY_4_QUAD,
    assert_all_responses_successful,
    dispatch_single_request,
    execute_concurrent_workload,
    run_concurrency_test_harness,
)

VALID_TASK_STATUSES = {"initializing", "queued", "active", "post-processing", "completed", "failed"}
FORBIDDEN_PLACEHOLDERS = {"unknown", "none", "null", "undefined", "(0/0)"}


def _build_telemetry_specs(test_wav: str, count: int = 3) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    for _ in range(count):
        specs.append({"endpoint": "/asr", "local_path": test_wav})
        specs.append({"endpoint": "/detect-language", "local_path": test_wav})
    return specs


def _validate_status_field(status: str | None) -> None:
    if status is not None:
        assert status in VALID_TASK_STATUSES
        assert status.lower() not in FORBIDDEN_PLACEHOLDERS


def _validate_stage_field(stage: str | None) -> None:
    if stage is not None:
        assert stage.lower() not in FORBIDDEN_PLACEHOLDERS


def _validate_single_task_entry(task: dict[str, Any]) -> None:
    _validate_status_field(task.get("status"))
    _validate_stage_field(task.get("stage"))


def _assert_status_payload_valid(data: dict[str, Any]) -> None:
    assert "active_sessions" in data
    assert "engines" in data
    assert "scheduler" in data
    for task in data.get("tasks", []):
        _validate_single_task_entry(task)


def test_telemetry_status_polling_invariants_during_concurrency(sample_wav: str):
    """Verify real-time /status polling during concurrent bursts returns valid canonical enums without placeholders."""
    with run_concurrency_test_harness(HW_TOPOLOGY_2_DUAL) as client:
        specs = _build_telemetry_specs(sample_wav, 3)
        release_event = threading.Event()
        arrived_lock = threading.Lock()
        arrived_count = 0
        all_arrived = threading.Event()
        hw_units = len(HW_TOPOLOGY_2_DUAL)

        def gated_transcription(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            nonlocal arrived_count
            with arrived_lock:
                arrived_count += 1
                if arrived_count >= hw_units:
                    all_arrived.set()
            release_event.wait(timeout=30.0)
            return {"text": "hello", "segments": []}

        def gated_voting(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            nonlocal arrived_count
            with arrived_lock:
                arrived_count += 1
                if arrived_count >= hw_units:
                    all_arrived.set()
            release_event.wait(timeout=30.0)
            return {"detected_language": "en", "confidence": 0.95}

        with (
            mock.patch("modules.api.routes.asr.model_manager.run_transcription", side_effect=gated_transcription),
            mock.patch(
                "modules.inference.pipeline.language_detection.run_voting_detection",
                side_effect=gated_voting,
            ),
        ):
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                futures = [executor.submit(dispatch_single_request, client, s) for s in specs]
                try:
                    assert all_arrived.wait(timeout=20.0)
                    status_resp = client.get("/status")
                    assert status_resp.status_code == 200
                    _assert_status_payload_valid(status_resp.get_json())
                    workload_resp = [f.result() for f in futures]
                    assert_all_responses_successful(workload_resp)
                finally:
                    release_event.set()


def _fetch_single_order(client: FlaskCompatibleClient) -> list[str]:
    resp = client.get("/status")
    assert resp.status_code == 200
    tasks = resp.get_json().get("tasks", [])
    return [t.get("task_id", "") for t in tasks]


def _assert_stable_orders(orders: list[list[str]]) -> None:
    assert any(len(order) > 0 for order in orders)
    assert all(order == orders[0] for order in orders)


def test_telemetry_ordering_stability_across_calls(sample_wav: str):
    """Verify consecutive /status calls maintain stable, deterministic task ordering."""
    with run_concurrency_test_harness(HW_TOPOLOGY_2_DUAL) as client:
        scheduler.STATE.task_registry["order-task-1"] = {
            "task_id": "order-task-1",
            "task_type": "ASR Task 1",
            "status": "active",
            "stage": "Inference",
            "is_priority": False,
            "unit_id": "NPU.0",
            "created_at": 100.0,
            "start_time": 100.0,
        }
        scheduler.STATE.task_registry["order-task-2"] = {
            "task_id": "order-task-2",
            "task_type": "LD Task 2",
            "status": "active",
            "stage": "Language Detection",
            "is_priority": True,
            "unit_id": "GPU.0",
            "created_at": 101.0,
            "start_time": 101.0,
        }
        try:
            orders = [_fetch_single_order(client) for _ in range(3)]
            _assert_stable_orders(orders)
        finally:
            scheduler.STATE.task_registry.pop("order-task-1", None)
            scheduler.STATE.task_registry.pop("order-task-2", None)

        specs = [{"endpoint": "/asr", "local_path": sample_wav} for _ in range(4)]
        responses = execute_concurrent_workload(client, specs)
        assert_all_responses_successful(responses)


def _assert_dashboard_html(client: Any) -> None:
    """Assert dashboard HTML route responds with expected markup."""
    dashboard_html_resp = client.get("/dashboard")
    assert dashboard_html_resp.status_code == 200
    assert b"Whisper Pro Dashboard" in dashboard_html_resp.data
    assert b"stats-grid" in dashboard_html_resp.data


def _assert_status_payload(client: Any) -> None:
    """Assert status JSON route responds with active session metrics."""
    status_resp = client.get("/status")
    assert status_resp.status_code == 200
    status_data = status_resp.get_json()
    assert "active_sessions" in status_data
    assert "hardware_units" in status_data


def _assert_dashboard_html_and_status(client: Any) -> None:
    """Assert dashboard HTML and status JSON endpoints return valid content."""
    _assert_dashboard_html(client)
    _assert_status_payload(client)


def _run_concurrent_dashboard_specs(client: Any, sample_wav: str) -> None:
    """Execute concurrent requests while sampling dashboard endpoints."""
    specs = [
        {"endpoint": "/asr", "local_path": sample_wav},
        {"endpoint": "/detect-language", "local_path": sample_wav},
        {"endpoint": "/v1/audio/transcriptions", "local_path": sample_wav},
        {"endpoint": "/v1/audio/translations", "local_path": sample_wav},
    ]
    release_event = threading.Event()
    arrived_lock = threading.Lock()
    arrived_count = 0
    all_arrived = threading.Event()
    expected_requests = len(specs)

    def gated_transcription(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal arrived_count
        with arrived_lock:
            arrived_count += 1
            if arrived_count >= expected_requests:
                all_arrived.set()
        release_event.wait(timeout=30.0)
        return {"text": "hello", "segments": []}

    def gated_voting(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal arrived_count
        with arrived_lock:
            arrived_count += 1
            if arrived_count >= expected_requests:
                all_arrived.set()
        release_event.wait(timeout=30.0)
        return {"detected_language": "en", "confidence": 0.95}

    with (
        mock.patch("modules.api.routes.asr.model_manager.run_transcription", side_effect=gated_transcription),
        mock.patch(
            "modules.inference.pipeline.language_detection.run_voting_detection",
            side_effect=gated_voting,
        ),
    ):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(dispatch_single_request, client, s) for s in specs]
            try:
                assert all_arrived.wait(timeout=20.0)
                _assert_dashboard_html_and_status(client)
                results = [f.result() for f in futures]
                assert_all_responses_successful(results)
            finally:
                release_event.set()


def test_dashboard_ui_html_rendering_during_concurrency(sample_wav: str):
    """Verify dashboard UI HTML template and status telemetry rendering during live multi-task concurrency."""
    with run_concurrency_test_harness(HW_TOPOLOGY_4_QUAD, confidence=0.99) as client:
        _run_concurrent_dashboard_specs(client, sample_wav)
