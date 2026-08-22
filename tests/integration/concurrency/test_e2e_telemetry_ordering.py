"""End-to-end tests for status polling, telemetry invariants, and deterministic task ordering during concurrency."""

from __future__ import annotations

import concurrent.futures
import threading
from collections.abc import Callable
from typing import Any
from unittest import mock

from modules.inference import scheduler
from modules.inference.runtime import model_manager
from modules.monitoring import telemetry
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


def _snapshot_active_unit_assignment() -> dict[str, Any]:
    """Read the scheduler's ground truth: the single currently-active task's real unit_id/stage.

    Returns a shallow copy of the selected task dict taken while the lock is
    held, so callers receive an immutable-in-time snapshot rather than the live
    registry entry (which a concurrent writer could mutate after release).
    """
    with scheduler.STATE.task_registry_lock:
        active_tasks = [t for t in scheduler.STATE.task_registry.values() if t.get("status") == "active"]
        assert len(active_tasks) == 1, f"expected exactly one active task, found {active_tasks}"
        return dict(active_tasks[0])


def _assert_unit_state_matches(
    uid: str,
    expected_busy: bool,
    assigned_unit_id: str,
    hw_by_id: dict[str, Any],
    source: str,
) -> None:
    """Assert a single hardware unit's reported busy/idle state (from the given
    telemetry `source`) matches the scheduler's ground-truth assignment."""
    assert uid in hw_by_id, f"hardware unit {uid} missing from {source} telemetry"
    assert (hw_by_id[uid]["whisper_status"] == "busy") is expected_busy, (
        f"{source} hardware_units drifted from scheduler assignment for {uid}: "
        f"scheduler assigned={assigned_unit_id}, telemetry={hw_by_id[uid]['whisper_status']}"
    )


def _assert_hardware_units_match_scheduler_assignment(
    known_unit_ids: set[str],
    assigned_unit_id: str,
    status_hw_by_id: dict[str, Any],
    direct_hw_by_id: dict[str, Any],
) -> None:
    """Cross-check every hardware unit's reported busy/idle state, from both the
    /status endpoint and telemetry.get_service_stats() directly, against the
    scheduler's ground-truth unit assignment."""
    for uid in known_unit_ids:
        expected_busy = uid == assigned_unit_id
        _assert_unit_state_matches(uid, expected_busy, assigned_unit_id, status_hw_by_id, "/status")
        _assert_unit_state_matches(uid, expected_busy, assigned_unit_id, direct_hw_by_id, "get_service_stats()")


def _make_gated_transcription(arrived: threading.Event, release_event: threading.Event) -> Callable[..., dict[str, Any]]:
    """Build a run_transcription stand-in that acquires a real unit via
    model_lock_ctx, marks the Inference stage (mirroring the real pipeline),
    signals `arrived`, then blocks on `release_event` -- so the test can
    observe/cross-check state while the task is gated mid-run."""

    def gated_transcription(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        with model_manager.model_lock_ctx(priority=False):
            scheduler.update_task_progress(None, "Inference")
            arrived.set()
            release_event.wait(timeout=30.0)
        return {"text": "hello", "segments": []}

    return gated_transcription


def _crosscheck_telemetry_against_scheduler(client: Any, known_unit_ids: set[str]) -> str:
    """Read the scheduler's ground-truth unit assignment for the in-flight task
    and cross-check it against both the /status endpoint and
    telemetry.get_service_stats(). Returns the assigned unit id."""
    scheduler_ground_truth = _snapshot_active_unit_assignment()
    assigned_unit_id = scheduler_ground_truth["unit_id"]
    assert assigned_unit_id in known_unit_ids

    # Cross-check via the public /status endpoint (what the dashboard actually reads)...
    status_resp = client.get("/status")
    assert status_resp.status_code == 200
    status_hw_by_id = {u["id"]: u for u in status_resp.get_json()["hardware_units"]}

    # ...and directly via telemetry.get_service_stats() in the very same process/run.
    direct_stats = telemetry.get_service_stats()
    direct_hw_by_id = {u["id"]: u for u in direct_stats["hardware_units"]}

    _assert_hardware_units_match_scheduler_assignment(known_unit_ids, assigned_unit_id, status_hw_by_id, direct_hw_by_id)
    return assigned_unit_id


def test_telemetry_hardware_state_crosschecks_scheduler_unit_assignment(sample_wav: str):
    """Gap 7.3: telemetry's reported hardware-unit busy/idle state must match what the
    scheduler *actually did* in the same run (scheduler.STATE.task_registry unit_id
    assignment), not merely be internally self-consistent with itself.

    Drives a real in-flight task through model_lock_ctx (the same code path
    concurrency.py uses to acquire a hardware unit and record unit_id/status on the
    task registry), reads the scheduler's ground-truth assignment while the task is
    gated mid-run, and cross-checks it against telemetry.get_service_stats() /
    the /status endpoint's hardware_units payload for the exact same moment.
    """
    with run_concurrency_test_harness(HW_TOPOLOGY_2_DUAL) as client:
        release_event = threading.Event()
        arrived = threading.Event()
        gated_transcription = _make_gated_transcription(arrived, release_event)

        with mock.patch("modules.api.routes.asr.model_manager.run_transcription", side_effect=gated_transcription):
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future = executor.submit(dispatch_single_request, client, {"endpoint": "/asr", "local_path": sample_wav})
                try:
                    assert arrived.wait(timeout=20.0)
                    known_unit_ids = {unit["id"] for unit in HW_TOPOLOGY_2_DUAL}
                    _crosscheck_telemetry_against_scheduler(client, known_unit_ids)
                finally:
                    release_event.set()
                result = future.result(timeout=30)
                assert result["status_code"] == 200
