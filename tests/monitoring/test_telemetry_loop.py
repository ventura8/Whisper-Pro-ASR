"""Tests for modules/monitoring/telemetry.py."""

import json
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from modules.monitoring import telemetry
from tests.monitoring._telemetry_test_helpers import get_service_stats_with_common_patches

_FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "task_ordering_fixture.json"


def _seed_task_registry(entries: dict[str, dict[str, Any]]) -> None:
    from modules.inference import scheduler

    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry.clear()
        scheduler.STATE.task_registry.update(entries)


def _task_orderings(count: int) -> list[list[str | None]]:
    orderings = []
    for _ in range(count):
        orderings.append([task.get("task_id") for task in telemetry.get_service_stats()["tasks"]])
    return orderings


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf"), 60.0, 0, None, "not-a-number"],
    ids=["nan", "inf", "-inf", "relative-offset", "zero", "none", "non-numeric"],
)
def test_resolve_epoch_timestamp_rejects_invalid_values(value: Any) -> None:
    """`_resolve_epoch_timestamp` underpins the stale-active-task reaper's timestamp
    handling; a bad value here would either wrongly reap a live task or wrongly hide
    a genuinely stale one. NaN/infinite values, missing values, non-numeric values, and
    values below `_EPOCH_TIMESTAMP_THRESHOLD` (relative/synthetic offsets, not real
    wall-clock epochs) must all resolve to None."""
    assert telemetry._resolve_epoch_timestamp(value) is None


def test_resolve_epoch_timestamp_accepts_valid_epoch_value() -> None:
    """A value at/above `_EPOCH_TIMESTAMP_THRESHOLD` is a real wall-clock epoch and
    must pass through unchanged."""
    assert telemetry._resolve_epoch_timestamp(200_000_000.0) == 200_000_000.0


def test_resolve_epoch_start_active_falls_back_to_start_time_when_start_active_invalid() -> None:
    """An invalid/relative `start_active` (e.g. a relative offset from a test fixture,
    or NaN) must not mask an otherwise-valid `start_time` -- the two are resolved
    independently, not via `or`, so this fallback is exercised directly."""
    now = 300_000_000.0

    task = {"start_active": 60.0, "start_time": 200_000_000.0}
    assert telemetry._resolve_epoch_start_active(task, now) == 200_000_000.0

    task_with_nan_start_active = {"start_active": float("nan"), "start_time": 200_000_000.0}
    assert telemetry._resolve_epoch_start_active(task_with_nan_start_active, now) == 200_000_000.0

    task_with_no_valid_timestamps = {"start_active": 60.0, "start_time": 60.0}
    assert telemetry._resolve_epoch_start_active(task_with_no_valid_timestamps, now) is None


def test_resolve_epoch_start_active_rejects_future_start_active_falls_back_to_start_time() -> None:
    """A `start_active` later than `now` (clock skew, a bad write) must be rejected
    -- not trusted -- since it would otherwise make `now - last_signal` negative
    and permanently hide staleness. A valid (non-future) `start_time` is used instead."""
    now = 300_000_000.0
    task = {"start_active": 400_000_000.0, "start_time": 200_000_000.0}
    assert telemetry._resolve_epoch_start_active(task, now) == 200_000_000.0

    task_both_future = {"start_active": 400_000_000.0, "start_time": 500_000_000.0}
    assert telemetry._resolve_epoch_start_active(task_both_future, now) is None


def test_telemetry_worker_unit(clean_telemetry: None):
    """Test a single execution of the telemetry worker logic."""
    with mock.patch("modules.core.config.TELEMETRY_RETENTION_HOURS", 0):
        with mock.patch("modules.core.utils.get_system_telemetry", return_value={"cpu": 10}):
            with mock.patch("modules.monitoring.metrics_discovery.get_nvidia_metrics", return_value=[]):
                with mock.patch("modules.monitoring.metrics_discovery.get_intel_gpu_load", return_value=0):
                    with mock.patch("modules.monitoring.metrics_discovery.get_npu_load", return_value=0):
                        # Seed with dummy entry so that appending makes len=2 > max_points=0, triggering pop(0)
                        telemetry.TELEMETRY_HISTORY.clear()
                        telemetry.TELEMETRY_HISTORY.append({"system": {"cpu": 5}})

                        # Mock the loop condition to run exactly once, then stay set to True
                        # Use a side effect that doesn't exhaust
                        def side_effect(*args, **kwargs):
                            if not hasattr(side_effect, "counter"):
                                side_effect.counter = 0
                            side_effect.counter += 1
                            return side_effect.counter > 1

                        with mock.patch.object(telemetry._STOP_EVENT, "is_set", side_effect=side_effect):
                            telemetry._telemetry_worker()

                        # Use >= 1 because some background thread might have sneaked in if not properly stopped
                        # but with the clear() above it should be 1.
                        assert len(telemetry.TELEMETRY_HISTORY) >= 1
                        # Find our mocked entry
                        found = any(entry.get("system", {}).get("cpu") == 10 for entry in telemetry.TELEMETRY_HISTORY)
                        assert found, f"Mocked CPU telemetry not found in history: {telemetry.TELEMETRY_HISTORY}"


def test_get_service_stats_structure(clean_telemetry: None):
    """Test that get_service_stats returns the expected schema."""
    _seed_task_registry(
        {
            "t1": {"status": "active", "stage": "transcribing", "unit_id": "CPU"},
            "t2": {"status": "active", "stage": "vocal isolation", "unit_id": "GPU"},
        }
    )

    stats = get_service_stats_with_common_patches(uvr_loaded=False)

    assert all(key in stats for key in ["version", "active_sessions", "tasks", "engines", "hardware_units"])
    assert stats["engines"]["whisper"]["status"] == "busy"
    assert stats["engines"]["uvr"]["status"] == "busy"
    assert all({"whisper_status", "uvr_status"}.issubset(unit) for unit in stats["hardware_units"])


def test_get_service_stats_tasks_sorted_by_start_time(clean_telemetry: None):
    """Tasks returned by telemetry should be ordered per task_status_display_specification_skill.

    Order: Active tasks first (by start_time), then all non-active tasks together
    by start_time (deterministic with task_id tie-breaker).
    """
    from modules.inference import scheduler

    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry.clear()
        # Insert out-of-order on purpose to verify three-tier sorting logic.
        scheduler.STATE.task_registry["standard_queued_2"] = {
            "task_id": "standard_queued_2",
            "status": "queued",
            "start_time": 300.0,
            "is_priority": False,
            "stage": "Waiting for Hardware",
        }
        scheduler.STATE.task_registry["active_2"] = {
            "task_id": "active_2",
            "status": "active",
            "start_time": 200.0,
            "stage": "Inference",
        }
        scheduler.STATE.task_registry["priority_queued"] = {
            "task_id": "priority_queued",
            "status": "queued",
            "start_time": 150.0,
            "is_priority": True,
            "stage": "Initializing",
        }
        scheduler.STATE.task_registry["active_1"] = {
            "task_id": "active_1",
            "status": "active",
            "start_time": 100.0,
            "is_priority": False,
            "stage": "Language Detection",
        }
        scheduler.STATE.task_registry["standard_queued_1"] = {
            "task_id": "standard_queued_1",
            "status": "queued",
            "start_time": 250.0,
            "is_priority": False,
            "stage": "Waiting for Hardware",
        }

    stats = get_service_stats_with_common_patches()

    # Expected order: active tasks first, then remaining tasks by start_time.
    task_order = [t.get("task_id") for t in stats["tasks"]]
    assert task_order == ["active_1", "active_2", "priority_queued", "standard_queued_1", "standard_queued_2"]


def test_task_ordering_deterministic_across_calls(clean_telemetry: None):
    """Verify /status returns active-first, then deterministic start_time/task_id ordering."""
    _seed_task_registry(
        {
            "t1": {"task_id": "t1", "status": "queued", "start_time": 100.0, "is_priority": False, "stage": "Stage 0"},
            "t2": {"task_id": "t2", "status": "active", "start_time": 150.0, "is_priority": False, "stage": "Stage 1"},
            "t3": {"task_id": "t3", "status": "queued", "start_time": 120.0, "is_priority": True, "stage": "Stage 2"},
            "t4": {"task_id": "t4", "status": "failed", "start_time": 90.0, "is_priority": True, "stage": "Stage 3"},
            "t0": {"task_id": "t0", "status": "queued", "start_time": 100.0, "is_priority": True, "stage": "Stage 4"},
        }
    )

    with mock.patch("modules.monitoring.history_manager.get_history_stats", return_value=([], {})):
        with mock.patch("modules.inference.runtime.model_manager.is_engine_actually_loaded", return_value=True):
            with mock.patch("modules.inference.runtime.model_manager.is_uvr_actually_loaded", return_value=True):
                orderings = _task_orderings(5)

    assert all(ordering == orderings[0] for ordering in orderings[1:])
    assert orderings[0] == ["t2", "t4", "t0", "t1", "t3"]


def _swap_tasks_by_id(tasks: list[dict[str, Any]], task_id_a: str, task_id_b: str) -> None:
    """Swap the list positions of the tasks with the given `task_id` values, in place."""
    idx_a = next(i for i, t in enumerate(tasks) if t["task_id"] == task_id_a)
    idx_b = next(i for i, t in enumerate(tasks) if t["task_id"] == task_id_b)
    tasks[idx_a], tasks[idx_b] = tasks[idx_b], tasks[idx_a]


def test_task_sort_key_matches_shared_fixture_active_order():
    """`_task_sort_key` must reproduce the shared fixture's expected active-task order.

    This fixture (tests/fixtures/task_ordering_fixture.json) is also consumed by the JS
    test suite (tests/js/dashboard_main.test.js) so that `_task_sort_key` (Python) and
    `_compareTaskOrder` (JS, modules/monitoring/templates/dashboard/features/active_tasks.js)
    are verified against IDENTICAL input data. A regression where the two comparators
    diverge will fail on at least one side of this shared fixture.
    """
    fixture = json.loads(_FIXTURE_PATH.read_text())
    tasks = [dict(task) for task in fixture["tasks"]]

    # task_a/task_b share start_time=100 and already appear in input in their
    # expected tie-broken output order (task_a before task_b) -- swap just
    # that pair here so a stable sort of the raw fixture order can't pass this
    # test without `_task_sort_key` actually applying the task_id tie-break.
    # The fixture JSON itself is left untouched since the JS suite
    # (tests/js/dashboard_main.test.js) consumes the same file directly.
    _swap_tasks_by_id(tasks, "task_a", "task_b")

    ordered = sorted(tasks, key=telemetry._task_sort_key)
    ordered_ids = [task["task_id"] for task in ordered]

    assert ordered_ids == fixture["expected_active_task_order"]


def test_get_minimal_stats():
    """Test health check stats."""
    _seed_task_registry(
        {
            "active_0": {"status": "active"},
            "active_1": {"status": "active"},
            "active_2": {"status": "active"},
            "active_3": {"status": "active"},
            "active_4": {"status": "active"},
            "queued_0": {"status": "queued"},
            "queued_1": {"status": "queued"},
        }
    )

    stats = telemetry.get_minimal_stats()
    assert stats["status"] == "healthy"
    assert stats["active"] == 5
    assert stats["queued"] == 2


def test_start_telemetry_loop(clean_telemetry: None):
    """Test starting the background loop."""
    with mock.patch("threading.Thread") as mock_thread:
        stop_event = telemetry.start_telemetry_loop()
        assert stop_event == telemetry._STOP_EVENT
        mock_thread.assert_called_once()
        assert mock_thread.call_args[1]["target"] == telemetry._telemetry_worker


def test_get_service_stats_normalizes_none_like_stage_values(clean_telemetry: None):
    """Dashboard task stage must never be None-like in API payload."""
    from modules.inference import scheduler

    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry.clear()
        scheduler.STATE.task_registry["none_stage"] = {
            "task_id": "none_stage",
            "status": "active",
            "stage": None,
            "start_time": 1.0,
        }
        scheduler.STATE.task_registry["string_none_stage"] = {
            "task_id": "string_none_stage",
            "status": "queued",
            "stage": "None",
            "start_time": 2.0,
        }

    stats = get_service_stats_with_common_patches()

    task_by_id = {t.get("task_id"): t for t in stats["tasks"]}
    assert task_by_id["none_stage"]["stage"] == "Active"
    assert task_by_id["string_none_stage"]["stage"] == "Queued"


def test_get_service_stats_blocks_placeholder_status_and_stage_values(clean_telemetry: None):
    """Dashboard payload must never expose placeholder-like status/stage values."""
    _seed_task_registry(
        {
            "placeholder_task": {
                "task_id": "placeholder_task",
                "status": "unknown",
                "stage": "resuming",
                "start_time": 10.0,
            },
            "ratio_placeholder": {
                "task_id": "ratio_placeholder",
                "status": None,
                "stage": "(0/0)",
                "start_time": 20.0,
            },
        }
    )

    stats = get_service_stats_with_common_patches()

    task_by_id = {task.get("task_id"): task for task in stats["tasks"]}
    assert {
        "placeholder_task": ("initializing", "Initializing"),
        "ratio_placeholder": ("initializing", "Initializing"),
    } == {task_id: (task["status"], task["stage"]) for task_id, task in task_by_id.items()}
