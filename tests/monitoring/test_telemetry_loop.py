"""Tests for modules/monitoring/telemetry.py."""

from unittest import mock

import pytest

from modules.monitoring import telemetry


@pytest.fixture
def clean_telemetry():
    """Reset telemetry history and ensure background loop is stopped."""
    telemetry._STOP_EVENT.set()
    telemetry.TELEMETRY_HISTORY.clear()
    telemetry._STOP_EVENT.clear()
    yield
    telemetry._STOP_EVENT.set()


def test_telemetry_worker_unit(clean_telemetry):
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


def test_get_service_stats_structure(clean_telemetry):
    """Test that get_service_stats returns the expected schema."""
    from modules.inference import model_manager, scheduler

    mock_units = [
        {"id": "CPU", "type": "CPU", "name": "CPU"},
        {"id": "GPU", "type": "GPU", "name": "GPU"},
        {"id": "NPU", "type": "NPU", "name": "NPU"},
        {"id": "AUTO", "type": "AUTO", "name": "AUTO"},
    ]

    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry.clear()
        scheduler.STATE.task_registry["t1"] = {"status": "active", "stage": "transcribing", "unit_id": "CPU"}
        scheduler.STATE.task_registry["t2"] = {"status": "active", "stage": "vocal isolation", "unit_id": "GPU"}

    mock_preprocessor = mock.MagicMock()
    mock_preprocessor.separator = mock.MagicMock()

    with mock.patch("modules.core.config.HARDWARE_UNITS", mock_units):
        with mock.patch.dict(model_manager.MODEL_POOL, {"NPU": mock.MagicMock()}):
            with mock.patch.dict(model_manager.PREPROCESSOR_POOL, {"NPU": mock_preprocessor}):
                with mock.patch("modules.monitoring.history_manager.get_history_stats", return_value=([], {})):
                    with mock.patch("modules.monitoring.metrics_discovery.get_nvidia_metrics", return_value=[]):
                        with mock.patch("modules.inference.model_manager.is_engine_actually_loaded", return_value=True):
                            with mock.patch(
                                "modules.inference.model_manager.is_uvr_actually_loaded", return_value=False
                            ):
                                stats = telemetry.get_service_stats()

                                assert "version" in stats
                                assert "active_sessions" in stats
                                assert "tasks" in stats
                                assert "engines" in stats
                                assert stats["engines"]["whisper"]["status"] == "busy"
                                assert stats["engines"]["uvr"]["status"] == "busy"
                                assert "hardware_units" in stats
                                for u in stats["hardware_units"]:
                                    assert "whisper_status" in u
                                    assert "uvr_status" in u


def test_get_service_stats_tasks_sorted_by_start_time(clean_telemetry):
    """Tasks returned by telemetry should be ordered per task_status_display_specification_skill.

    Order: Active tasks first (by start_time), then priority-queued (by start_time),
    then standard-queued (by start_time).
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

    with mock.patch("modules.monitoring.history_manager.get_history_stats", return_value=([], {})):
        with mock.patch("modules.inference.model_manager.is_engine_actually_loaded", return_value=True):
            with mock.patch("modules.inference.model_manager.is_uvr_actually_loaded", return_value=True):
                stats = telemetry.get_service_stats()

    # Expected order: active tasks (100, 200), priority queued (150), standard queued (250, 300)
    task_order = [t.get("task_id") for t in stats["tasks"]]
    assert task_order == ["active_1", "active_2", "priority_queued", "standard_queued_1", "standard_queued_2"]


def test_task_ordering_deterministic_across_calls(clean_telemetry):
    """Verify /status endpoint returns tasks in deterministic order across multiple calls."""
    from modules.inference import scheduler

    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry.clear()
        for i, (tid, status, start, is_prio) in enumerate(
            [
                ("t1", "queued", 100.0, False),
                ("t2", "active", 150.0, False),
                ("t3", "queued", 120.0, True),
            ]
        ):
            scheduler.STATE.task_registry[tid] = {
                "task_id": tid,
                "status": status,
                "start_time": start,
                "is_priority": is_prio,
                "stage": f"Stage {i}",
            }

    with mock.patch("modules.monitoring.history_manager.get_history_stats", return_value=([], {})):
        with mock.patch("modules.inference.model_manager.is_engine_actually_loaded", return_value=True):
            with mock.patch("modules.inference.model_manager.is_uvr_actually_loaded", return_value=True):
                # Call get_service_stats multiple times
                orderings = []
                for _ in range(5):
                    stats = telemetry.get_service_stats()
                    orderings.append([t.get("task_id") for t in stats["tasks"]])

    # All orderings should be identical
    for i in range(1, 5):
        assert orderings[i] == orderings[0], f"Ordering diverged at call {i}"

    # Verify expected order (t2 active first, then t3 priority-queued, then t1 standard-queued)
    assert orderings[0] == ["t2", "t3", "t1"]


def test_get_minimal_stats():
    """Test health check stats."""
    from modules.inference import scheduler

    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry.clear()
        # Mock 5 active tasks
        for i in range(5):
            scheduler.STATE.task_registry[f"active_{i}"] = {"status": "active"}
        # Mock 2 queued tasks
        for i in range(2):
            scheduler.STATE.task_registry[f"queued_{i}"] = {"status": "queued"}

    stats = telemetry.get_minimal_stats()
    assert stats["status"] == "healthy"
    assert stats["active"] == 5
    assert stats["queued"] == 2


def test_start_telemetry_loop(clean_telemetry):
    """Test starting the background loop."""
    with mock.patch("threading.Thread") as mock_thread:
        stop_event = telemetry.start_telemetry_loop()
        assert stop_event == telemetry._STOP_EVENT
        mock_thread.assert_called_once()
        assert mock_thread.call_args[1]["target"] == telemetry._telemetry_worker


def test_get_service_stats_normalizes_none_like_stage_values(clean_telemetry):
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

    with mock.patch("modules.monitoring.history_manager.get_history_stats", return_value=([], {})):
        with mock.patch("modules.inference.model_manager.is_engine_actually_loaded", return_value=True):
            with mock.patch("modules.inference.model_manager.is_uvr_actually_loaded", return_value=True):
                stats = telemetry.get_service_stats()

    task_by_id = {t.get("task_id"): t for t in stats["tasks"]}
    assert task_by_id["none_stage"]["stage"] == "Active"
    assert task_by_id["string_none_stage"]["stage"] == "Queued"


def test_get_service_stats_blocks_placeholder_status_and_stage_values(clean_telemetry):
    """Dashboard payload must never expose placeholder-like status/stage values."""
    from modules.inference import scheduler

    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry.clear()
        scheduler.STATE.task_registry["placeholder_task"] = {
            "task_id": "placeholder_task",
            "status": "unknown",
            "stage": "resuming",
            "start_time": 10.0,
        }
        scheduler.STATE.task_registry["ratio_placeholder"] = {
            "task_id": "ratio_placeholder",
            "status": None,
            "stage": "(0/0)",
            "start_time": 20.0,
        }

    with mock.patch("modules.monitoring.history_manager.get_history_stats", return_value=([], {})):
        with mock.patch("modules.inference.model_manager.is_engine_actually_loaded", return_value=True):
            with mock.patch("modules.inference.model_manager.is_uvr_actually_loaded", return_value=True):
                stats = telemetry.get_service_stats()

    task_by_id = {t.get("task_id"): t for t in stats["tasks"]}

    assert task_by_id["placeholder_task"]["status"] == "initializing"
    assert task_by_id["placeholder_task"]["stage"] == "Initializing"
    assert task_by_id["ratio_placeholder"]["status"] == "initializing"
    assert task_by_id["ratio_placeholder"]["stage"] == "Initializing"
