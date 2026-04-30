"""Tests for modules/monitoring/telemetry.py."""
import time
import threading
from unittest import mock
import pytest
from modules.monitoring import telemetry, metrics_discovery
from modules import config


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
    with mock.patch("modules.utils.get_system_telemetry", return_value={"cpu": 10}):
        with mock.patch("modules.monitoring.metrics_discovery.get_nvidia_metrics", return_value=[]):
            with mock.patch("modules.monitoring.metrics_discovery.get_intel_gpu_load", return_value=0):
                with mock.patch("modules.monitoring.metrics_discovery.get_npu_load", return_value=0):

                    # Ensure history is empty before start
                    telemetry.TELEMETRY_HISTORY.clear()

                    # Mock the loop condition to run exactly once, then stay set to True
                    # Use a side effect that doesn't exhaust
                    def side_effect(*args, **kwargs):
                        if not hasattr(side_effect, 'counter'):
                            side_effect.counter = 0
                        side_effect.counter += 1
                        return side_effect.counter > 1

                    with mock.patch.object(telemetry._STOP_EVENT, 'is_set', side_effect=side_effect):
                        telemetry._telemetry_worker()

                    # Use >= 1 because some background thread might have sneaked in if not properly stopped
                    # but with the clear() above it should be 1.
                    assert len(telemetry.TELEMETRY_HISTORY) >= 1
                    # Find our mocked entry
                    found = any(entry.get("system", {}).get("cpu") ==
                                10 for entry in telemetry.TELEMETRY_HISTORY)
                    assert found, f"Mocked CPU telemetry not found in history: {telemetry.TELEMETRY_HISTORY}"


def test_get_service_stats_structure(clean_telemetry):
    """Test that get_service_stats returns the expected schema."""
    with mock.patch("modules.monitoring.history_manager.get_history_stats", return_value=([], {})):
        with mock.patch("modules.monitoring.metrics_discovery.get_nvidia_metrics", return_value=[]):
            with mock.patch("modules.inference.model_manager.is_engine_actually_loaded", return_value=True):
                with mock.patch("modules.inference.model_manager.is_uvr_actually_loaded", return_value=False):
                    stats = telemetry.get_service_stats()

                    assert "version" in stats
                    assert "active_sessions" in stats
                    assert "tasks" in stats
                    assert "engines" in stats
                    assert stats["engines"]["whisper"]["status"] == "loaded"
                    assert stats["engines"]["uvr"]["status"] == "ready"


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
