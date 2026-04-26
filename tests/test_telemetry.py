
import os
import json
import pytest
from unittest import mock
from modules import telemetry_manager


@pytest.fixture
def mock_telemetry_file(tmp_path):
    temp_file = tmp_path / "telemetry_history.json"
    with mock.patch("modules.telemetry_manager.TELEMETRY_FILE", str(temp_file)):
        yield temp_file


def test_get_telemetry_history_empty(mock_telemetry_file):
    assert telemetry_manager.get_telemetry_history() == []


def test_record_snapshot(mock_telemetry_file):
    stats = {
        'system': {
            'cpu_percent': 10.0,
            'app_cpu_percent': 5.0,
            'memory_percent': 50.0,
            'app_memory_gb': 1.0
        },
        'telemetry': {
            'nvidia': [{'util': 20}],
            'intel_gpu_load': 10,
            'npu_load': 5
        }
    }
    telemetry_manager.record_snapshot(stats)

    history = telemetry_manager.get_telemetry_history()
    assert len(history) == 1
    assert history[0]['cpu_sys'] == 10.0
    assert history[0]['nvidia_util'] == [20]


def test_record_snapshot_pruning(mock_telemetry_file):
    # Setup history with an old snapshot
    old_time = int(telemetry_manager.time.time()) - 100000  # ~27 hours ago
    history = [{"timestamp": old_time, "cpu_sys": 1.0}]
    with open(mock_telemetry_file, 'w') as f:
        json.dump(history, f)

    stats = {
        'system': {'cpu_percent': 10.0, 'app_cpu_percent': 5.0, 'memory_percent': 50.0, 'app_memory_gb': 1.0},
        'telemetry': {}
    }
    # With default 24h retention, the old snapshot should be pruned
    telemetry_manager.record_snapshot(stats)

    new_history = telemetry_manager.get_telemetry_history()
    assert len(new_history) == 1
    assert new_history[0]['cpu_sys'] == 10.0


def test_record_snapshot_limit(mock_telemetry_file):
    history = [{"timestamp": int(telemetry_manager.time.time()),
                "cpu_sys": float(i)} for i in range(2005)]
    with open(mock_telemetry_file, 'w') as f:
        json.dump(history, f)

    stats = {
        'system': {'cpu_percent': 99.0, 'app_cpu_percent': 5.0, 'memory_percent': 50.0, 'app_memory_gb': 1.0},
        'telemetry': {}
    }
    telemetry_manager.record_snapshot(stats)

    new_history = telemetry_manager.get_telemetry_history()
    assert len(new_history) == 2000
    assert new_history[-1]['cpu_sys'] == 99.0


def test_update_retention():
    telemetry_manager.update_retention(telemetry_hours=48, log_days=14)
    assert os.environ["TELEMETRY_RETENTION_HOURS"] == "48"
    assert os.environ["LOG_RETENTION_DAYS"] == "14"


def test_get_telemetry_history_corrupt(mock_telemetry_file):
    with open(mock_telemetry_file, 'w') as f:
        f.write("corrupt json")
    assert telemetry_manager.get_telemetry_history() == []
