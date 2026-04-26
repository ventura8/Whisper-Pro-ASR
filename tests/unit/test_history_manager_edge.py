"""
Edge case tests for the History Manager.
"""
# pylint: disable=protected-access
import json
from unittest import mock
from modules.monitoring import history_manager


def test_history_manager_corrupt_file(tmp_path):
    """Test history manager behavior with corrupt JSON files."""
    corrupt_file = tmp_path / "corrupt_history.json"
    with open(corrupt_file, "w", encoding="utf-8") as f:
        f.write("invalid json")

    with mock.patch("modules.monitoring.history_manager.HISTORY_FILE", str(corrupt_file)):
        history_manager._HISTORY_CACHE = []
        history_manager._ensure_loaded()
        assert history_manager._HISTORY_CACHE == []


def test_history_manager_migration_fallback(tmp_path):
    """Test history manager behavior when the history file is missing."""
    migration_file = tmp_path / "old_history.json"
    history = [{"task_id": "old"}]
    with open(migration_file, "w", encoding="utf-8") as f:
        json.dump(history, f)

    with mock.patch("modules.monitoring.history_manager.HISTORY_FILE", str(migration_file)):
        history_manager._HISTORY_CACHE = []
        history_manager._ensure_loaded()
        assert history_manager._HISTORY_CACHE == history


def test_history_manager_flush_error():
    """Test history manager resilience to flush errors."""
    with mock.patch("builtins.open", side_effect=OSError("Disk Full")):
        history_manager._HISTORY_CACHE = [{"data": 1}]
        # Should not raise exception
        history_manager.flush_history()
