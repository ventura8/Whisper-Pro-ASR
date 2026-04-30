"""Tests for modules/monitoring/history_manager.py."""
import os
import json
import time
from unittest import mock
import pytest
from modules.monitoring import history_manager
from modules import config


@pytest.fixture
def clean_history_cache(tmp_path):
    """Reset history cache and use a temporary file."""
    history_manager._HISTORY_CACHE = []
    history_manager._STATS_CACHE = None

    # Use tmp_path for persistent file
    temp_file = tmp_path / "task_history.json"
    with mock.patch("modules.monitoring.history_manager.HISTORY_FILE", str(temp_file)):
        yield temp_file


def test_log_completed_task(clean_history_cache):
    """Test logging a task to history."""
    task_data = {
        "task_id": "123",
        "video_duration": 120,
        "start_time": time.time() - 10
    }
    history_manager.log_completed_task(task_data)

    history = history_manager.get_history()
    assert len(history) == 1
    assert history[0]["task_id"] == "123"
    assert history[0]["total_elapsed_sec"] >= 10
    assert "completed_at" in history[0]


def test_history_stats(clean_history_cache):
    """Test history stats calculation."""
    # Log two tasks for today
    history_manager.log_completed_task({"task_id": "1", "video_duration": 60})
    history_manager.log_completed_task({"task_id": "2", "video_duration": 40})

    history, stats = history_manager.get_history_stats()
    assert stats["count_all_time"] == 2
    assert stats["all_time"] == 100
    assert stats["today"] == 100


def test_history_persistence(clean_history_cache):
    """Test that history is saved to SSD and reloaded."""
    history_manager.log_completed_task({"task_id": "p1", "video_duration": 50})

    # Force reload by clearing cache
    history_manager._HISTORY_CACHE = []
    history_manager._STATS_CACHE = None

    history = history_manager.get_history()
    assert len(history) == 1
    assert history[0]["task_id"] == "p1"


def test_history_limit(clean_history_cache):
    """Test that history is limited to MAX_HISTORY_DISK."""
    with mock.patch("modules.monitoring.history_manager.MAX_HISTORY_DISK", 2), \
            mock.patch("modules.monitoring.history_manager.MAX_HISTORY_RAM", 2):
        history_manager.log_completed_task({"task_id": "1"})
        history_manager.log_completed_task({"task_id": "2"})
        history_manager.log_completed_task({"task_id": "3"})

        history = history_manager.get_history()
        assert len(history) == 2
        assert history[0]["task_id"] == "3"


def test_ensure_loaded_corrupt(clean_history_cache):
    """Test resilience to corrupt JSON on SSD."""
    with open(clean_history_cache, 'w') as f:
        f.write("corrupt")

    history_manager._HISTORY_CACHE = []
    history_manager._ensure_loaded()
    assert history_manager._HISTORY_CACHE == []


def test_history_manager_exceptions():
    """Cover exception handling in log_completed_task."""
    # Passing None to a dict operation should trigger TypeError
    history_manager.log_completed_task(None)


def test_history_manager_stats_cache():
    """Cover stats cache hit branch."""
    history_manager._STATS_CACHE = {"cached": True}
    history, stats = history_manager.get_history_stats()
    assert stats["cached"] is True
    history_manager._STATS_CACHE = None


def test_history_manager_clear_logic(tmp_path):
    """Cover clear_history and disk removal failure."""
    history_file = tmp_path / "test_history.json"
    history_file.write_text("[]")

    with mock.patch("modules.monitoring.history_manager.HISTORY_FILE", str(history_file)):
        history_manager.clear_history()
        assert not os.path.exists(str(history_file))

    # Test removal failure (OSError)
    with mock.patch("os.path.exists", return_value=True), \
            mock.patch("os.remove", side_effect=OSError("Permission denied")):
        history_manager.clear_history()


def test_history_manager_stats_aggregation():
    """Cover history stats logic with actual aggregation."""
    history = [{"video_duration": 10, "task_id": "1"}, {"video_duration": 20, "task_id": "2"}]
    with mock.patch("modules.monitoring.history_manager.get_history", return_value=history):
        history_manager._STATS_CACHE = None
        _, stats = history_manager.get_history_stats()
        assert stats["all_time"] == 30
