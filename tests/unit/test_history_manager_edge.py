"""
Edge case tests for the History Manager.
"""

import json
import threading
from unittest import mock

from modules.monitoring import history_manager


def test_history_manager_corrupt_file(tmp_path):
    """Test history manager behavior with corrupt JSON files."""
    corrupt_file = tmp_path / "corrupt_history.json"
    with open(corrupt_file, "w", encoding="utf-8") as f:
        f.write("invalid json")

    with mock.patch("modules.monitoring.history_manager.HISTORY_FILE", str(corrupt_file)):
        history_manager.HISTORY_CACHE = []
        history_manager.ensure_loaded()
        assert not history_manager.HISTORY_CACHE


def test_history_manager_migration_fallback(tmp_path):
    """Test history manager behavior when the history file is missing."""
    migration_file = tmp_path / "old_history.json"
    history = [{"task_id": "old"}]
    with open(migration_file, "w", encoding="utf-8") as f:
        json.dump(history, f)

    with mock.patch("modules.monitoring.history_manager.HISTORY_FILE", str(migration_file)):
        history_manager.HISTORY_CACHE = []
        history_manager.ensure_loaded()
        assert history_manager.HISTORY_CACHE == history


def test_history_manager_flush_error():
    """Test history manager resilience to flush errors."""
    with mock.patch("builtins.open", side_effect=OSError("Disk Full")):
        history_manager.HISTORY_CACHE = [{"data": 1}]
        # Should not raise exception
        history_manager.flush_history()


def test_analytics_lock_prevents_load_update_race(tmp_path):
    """Overlapping ensure_analytics_loaded and update_analytics should keep cache/file consistent."""
    analytics_file = tmp_path / "analytics_stats.json"
    with open(analytics_file, "w", encoding="utf-8") as f:
        json.dump({}, f)

    task = {
        "status": "completed",
        "video_duration": 3.0,
        "type": "Transcription",
        "completed_at": "2026-07-07 10:00:00",
        "request_json": {},
    }

    prev_analytics_cache = history_manager.ANALYTICS_CACHE
    prev_stats_cache = history_manager.STATS_CACHE
    prev_stats_cache_date = history_manager.STATS_CACHE_DATE

    with mock.patch("modules.monitoring.history_manager.ANALYTICS_FILE", str(analytics_file)):
        try:
            history_manager.ANALYTICS_CACHE = None
            history_manager.STATS_CACHE = None
            history_manager.STATS_CACHE_DATE = None

            def load_worker():
                history_manager.ensure_analytics_loaded()

            def update_worker():
                history_manager.update_analytics(task)

            t1 = threading.Thread(target=load_worker)
            t2 = threading.Thread(target=update_worker)
            t1.start()
            t2.start()
            t1.join(timeout=2)
            t2.join(timeout=2)

            assert not t1.is_alive()
            assert not t2.is_alive()
            assert history_manager.ANALYTICS_CACHE is not None
            with open(analytics_file, "r", encoding="utf-8") as f:
                persisted = json.load(f)
            assert persisted.get("2026-07-07", {}).get("count", 0) >= 1
        finally:
            history_manager.ANALYTICS_CACHE = prev_analytics_cache
            history_manager.STATS_CACHE = prev_stats_cache
            history_manager.STATS_CACHE_DATE = prev_stats_cache_date
