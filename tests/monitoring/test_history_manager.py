"""Tests for modules/monitoring/history_manager.py."""

import datetime
import json
import os
import time
from unittest import mock

import pytest

from modules.monitoring import history_manager


@pytest.fixture(autouse=True)
def reset_history_cache(tmp_path):
    """Reset history cache and use a temporary file for every test."""
    history_manager.HISTORY_CACHE = []
    history_manager.ANALYTICS_CACHE = None
    history_manager.STATS_CACHE = None

    # Use tmp_path for persistent file
    temp_file = tmp_path / "task_history.json"
    temp_analytics_file = tmp_path / "analytics_stats.json"
    with (
        mock.patch("modules.monitoring.history_manager.HISTORY_FILE", str(temp_file)),
        mock.patch("modules.monitoring.history_manager.ANALYTICS_FILE", str(temp_analytics_file)),
    ):
        yield temp_file


def test_log_completed_task():
    """Test logging a task to history."""
    task_data = {
        "task_id": "123",
        "type": "Transcription",
        "video_duration": 120,
        "start_time": time.time() - 10,
        "result": {"segments": [{"start": 0, "end": 10, "text": "Hello"}, {"start": 10, "end": 20, "text": "World"}]},
    }
    history_manager.log_completed_task(task_data)

    task_history = history_manager.get_history()
    assert len(task_history) == 1
    assert task_history[0]["task_id"] == "123"
    assert task_history[0]["total_elapsed_sec"] >= 10
    assert task_history[0]["segments_processed"] == 2
    assert "completed_at" in task_history[0]

    # Test language detection type
    ld_data = {
        "task_id": "456",
        "type": "Language Detection",
        "start_time": time.time() - 2,
        "result": {"segments_processed": 5},
    }
    history_manager.log_completed_task(ld_data)
    task_history = history_manager.get_history()
    assert task_history[0]["task_id"] == "456"
    assert task_history[0]["segments_processed"] == 5


def test_history_stats():
    """Test history stats calculation."""
    # Log tasks with different types/endpoints
    history_manager.log_completed_task({"task_id": "1", "video_duration": 60.0, "endpoint": "/asr"})
    history_manager.log_completed_task({"task_id": "2", "video_duration": 40.0, "endpoint": "/detect-language"})
    history_manager.log_completed_task({"task_id": "3", "video_duration": 50.0, "endpoint": "/v1/audio/transcriptions"})

    _history, stats = history_manager.get_history_stats()
    assert stats["count_all_time"] == 3
    assert stats["all_time"] == 150.0
    assert stats["today"] == 150.0
    assert stats["asr"]["count"] == 1
    assert stats["asr"]["duration"] == 60.0
    assert stats["detectlang"]["count"] == 1
    assert stats["detectlang"]["duration"] == 40.0
    assert stats["audio"]["count"] == 1
    assert stats["audio"]["duration"] == 50.0


def test_history_persistence():
    """Test that history is saved to SSD and reloaded."""
    history_manager.log_completed_task({"task_id": "p1", "video_duration": 50})

    # Force reload by clearing cache
    history_manager.HISTORY_CACHE = []
    history_manager.STATS_CACHE = None

    task_history = history_manager.get_history()
    assert len(task_history) == 1
    assert task_history[0]["task_id"] == "p1"


def test_history_limit():
    """Test that history is limited to MAX_HISTORY_DISK."""
    with (
        mock.patch("modules.monitoring.history_manager.MAX_HISTORY_DISK", 2),
        mock.patch("modules.monitoring.history_manager.MAX_HISTORY_RAM", 2),
    ):
        history_manager.log_completed_task({"task_id": "1"})
        history_manager.log_completed_task({"task_id": "2"})
        history_manager.log_completed_task({"task_id": "3"})

        task_history = history_manager.get_history()
        assert len(task_history) == 2
        assert task_history[0]["task_id"] == "3"


def test_ensure_loaded_corrupt(request):
    """Test resilience to corrupt JSON on SSD."""
    temp_file = request.getfixturevalue("reset_history_cache")
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write("corrupt")

    history_manager.HISTORY_CACHE = []
    history_manager.ensure_loaded()
    assert not history_manager.HISTORY_CACHE


def test_history_manager_exceptions():
    """Cover exception handling in log_completed_task."""
    # Passing None to a dict operation should trigger TypeError
    history_manager.log_completed_task(None)


def test_history_manager_stats_cache():
    """Cover stats cache hit branch."""
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    history_manager.STATS_CACHE = {"cached": True}
    history_manager.STATS_CACHE_DATE = today_str
    _history, stats = history_manager.get_history_stats()
    assert stats["cached"] is True
    history_manager.STATS_CACHE = None
    history_manager.STATS_CACHE_DATE = None


def test_history_manager_clear_logic(tmp_path):
    """Cover clear_history and disk removal failure."""
    history_file = tmp_path / "test_history.json"
    history_file.write_text("[]")

    with mock.patch("modules.monitoring.history_manager.HISTORY_FILE", str(history_file)):
        history_manager.clear_history()
        assert not os.path.exists(str(history_file))

    # Test removal failure (OSError)
    with (
        mock.patch("os.path.exists", return_value=True),
        mock.patch("os.remove", side_effect=OSError("Permission denied")),
    ):
        history_manager.clear_history()
        assert history_manager.HISTORY_CACHE == []
        assert history_manager.UNSAVED_COUNT == 0


def test_history_manager_stats_aggregation():
    """Cover history stats logic with actual aggregation."""
    history_manager.ANALYTICS_CACHE = {
        "2026-05-26": {
            "count": 2,
            "duration": 30.0,
            "asr": {"count": 1, "duration": 10.0},
            "detectlang": {"count": 1, "duration": 20.0},
            "audio": {"count": 0, "duration": 0.0},
        }
    }
    history_manager.STATS_CACHE = None
    _, stats = history_manager.get_history_stats()
    assert stats["all_time"] == 30.0
    assert stats["count_all_time"] == 2
    assert stats["asr"]["count"] == 1
    assert stats["asr"]["duration"] == 10.0
    assert stats["detectlang"]["count"] == 1
    assert stats["detectlang"]["duration"] == 20.0


def test_history_stats_persistent_on_clear():
    """Test that analytics stats are preserved when history is cleared."""
    task_data = {"task_id": "1", "video_duration": 60.0, "completed_at": "2026-05-26 12:00:00"}
    history_manager.log_completed_task(task_data)

    # Verify history is saved and stats calculate correctly
    task_history = history_manager.get_history()
    assert len(task_history) == 1
    _, stats = history_manager.get_history_stats()
    assert stats["count_all_time"] == 1
    assert stats["all_time"] == 60.0

    # Clear history
    history_manager.clear_history()

    # History list should be empty
    assert len(history_manager.get_history()) == 0

    # Stats should still be present!
    _, stats_after_clear = history_manager.get_history_stats()
    assert stats_after_clear["count_all_time"] == 1
    assert stats_after_clear["all_time"] == 60.0


def test_get_analytics_data():
    """Test retrieving combined cumulative and daily analytics data."""
    history_manager.log_completed_task(
        {
            "task_id": "analytics_test_1",
            "video_duration": 45.0,
            "completed_at": "2026-05-26 12:00:00",
            "endpoint": "/asr",
        }
    )

    data = history_manager.get_analytics_data()
    assert "cumulative" in data
    assert "daily" in data
    assert data["cumulative"]["count_all_time"] == 1
    assert data["cumulative"]["all_time"] == 45.0
    assert "2026-05-26" in data["daily"]
    assert data["daily"]["2026-05-26"]["count"] == 1
    assert data["daily"]["2026-05-26"]["duration"] == 45.0
    assert data["daily"]["2026-05-26"]["asr"]["count"] == 1
    assert data["daily"]["2026-05-26"]["asr"]["duration"] == 45.0


def test_get_analytics_data_returns_daily_snapshot():
    """Returned analytics payload must not expose live ANALYTICS_CACHE object."""
    history_manager.log_completed_task(
        {
            "task_id": "analytics_snapshot_1",
            "video_duration": 10.0,
            "completed_at": "2026-05-27 12:00:00",
            "endpoint": "/asr",
        }
    )

    data = history_manager.get_analytics_data()
    data["daily"]["2026-05-27"]["count"] = 999

    assert history_manager.ANALYTICS_CACHE["2026-05-27"]["count"] == 1


def test_categorize_task():
    """Test categorize_task with different keys, endpoints and fallbacks."""
    # Endpoint mapping
    assert history_manager.categorize_task({"endpoint": "/asr"}) == "asr"
    assert history_manager.categorize_task({"endpoint": "/detect-language"}) == "detectlang"
    assert history_manager.categorize_task({"endpoint": "/detectlang"}) == "detectlang"
    assert history_manager.categorize_task({"endpoint": "/v1/audio/transcriptions"}) == "audio"

    # Type fallback
    assert history_manager.categorize_task({"type": "Language Detection"}) == "detectlang"
    assert history_manager.categorize_task({"type": "Translation"}) == "audio"

    # Request JSON fallback
    assert history_manager.categorize_task({"request_json": {"response_format": "json"}}) == "audio"
    assert history_manager.categorize_task({"request_json": {"file": "test.wav"}}) == "asr"

    # Default fallback
    assert history_manager.categorize_task({}) == "asr"


def test_rebuild_analytics_from_history():
    """Test rebuild_analytics_from_history correctly parses and structures task cache."""
    t_time = 1779930000.0
    expected_date = datetime.datetime.fromtimestamp(t_time).strftime("%Y-%m-%d")
    history_manager.HISTORY_CACHE = [
        {"status": "completed", "video_duration": 10.0, "completed_at": "2026-06-20 12:00:00", "endpoint": "/asr"},
        {
            "status": "completed",
            "video_duration": 20.0,
            "completed_at": "2026-06-20 13:00:00",
            "endpoint": "/detect-language",
        },
        {"status": "completed", "video_duration": 30.0, "start_time": t_time, "endpoint": "/v1/audio/translations"},
        {
            "status": "failed",  # Non-completed task should be ignored
            "video_duration": 40.0,
            "completed_at": "2026-06-20 14:00:00",
            "endpoint": "/asr",
        },
    ]

    history_manager.rebuild_analytics_from_history()
    cache = history_manager.ANALYTICS_CACHE
    assert "2026-06-20" in cache
    assert expected_date in cache

    day1 = cache["2026-06-20"]
    assert day1["count"] == 2
    assert day1["duration"] == 30.0
    assert day1["asr"]["count"] == 1
    assert day1["asr"]["duration"] == 10.0
    assert day1["detectlang"]["count"] == 1
    assert day1["detectlang"]["duration"] == 20.0

    day2 = cache[expected_date]
    assert day2["count"] == 1
    assert day2["duration"] == 30.0
    assert day2["audio"]["count"] == 1
    assert day2["audio"]["duration"] == 30.0


def test_ensure_analytics_loaded_backfill(tmp_path):
    """Test that ensure_analytics_loaded detects legacy format and triggers rebuild/save."""
    analytics_file = tmp_path / "analytics_stats.json"
    # Legacy data (lacks category structures like 'asr', etc.)
    legacy_data = {"2026-06-20": {"count": 5, "duration": 120.0}}
    analytics_file.write_text(json.dumps(legacy_data))

    # Populate history to rebuild from
    history_manager.HISTORY_CACHE = [
        {"status": "completed", "video_duration": 120.0, "completed_at": "2026-06-20 10:00:00", "endpoint": "/asr"}
    ]

    with mock.patch("modules.monitoring.history_manager.ANALYTICS_FILE", str(analytics_file)):
        history_manager.ANALYTICS_CACHE = None
        history_manager.ensure_analytics_loaded()

        # Check cache is updated and contains categories
        cache = history_manager.ANALYTICS_CACHE
        assert "2026-06-20" in cache
        assert "asr" in cache["2026-06-20"]
        assert cache["2026-06-20"]["asr"]["count"] == 5

        # Check file was also written
        with open(analytics_file, "r", encoding="utf-8") as f:
            written_data = json.load(f)
        assert "asr" in written_data["2026-06-20"]


def test_ensure_loaded_backfills_filenames(tmp_path):
    """Verify ensure_loaded correctly cleans and backfills generic filenames from request_json."""
    history_file = tmp_path / "task_history.json"
    dummy_history = [
        {"task_id": "1", "filename": "audio_file", "request_json": {"video_file": "/movies/my_awesome_video.mp4"}},
        {"task_id": "2", "filename": "Unknown Media", "request_json": {"local_path": "/audio/podcast.wav"}},
        {
            "task_id": "3",
            "filename": "already_correct.mp3",
            "request_json": {"video_file": "should_not_overwrite_this.mp4"},
        },
    ]
    history_file.write_text(json.dumps(dummy_history), encoding="utf-8")

    with mock.patch("modules.monitoring.history_manager.HISTORY_FILE", str(history_file)):
        history_manager.HISTORY_CACHE = []
        history_manager.ensure_loaded()

        cache = history_manager.HISTORY_CACHE
        assert len(cache) == 3
        # Check backfilled fields
        assert cache[0]["filename"] == "my_awesome_video.mp4"
        assert cache[1]["filename"] == "podcast.wav"
        assert cache[2]["filename"] == "already_correct.mp3"


def test_ensure_analytics_loaded_preserves_old_days(tmp_path):
    """Test that ensure_analytics_loaded preserves historical days not in task history."""
    analytics_file = tmp_path / "analytics_stats.json"
    legacy_data = {"2026-06-19": {"count": 10, "duration": 500.0}, "2026-06-20": {"count": 5, "duration": 120.0}}
    analytics_file.write_text(json.dumps(legacy_data), encoding="utf-8")

    # History only contains task for 2026-06-20
    history_manager.HISTORY_CACHE = [
        {"status": "completed", "video_duration": 120.0, "completed_at": "2026-06-20 10:00:00", "endpoint": "/asr"}
    ]

    with mock.patch("modules.monitoring.history_manager.ANALYTICS_FILE", str(analytics_file)):
        history_manager.ANALYTICS_CACHE = None
        history_manager.ensure_analytics_loaded()

        cache = history_manager.ANALYTICS_CACHE
        # 2026-06-20 should be rebuilt from history
        assert "2026-06-20" in cache
        assert cache["2026-06-20"]["asr"]["count"] == 5
        # 2026-06-19 should be preserved and backfilled
        assert "2026-06-19" in cache
        assert cache["2026-06-19"]["asr"]["count"] == 10
        assert cache["2026-06-19"]["asr"]["duration"] == 500.0


def test_ensure_analytics_loaded_preserves_already_categorized_overlapping(tmp_path):
    """Verify ensure_analytics_loaded preserves fully categorized old days even if they overlap with rebuilt history."""
    analytics_file = tmp_path / "analytics_stats.json"
    categorized_data = {
        "2026-06-20": {
            "count": 50,
            "duration": 1000.0,
            "asr": {"count": 40, "duration": 800.0},
            "detectlang": {"count": 10, "duration": 200.0},
            "audio": {"count": 0, "duration": 0.0},
        }
    }
    analytics_file.write_text(json.dumps(categorized_data), encoding="utf-8")

    # History contains built task for 2026-06-20, which would normally overwrite it
    history_manager.HISTORY_CACHE = [
        {"status": "completed", "video_duration": 120.0, "completed_at": "2026-06-20 10:00:00", "endpoint": "/asr"}
    ]

    with mock.patch("modules.monitoring.history_manager.ANALYTICS_FILE", str(analytics_file)):
        history_manager.ANALYTICS_CACHE = None
        history_manager.ensure_analytics_loaded()

        cache = history_manager.ANALYTICS_CACHE
        assert "2026-06-20" in cache
        # The fully categorized entry from old_cache should be completely preserved!
        assert cache["2026-06-20"]["count"] == 50
        assert cache["2026-06-20"]["asr"]["count"] == 40
        assert cache["2026-06-20"]["detectlang"]["count"] == 10


def test_ensure_analytics_loaded_merges_uncategorized_overlapping(tmp_path):
    """Verify ensure_analytics_loaded merges uncategorized legacy days overlapping with rebuilt history."""
    analytics_file = tmp_path / "analytics_stats.json"
    legacy_data = {"2026-06-20": {"count": 5, "duration": 500.0}}
    analytics_file.write_text(json.dumps(legacy_data), encoding="utf-8")

    # History has rebuilt tasks for 2026-06-20 with smaller total count
    history_manager.HISTORY_CACHE = [
        {
            "status": "completed",
            "video_duration": 100.0,
            "completed_at": "2026-06-20 10:00:00",
            "endpoint": "/detect-language",
        }
    ]

    with mock.patch("modules.monitoring.history_manager.ANALYTICS_FILE", str(analytics_file)):
        history_manager.ANALYTICS_CACHE = None
        history_manager.ensure_analytics_loaded()

        cache = history_manager.ANALYTICS_CACHE
        assert "2026-06-20" in cache
        assert cache["2026-06-20"]["count"] == 5
        assert cache["2026-06-20"]["duration"] == 500.0
        # Rebuilt 1 detectlang task should be merged:
        assert cache["2026-06-20"]["detectlang"]["count"] == 1
        assert cache["2026-06-20"]["detectlang"]["duration"] == 100.0
        # The remaining 4 pruned tasks should default to "asr":
        assert cache["2026-06-20"]["asr"]["count"] == 4
        assert cache["2026-06-20"]["asr"]["duration"] == 400.0
