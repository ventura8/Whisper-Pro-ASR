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
        mock.patch("modules.monitoring.history_manager.LEGACY_HISTORY_FILES", []),
        mock.patch("modules.monitoring.history_manager.LEGACY_ANALYTICS_FILES", []),
    ):
        yield temp_file


def test_log_completed_task_transcription():
    """Test logging a transcription task to history."""
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
    assert (
        task_history[0]["task_id"],
        task_history[0]["segments_processed"],
        "completed_at" in task_history[0],
    ) == ("123", 2, True)


def test_log_completed_task_language_detection():
    """Test logging a language-detection task to history."""
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
    assert stats == {
        "all_time": 150.0,
        "today": 150.0,
        "this_month": 150.0,
        "this_year": 150.0,
        "count_all_time": 3,
        "count_today": 3,
        "asr": {"count": 2, "duration": 110.0},
        "detectlang": {"count": 1, "duration": 40.0},
        "audio": {"count": 0, "duration": 0.0},
    }


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


def test_load_history_cache_from_disk_prefers_primary_entries() -> None:
    """Primary history entries should take precedence over any legacy source."""
    primary_entries = [{"task_id": "primary-1"}]
    with (
        mock.patch("modules.monitoring.history_manager._load_primary_history_entries", return_value=primary_entries),
        mock.patch("modules.monitoring.history_manager._load_legacy_history_entries") as legacy_loader,
    ):
        entries, imported_from_legacy = history_manager._load_history_cache_from_disk()

    assert entries == primary_entries
    assert imported_from_legacy is False
    legacy_loader.assert_not_called()


def test_load_history_cache_from_disk_returns_empty_without_legacy_candidates() -> None:
    """Missing primary and missing legacy data should return empty history without legacy-import flag."""
    with (
        mock.patch("modules.monitoring.history_manager._load_primary_history_entries", return_value=None),
        mock.patch("modules.monitoring.history_manager._load_legacy_history_entries", return_value=None),
    ):
        entries, imported_from_legacy = history_manager._load_history_cache_from_disk()

    assert entries == []
    assert imported_from_legacy is False


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
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    history_manager.ANALYTICS_CACHE = {
        today_str: {
            "count": 2,
            "duration": 30.0,
            "asr": {"count": 1, "duration": 10.0},
            "detectlang": {"count": 1, "duration": 20.0},
            "audio": {"count": 0, "duration": 0.0},
        }
    }
    history_manager.STATS_CACHE = None
    _, stats = history_manager.get_history_stats()
    assert stats == {
        "all_time": 30.0,
        "today": 30.0,
        "this_month": 30.0,
        "this_year": 30.0,
        "count_all_time": 2,
        "count_today": 2,
        "asr": {"count": 1, "duration": 10.0},
        "detectlang": {"count": 1, "duration": 20.0},
        "audio": {"count": 0, "duration": 0.0},
    }


def test_history_stats_persistent_on_clear():
    """Test that analytics stats are preserved when history is cleared."""
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    task_data = {"task_id": "1", "video_duration": 60.0, "completed_at": f"{today_str} 12:00:00"}
    history_manager.log_completed_task(task_data)

    # Verify history is saved and stats calculate correctly
    task_history = history_manager.get_history()
    assert len(task_history) == 1
    _, stats = history_manager.get_history_stats()

    # Clear history
    history_manager.clear_history()

    # History list should be empty
    assert len(history_manager.get_history()) == 0

    # Stats should still be present!
    _, stats_after_clear = history_manager.get_history_stats()
    assert (stats["count_all_time"], stats["all_time"], stats_after_clear) == (
        1,
        60.0,
        {
            "all_time": 60.0,
            "today": 60.0,
            "this_month": 60.0,
            "this_year": 60.0,
            "count_all_time": 1,
            "count_today": 1,
            "asr": {"count": 1, "duration": 60.0},
            "detectlang": {"count": 0, "duration": 0.0},
            "audio": {"count": 0, "duration": 0.0},
        },
    )


def test_get_analytics_data():
    """Test retrieving combined cumulative and daily analytics data."""
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    history_manager.log_completed_task(
        {
            "task_id": "analytics_test_1",
            "video_duration": 45.0,
            "completed_at": f"{today_str} 12:00:00",
            "endpoint": "/asr",
        }
    )

    data = history_manager.get_analytics_data()
    assert data == {
        "cumulative": {
            "all_time": 45.0,
            "today": 45.0,
            "this_month": 45.0,
            "this_year": 45.0,
            "count_all_time": 1,
            "count_today": 1,
            "asr": {"count": 1, "duration": 45.0},
            "detectlang": {"count": 0, "duration": 0.0},
            "audio": {"count": 0, "duration": 0.0},
        },
        "daily": {
            history_manager.ANALYTICS_SCHEMA_KEY: history_manager.ANALYTICS_SCHEMA_VERSION,
            today_str: {
                "count": 1,
                "duration": 45.0,
                "asr": {"count": 1, "duration": 45.0},
                "detectlang": {"count": 0, "duration": 0.0},
                "audio": {"count": 0, "duration": 0.0},
            },
        },
    }


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


@pytest.mark.parametrize(
    ("task_data", "expected"),
    [
        ({"endpoint": "/asr"}, "asr"),
        ({"endpoint": "/detect-language"}, "detectlang"),
        ({"endpoint": "/detectlang"}, "detectlang"),
        ({"endpoint": "/v1/audio/transcriptions"}, "asr"),
        ({"type": "Language Detection"}, "detectlang"),
        ({"type": "Translation"}, "audio"),
        ({"request_json": {"response_format": "json"}}, "audio"),
        ({"request_json": {"file": "test.wav"}}, "asr"),
        ({}, "asr"),
    ],
)
def test_categorize_task(task_data, expected):
    """Test categorize_task with different keys, endpoints and fallbacks."""
    assert history_manager.categorize_task(task_data) == expected


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
    assert set(cache) == {history_manager.ANALYTICS_SCHEMA_KEY, "2026-06-20", expected_date}
    assert cache[history_manager.ANALYTICS_SCHEMA_KEY] == history_manager.ANALYTICS_SCHEMA_VERSION
    assert cache["2026-06-20"] == {
        "count": 2,
        "duration": 30.0,
        "asr": {"count": 1, "duration": 10.0},
        "detectlang": {"count": 1, "duration": 20.0},
        "audio": {"count": 0, "duration": 0.0},
    }
    assert cache[expected_date] == {
        "count": 1,
        "duration": 30.0,
        "asr": {"count": 1, "duration": 30.0},
        "detectlang": {"count": 0, "duration": 0.0},
        "audio": {"count": 0, "duration": 0.0},
    }


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


def test_log_completed_task_backfills_generic_filename(tmp_path):
    """History writes should recover Bazarr-style filenames from request_json.local_path."""
    history_file = tmp_path / "task_history.json"
    task_data = {
        "task_id": "bazarr-1",
        "filename": "",
        "request_json": {
            "local_path": "/tv/Doc - In Your Hands/Season 3/Doc (IT) - S03E01 - Awakenings WEBDL-1080p.mkv",
            "audio_file": "",
        },
        "status": "completed",
        "logs": ["19:00:01 sample log"],
    }

    with mock.patch("modules.monitoring.history_manager.HISTORY_FILE", str(history_file)):
        history_manager.HISTORY_CACHE = []
        history_manager.log_completed_task(task_data)

        assert history_manager.HISTORY_CACHE[0]["filename"] == "Doc (IT) - S03E01 - Awakenings WEBDL-1080p.mkv"


def test_get_history_backfills_filename_from_bazarr_json_path_key(tmp_path):
    """Legacy history entries may store the media path as request_json object key."""
    history_file = tmp_path / "task_history.json"
    path = "/tv/Doc - In Your Hands/Season 3/Doc (IT) - S03E01 - Awakenings WEBDL-1080p.mkv"
    history_file.write_text(
        json.dumps(
            [
                {
                    "task_id": "bazarr-key-1",
                    "filename": "Unknown Media",
                    "request_json": {path: ""},
                    "status": "failed",
                }
            ]
        ),
        encoding="utf-8",
    )

    with mock.patch("modules.monitoring.history_manager.HISTORY_FILE", str(history_file)):
        history_manager.HISTORY_CACHE = []
        history = history_manager.get_history()

    assert history[0]["filename"] == "Doc (IT) - S03E01 - Awakenings WEBDL-1080p.mkv"


def test_get_history_backfills_request_json_from_bazarr_path_key(tmp_path):
    """Serving history should normalize Bazarr path-as-key request payloads."""
    history_file = tmp_path / "task_history.json"
    path = "/tv/Doc - In Your Hands/Season 3/Doc (IT) - S03E01 - Awakenings WEBDL-1080p.mkv"
    history_file.write_text(
        json.dumps(
            [
                {
                    "task_id": "bazarr-key-req-1",
                    "filename": "Unknown Media",
                    "request_json": {path: ""},
                    "status": "failed",
                }
            ]
        ),
        encoding="utf-8",
    )

    with mock.patch("modules.monitoring.history_manager.HISTORY_FILE", str(history_file)):
        history_manager.HISTORY_CACHE = []
        history = history_manager.get_history()

    assert history[0]["request_json"] == {"local_path": path}


def test_get_history_backfills_generic_filename_without_reload(tmp_path):
    """Serving history should repair generic filenames already present in RAM cache."""
    history_file = tmp_path / "task_history.json"
    history_manager.HISTORY_CACHE = [
        {
            "task_id": "live-1",
            "filename": "Unknown Media",
            "request_json": {"local_path": "/media/show/episode.mkv"},
        }
    ]

    with mock.patch("modules.monitoring.history_manager.HISTORY_FILE", str(history_file)):
        history = history_manager.get_history()

    assert history[0]["filename"] == "episode.mkv"


def test_ensure_loaded_imports_legacy_history_when_primary_missing(tmp_path):
    """Upgrade path should import history from legacy data location when new state file is absent."""
    new_history_file = tmp_path / "state" / "task_history.json"
    legacy_history_file = tmp_path / "legacy" / "task_history.json"
    legacy_history_file.parent.mkdir(parents=True, exist_ok=True)
    legacy_history_file.write_text(
        json.dumps(
            [
                {
                    "task_id": "legacy-1",
                    "filename": "audio_file",
                    "request_json": {"local_path": "/media/legacy_movie.mkv"},
                    "status": "completed",
                }
            ]
        ),
        encoding="utf-8",
    )

    with (
        mock.patch("modules.monitoring.history_manager.HISTORY_FILE", str(new_history_file)),
        mock.patch("modules.monitoring.history_manager.LEGACY_HISTORY_FILES", [str(legacy_history_file)]),
    ):
        history_manager.HISTORY_CACHE = []
        history_manager.ensure_loaded()

        cache = history_manager.HISTORY_CACHE
        assert len(cache) == 1
        assert cache[0]["task_id"] == "legacy-1"
        assert cache[0]["filename"] == "legacy_movie.mkv"
        assert new_history_file.exists()


def test_ensure_analytics_loaded_imports_legacy_when_primary_missing(tmp_path):
    """Upgrade path should import analytics from legacy data location when new state file is absent."""
    new_analytics_file = tmp_path / "state" / "analytics_stats.json"
    legacy_analytics_file = tmp_path / "legacy" / "analytics_stats.json"
    legacy_analytics_file.parent.mkdir(parents=True, exist_ok=True)
    legacy_payload = {
        history_manager.ANALYTICS_SCHEMA_KEY: history_manager.ANALYTICS_SCHEMA_VERSION,
        "2026-07-01": {
            "count": 2,
            "duration": 120.0,
            "asr": {"count": 2, "duration": 120.0},
            "detectlang": {"count": 0, "duration": 0.0},
            "audio": {"count": 0, "duration": 0.0},
        },
    }
    legacy_analytics_file.write_text(json.dumps(legacy_payload), encoding="utf-8")

    with (
        mock.patch("modules.monitoring.history_manager.ANALYTICS_FILE", str(new_analytics_file)),
        mock.patch("modules.monitoring.history_manager.LEGACY_ANALYTICS_FILES", [str(legacy_analytics_file)]),
    ):
        history_manager.ANALYTICS_CACHE = None
        history_manager.ensure_analytics_loaded()

        assert history_manager.ANALYTICS_CACHE["2026-07-01"]["count"] == 2
        assert new_analytics_file.exists()


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
        assert cache == {
            history_manager.ANALYTICS_SCHEMA_KEY: history_manager.ANALYTICS_SCHEMA_VERSION,
            "2026-06-19": {
                "count": 10,
                "duration": 500.0,
                "asr": {"count": 10, "duration": 500.0},
                "detectlang": {"count": 0, "duration": 0.0},
                "audio": {"count": 0, "duration": 0.0},
            },
            "2026-06-20": {
                "count": 5,
                "duration": 120.0,
                "asr": {"count": 5, "duration": 120.0},
                "detectlang": {"count": 0, "duration": 0.0},
                "audio": {"count": 0, "duration": 0.0},
            },
        }


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
        assert cache["2026-06-20"]["asr"]["count"] == 50
        assert cache["2026-06-20"]["detectlang"]["count"] == 0


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
        assert cache == {
            history_manager.ANALYTICS_SCHEMA_KEY: history_manager.ANALYTICS_SCHEMA_VERSION,
            "2026-06-20": {
                "count": 5,
                "duration": 500.0,
                "detectlang": {"count": 1, "duration": 100.0},
                "asr": {"count": 4, "duration": 400.0},
                "audio": {"count": 0, "duration": 0.0},
            },
        }
