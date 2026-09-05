"""Daily analytics: categorisation, rebuild from history, and legacy migration.

Split out of test_history_manager.py, which had grown past the project's module-length
limit. Analytics are stored separately from history and deliberately outlive it -- clearing
the task list must not erase the totals -- so these cover the merge and backfill paths that
keep older, less-structured files readable.
"""

import datetime
import json
from unittest import mock

import pytest

from modules.monitoring import history_manager

# Every test here mutates history_manager's module-level caches and writes files.
pytestmark = pytest.mark.usefixtures("reset_history_cache")


@pytest.fixture(name="frozen_today")
def _frozen_today():
    """Pin "today" for the whole test, fixture data and assertions alike.

    These tests stamp a task with today's date and then assert the totals land under it. Read
    twice from the clock, a run crossing midnight writes one date and asserts another, and
    the failure is a once-a-day mystery that never reproduces. One value, captured once, and
    the code under test made to agree with it.
    """
    today = datetime.datetime.now()
    with mock.patch("modules.monitoring.history_manager.datetime") as fake:
        fake.now.return_value = today
        fake.fromtimestamp.side_effect = datetime.datetime.fromtimestamp
        yield today.strftime("%Y-%m-%d")


def test_get_analytics_data(frozen_today):
    """Test retrieving combined cumulative and daily analytics data."""
    today_str = frozen_today
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
        # The stored breakdown is the accumulated record; the rebuild only sees whatever
        # history rows survived the cap, so it must not overwrite the categories.
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


def test_history_manager_stats_aggregation(frozen_today):
    """Cover history stats logic with actual aggregation."""
    today_str = frozen_today
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
