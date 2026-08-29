"""Tests for modules/monitoring/history_manager.py."""

import datetime
import json
import os
import time
from unittest import mock

import pytest

from modules.monitoring import history_helpers, history_manager

# Every test here mutates history_manager's module-level caches and writes files.
pytestmark = pytest.mark.usefixtures("reset_history_cache")


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


class TestTruncateLargeSegments:
    """History truncation must not reach back into the caller's response.

    ``task_data["result"]`` is the same object the request handler returns to the client,
    and ``log_completed_task`` only shallow-copies ``task_data`` afterwards. Truncating in
    place therefore clipped real transcripts on the way out. Measured on a 20-minute clip:
    169 segments reaching 1202s were delivered as 100 segments ending at 765s.
    """

    def _result(self, n_segments):
        return {
            "segments": [{"start": float(i), "end": float(i) + 1.0, "text": f"seg {i}"} for i in range(n_segments)],
            "language": "es",
        }

    def test_callers_segments_are_not_truncated(self):
        result = self._result(169)
        task_data = {"result": result}

        history_helpers.truncate_large_segments(task_data)

        assert len(result["segments"]) == 169
        assert result["segments"][-1]["end"] == 169.0
        assert "segments_truncated" not in result

    def test_history_copy_is_truncated(self):
        task_data = {"result": self._result(169)}

        history_helpers.truncate_large_segments(task_data)

        stored = task_data["result"]
        assert len(stored["segments"]) == 100
        assert stored["segments_total_count"] == 169
        assert stored["segments_truncated"] is True

    def test_other_result_fields_survive_truncation(self):
        task_data = {"result": self._result(169)}

        history_helpers.truncate_large_segments(task_data)

        assert task_data["result"]["language"] == "es"

    def test_result_at_or_below_limit_is_left_alone(self):
        result = self._result(100)
        task_data = {"result": result}

        history_helpers.truncate_large_segments(task_data)

        assert task_data["result"] is result
        assert "segments_truncated" not in result

    def test_missing_result_is_a_noop(self):
        task_data = {}
        history_helpers.truncate_large_segments(task_data)
        assert task_data == {}

    def test_empty_segments_is_a_noop(self):
        task_data = {"result": {"segments": []}}
        history_helpers.truncate_large_segments(task_data)
        assert task_data["result"]["segments"] == []


class TestHistoryEntryTruncationIsolation:
    """The history file gets the small copy; the caller keeps the whole transcript.

    This is the regression behind the long-recorded long-form "coverage" defect. Keeping the
    *history file* small clipped the very dict the request handler returns, so every response
    over 100 segments reached the client truncated -- 169 segments ending at 1202s delivered
    as 100 ending at 765s. Both halves are asserted here, because fixing one without the
    other is exactly how this was wrong twice.
    """

    @staticmethod
    def _task_with_segments(count: int) -> dict:
        return {
            "task_id": "truncation-regression",
            "type": "Transcription",
            "endpoint": "/asr",
            "video_duration": 1203.0,
            "result": {
                "text": "full transcript",
                "segments": [{"start": float(i), "end": float(i) + 1.0, "text": f"segment {i}"} for i in range(count)],
            },
        }

    def test_the_callers_task_data_keeps_every_segment(self):
        task_data = self._task_with_segments(169)

        history_manager.log_completed_task(task_data)

        assert len(task_data["result"]["segments"]) == 169
        assert task_data["result"]["segments"][-1]["text"] == "segment 168"
        # The caller's copy is the response; it must carry no truncation markers at all.
        assert "segments_truncated" not in task_data["result"]
        assert "segments_total_count" not in task_data["result"]

    def test_the_inserted_history_entry_is_clipped_and_says_so(self):
        history_manager.log_completed_task(self._task_with_segments(169))

        entry = history_manager.HISTORY_CACHE[0]
        assert len(entry["result"]["segments"]) == 100
        assert entry["result"]["segments_total_count"] == 169
        assert entry["result"]["segments_truncated"] is True

    def test_a_transcript_within_the_cap_is_stored_whole_and_unmarked(self):
        """The boundary matters: a 100-segment result must not be labelled truncated."""
        history_manager.log_completed_task(self._task_with_segments(100))

        entry = history_manager.HISTORY_CACHE[0]
        assert len(entry["result"]["segments"]) == 100
        assert "segments_truncated" not in entry["result"]
