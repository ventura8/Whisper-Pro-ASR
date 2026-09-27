"""Focused tests for history legacy migration and legacy path helpers."""

import json
import os
from pathlib import Path
from unittest import mock

import pytest

from modules.monitoring import history_helpers, history_manager


@pytest.fixture(autouse=True, name="reset_history_cache")
def _reset_history_cache(tmp_path: Path):
    """Reset history cache and use a temporary file for each test."""
    history_manager.HISTORY_CACHE = []
    history_manager.ANALYTICS_CACHE = None
    history_manager.STATS_CACHE = None

    temp_file = tmp_path / "task_history.json"
    temp_analytics_file = tmp_path / "analytics_stats.json"
    with (
        mock.patch("modules.monitoring.history_manager.HISTORY_FILE", str(temp_file)),
        mock.patch("modules.monitoring.history_manager.ANALYTICS_FILE", str(temp_analytics_file)),
        mock.patch("modules.monitoring.history_manager.LEGACY_HISTORY_FILES", []),
        mock.patch("modules.monitoring.history_manager.LEGACY_ANALYTICS_FILES", []),
    ):
        yield temp_file


def test_history_imports_legacy_state_folder(reset_history_cache: Path, tmp_path: Path, monkeypatch) -> None:
    """When current history file is missing, runtime should import from legacy state path candidates."""
    temp_file = reset_history_cache
    legacy_file = tmp_path / "legacy_state_task_history.json"
    legacy_payload = [{"task_id": "legacy-1", "status": "completed", "video_duration": 10.0}]
    legacy_file.write_text(json.dumps(legacy_payload), encoding="utf-8")

    monkeypatch.setattr(history_manager, "HISTORY_CACHE", [])
    monkeypatch.setattr(history_manager, "STATS_CACHE", None)

    with mock.patch("modules.monitoring.history_manager.LEGACY_HISTORY_FILES", [str(legacy_file)]):
        task_history = history_manager.get_history()

    assert len(task_history) == 1
    assert task_history[0]["task_id"] == "legacy-1"
    with open(temp_file, "r", encoding="utf-8") as persisted:
        persisted_data = json.load(persisted)
    assert persisted_data[0]["task_id"] == "legacy-1"


def test_load_legacy_history_entries_returns_none_for_unreadable_legacy_data(tmp_path: Path) -> None:
    """Unreadable legacy JSON should be treated as no legacy history available."""
    legacy_file = tmp_path / "legacy_task_history.json"
    legacy_file.write_text("not-json", encoding="utf-8")

    with (
        mock.patch("modules.monitoring.history_manager.LEGACY_HISTORY_FILES", [str(legacy_file)]),
        mock.patch("modules.monitoring.history_manager.HISTORY_FILE", str(tmp_path / "state" / "task_history.json")),
    ):
        legacy_loaded = history_manager._load_legacy_history_entries()

    assert legacy_loaded is None


def test_find_legacy_file_skips_current_history_file_and_duplicate_candidates(tmp_path: Path) -> None:
    """Legacy lookup should ignore current HISTORY_FILE path and duplicate legacy candidates."""
    current_file = tmp_path / "state" / "task_history.json"
    legacy_file = tmp_path / "legacy" / "task_history.json"
    current_file.parent.mkdir(parents=True, exist_ok=True)
    legacy_file.parent.mkdir(parents=True, exist_ok=True)
    current_file.write_text("[]", encoding="utf-8")
    legacy_file.write_text("[]", encoding="utf-8")

    selected = history_manager._find_legacy_file(
        [str(current_file), str(legacy_file), str(legacy_file)],
        str(current_file),
    )

    assert selected == os.path.abspath(str(legacy_file))


def test_normalize_legacy_candidate_returns_none_for_falsy_values() -> None:
    """Falsy legacy candidates should normalize to None."""
    assert history_helpers.normalize_legacy_candidate("") is None
    assert history_helpers.normalize_legacy_candidate(None) is None


def test_normalize_legacy_candidate_returns_absolute_path(tmp_path: Path) -> None:
    """Valid candidate path should normalize to absolute path."""
    relative_candidate = os.path.relpath(str(tmp_path / "legacy" / "task_history.json"), os.getcwd())
    assert history_helpers.normalize_legacy_candidate(relative_candidate) == os.path.abspath(relative_candidate)


def test_iter_unique_legacy_paths_excludes_current_deduplicates_and_preserves_first_seen_order(tmp_path: Path) -> None:
    """Legacy path iterator should skip current file, dedupe normalized paths, and keep first-seen order."""
    current_file = tmp_path / "state" / "task_history.json"
    first_unique = tmp_path / "legacy_a" / "task_history.json"
    second_unique = tmp_path / "legacy_b" / "task_history.json"
    current_file.parent.mkdir(parents=True, exist_ok=True)
    first_unique.parent.mkdir(parents=True, exist_ok=True)
    second_unique.parent.mkdir(parents=True, exist_ok=True)

    results = list(
        history_helpers.iter_unique_legacy_paths(
            [
                "",
                None,
                str(current_file),
                str(first_unique),
                os.path.relpath(str(first_unique), os.getcwd()),
                str(second_unique),
                str(first_unique),
            ],
            str(current_file),
        )
    )

    assert results == [os.path.abspath(str(first_unique)), os.path.abspath(str(second_unique))]
