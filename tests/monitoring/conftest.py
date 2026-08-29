"""Shared fixtures for tests/monitoring/, reused by test_telemetry_loop.py and
test_telemetry_liveness.py to avoid duplicating the same background-loop
reset fixture in both files.
"""

from collections.abc import Iterator
from unittest import mock

import pytest

from modules.monitoring import history_manager, telemetry


@pytest.fixture
def clean_telemetry() -> Iterator[None]:
    """Reset telemetry history and ensure the background loop stays stopped.

    Does NOT clear `_STOP_EVENT`: a prior test's worker thread (if any were ever
    left running) checks `is_set()` only once per ~2s loop iteration, so clearing
    the event here -- immediately after telling it to stop -- would race a worker
    that hasn't actually exited yet, letting it resume and append telemetry into
    the just-cleared TELEMETRY_HISTORY. No current test in this suite needs the
    event pre-cleared (test_telemetry_worker_unit fully mocks `is_set`, and
    test_start_telemetry_loop fully mocks `threading.Thread`); a test that
    genuinely needs to run the worker should clear the event itself, scoped to
    just that exercise.
    """
    telemetry._STOP_EVENT.set()
    telemetry.TELEMETRY_HISTORY.clear()
    yield
    telemetry._STOP_EVENT.set()


@pytest.fixture
def reset_history_cache(tmp_path):
    """Give one test its own history/analytics files and a cleared cache.

    Shared by test_history_manager.py and test_history_analytics.py, which were one module
    until it outgrew the project's module-length limit. Deliberately not autouse: this
    conftest is visible to every module in tests/monitoring, and the telemetry tests must
    not have history_manager's paths swapped out from under them. The two history modules
    opt in with a module-level usefixtures mark.
    """
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

    # Cleared on the way out as well as in. These caches are module globals, so whatever the
    # test loaded from its tmp_path files outlives the patches that pointed at them -- a
    # later test then reads history from files that no longer exist, and the staleness looks
    # like a history_manager bug rather than leaked fixture state.
    history_manager.HISTORY_CACHE = []
    history_manager.ANALYTICS_CACHE = None
    history_manager.STATS_CACHE = None
