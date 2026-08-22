"""Shared fixtures for tests/monitoring/, reused by test_telemetry_loop.py and
test_telemetry_liveness.py to avoid duplicating the same background-loop
reset fixture in both files.
"""

from collections.abc import Iterator

import pytest

from modules.monitoring import telemetry


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
