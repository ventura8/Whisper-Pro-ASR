"""Shared thread-rendezvous polling helper, reused by the concurrency test
harness (`tests/integration/concurrency/concurrency_fixtures.py`) and the
priority-preemption test helpers (`tests/inference/scheduler/priority/_preemption_test_helpers.py`)
to avoid maintaining two verbatim copies of the same poll-loop implementation.
"""

from __future__ import annotations

import time
from collections.abc import Callable


def poll_until(predicate: Callable[[], bool], timeout: float = 5.0, interval: float = 0.005) -> bool:
    """Poll `predicate` (a zero-arg callable) until it returns truthy or `timeout`
    elapses. Returns the final truthiness of `predicate()`. Centralizes the
    "while not <condition> and time.monotonic() < deadline: sleep()" pattern used
    throughout the concurrency and priority-preemption test suites' thread-rendezvous
    waits, so individual tests don't each carry their own branching poll loop.

    Uses `time.monotonic()` (immune to system-clock adjustments) for the deadline
    and elapsed checks.
    """
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(interval)
    return bool(predicate())
