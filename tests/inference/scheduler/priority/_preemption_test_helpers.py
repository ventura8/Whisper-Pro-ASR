"""Shared helpers for the priority-preemption test modules
(`test_priority_preemption_sla.py`, `test_priority_preemption_self_heal.py`)
split out of the former, oversized `test_priority_preemption_gaps.py` to keep
each module under the repo's 500-line-per-file limit.

Also reused by `tests/inference/runtime/test_concurrency_reentrancy.py`
(`_poll_until`) to avoid a third verbatim copy of the same poll-loop helper.
"""

import contextlib
from collections.abc import Iterator

from modules.inference import scheduler
from tests.polling_helpers import poll_until

_poll_until = poll_until


def _restore_scheduler_state_body() -> Iterator[None]:
    """Snapshot `scheduler.STATE` before each test and restore that exact
    object afterward. Some tests in these modules replace `scheduler.STATE`
    wholesale (e.g. `scheduler.STATE = SchedulerState()`) to get a clean
    slate for simulated-crash scenarios; without restoring, that replacement
    leaks into later tests in the suite.

    Plain (undecorated) generator so each importing module can register it
    as its own autouse fixture via a thin local wrapper -- keeping the
    fixture registration explicit per module without an unused-import lint
    suppression."""
    previous_state = scheduler.STATE
    try:
        yield
    finally:
        scheduler.STATE = previous_state


@contextlib.contextmanager
def _capture_worker_exc(errors: list[Exception]) -> Iterator[None]:
    """Context manager for worker threads: captures any raised exception into
    `errors` and re-raises so the thread terminates with an exception trace
    rather than silently swallowing it. Using a named context manager avoids
    a bare top-level `except Exception` in the worker body itself."""
    try:
        yield
    except Exception as exc:  # worker-thread isolation: captured and re-raised
        errors.append(exc)
        raise


def _assert_no_worker_errors(errors: list[Exception]) -> None:
    """Assert no worker thread captured an exception via `_capture_worker_exc`."""
    assert not errors, f"worker thread(s) raised: {errors}"
