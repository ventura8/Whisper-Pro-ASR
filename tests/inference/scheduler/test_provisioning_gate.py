"""Tasks arriving during startup provisioning wait in the queue, not on hardware.

A first start downloads several GB while the API is already serving. Requests that land in
that window must not be rejected and must not claim a hardware unit -- they sit in the
scheduler queue showing download progress, then proceed normally once provisioning ends.
Getting this wrong is invisible in a warm-cache test run and only appears on a fresh
deployment, which is the one moment nobody is watching a test suite.
"""

# pylint: disable=protected-access
# The gate lives inside the acquisition loop's step function, which has no public entry
# point that does not also block on real hardware.

from unittest import mock

import pytest

from modules.inference import scheduler
from modules.inference.runtime import concurrency


@pytest.fixture(name="quiet_scheduler")
def _quiet_scheduler():
    """Stub the scheduler side effects so a step can be driven without a live task."""
    with (
        mock.patch.object(scheduler, "update_task_metadata") as metadata,
        mock.patch.object(scheduler, "update_task_progress") as progress,
        mock.patch.object(scheduler, "increment_queued_session") as queued,
    ):
        yield {"metadata": metadata, "progress": progress, "queued": queued}


def _gate(active: bool, percent: int = 0):
    return (
        mock.patch.object(concurrency.model_provisioning, "should_gate_tasks", return_value=active),
        mock.patch.object(concurrency.model_provisioning, "get_progress", return_value={"percent": percent}),
    )


def test_a_gated_task_gets_no_unit(quiet_scheduler):
    """The point of gating here rather than in init_unit: no unit is held while waiting."""
    gate, progress = _gate(True, percent=17)
    with gate, progress, scheduler.STATE.cond:
        with mock.patch.object(scheduler.STATE.cond, "wait") as wait:
            unit, borrowed, queued_added = concurrency._loop_step_acquire("t1", False, False)

    assert unit is None and borrowed is False
    assert queued_added is True, "the task must be registered as queued exactly once"
    wait.assert_called_once()
    quiet_scheduler["metadata"].assert_called_once_with(status="queued")
    quiet_scheduler["queued"].assert_called_once()


def test_the_gate_runs_before_any_hardware_allocation(quiet_scheduler):
    """Allocation must not even be attempted while provisioning holds the gate."""
    gate, progress = _gate(True)
    with gate, progress, scheduler.STATE.cond:
        with (
            mock.patch.object(scheduler.STATE.cond, "wait"),
            mock.patch.object(concurrency, "_try_allocating_unit") as allocate,
            mock.patch.object(concurrency, "_is_task_waiting_for_earlier_fifo") as fifo,
        ):
            concurrency._loop_step_acquire("t1", False, False)

    allocate.assert_not_called()
    fifo.assert_not_called(), "the FIFO check is downstream of the gate"


def test_the_reported_stage_tracks_the_download_percentage(quiet_scheduler):
    """Refreshed every iteration, so the dashboard advances instead of freezing at 0%."""
    seen = []
    for percent in (0, 40, 95):
        gate, progress = _gate(True, percent=percent)
        with gate, progress, scheduler.STATE.cond:
            with mock.patch.object(scheduler.STATE.cond, "wait"):
                concurrency._loop_step_acquire("t1", False, True)
        seen.append(quiet_scheduler["progress"].call_args.args[1])

    assert seen == ["Downloading Model (0%)", "Downloading Model (40%)", "Downloading Model (95%)"]


def test_an_already_queued_task_is_not_counted_twice(quiet_scheduler):
    """queued_added is the once-only latch; a gate that re-counts inflates the queue depth."""
    gate, progress = _gate(True)
    with gate, progress, scheduler.STATE.cond:
        with mock.patch.object(scheduler.STATE.cond, "wait"):
            _, _, queued_added = concurrency._loop_step_acquire("t1", False, True)

    assert queued_added is True
    quiet_scheduler["queued"].assert_not_called()
    quiet_scheduler["metadata"].assert_not_called()


def test_waiters_are_woken_so_the_gate_clearing_is_noticed(quiet_scheduler):
    """Without the notify, a task can sleep past the end of provisioning."""
    gate, progress = _gate(True)
    with gate, progress, scheduler.STATE.cond:
        with (
            mock.patch.object(scheduler.STATE.cond, "wait"),
            mock.patch.object(scheduler.STATE.cond, "notify_all") as notify,
        ):
            concurrency._loop_step_acquire("t1", False, False)

    notify.assert_called_once()


def test_hardware_allocation_resumes_once_the_gate_clears(quiet_scheduler):
    """The whole point: a gated task is delayed, never failed."""
    unit = {"id": "cuda:0", "type": "CUDA", "name": "NVIDIA GPU 0"}
    gate, progress = _gate(False)
    with gate, progress, scheduler.STATE.cond:
        with (
            mock.patch.object(concurrency, "_is_task_waiting_for_earlier_fifo", return_value=False),
            mock.patch.object(concurrency, "_try_allocating_unit", return_value=(unit, False)),
        ):
            acquired, borrowed, queued_added = concurrency._loop_step_acquire("t1", False, True)

    assert acquired is unit
    assert borrowed is False
    assert queued_added is True, "the queued latch is carried through, not reset"


def test_fifo_order_still_applies_after_provisioning(quiet_scheduler):
    """Clearing the gate must not let a late task jump an earlier one."""
    gate, progress = _gate(False)
    with gate, progress, scheduler.STATE.cond:
        with (
            mock.patch.object(scheduler.STATE.cond, "wait"),
            mock.patch.object(concurrency, "_is_task_waiting_for_earlier_fifo", return_value=True),
            mock.patch.object(concurrency, "_try_allocating_unit") as allocate,
        ):
            acquired, _, _ = concurrency._loop_step_acquire("t2", False, False)

    assert acquired is None
    allocate.assert_not_called()
