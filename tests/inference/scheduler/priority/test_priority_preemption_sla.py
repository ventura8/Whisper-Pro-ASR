"""Priority-preemption SLA/queueing tests for section 3 (priority preemption) of
docs/E2E_TEST_PLAN_ORCHESTRATION.md.

Covers:
- 3.1: numeric wall-clock SLA for "ASR paused by an arriving language-detection
  request" (distinct from the self-preemption bypass `<0.2s` check in
  test_priority_concurrency_core_tests.py:test_priority_does_not_preempt_itself).
- 3.6: all units already running language-detection (priority) tasks -> a new
  language-detection request must queue within its priority class rather than
  attempt to preempt another priority task.

Split out of the former, oversized `test_priority_preemption_gaps.py` (past the
repo's 500-line-per-file limit) together with `test_priority_preemption_self_heal.py`,
which covers the 3.7/3.9 scenarios. Shared thread-rendezvous helpers live in
`_preemption_test_helpers.py`.
"""

import threading
import time
from unittest import mock

import pytest

from modules.core import utils
from modules.inference import scheduler
from modules.inference.runtime import concurrency, model_manager
from modules.inference.scheduler import state_helpers as scheduler_state_helpers
from tests.inference.scheduler.priority._preemption_test_helpers import (
    _assert_no_worker_errors,
    _capture_worker_exc,
    _poll_until,
    _restore_scheduler_state_body,
)
from tests.inference.scheduler.priority.test_priority_concurrency import (
    _setup_priority_scheduler,
)


@pytest.fixture(autouse=True)
def _restore_scheduler_state():
    """Registers `_restore_scheduler_state_body` as this module's autouse fixture."""
    yield from _restore_scheduler_state_body()


# Generous CI-safe SLA bound for "ASR paused after a language-detection
# request arrives". The tight 0.2s bound used elsewhere in this suite covers
# a synchronous bypass check with no actual wait loop involved. This scenario
# instead has to go through real thread scheduling plus two independent
# 0.1s-resolution condition-variable poll loops (the ASR task's own
# `_check_preemption` cadence and the priority task's
# `wait_for_pause_confirmation` poll), so 1.5s gives ~10x headroom over the
# theoretical worst case (~0.2s of polling latency) to absorb CI scheduler
# jitter without being so loose it stops catching real regressions.
PAUSE_SLA_SECONDS = 1.5


def _run_asr_saturating(
    events: list[str],
    errors: list[Exception],
    name: str,
    iterations: int = 200,
    step_delay: float = 0.01,
) -> None:
    """Simulate a chunked ASR task that checks for preemption every chunk."""
    utils.THREAD_CONTEXT.reset()
    events.append(f"asr_{name}_start")
    model_manager.increment_active_session()
    try:
        with _capture_worker_exc(errors):
            with model_manager.early_task_registration(is_priority=False):
                with model_manager.model_lock_ctx(priority=False) as (_, unit_id):
                    events.append(f"asr_{name}_running_on_{unit_id}")
                    for _ in range(iterations):
                        time.sleep(step_delay)
                        model_manager._check_preemption()
                    events.append(f"asr_{name}_done")
    finally:
        model_manager.decrement_active_session()


def _run_language_detection_once(events: list[str], errors: list[Exception], delay: float = 0.05) -> None:
    """Run a single language-detection (priority) task to completion, recording
    its start/end into `events`. Shared body for the priority-side thread used
    by the pause-SLA test.
    """
    utils.THREAD_CONTEXT.reset()
    model_manager.increment_active_session()
    try:
        with _capture_worker_exc(errors):
            with model_manager.early_task_registration(task_type="Language Detection", is_priority=True):
                model_manager.wait_for_priority()
                with model_manager.model_lock_ctx(priority=True) as (_, unit_id):
                    events.append(f"ld_running_on_{unit_id}")
                    time.sleep(delay)
                    events.append("ld_done")
    finally:
        model_manager.decrement_active_session()


def _assert_pause_observed_within_sla(pause_confirmed_at: dict, arrival: float) -> None:
    """Assert the pause was observed at all, and within the SLA bound."""
    assert "t" in pause_confirmed_at, "ASR pause was never observed"
    elapsed = pause_confirmed_at["t"] - arrival
    assert elapsed < PAUSE_SLA_SECONDS, f"ASR pause took {elapsed:.3f}s, exceeding SLA of {PAUSE_SLA_SECONDS}s"


def _assert_both_threads_finished_cleanly(
    t_asr: threading.Thread, t_ld: threading.Thread, events: list[str], errors: list[Exception]
) -> None:
    """Assert both threads finished (no hang), raised nothing, and both reached completion."""
    t_asr.join(timeout=10.0)
    assert not t_asr.is_alive()
    assert not t_ld.is_alive()
    _assert_no_worker_errors(errors)
    assert "asr_1_done" in events
    assert "ld_done" in events


def _assert_pause_sla_result(
    t_asr: threading.Thread,
    t_ld: threading.Thread,
    *,
    events: list[str],
    errors: list[Exception],
    pause_confirmed_at: dict,
    arrival: float,
) -> None:
    """Assert the pause-SLA test's post-conditions: pause was observed within
    the SLA bound, and both threads finished cleanly."""
    _assert_pause_observed_within_sla(pause_confirmed_at, arrival)
    _assert_both_threads_finished_cleanly(t_asr, t_ld, events, errors)


def test_asr_pause_sla_wall_clock_under_bound():
    """3.1: ASR paused by an arriving language-detection request within an SLA bound.

    Drives a real ASR task to a running/saturated state on the only hardware
    unit, then submits a language-detection (priority) request and measures
    wall-clock time from request-arrival to the ASR task's pause-confirmed
    state (`unit_sync[unit]["pause_confirmed"]` set), which is the concrete
    signal the ASR side yielded control back to the scheduler.
    """
    hw_list = [{"id": "NPU.0", "type": "NPU", "name": "Intel NPU"}]

    with (
        mock.patch("modules.core.config.HARDWARE_UNITS", hw_list),
        mock.patch("modules.inference.runtime.model_manager.unload_models"),
    ):
        _setup_priority_scheduler(hw_list)

        events = []
        errors = []
        t_asr = threading.Thread(target=_run_asr_saturating, args=(events, errors, "1"), daemon=True)
        t_asr.start()

        # Wait until ASR is actually running on the unit before submitting LD.
        reached_running = _poll_until(lambda: "asr_1_running_on_NPU.0" in events)
        assert reached_running, "ASR task never reached running state"

        # Spy on the concrete "ASR paused" signal (`_set_pause_confirmed`,
        # called from the ASR thread once it observes the pause request and
        # yields) to capture the exact wall-clock moment it fires, avoiding
        # any race in a separate polling observer.
        pause_confirmed_at = {}
        original_set_pause_confirmed = concurrency._set_pause_confirmed

        def _spy_set_pause_confirmed(u_sync, pause_generation):
            original_set_pause_confirmed(u_sync, pause_generation)
            pause_confirmed_at.setdefault("t", time.time())

        arrival = time.time()

        with mock.patch.object(concurrency, "_set_pause_confirmed", side_effect=_spy_set_pause_confirmed):
            t_ld = threading.Thread(target=_run_language_detection_once, args=(events, errors), daemon=True)
            t_ld.start()
            _poll_until(lambda: "t" in pause_confirmed_at, timeout=10.0)
            t_ld.join(timeout=10.0)

        _assert_pause_sla_result(t_asr, t_ld, events=events, errors=errors, pause_confirmed_at=pause_confirmed_at, arrival=arrival)


def _run_holding_priority(events: list[str], errors: list[Exception], release_gate: threading.Event, name: str) -> None:
    """Run a language-detection (priority) task that holds its unit until
    `release_gate` is set. Shared body for the units-saturated test."""
    utils.THREAD_CONTEXT.reset()
    model_manager.increment_active_session()
    try:
        with _capture_worker_exc(errors):
            with model_manager.early_task_registration(task_type="Language Detection", is_priority=True):
                model_manager.wait_for_priority()
                with model_manager.model_lock_ctx(priority=True) as (_, unit_id):
                    events.append(f"ld_{name}_running_on_{unit_id}")
                    release_gate.wait(timeout=10.0)
                    events.append(f"ld_{name}_done")
    finally:
        model_manager.decrement_active_session()


def _assert_third_ld_did_not_dispatch(events: list[str]) -> None:
    """Assert the third LD request did not dispatch onto either unit."""
    assert "ld_C_running_on_NPU.0" not in events
    assert "ld_C_running_on_GPU.0" not in events


def _assert_no_preemption_was_attempted() -> None:
    """Assert no unit has an outstanding pause request / preemptible entry."""
    assert not scheduler.STATE.unit_sync["NPU.0"]["pause_requested"].is_set()
    assert not scheduler.STATE.unit_sync["GPU.0"]["pause_requested"].is_set()
    assert not scheduler.STATE.preemptible_units


def _assert_third_ld_queued_without_preemption(events: list[str]) -> None:
    """Assert a third LD request, arriving while all units are saturated with
    other LD (priority-class) tasks, queues rather than attempts preemption."""
    _assert_third_ld_did_not_dispatch(events)
    _assert_no_preemption_was_attempted()


def test_all_units_saturated_with_language_detection_queues_not_preempts():
    """3.6: LD request arrives when ALL units already run LD tasks -> queues, no preemption attempt."""
    hw_list = [
        {"id": "NPU.0", "type": "NPU", "name": "Intel NPU"},
        {"id": "GPU.0", "type": "GPU", "name": "Intel GPU"},
    ]

    with (
        mock.patch("modules.core.config.HARDWARE_UNITS", hw_list),
        mock.patch("modules.inference.runtime.model_manager.unload_models"),
    ):
        _setup_priority_scheduler(hw_list)

        events = []
        errors = []
        release_gate = threading.Event()

        # Saturate both units with LD (priority-class) tasks.
        t1, t2 = _start_saturating_threads(events, errors, release_gate)

        # No standard (ASR) task exists, so preemption must never be attempted:
        # pause_requested must stay clear on both units the whole time a third
        # LD request is queued behind the saturated priority class. Spy on
        # request_pause_for_target itself (not just the end-state flags) so a
        # transient request-then-clear during the queued window would also be
        # caught, not just a request still outstanding at the moment we check.
        pause_request_calls: list[tuple] = []
        original_request_pause_for_target = scheduler_state_helpers.request_pause_for_target

        def _spy_request_pause_for_target(*args, **kwargs):
            pause_request_calls.append((args, kwargs))
            return original_request_pause_for_target(*args, **kwargs)

        events.append("third_ld_start")
        with mock.patch.object(scheduler_state_helpers, "request_pause_for_target", side_effect=_spy_request_pause_for_target):
            t3 = threading.Thread(target=_run_holding_priority, args=(events, errors, release_gate, "C"), daemon=True)
            t3.start()

            # Wait until the third LD task is actually queued (observable via
            # STATE.queued_sessions reaching the expected count) rather than
            # relying on a fixed sleep to give it "a chance" to (wrongly) attempt
            # preemption -- mirrors the pattern in
            # tests/inference/runtime/test_concurrency_reentrancy.py
            # (test_contention_blocks_are_global_not_thread_local).
            reached_queued = _poll_until(lambda: scheduler.STATE.queued_sessions == 1, timeout=5.0)
            assert reached_queued, "third LD task never reached queued state"
            _assert_third_ld_queued_without_preemption(events)
            assert not pause_request_calls, f"request_pause_for_target was called during the queued window: {pause_request_calls}"

        # Free one unit; the third LD task should now dispatch onto it.
        release_gate.set()
        _assert_all_three_ld_threads_finished(t1, t2, t3, events, errors)


def _start_saturating_threads(
    events: list[str], errors: list[Exception], release_gate: threading.Event
) -> tuple[threading.Thread, threading.Thread]:
    """Start two LD (priority-class) threads and wait until both are confirmed
    running, saturating both hardware units. Returns the two started threads."""
    t1 = threading.Thread(target=_run_holding_priority, args=(events, errors, release_gate, "A"), daemon=True)
    t2 = threading.Thread(target=_run_holding_priority, args=(events, errors, release_gate, "B"), daemon=True)
    t1.start()
    t2.start()

    def _both_running():
        return any(e.startswith("ld_A_running_on_") for e in events) and any(e.startswith("ld_B_running_on_") for e in events)

    _poll_until(_both_running, interval=0.01)
    assert _both_running()
    return t1, t2


def _assert_all_three_ld_threads_finished(
    t1: threading.Thread, t2: threading.Thread, t3: threading.Thread, events: list[str], errors: list[Exception]
) -> None:
    """Assert all three LD threads finished cleanly and the third completed."""
    t1.join(timeout=10.0)
    t2.join(timeout=10.0)
    t3.join(timeout=10.0)

    assert not t1.is_alive() and not t2.is_alive() and not t3.is_alive()
    _assert_no_worker_errors(errors)
    assert "ld_C_done" in events
    assert any(e.startswith("ld_C_running_on_") for e in events)
