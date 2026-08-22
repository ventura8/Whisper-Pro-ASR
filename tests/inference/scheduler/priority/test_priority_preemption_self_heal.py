"""Priority-preemption self-heal/race tests for section 3 (priority preemption) of
docs/E2E_TEST_PLAN_ORCHESTRATION.md.

Covers:
- 3.7: ASR task finishes naturally at (as close as deterministically achievable
  to) the exact moment preemption is being triggered against it -> no
  double-release / lock error, state stays consistent.
- 3.9: simulated worker crash/timeout during the pause-request handshake ->
  the pause-wait paths (`state_helpers.wait_for_pause_confirmation` on the
  priority side, `concurrency._wait_for_resume_signal` on the ASR side) must
  not hang forever. NOTE: neither path has a fixed wall-clock timeout
  constant; both instead self-heal by re-polling registry/state on every
  `cond.wait(timeout=0.1)` tick and bailing out once the condition that
  required waiting no longer holds (see `should_skip_pause_confirmation` and
  `_can_resume_preempted_unit`). This test exercises that self-heal fallback
  and documents it as the existing "timeout" behavior rather than a bug.

Split out of the former, oversized `test_priority_preemption_gaps.py` (past the
repo's 500-line-per-file limit) together with `test_priority_preemption_sla.py`,
which covers the 3.1/3.6 scenarios. Shared thread-rendezvous helpers live in
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


def _run_asr_racing_completion(events: list[str], errors: list[Exception], barrier: threading.Barrier) -> None:
    """ASR thread body for the natural-completion-vs-preemption race test: runs
    to the barrier rendezvous, then makes its single final preemption check,
    racing the priority thread's pause request for the same unit."""
    utils.THREAD_CONTEXT.reset()
    model_manager.increment_active_session()
    try:
        with _capture_worker_exc(errors):
            with model_manager.early_task_registration(is_priority=False):
                with model_manager.model_lock_ctx(priority=False) as (_, unit_id):
                    events.append(f"asr_running_on_{unit_id}")
                    barrier.wait(timeout=5.0)
                    model_manager._check_preemption()
                    events.append("asr_done")
    finally:
        model_manager.decrement_active_session()


def _run_priority_racing_completion(events: list[str], errors: list[Exception], barrier: threading.Barrier) -> None:
    """Priority (language-detection) thread body for the same race test."""
    utils.THREAD_CONTEXT.reset()
    model_manager.increment_active_session()
    try:
        with _capture_worker_exc(errors):
            with model_manager.early_task_registration(task_type="Language Detection", is_priority=True):
                barrier.wait(timeout=5.0)
                model_manager.wait_for_priority()
                events.append("prio_waited")
                with model_manager.model_lock_ctx(priority=True) as (_, unit_id):
                    events.append(f"prio_running_on_{unit_id}")
                    events.append("prio_done")
    finally:
        model_manager.decrement_active_session()


def _assert_race_threads_did_not_hang(t_asr: threading.Thread, t_prio: threading.Thread, errors: list[Exception]) -> None:
    """Assert both racing threads finished (no hang) and raised nothing."""
    assert not t_asr.is_alive(), "ASR thread hung"
    assert not t_prio.is_alive(), "priority thread hung"
    _assert_no_worker_errors(errors)


def _assert_race_threads_completed_cleanly(
    t_asr: threading.Thread, t_prio: threading.Thread, events: list[str], errors: list[Exception]
) -> None:
    """Assert both racing threads finished (no hang), raised nothing, and both
    reached their completion event."""
    _assert_race_threads_did_not_hang(t_asr, t_prio, errors)
    assert "asr_done" in events
    assert "prio_done" in events


def _assert_race_left_scheduler_state_consistent() -> None:
    """Assert scheduler state (preemptible units, hw pool, semaphore permit) is
    fully consistent after the race -- no lingering entry, no leaked permit."""
    assert "NPU.0" not in scheduler.STATE.preemptible_units
    assert scheduler.STATE.hw_pool.qsize() == 1
    # Semaphore should be back to full capacity (1 unit -> 1 permit).
    acquired = scheduler.STATE.model_lock.acquire(blocking=False)
    assert acquired, "model_lock permit was leaked by the race"
    scheduler.STATE.model_lock.release()


def _assert_race_left_consistent_state(
    t_asr: threading.Thread, t_prio: threading.Thread, events: list[str], errors: list[Exception]
) -> None:
    """Post-race assertions: both threads finished cleanly, no exceptions, and
    scheduler state is fully consistent afterwards."""
    _assert_race_threads_completed_cleanly(t_asr, t_prio, events, errors)
    _assert_race_left_scheduler_state_consistent()


def test_asr_natural_completion_races_with_preemption_trigger():
    """3.7: ASR finishes naturally right as preemption is triggered -> no double-release/lock error.

    Uses a barrier to force the interleaving deterministically: the ASR task's
    single, final `_check_preemption()` call is made to land at (as close as
    threads allow) the exact moment the priority task sets the pause request
    for that unit. Repeated multiple times to shake out any race rather than
    relying on single-shot timing luck.
    """
    hw_list = [{"id": "NPU.0", "type": "NPU", "name": "Intel NPU"}]

    for _ in range(15):
        with (
            mock.patch("modules.core.config.HARDWARE_UNITS", hw_list),
            mock.patch("modules.inference.runtime.model_manager.unload_models"),
        ):
            _setup_priority_scheduler(hw_list)

            events = []
            errors = []
            barrier = threading.Barrier(2)

            t_asr = threading.Thread(target=_run_asr_racing_completion, args=(events, errors, barrier), daemon=True)
            t_prio = threading.Thread(target=_run_priority_racing_completion, args=(events, errors, barrier), daemon=True)
            t_asr.start()
            # Ensure ASR reaches the barrier first so both threads race the
            # rendezvous at (as close as possible to) the same wall-clock
            # instant. Poll the ASR thread's own "running" event (appended
            # immediately before its barrier.wait()) instead of a fixed sleep.
            reached_barrier = _poll_until(lambda ev=events: any(e.startswith("asr_running_on_") for e in ev))
            assert reached_barrier, "ASR task never reached the barrier rendezvous"
            t_prio.start()

            t_asr.join(timeout=10.0)
            t_prio.join(timeout=10.0)

            _assert_race_left_consistent_state(t_asr, t_prio, events, errors)


def _assert_pause_wait_self_healed(t_wait: threading.Thread, errors: list[Exception], result: dict) -> None:
    """Assert the waiter thread finished cleanly and self-healed via the registry-driven
    fallback, reporting outcome=True rather than hanging or raising."""
    assert not t_wait.is_alive(), "wait_for_pause_confirmation hung after simulated worker crash"
    _assert_no_worker_errors(errors)
    assert result.get("outcome") is True


def test_pause_confirmation_self_heals_on_simulated_worker_crash():
    """3.9: worker crash mid-pause-handshake must not hang the priority-side wait forever.

    `state_helpers.wait_for_pause_confirmation` has no fixed wall-clock
    timeout constant. Instead, on every 0.1s condition-variable poll it
    re-evaluates `should_skip_pause_confirmation`, which returns True once
    the targeted unit no longer has an active standard task in the registry.
    This test simulates a worker crash (the ASR task's registry entry is
    force-removed/marked non-active without ever setting `pause_confirmed`)
    and asserts the wait unblocks quickly via that fallback rather than
    hanging indefinitely.

    This is a design-gap flag, not just a test gap: there is no explicit,
    named timeout constant for this handshake -- the "timeout" behavior is an
    emergent property of the registry-driven self-heal check. That is
    reported as-is; no production timeout logic is added here per the
    instruction to bias toward reporting over fixing in this high-risk
    concurrency path.
    """
    from modules.inference.scheduler import SchedulerState

    hw_list = [{"id": "NPU.0", "type": "NPU", "name": "Intel NPU"}]
    with mock.patch("modules.core.config.HARDWARE_UNITS", hw_list):
        scheduler.STATE = SchedulerState()

        task_id = "crashed-asr-task"
        with scheduler.STATE.task_registry_lock:
            scheduler.STATE.task_registry[task_id] = {
                "status": "active",
                "is_priority": False,
                "unit_id": "NPU.0",
            }

        pause_gen, wait_confirm = scheduler_state_helpers.request_pause_for_target(scheduler.STATE, "NPU.0")
        assert wait_confirm is True

        result = {}
        errors: list[Exception] = []

        def wait_for_confirm():
            with _capture_worker_exc(errors):
                start = time.time()
                outcome = scheduler_state_helpers.wait_for_pause_confirmation(
                    scheduler.STATE, target_unit_id="NPU.0", expected_generation=pause_gen
                )
                result["elapsed"] = time.time() - start
                result["outcome"] = outcome

        # Observe the waiter actually entering its poll loop via its first
        # should_skip_pause_confirmation check (made on every iteration, including
        # the first, before the loop's cond.wait) instead of a fixed sleep.
        entered_poll_loop = threading.Event()
        original_should_skip_pause_confirmation = scheduler_state_helpers.should_skip_pause_confirmation

        def _spy_should_skip_pause_confirmation(state, target_unit_id):
            entered_poll_loop.set()
            return original_should_skip_pause_confirmation(state, target_unit_id)

        with mock.patch.object(scheduler_state_helpers, "should_skip_pause_confirmation", side_effect=_spy_should_skip_pause_confirmation):
            t_wait = threading.Thread(target=wait_for_confirm, daemon=True)
            t_wait.start()

            # Simulate the worker crashing only once the waiter has confirmed it
            # entered its poll loop: it disappears from the registry before ever
            # setting pause_confirmed.
            assert entered_poll_loop.wait(timeout=5.0), "waiter never entered its poll loop"
            with scheduler.STATE.task_registry_lock:
                del scheduler.STATE.task_registry[task_id]
            with scheduler.STATE.cond:
                scheduler.STATE.cond.notify_all()

            t_wait.join(timeout=5.0)
        _assert_pause_wait_self_healed(t_wait, errors, result)
        # Bounded by the self-heal poll interval (0.1s) with a 10x factor for scheduling
        # slack plus a fixed 0.5s overhead -- generous for CI, still catches genuine hangs.
        _SELF_HEAL_POLL_INTERVAL = 0.1
        assert result["elapsed"] < _SELF_HEAL_POLL_INTERVAL * 10 + 0.5


def test_resume_wait_self_heals_on_simulated_priority_crash():
    """3.9 (ASR side): if the priority task vanishes mid-pause, the paused ASR worker's
    resume wait (`concurrency._wait_for_resume_signal` / `_can_resume_preempted_unit`)
    must not hang forever either -- it self-heals once no queued/active priority
    tasks remain in the registry.
    """
    from modules.inference.scheduler import SchedulerState

    hw_list = [{"id": "NPU.0", "type": "NPU", "name": "Intel NPU"}]
    with mock.patch("modules.core.config.HARDWARE_UNITS", hw_list):
        scheduler.STATE = SchedulerState()

        priority_task_id = "crashed-priority-task"
        with scheduler.STATE.task_registry_lock:
            scheduler.STATE.task_registry[priority_task_id] = {
                "status": "active",
                "is_priority": True,
                "unit_id": None,
            }

        u_sync = scheduler.STATE.unit_sync["NPU.0"]
        u_sync["resume_event"].clear()
        u_sync["pause_requested"].set()
        scheduler.STATE.preemptible_units.add("NPU.0")

        result = {}
        errors: list[Exception] = []

        def wait_for_resume():
            with _capture_worker_exc(errors):
                start = time.time()
                # Production code always calls this while holding STATE.cond
                # (see concurrency._handle_preemption_pause_resume).
                with scheduler.STATE.cond:
                    concurrency._wait_for_resume_signal("NPU.0", u_sync, u_sync["pause_requested"])
                result["elapsed"] = time.time() - start

        # Observe the waiter actually entering its poll loop via its first
        # _can_resume_preempted_unit check (made on every iteration, including
        # the first, before the loop's cond.wait) instead of a fixed sleep.
        entered_poll_loop = threading.Event()
        original_can_resume_preempted_unit = concurrency._can_resume_preempted_unit

        def _spy_can_resume_preempted_unit(unit_id, u_sync_arg, pause_req_evt):
            entered_poll_loop.set()
            return original_can_resume_preempted_unit(unit_id, u_sync_arg, pause_req_evt)

        with mock.patch.object(concurrency, "_can_resume_preempted_unit", side_effect=_spy_can_resume_preempted_unit):
            t_wait = threading.Thread(target=wait_for_resume, daemon=True)
            t_wait.start()

            # Simulate the priority task crashing only once the waiter has confirmed
            # it entered its poll loop: it disappears from the registry without ever
            # releasing the resume event itself.
            assert entered_poll_loop.wait(timeout=5.0), "waiter never entered its poll loop"
            with scheduler.STATE.task_registry_lock:
                del scheduler.STATE.task_registry[priority_task_id]
            with scheduler.STATE.cond:
                scheduler.STATE.cond.notify_all()

            t_wait.join(timeout=5.0)
        assert not t_wait.is_alive(), "_wait_for_resume_signal hung after simulated priority-task crash"
        _assert_no_worker_errors(errors)
        # Bounded by the self-heal poll interval (0.1s) with a 10x factor for scheduling
        # slack plus a fixed 0.5s overhead -- generous for CI, still catches genuine hangs.
        _SELF_HEAL_POLL_INTERVAL = 0.1
        assert result["elapsed"] < _SELF_HEAL_POLL_INTERVAL * 10 + 0.5
