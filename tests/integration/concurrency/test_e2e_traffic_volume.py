"""End-to-end traffic volume tests (1, 5, 10 LD, ASR, and v1 calls, mixed bursts)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any
from unittest import mock

import pytest

from modules.inference import scheduler
from modules.inference.runtime import model_manager
from tests.integration.concurrency.concurrency_fixtures import (
    HW_TOPOLOGY_2_DUAL,
    HW_TOPOLOGY_4_QUAD,
    assert_all_responses_successful,
    auto_confirm_priority_waits,
    execute_concurrent_workload,
    registered_worker_session,
    reset_scheduler_state_and_pools,
    run_concurrency_test_harness,
)
from tests.polling_helpers import poll_until


@pytest.mark.parametrize("volume", [1, 5, 10], ids=["1-call", "5-calls", "10-calls"])
def test_e2e_pure_ld_volume_tiers(sample_wav: str, volume: int):
    """Verify 1, 5, and 10 concurrent priority LD calls complete successfully."""
    with run_concurrency_test_harness(HW_TOPOLOGY_2_DUAL) as client:
        specs = [{"endpoint": "/detect-language" if i % 2 == 0 else "/detectlang", "local_path": sample_wav} for i in range(volume)]
        responses = execute_concurrent_workload(client, specs)
        assert_all_responses_successful(responses)


@pytest.mark.parametrize("volume", [1, 5, 10], ids=["1-call", "5-calls", "10-calls"])
def test_e2e_pure_asr_volume_tiers(sample_wav: str, volume: int):
    """Verify 1, 5, and 10 concurrent standard ASR calls complete successfully."""
    with run_concurrency_test_harness(HW_TOPOLOGY_2_DUAL) as client:
        specs = [{"endpoint": "/asr", "local_path": sample_wav} for _ in range(volume)]
        responses = execute_concurrent_workload(client, specs)
        assert_all_responses_successful(responses)


@pytest.mark.parametrize("volume", [1, 5, 10], ids=["1-call", "5-calls", "10-calls"])
def test_e2e_pure_v1_volume_tiers(sample_wav: str, volume: int):
    """Verify 1, 5, and 10 concurrent v1 transcription & translation calls complete successfully."""
    with run_concurrency_test_harness(HW_TOPOLOGY_2_DUAL) as client:
        specs = []
        for i in range(volume):
            endpoint = "/v1/audio/transcriptions" if i % 2 == 0 else "/v1/audio/translations"
            specs.append({"endpoint": endpoint, "local_path": sample_wav})

        responses = execute_concurrent_workload(client, specs)
        assert_all_responses_successful(responses)


def test_e2e_heavy_mixed_endpoint_burst(sample_wav: str):
    """Verify heavy 25-request mixed endpoint burst (ASR + v1 transcriptions + v1 translations + LD)."""
    with run_concurrency_test_harness(HW_TOPOLOGY_4_QUAD, confidence=0.99) as client:
        specs = []
        for _ in range(5):
            specs.append({"endpoint": "/asr", "local_path": sample_wav})
            specs.append({"endpoint": "/v1/audio/transcriptions", "local_path": sample_wav})
            specs.append({"endpoint": "/v1/audio/translations", "local_path": sample_wav})
            specs.append({"endpoint": "/detect-language", "local_path": sample_wav})
            specs.append({"endpoint": "/detectlang", "local_path": sample_wav})

        assert len(specs) == 25
        responses = execute_concurrent_workload(client, specs, max_workers=25)
        assert_all_responses_successful(responses)


def test_standard_task_not_starved_by_repeated_priority_arrivals() -> None:
    """A standard (ASR) task waiting behind repeated priority-class (detect-language) arrivals
    must still eventually run within a bounded wait — no infinite/unbounded starvation.

    Drives load at the scheduler layer (mirroring tests/inference/scheduler/priority/
    test_priority_fifo_ordering*.py's thread-based harness) rather than through the HTTP client,
    so hold times are controllable and the timing assertion is meaningful rather than trivially
    satisfied by mocked-out, near-instant HTTP handlers.

    A single CPU unit is used so every priority arrival must fully occupy (and release) the only
    unit before the standard task can acquire it. The unit is deliberately occupied by the FIRST
    priority arrival *before* the standard task even starts (confirmed via its own
    "prio_0_acquired_" event, not a fixed sleep), so "standard" is guaranteed to register into
    real, active contention rather than winning a race against a momentarily-idle unit — without
    this, the standard task could simply acquire the free unit immediately and the test would
    pass trivially without exercising any actual starvation-avoidance behavior.

    Priority arrivals then stream on a background thread across an extended window (not a
    small fixed finite batch fired and joined up front) -- each arrival started only once the
    previous has fully RELEASED the unit -- until the standard task finally acquires (or the
    bound is exceeded), at which point the stream is stopped. This is deliberate: with a small
    batch fired and joined entirely before the standard task even starts waiting, the standard
    task's wait would only begin once no priority-class contender remained, trivially winning
    regardless of whether the scheduler is actually fair -- the test could not distinguish
    correct FIFO-fair scheduling from a regression that lets priority arrivals perpetually cut
    ahead of standard work. Running the stream concurrently with the standard task's wait closes
    that gap: standard's acquisition is only observed once it happens while priority arrivals are
    still actively being fired, not merely after a batch already ran dry. Each arrival must still
    fully release before the next one starts (not merely be confirmed acquired) -- the scheduler
    intentionally keeps a unit paused for priority work as long as another priority request is
    already queued (`keep_pause_for_backlog` in modules/inference/scheduler/__init__.py), to avoid
    wasteful pause/resume churn, so a truly gapless stream would keep the backlog permanently
    saturated and starve standard by test construction, not by a real regression. The natural gap
    between one release and the next arrival is exactly the window the scheduler needs to hand the
    unit to standard if it's fair. Bound = 60 iterations * 0.05s hold * 5 (generous safety margin
    for scheduling/registration overhead under test load) + 5s fixed overhead = 20s, well above
    realistic contention time but still a real, finite bound.
    """
    hw_list = [{"id": "CPU", "type": "CPU", "name": "Host CPU"}]
    with mock.patch("modules.core.config.HARDWARE_UNITS", hw_list):
        snapshot = _snapshot_scheduler_and_pools()
        threads: list[threading.Thread] = []
        try:
            _reset_scheduler_and_pools_for_starvation_test(hw_list)
            events, lock, standard_registered, standard_acquired = _starvation_test_state()
            errors: list[Exception] = []

            expected_priority_arrivals = 60
            priority_hold = 0.05
            bound_seconds = expected_priority_arrivals * priority_hold * 5 + 5.0
            # Hard safety cap on the stream, decoupled from the bound_seconds sizing above:
            # at full speed (no scheduling slack) the stream could complete roughly
            # bound_seconds / priority_hold arrivals within the bound. This cap must sit
            # well above that so it only guards against a genuine infinite-loop bug (e.g.
            # stop_event never getting set) -- if it were close to expected_priority_arrivals,
            # the stream could exhaust it and go quiet well before bound_seconds elapses,
            # letting the standard task win once the (capped) priority supply runs dry
            # instead of proving it wins while contention is still genuinely ongoing.
            max_priority_arrivals = int(bound_seconds / priority_hold) * 100

            run_standard = _make_standard_task_runner(events, lock, standard_registered, standard_acquired, errors)
            run_priority = _make_priority_task_runner(events, lock, priority_hold, errors)

            with auto_confirm_priority_waits():
                t_standard, acquired_in_time, elapsed, fired_count = _drive_starvation_scenario(
                    run_standard,
                    run_priority,
                    events,
                    threads,
                    standard_registered=standard_registered,
                    standard_acquired=standard_acquired,
                    bound_seconds=bound_seconds,
                    errors=errors,
                    max_priority_arrivals=max_priority_arrivals,
                )

            t_standard.join(timeout=5.0)

            assert not errors, f"worker thread(s) raised: {errors}"

            _assert_standard_task_not_starved(
                acquired_in_time,
                elapsed,
                bound_seconds,
                t_standard,
                events,
                hw_list=hw_list,
                num_priority_arrivals=fired_count,
            )
            # Note: fired_count can legitimately be as low as 0 -- standard winning the very
            # first gap after arrival 0 releases is the fastest possible proof of fairness, not
            # a weaker one, since the stream was already running concurrently with standard's
            # wait by construction (started before the wait began). Asserting a minimum count
            # here would incorrectly penalize a scheduler for being promptly fair.
        finally:
            # Every started worker is joined here regardless of which path above raised
            # (an assertion, a timeout, or a priority-worker failure), so no thread from
            # this test can outlive it and leak into a later test.
            for t in threads:
                t.join(timeout=5.0)
            _restore_scheduler_and_pools(snapshot)


def _starvation_test_state() -> tuple[list[str], threading.Lock, threading.Event, threading.Event]:
    """Build the shared events list, lock, and coordination events for the starvation test."""
    return [], threading.Lock(), threading.Event(), threading.Event()


def _occupy_unit_with_first_priority_arrival(
    run_priority: Callable[[int], None], events: list[str], threads: list[threading.Thread]
) -> threading.Thread:
    """Start priority arrival 0 and wait until it's confirmed holding the unit,
    so the standard task (started afterward) registers into genuine contention
    rather than an idle unit."""
    t_priority_0 = threading.Thread(target=run_priority, args=(0,), daemon=True)
    threads.append(t_priority_0)
    t_priority_0.start()
    occupied = poll_until(lambda: any(e.startswith("prio_0_acquired_") for e in events))
    assert occupied, "first priority arrival never acquired the unit"
    return t_priority_0


def _register_standard_task_under_contention(
    run_standard: Callable[[], None], standard_registered: threading.Event, threads: list[threading.Thread]
) -> threading.Thread:
    """Start the standard task and wait until it's confirmed registered (queued
    behind the priority arrival occupying the unit)."""
    t_standard = threading.Thread(target=run_standard, daemon=True)
    threads.append(t_standard)
    t_standard.start()
    assert standard_registered.wait(timeout=5.0), "standard task never registered"
    return t_standard


def _run_continuous_priority_stream(
    run_priority: Callable[[int], None],
    events: list[str],
    threads: list[threading.Thread],
    *,
    stop_event: threading.Event,
    start_idx: int,
    errors: list[Exception],
    max_priority_arrivals: int,
    bound_seconds: float,
) -> int:
    """Keep firing sequential priority-class arrivals -- each one started only once
    the previous has fully RELEASED the unit (not merely acquired it) -- until
    `stop_event` is set, `max_priority_arrivals` is reached, or the wall-clock
    deadline (derived from `bound_seconds`) expires, then return the total
    count fired. Running on a background thread means a caller waiting on some other
    event (e.g. the standard task's acquisition) observes that event, if it ever
    fires, concurrently with this stream instead of only after a finite batch has
    run dry -- but each arrival must still fully release before the next one starts,
    leaving a brief natural gap where no priority request is queued. This gap is
    required: the scheduler's anti-thrash guard (`keep_pause_for_backlog =
    queued_priority_count >= STATE.accel_limit` in
    modules/inference/scheduler/__init__.py) deliberately keeps a unit paused for
    priority work as long as another priority request is already queued, to avoid
    wasteful pause/resume churn. Starting the next arrival before the current one
    releases (i.e. the moment it's merely confirmed *acquired*) would keep the
    backlog permanently saturated and starve the standard task by design, not by a
    real regression -- that is not the fairness guarantee this test is meant to
    verify. `max_priority_arrivals` is a secondary hard cap (in case `stop_event` is
    somehow never set and the deadline arithmetic is somehow wrong) rather than the
    primary stopping mechanism. A failure to acquire or release in time is recorded
    into `errors` (the same mechanism worker bodies use) and stops the stream,
    instead of raising a bare assertion on this background thread -- an uncaught
    exception here would otherwise vanish into the thread's default handler, leaving
    the caller to time out on an unrelated wait with no indication of the real
    failure."""
    deadline = time.monotonic() + bound_seconds
    idx = start_idx
    fired = 0
    while not stop_event.is_set() and fired < max_priority_arrivals and time.monotonic() < deadline:
        if not _fire_and_await_one_priority_arrival(run_priority, idx, events, threads, errors):
            break
        idx += 1
        fired += 1
    return fired


def _fire_and_await_one_priority_arrival(
    run_priority: Callable[[int], None],
    idx: int,
    events: list[str],
    threads: list[threading.Thread],
    errors: list[Exception],
) -> bool:
    """Start priority arrival `idx`, wait for it to acquire, then wait for it to fully
    release. Returns True on a clean acquire-then-release cycle, or records the failure
    into `errors` and returns False."""
    t_priority = threading.Thread(target=run_priority, args=(idx,), daemon=True)
    threads.append(t_priority)
    t_priority.start()
    acquired = poll_until(lambda i=idx: any(e.startswith(f"prio_{i}_acquired_") for e in events))
    if not acquired:
        errors.append(AssertionError(f"priority arrival {idx} never acquired the unit"))
        return False
    t_priority.join(timeout=5.0)
    if t_priority.is_alive():
        errors.append(AssertionError(f"priority arrival {idx} did not release the unit in time"))
        return False
    return True


def _drive_starvation_scenario(
    run_standard: Callable[[], None],
    run_priority: Callable[[int], None],
    events: list[str],
    threads: list[threading.Thread],
    *,
    standard_registered: threading.Event,
    standard_acquired: threading.Event,
    bound_seconds: float,
    errors: list[Exception],
    max_priority_arrivals: int,
) -> tuple[threading.Thread, bool, float, int]:
    """Occupy the unit with priority arrival 0, register the standard task into that
    contention, then start a continuously-streaming priority-arrival background thread
    and wait for standard to finally acquire (or the bound to elapse) while that stream
    is still running. Returns (standard_thread, acquired_in_time, elapsed_seconds,
    fired_count) where `fired_count` is how many priority arrivals (beyond arrival 0)
    the stream fired before being stopped."""
    t_priority_0 = _occupy_unit_with_first_priority_arrival(run_priority, events, threads)
    t_standard = _register_standard_task_under_contention(run_standard, standard_registered, threads)

    # Timing measurement begins only once registration under active contention is
    # confirmed, not from thread-start (which would include scheduling slack).
    # Uses time.monotonic() (immune to system-clock adjustments), matching the
    # deadline arithmetic inside _run_continuous_priority_stream.
    start = time.monotonic()

    stop_stream = threading.Event()
    fired_count_box: list[int] = []
    t_stream = threading.Thread(
        target=lambda: fired_count_box.append(
            _run_continuous_priority_stream(
                run_priority,
                events,
                threads,
                stop_event=stop_stream,
                start_idx=1,
                errors=errors,
                max_priority_arrivals=max_priority_arrivals,
                bound_seconds=bound_seconds,
            )
        ),
        daemon=True,
    )
    threads.append(t_stream)
    t_stream.start()

    try:
        # Standard's wait races directly against the still-running priority stream: if
        # standard acquires, the stream is stopped right after -- proving the acquisition
        # happened while priority arrivals were still actively being fired, not merely
        # once a finite batch had already exhausted itself.
        acquired_in_time = standard_acquired.wait(timeout=bound_seconds)
        elapsed = time.monotonic() - start
    finally:
        # Guaranteed even if the wait above raises, so the stream thread is always
        # stopped and joined before this function returns -- otherwise it could keep
        # appending new threads to the shared `threads` list while the caller's own
        # cleanup loop is iterating it (a "list changed size during iteration" risk).
        stop_stream.set()
        t_stream.join(timeout=bound_seconds + 5.0)
    assert not t_stream.is_alive(), "priority stream thread did not stop"
    fired_count = fired_count_box[0] if fired_count_box else 0

    t_priority_0.join(timeout=5.0)
    assert not t_priority_0.is_alive(), "priority arrival 0 did not complete"
    assert not errors, f"priority arrivals raised: {errors}"

    return t_standard, acquired_in_time, elapsed, fired_count


def _snapshot_scheduler_and_pools() -> dict[str, Any]:
    """Capture the current scheduler state and model/preprocessor/diarize/align pool
    contents so they can be restored after this test perturbs them, regardless of outcome."""
    return {
        "state": scheduler.STATE,
        "model_pool": dict(model_manager.MODEL_POOL),
        "preprocessor_pool": dict(model_manager.PREPROCESSOR_POOL),
        "diarize_pool": dict(model_manager.DIARIZE_POOL),
        "align_pool": dict(model_manager.ALIGN_POOL),
    }


def _restore_scheduler_and_pools(snapshot: dict[str, Any]) -> None:
    """Restore scheduler state and model/preprocessor/diarize/align pool contents
    from a snapshot taken by `_snapshot_scheduler_and_pools`."""
    # Cancel any pending idle-unload timer before swapping state/pools out from under
    # it, so a stale timer can't fire later against replaced state and leak into a
    # subsequent test.
    model_manager.cancel_idle_cleanup()
    scheduler.STATE = snapshot["state"]
    model_manager.MODEL_POOL.clear()
    model_manager.MODEL_POOL.update(snapshot["model_pool"])
    model_manager.PREPROCESSOR_POOL.clear()
    model_manager.PREPROCESSOR_POOL.update(snapshot["preprocessor_pool"])
    model_manager.DIARIZE_POOL.clear()
    model_manager.DIARIZE_POOL.update(snapshot["diarize_pool"])
    model_manager.ALIGN_POOL.clear()
    model_manager.ALIGN_POOL.update(snapshot["align_pool"])


def _reset_scheduler_and_pools_for_starvation_test(hw_list: list[dict[str, str]]) -> None:
    """Reset scheduler state and (re)populate the model/preprocessor pools with
    mocks for the given hardware list."""
    reset_scheduler_state_and_pools()
    for unit in hw_list:
        model_manager.MODEL_POOL[unit["id"]] = mock.MagicMock()
        model_manager.PREPROCESSOR_POOL[unit["id"]] = mock.MagicMock()


def _make_standard_task_runner(
    events: list[str],
    lock: threading.Lock,
    standard_registered: threading.Event,
    standard_acquired: threading.Event,
    errors: list[Exception],
) -> Callable[[], None]:
    """Build the thread body for the standard (ASR) task under test: registers
    (signaling `standard_registered`), then blocks (via model_lock_ctx) until it
    can acquire the unit (signaling `standard_acquired`). Any exception raised
    inside is captured into `errors` and re-raised, so a failure surfaces as a
    real error instead of a confusing downstream timeout."""

    def run_standard() -> None:
        try:
            with registered_worker_session(is_priority=False):
                with lock:
                    events.append("standard_registered")
                standard_registered.set()
                with model_manager.model_lock_ctx(priority=False) as (_, unit_id):
                    with lock:
                        events.append(f"standard_acquired_{unit_id}")
                    standard_acquired.set()
        except Exception as exc:  # worker-thread isolation: captured and re-raised
            errors.append(exc)
            raise

    return run_standard


def _make_priority_task_runner(
    events: list[str], lock: threading.Lock, priority_hold: float, errors: list[Exception]
) -> Callable[[int], None]:
    """Build the thread body for a single priority (detect-language) arrival:
    acquires the unit, holds it briefly, then releases. Any exception raised
    inside is captured into `errors` and re-raised, so a failure surfaces as a
    real error instead of a confusing downstream timeout."""

    def run_priority(idx: int) -> None:
        try:
            with registered_worker_session(is_priority=True):
                model_manager.wait_for_priority()
                with model_manager.model_lock_ctx(priority=True) as (_, unit_id):
                    with lock:
                        events.append(f"prio_{idx}_acquired_{unit_id}")
                    time.sleep(priority_hold)
        except Exception as exc:  # worker-thread isolation: captured and re-raised
            errors.append(exc)
            raise

    return run_priority


def _assert_standard_task_eventually_ran(
    acquired_in_time: bool, elapsed: float, bound_seconds: float, t_standard: threading.Thread, events: list[str]
) -> None:
    """Assert the standard task eventually acquired hardware within the bound."""
    assert acquired_in_time, f"Standard task was starved: did not acquire hardware within the {bound_seconds}s bound"
    assert elapsed < bound_seconds
    assert not t_standard.is_alive()
    assert any(event.startswith("standard_acquired_") for event in events)


def _assert_standard_task_not_starved(
    acquired_in_time: bool,
    elapsed: float,
    bound_seconds: float,
    t_standard: threading.Thread,
    events: list[str],
    *,
    hw_list: list[dict[str, str]],
    num_priority_arrivals: int,
) -> None:
    """Assert the standard task eventually acquired hardware within the bound,
    and that every priority arrival (including arrival 0) also ran to completion."""
    _assert_standard_task_eventually_ran(acquired_in_time, elapsed, bound_seconds, t_standard, events)
    total_priority_arrivals = num_priority_arrivals + 1  # + arrival 0, started ahead to occupy the unit
    assert (
        sum(1 for event in events if event.startswith("prio_") and event.endswith(tuple(f"acquired_{u['id']}" for u in hw_list)))
        == total_priority_arrivals
    )
