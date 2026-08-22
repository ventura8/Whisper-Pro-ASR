"""E2E tests closing the P2 gaps in doc section 2 ("Hybrid split architecture"):

  2.5 NPU saturated -> GPU-side work stays unblocked (no false cross-unit contention).
  2.6 Mixed batch -> GPU pipeline throughput independent of NPU load.
  2.8 NPU becomes unavailable mid-task -> in-flight NPU task fails/cleans up without
      corrupting a concurrently-running GPU task's lock/registry state.

These tests drive `model_manager.model_lock_ctx` directly (like
`test_e2e_preemption_yielding_stages.py`) rather than through HTTP endpoints, so unit
acquisition can be observed and timed precisely.

Determinism note: `scheduler.STATE.hw_pool` is a plain FIFO `queue.Queue` seeded from
`config.HARDWARE_UNITS` in insertion order (see `state_helpers.build_scheduler_state`).
For the 2-unit topology `[NPU.0, GPU.0]` this guarantees the first thread to reach
`hw_pool.get()` receives NPU.0 and the second receives GPU.0, as long as the first
thread's acquisition is confirmed (via an event) before the second thread starts.
This guarantee is only used by the 2.5 and 2.8 tests below, which do confirm the
first acquisition via an event before starting the second thread and then assert
the specific unit id. `test_gpu_throughput_independent_of_continuous_npu_load` (2.6)
deliberately does NOT rely on it -- it derives the busy/free unit dynamically via
`_free_unit_id_given_busy` instead of assuming which unit the continuous-load task
lands on, since prior test ordering effects on `hw_pool` are not guaranteed there.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Generator
from typing import Any

import pytest

from modules.inference import scheduler
from modules.inference.runtime import model_manager
from tests.integration.concurrency.concurrency_fixtures import (
    HW_TOPOLOGY_2_DUAL,
    registered_worker_session,
    setup_concurrency_harness,
)


@pytest.fixture
def dual_harness() -> Generator[None, None, None]:
    """Patch scheduler state with the 2-unit [NPU.0, GPU.0] hardware topology for the test's duration."""
    harness, _ = setup_concurrency_harness(HW_TOPOLOGY_2_DUAL)
    with harness:
        yield


def _worker_hold_unit(
    events: list[str],
    label: str,
    acquired_evt: threading.Event,
    release_evt: threading.Event,
    timing: dict[str, float],
) -> None:
    """Acquire a unit, signal once held, then block until told to release."""
    try:
        with registered_worker_session():
            with model_manager.model_lock_ctx(priority=False) as (_, unit_id):
                timing[f"{label}_acquired_at"] = time.time()
                events.append(f"{label}_acquired_{unit_id}")
                acquired_evt.set()
                released = release_evt.wait(timeout=10.0)
                if released:
                    events.append(f"{label}_released")
    finally:
        timing[f"{label}_thread_done_at"] = time.time()


def _worker_quick_task(events: list[str], label: str, timing: dict[str, float], work_sec: float = 0.05) -> None:
    """Acquire a unit, do a short bit of work, release. Records start/end timing."""
    with registered_worker_session():
        start = time.time()
        timing[f"{label}_submit_to_acquire_start"] = start
        with model_manager.model_lock_ctx(priority=False) as (_, unit_id):
            acquired = time.time()
            timing[f"{label}_wait_before_acquire"] = acquired - start
            events.append(f"{label}_acquired_{unit_id}")
            time.sleep(work_sec)
            events.append(f"{label}_done")
        timing[f"{label}_total_duration"] = time.time() - start


def _run_worker_capturing_errors(target: Callable[..., None], args: tuple[Any, ...]) -> list[BaseException]:
    """Run `target(*args)` on a background thread, join it, and return any
    exception it raised (instead of letting it vanish into the thread's
    default top-level handler)."""
    errors: list[BaseException] = []

    def _wrapped() -> None:
        try:
            target(*args)
        except BaseException as exc:  # worker-thread isolation: captured and re-raised
            errors.append(exc)
            raise

    t = threading.Thread(target=_wrapped, daemon=True)
    t.start()
    t.join(timeout=5.0)
    assert not t.is_alive(), f"worker thread {target.__name__} did not finish within timeout"
    return errors


def _measure_baseline_gpu_duration() -> float:
    """Run a single quick task and return its total duration, for comparison against loaded runs.

    Runs on whichever unit is currently free -- some call sites measure this
    while another unit is already held/saturated by a concurrent task, so
    "baseline" here means "unburdened on the unit it lands on", not
    "measured on a fully idle system"."""
    baseline_timing: dict[str, float] = {}
    errors = _run_worker_capturing_errors(_worker_quick_task, ([], "baseline", baseline_timing, 0.05))
    if errors:
        raise errors[0]
    return baseline_timing["baseline_total_duration"]


def _worker_raise_mid_task(events: list[str], label: str, acquired_evt: threading.Event) -> None:
    """Acquire a unit, signal, then raise mid-operation to simulate device loss."""
    with registered_worker_session():
        with model_manager.model_lock_ctx(priority=False) as (_, unit_id):
            events.append(f"{label}_acquired_{unit_id}")
            acquired_evt.set()
            time.sleep(0.03)
            raise RuntimeError("Simulated NPU driver reset / device loss")


# --------------------------------------------------------------------------- #
# 2.5: NPU saturated -> GPU-targeted work proceeds independently, unblocked.
# --------------------------------------------------------------------------- #


def _assert_gpu_task_ran_to_completion(t_gpu: threading.Thread | None, events: list[str]) -> None:
    """Assert the GPU task thread finished and its acquire/done events fired."""
    if t_gpu is not None:
        assert not t_gpu.is_alive()
    assert "gpu_transcribe_acquired_GPU.0" in events
    assert "gpu_transcribe_done" in events


def _assert_gpu_task_completed_unblocked(
    t_gpu: threading.Thread, events: list[str], gpu_elapsed: float, baseline_duration: float = 0.0
) -> None:
    """Assert the GPU task finished quickly and without waiting on the saturated NPU unit.

    The threshold is workload-relative: baseline_duration * 10 + 0.5s gives enough
    headroom for CI scheduling overhead while still catching actual blocking regressions.
    A fixed 1.0s threshold would be fragile on slow CI runners.
    """
    _assert_gpu_task_ran_to_completion(t_gpu, events)
    # GPU task must not have waited on the saturated NPU unit at all: total
    # wall time should be close to its own work duration, not the NPU hold.
    bound = baseline_duration * 10 + 0.5 if baseline_duration > 0 else 1.0
    assert gpu_elapsed < bound, f"GPU task took {gpu_elapsed:.3f}s (bound={bound:.3f}s); looks blocked by saturated NPU unit"
    # NPU task is still holding its unit (hasn't been released) at this point.
    assert "npu_saturating_released" not in events


@pytest.mark.usefixtures("dual_harness")
def test_gpu_transcription_unblocked_while_npu_saturated() -> None:
    """Verify a GPU-targeted task acquires GPU.0 and completes quickly while NPU.0 is saturated."""
    events: list[str] = []
    timing: dict[str, float] = {}
    npu_acquired = threading.Event()
    npu_release = threading.Event()

    t_npu = threading.Thread(
        target=_worker_hold_unit,
        args=(events, "npu_saturating", npu_acquired, npu_release, timing),
        daemon=True,
    )
    t_npu.start()
    try:
        assert npu_acquired.wait(timeout=2.0)
        # Confirm determinism: first thread through the FIFO hw_pool gets NPU.0.
        assert "npu_saturating_acquired_NPU.0" in events

        # Measure an unburdened baseline so the bound is workload-relative, not fixed.
        baseline_duration = _measure_baseline_gpu_duration()

        gpu_start = time.time()
        gpu_errors = _run_worker_capturing_errors(_worker_quick_task, (events, "gpu_transcribe", timing, 0.05))
        gpu_elapsed = time.time() - gpu_start

        assert not gpu_errors, f"gpu_transcribe worker raised: {gpu_errors}"
        _assert_gpu_task_completed_unblocked(
            None,  # thread already joined inside _run_worker_capturing_errors
            events,
            gpu_elapsed,
            baseline_duration,
        )
    finally:
        # Guaranteed even if an assertion above raises, so the NPU-holding
        # daemon thread never lingers past this test into fixture teardown.
        npu_release.set()
        t_npu.join(timeout=5.0)
    assert not t_npu.is_alive()


# --------------------------------------------------------------------------- #
# 2.6: mixed batch -> GPU-side throughput stays independent of ongoing NPU load.
# --------------------------------------------------------------------------- #


def _run_npu_load_loop(events: list[str], npu_acquired: threading.Event, npu_stop: threading.Event) -> None:
    """Hold a unit continuously (busy-waiting) until `npu_stop` is set, simulating continuous NPU load."""
    with registered_worker_session():
        with model_manager.model_lock_ctx(priority=False) as (_, unit_id):
            events.append(f"npu_loop_acquired_{unit_id}")
            npu_acquired.set()
            while not npu_stop.is_set():
                time.sleep(0.02)
            events.append("npu_loop_released")


def _free_unit_id_given_busy(events: list[str]) -> str:
    """Given the continuous-load task's acquisition event, return the id of the
    other (free) unit that subsequent GPU-side tasks must land on."""
    npu_busy_event = next(e for e in events if e.startswith("npu_loop_acquired_"))
    busy_unit_id = npu_busy_event.rsplit("_", 1)[-1]
    all_unit_ids = {unit["id"] for unit in HW_TOPOLOGY_2_DUAL}
    remaining_unit_ids = all_unit_ids - {busy_unit_id}
    return remaining_unit_ids.pop()


def _run_mixed_batch_and_collect_durations(events: list[str], timing: dict[str, float], free_unit_id: str) -> list[float]:
    """Run a back-to-back batch of GPU-side quick tasks, asserting each lands on
    `free_unit_id`, and return their total durations."""
    durations: list[float] = []
    for i in range(5):
        label = f"mixed_{i}"
        worker_errors = _run_worker_capturing_errors(_worker_quick_task, (events, label, timing, 0.05))
        assert not worker_errors, f"{label} worker raised: {worker_errors}"
        assert f"{label}_acquired_{free_unit_id}" in events
        durations.append(timing[f"{label}_total_duration"])
    return durations


def _assert_durations_near_baseline(durations: list[float], baseline_duration: float) -> None:
    """Assert each GPU task's duration stayed close to the unburdened baseline,
    i.e. NPU load did not leak into GPU throughput."""
    for i, d in enumerate(durations):
        # Workload-relative bound, matching _assert_gpu_task_completed_unblocked's formula:
        # generous headroom for CI scheduling overhead while still catching real leaks.
        bound = baseline_duration * 10 + 0.5
        within_bound = d < bound
        msg = f"GPU task {i} took {d:.3f}s vs baseline {baseline_duration:.3f}s (bound={bound:.3f}s); NPU load leaked into GPU throughput"
        assert within_bound, msg


@pytest.mark.usefixtures("dual_harness")
def test_gpu_throughput_independent_of_continuous_npu_load() -> None:
    """Verify a batch of GPU-side tasks completes near baseline duration despite continuous NPU load."""
    events: list[str] = []
    timing: dict[str, float] = {}
    npu_acquired = threading.Event()
    npu_stop = threading.Event()

    # Baseline: measure a lone GPU task's duration with no NPU contention.
    baseline_duration = _measure_baseline_gpu_duration()

    # Now saturate NPU.0 continuously and run a "mixed batch" of GPU-side quick
    # tasks (representing files that skip straight to GPU transcription) back to back.
    t_npu = threading.Thread(target=_run_npu_load_loop, args=(events, npu_acquired, npu_stop), daemon=True)
    t_npu.start()
    try:
        assert npu_acquired.wait(timeout=2.0)
        # Whichever unit the continuous-load task lands on (FIFO ordering may hand it
        # either unit depending on prior returns), the other unit is the free one that
        # subsequent GPU-side tasks must consistently land on and complete quickly.
        free_unit_id = _free_unit_id_given_busy(events)

        durations = _run_mixed_batch_and_collect_durations(events, timing, free_unit_id)
        _assert_durations_near_baseline(durations, baseline_duration)
    finally:
        # Guaranteed even if an assertion above raises, so the NPU-holding
        # daemon thread never lingers past this test into fixture teardown.
        npu_stop.set()
        t_npu.join(timeout=5.0)
    assert not t_npu.is_alive()
    assert "npu_loop_released" in events


# --------------------------------------------------------------------------- #
# 2.8: NPU becomes unavailable mid-run -> NPU task fails cleanly, GPU task unaffected.
# --------------------------------------------------------------------------- #


def _npu_worker_wrapper(events: list[str], npu_acquired: threading.Event, npu_exc: dict[str, Exception]) -> None:
    # _worker_raise_mid_task always raises RuntimeError to simulate device loss.
    # Caught broadly here (not just RuntimeError) so any other exception raised
    # during model_lock_ctx's teardown/unwind is also captured for assertion in
    # the main thread, rather than being silently swallowed by the thread's
    # default top-level handler. _assert_npu_task_failed_cleanly still asserts
    # the captured error is a RuntimeError, so an unexpected type correctly
    # fails the test instead of hanging.
    try:
        _worker_raise_mid_task(events, "npu_failing", npu_acquired)
    except Exception as exc:  # worker-thread isolation: captured and re-raised
        npu_exc["error"] = exc
        raise


def _assert_npu_task_failed_cleanly(npu_exc: dict[str, Exception]) -> None:
    """Assert the NPU-side task's simulated device-loss error surfaced (not swallowed)."""
    assert "error" in npu_exc
    assert isinstance(npu_exc["error"], RuntimeError)
    assert "Simulated NPU driver reset" in str(npu_exc["error"])


def _assert_gpu_task_unaffected(events: list[str]) -> None:
    """Assert the concurrently-running GPU task was completely unaffected by the NPU failure."""
    assert "gpu_concurrent_acquired_GPU.0" in events
    assert "gpu_concurrent_done" in events


def _assert_npu_unit_returned_to_pool() -> None:
    """Assert model_lock_ctx's finally-block released the NPU unit/semaphore on
    the exception path (no lock leak), so it's available again in the pool."""
    with scheduler.STATE.hw_pool.mutex:
        pool_ids = [u.get("id") for u in list(scheduler.STATE.hw_pool.queue) if isinstance(u, dict)]
    assert "NPU.0" in pool_ids


def _assert_npu_unit_recoverable_after_failure() -> None:
    """Assert a follow-up task can acquire some unit again without hanging,
    proving hardware recovery after the simulated device-loss failure.

    Does not assert the specific unit id: which of NPU.0/GPU.0 ends up at the
    head of the FIFO hw_pool after both threads release depends on their
    relative release timing (npu_failing sleeps 0.03s before raising,
    gpu_concurrent holds for 0.08s) -- not a structural guarantee like the
    module's other documented determinism note (which relies on an explicit
    acquisition-confirmed event, not relative sleep durations). Asserting a
    specific unit here would be a latent flakiness risk under CI scheduling
    jitter rather than a real behavioral requirement."""
    recovery_events: list[str] = []
    recovery_timing: dict[str, float] = {}
    recovery_acquired = threading.Event()
    release_evt = threading.Event()
    t_recover = threading.Thread(
        target=_worker_hold_unit,
        args=(recovery_events, "npu_recovered", recovery_acquired, release_evt, recovery_timing),
        daemon=True,
    )
    t_recover.start()
    assert recovery_acquired.wait(timeout=2.0)
    assert any(e.startswith("npu_recovered_acquired_") for e in recovery_events)
    release_evt.set()
    t_recover.join(timeout=5.0)
    assert not t_recover.is_alive()


def _start_gpu_worker_capturing_errors(events: list[str], timing: dict[str, float], gpu_errors: list[BaseException]) -> threading.Thread:
    """Start the concurrent GPU quick-task thread, capturing any raised exception
    into `gpu_errors` instead of letting it vanish into the thread's default
    top-level handler. Does not join -- the caller is responsible for that."""

    def _wrapped() -> None:
        try:
            _worker_quick_task(events, "gpu_concurrent", timing, 0.08)
        except BaseException as exc:  # worker-thread isolation: captured and re-raised
            gpu_errors.append(exc)
            raise

    t_gpu = threading.Thread(target=_wrapped, daemon=True)
    t_gpu.start()
    return t_gpu


def _run_npu_failure_and_concurrent_gpu_task(
    events: list[str],
    timing: dict[str, float],
    npu_acquired: threading.Event,
    npu_exc: dict[str, Exception],
) -> tuple[threading.Thread, threading.Thread | None]:
    """Start the failing-NPU worker and, once it's confirmed to hold NPU.0, a concurrent
    GPU quick task. Both are joined regardless of any assertion failure in between, so
    neither daemon thread lingers past this test into fixture teardown. Returns
    (t_npu, t_gpu) -- t_gpu is None if the NPU-side assertions failed before it started."""
    t_npu = threading.Thread(target=_npu_worker_wrapper, args=(events, npu_acquired, npu_exc), daemon=True)
    t_npu.start()
    gpu_errors: list[BaseException] = []
    t_gpu: threading.Thread | None = None
    try:
        assert npu_acquired.wait(timeout=2.0)
        assert "npu_failing_acquired_NPU.0" in events
        # Start a concurrent GPU task while the NPU task is mid-failure.
        t_gpu = _start_gpu_worker_capturing_errors(events, timing, gpu_errors)
    finally:
        t_npu.join(timeout=5.0)
        if t_gpu is not None:
            t_gpu.join(timeout=5.0)
    assert not gpu_errors, f"gpu_concurrent worker raised: {gpu_errors}"
    return t_npu, t_gpu


def _assert_npu_and_gpu_threads_finished(t_npu: threading.Thread, t_gpu: threading.Thread | None) -> None:
    """Assert both the NPU-side and GPU-side threads finished (no hang, no leftover daemon)."""
    assert not t_npu.is_alive()
    assert t_gpu is not None and not t_gpu.is_alive()


@pytest.mark.usefixtures("dual_harness")
def test_npu_device_loss_mid_task_does_not_corrupt_concurrent_gpu_task() -> None:
    """Verify NPU device loss mid-task fails cleanly and doesn't corrupt a concurrently-running GPU task."""
    events: list[str] = []
    timing: dict[str, float] = {}
    npu_acquired = threading.Event()
    npu_exc: dict[str, Exception] = {}

    t_npu, t_gpu = _run_npu_failure_and_concurrent_gpu_task(events, timing, npu_acquired, npu_exc)
    _assert_npu_and_gpu_threads_finished(t_npu, t_gpu)

    # (a) The NPU-side task must have actually failed (error surfaced, not swallowed).
    _assert_npu_task_failed_cleanly(npu_exc)

    # (b) The GPU task, running concurrently, must be completely unaffected.
    _assert_gpu_task_unaffected(events)

    # Cleanup correctness: the NPU unit must be back in the pool, and usable again.
    _assert_npu_unit_returned_to_pool()
    _assert_npu_unit_recoverable_after_failure()
