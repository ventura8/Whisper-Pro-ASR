"""Scheduler priority and FIFO behavior tests split from test_scheduler.py."""

import threading
import time
from unittest import mock

import pytest

from modules.core import logging_setup, utils
from modules.inference import scheduler
from modules.inference.scheduler import state_helpers as scheduler_state_helpers


@pytest.fixture(autouse=True)
def reset_state():
    """Reset global state and threading primitives before each test."""
    with mock.patch("modules.core.config.HARDWARE_UNITS", [{"id": "CPU", "type": "CPU", "name": "CPU"}]):
        scheduler.STATE = scheduler.SchedulerState()
        for attr in list(vars(utils.THREAD_CONTEXT).keys()):
            delattr(utils.THREAD_CONTEXT, attr)
        utils.THREAD_CONTEXT.is_priority = False
        utils.THREAD_CONTEXT.assigned_unit = None

        yield

        scheduler.STATE = scheduler.SchedulerState()


def test_priority_does_not_wait_for_unrelated_ffmpeg_once_target_is_preemptible():
    """Priority task should not be blocked by unrelated FFmpeg count."""
    scheduler.STATE.active_sessions = 2
    scheduler.STATE.task_registry["standard_task"] = {
        "task_id": "standard_task",
        "status": "active",
        "is_priority": False,
        "unit_id": "CPU",
        "stage": "Vocal Separation",
    }
    u_sync = scheduler.STATE.unit_sync.get("CPU")

    with utils.STANDARD_FFMPEG_COND:
        utils.STANDARD_FFMPEG_STATE["count"] = 1

    completion = threading.Event()
    wait_thread = None

    def _run_wait():
        scheduler.wait_for_priority()
        scheduler.release_priority()
        completion.set()

    try:
        wait_thread = threading.Thread(target=_run_wait)
        wait_thread.start()

        assert completion.wait(timeout=2.0), "Priority should proceed without waiting on unrelated FFmpeg"
        wait_thread.join(timeout=2.0)
        assert not wait_thread.is_alive()
    finally:
        with utils.STANDARD_FFMPEG_COND:
            utils.STANDARD_FFMPEG_STATE["count"] = 0
            utils.STANDARD_FFMPEG_COND.notify_all()
        scheduler.STATE.task_registry.pop("standard_task", None)
        if u_sync:
            u_sync["pause_confirmed"].clear()
        if wait_thread is not None:
            wait_thread.join(timeout=2.0)


def test_priority_task_starts_as_queued_for_dashboard_visibility():
    """Priority registration must keep task status as queued until hardware is acquired."""
    captured = {}

    with scheduler.early_task_registration(task_type="LD", stage="Language Detection", is_priority=True):
        task_id = getattr(utils.THREAD_CONTEXT, "task_id", None)
        assert task_id is not None
        with scheduler.STATE.task_registry_lock:
            task = scheduler.STATE.task_registry.get(task_id)
            assert task is not None
            captured["status"] = task.get("status")
            captured["stage"] = task.get("stage")

    assert captured["status"] == "queued"
    assert captured["stage"] == "Waiting for Priority Slot"


def test_priority_sequential_lock_is_single_permit_across_hardware_units():
    """Priority lock must enforce one-at-a-time execution regardless of hardware count."""
    with mock.patch(
        "modules.core.config.HARDWARE_UNITS",
        [
            {"id": "NPU.0", "type": "NPU", "name": "Intel NPU"},
            {"id": "GPU.0", "type": "GPU", "name": "Intel GPU"},
        ],
    ):
        scheduler.STATE = scheduler.SchedulerState()

    assert scheduler.STATE.priority_sequential_lock.acquire(blocking=False) is True
    assert scheduler.STATE.priority_sequential_lock.acquire(blocking=False) is False

    scheduler.STATE.priority_sequential_lock.release()


def test_unmark_unit_preemptible_removes_present_unit():
    """Ensure unmark removes an existing preemptible unit."""
    scheduler.mark_unit_preemptible("CPU")
    assert "CPU" in scheduler.STATE.preemptible_units

    scheduler.unmark_unit_preemptible("CPU")

    assert "CPU" not in scheduler.STATE.preemptible_units


def test_get_service_stats_minimal_returns_active_tasks():
    """Ensure minimal stats include active task entries."""
    scheduler.STATE.task_registry["t1"] = {
        "status": "active",
        "unit_type": "CPU",
        "unit_name": "Host CPU",
        "unit_id": "CPU",
        "stage": "Inference",
    }
    scheduler.STATE.task_registry["t2"] = {
        "status": "queued",
        "unit_type": "CPU",
        "unit_name": "Host CPU",
        "unit_id": "CPU",
        "stage": "Queued",
    }

    stats = scheduler.get_service_stats_minimal()

    assert len(stats["active_tasks"]) == 1
    assert stats["active_tasks"][0]["unit_id"] == "CPU"
    assert stats["active_tasks"][0]["stage"] == "Inference"


def test_is_uvr_loaded_reflects_state_flag():
    """Ensure is_uvr_loaded mirrors scheduler state."""
    scheduler.STATE.uvr_loaded = False
    assert scheduler.is_uvr_loaded() is False

    scheduler.STATE.uvr_loaded = True
    assert scheduler.is_uvr_loaded() is True


def test_has_earlier_task_only_blocks_waiting_same_priority_tasks():
    """FIFO gate should block only on earlier same-priority tasks waiting for hardware."""
    now = time.time()
    with scheduler.STATE.task_order_lock:
        scheduler.STATE.task_arrival_order["asr_1"] = now
        scheduler.STATE.task_arrival_order["asr_2"] = now + 1

    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry["asr_1"] = {
            "task_id": "asr_1",
            "is_priority": False,
            "status": "active",
            "unit_id": "NPU.0",
        }
        scheduler.STATE.task_registry["asr_2"] = {
            "task_id": "asr_2",
            "is_priority": False,
            "status": "queued",
            "unit_id": None,
        }

    assert scheduler.has_earlier_task("asr_2", is_priority=False) is False


def test_has_earlier_task_blocks_when_earlier_same_priority_is_waiting():
    """FIFO gate should block when an earlier same-priority task is still waiting."""
    now = time.time()
    with scheduler.STATE.task_order_lock:
        scheduler.STATE.task_arrival_order["asr_1"] = now
        scheduler.STATE.task_arrival_order["asr_2"] = now + 1

    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry["asr_1"] = {
            "task_id": "asr_1",
            "is_priority": False,
            "status": "queued",
            "unit_id": None,
        }
        scheduler.STATE.task_registry["asr_2"] = {
            "task_id": "asr_2",
            "is_priority": False,
            "status": "queued",
            "unit_id": None,
        }

    assert scheduler.has_earlier_task("asr_2", is_priority=False) is True


def test_has_earlier_task_returns_false_when_current_task_missing_and_priority_inferred():
    """When priority is inferred and current task is absent, helper should return False."""
    with scheduler.STATE.task_order_lock:
        scheduler.STATE.task_arrival_order["ghost_task"] = time.time()

    assert scheduler.has_earlier_task("ghost_task") is False


def test_has_earlier_task_infers_priority_from_registry_entry():
    """Inferred priority path should use current task registry metadata."""
    now = time.time()
    with scheduler.STATE.task_order_lock:
        scheduler.STATE.task_arrival_order["prio_1"] = now
        scheduler.STATE.task_arrival_order["prio_2"] = now + 1

    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry["prio_1"] = {
            "task_id": "prio_1",
            "is_priority": True,
            "status": "queued",
            "unit_id": None,
        }
        scheduler.STATE.task_registry["prio_2"] = {
            "task_id": "prio_2",
            "is_priority": True,
            "status": "queued",
            "unit_id": None,
        }

    assert scheduler.has_earlier_task("prio_2") is True


def test_priority_proceeds_without_ffmpeg_wait_when_no_preemption_needed():
    """Priority task does not block on FFmpeg drain when capacity is available and preemption is unnecessary."""
    with utils.STANDARD_FFMPEG_COND:
        utils.STANDARD_FFMPEG_STATE["count"] = 1

    completion = threading.Event()
    errors = []
    wait_thread = None

    def _run_wait():
        try:
            scheduler.wait_for_priority()
        except Exception as exc:
            errors.append(exc)
        finally:
            scheduler.release_priority()
            completion.set()

    try:
        wait_thread = threading.Thread(target=_run_wait)
        wait_thread.start()
        assert completion.wait(timeout=1.0), "Priority should not block on FFmpeg when capacity is available"
        wait_thread.join(timeout=1.0)
        assert not wait_thread.is_alive()
        assert not errors
    finally:
        with utils.STANDARD_FFMPEG_COND:
            utils.STANDARD_FFMPEG_STATE["count"] = 0
            utils.STANDARD_FFMPEG_COND.notify_all()
        if wait_thread is not None:
            wait_thread.join(timeout=1.0)


def test_wait_for_pause_confirmation_requires_matching_generation():
    """Pause confirmation should keep waiting until the expected generation token appears."""
    scheduler.STATE.task_registry["std"] = {
        "task_id": "std",
        "status": "active",
        "is_priority": False,
    }
    u_sync = scheduler.STATE.unit_sync["CPU"]
    u_sync["pause_confirmed"].set()
    u_sync["confirmed_generation"] = 2

    result = {}

    def _wait_for_match():
        result["ok"] = scheduler._wait_for_pause_confirmation("CPU", expected_generation=3)

    worker = threading.Thread(target=_wait_for_match)
    worker.daemon = True
    worker.start()
    worker.join(timeout=0.05)
    assert worker.is_alive()

    u_sync["confirmed_generation"] = 3
    worker.join(timeout=0.5)
    assert not worker.is_alive()
    assert result.get("ok") is True
    scheduler.STATE.task_registry.pop("std", None)


def test_wait_for_pause_confirmation_accepts_legacy_event_without_generation():
    """Legacy tests/setters that only toggle pause_confirmed should still be accepted."""
    scheduler.STATE.task_registry["active_std"] = {
        "task_id": "active_std",
        "status": "active",
        "is_priority": False,
        "unit_id": "CPU",
    }
    try:
        u_sync = scheduler.STATE.unit_sync["CPU"]
        u_sync["pause_confirmed"].set()
        u_sync["confirmed_generation"] = None

        assert scheduler._wait_for_pause_confirmation("CPU", expected_generation=7) is True
    finally:
        scheduler.STATE.task_registry.pop("active_std", None)
        u_sync = scheduler.STATE.unit_sync.get("CPU")
        if u_sync:
            u_sync["pause_confirmed"].clear()


def test_get_standard_task_state_uses_session_fallback_when_registry_empty():
    """Fallback session accounting should mark standard work as active when sessions are present."""
    scheduler.STATE.active_sessions = 1
    scheduler.STATE.priority_requests = 0

    active, initializing = scheduler._get_standard_task_state(None, None)

    assert active is True
    assert initializing is False


def test_get_standard_task_state_skips_session_fallback_for_priority_only_registry():
    """Priority-only registry entries must not be treated as active standard workload."""
    scheduler.STATE.active_sessions = 2
    scheduler.STATE.priority_requests = 1
    scheduler.STATE.task_registry["prio"] = {
        "task_id": "prio",
        "status": "active",
        "is_priority": True,
        "unit_id": "GPU.0",
    }

    active, initializing = scheduler._get_standard_task_state(None, None)

    assert active is False
    assert initializing is False


def test_wait_for_priority_does_not_pause_when_registry_has_only_priority_tasks():
    """Priority-only bursts must not request preemption/pause on any unit."""
    scheduler.STATE.active_sessions = 4
    scheduler.STATE.accel_limit = 2
    scheduler.STATE.task_registry["prio-a"] = {
        "task_id": "prio-a",
        "status": "active",
        "is_priority": True,
        "unit_id": "GPU.0",
    }

    scheduler.wait_for_priority()

    assert scheduler.STATE.pause_requested.is_set() is False


def test_wait_for_standard_task_to_activate_times_out():
    """The wait helper should return False when no standard task activates in time."""
    assert scheduler_state_helpers.wait_for_standard_task_to_activate(scheduler.STATE, None, None, timeout=0.01) is False


def test_wait_for_standard_task_to_activate_ignores_current_task(monkeypatch):
    """The wait helper should skip the current task while scanning for other active work."""
    state = scheduler.SchedulerState()
    task_id = "current-task"
    state.task_registry[task_id] = {
        "task_id": task_id,
        "status": "active",
        "is_priority": False,
    }

    time_list = [0.0, 0.0, 0.6]

    def _time_source():
        if len(time_list) > 1:
            return time_list.pop(0)
        return time_list[0]

    monkeypatch.setattr(scheduler_state_helpers.time, "time", _time_source)
    monkeypatch.setattr(scheduler_state_helpers.time, "sleep", lambda *_args, **_kwargs: None)

    assert scheduler_state_helpers.wait_for_standard_task_to_activate(state, task_id, None, timeout=0.5) is False


def test_has_preferred_idle_unit_returns_true_for_higher_ranked_idle_unit():
    """The helper should detect when an idle lower-tier unit is better than the target unit."""
    state = scheduler.SchedulerState()
    state.hw_pool.put({"id": "CPU"})

    assert (
        scheduler_state_helpers.has_preferred_idle_unit(
            state,
            [{"id": "GPU", "type": "GPU"}, {"id": "CPU", "type": "CPU"}],
            "GPU",
        )
        is True
    )


def test_request_pause_for_target_handles_fallback_and_pausing_target():
    """Pause request should handle both missing sync state and already-pausing target units."""
    scheduler.STATE.resume_event.set()
    fallback_generation, fallback_wait = scheduler._request_pause_for_target("missing-unit")

    assert (
        fallback_generation,
        fallback_wait,
        scheduler.STATE.pause_generation,
        scheduler.STATE.pause_requested.is_set(),
        scheduler.STATE.resume_event.is_set(),
    ) == (scheduler.STATE.pause_generation, False, scheduler.STATE.pause_generation, False, True)

    u_sync = scheduler.STATE.unit_sync["CPU"]
    u_sync["resume_event"].clear()
    already_pausing_generation, already_pausing_wait = scheduler._request_pause_for_target("CPU")

    assert (already_pausing_generation, already_pausing_wait) == (u_sync["pause_generation"], False)


def test_wait_for_pause_confirmation_returns_when_no_active_standard(monkeypatch):
    """Pause confirmation should not block indefinitely once no active standard task remains.

    wait_for_pause_confirmation polls via state.cond.wait(timeout=0.1) (see
    modules/inference/scheduler/state_helpers.py), not time.sleep -- the hook that
    flips the task to inactive must fire from that wait call, not a time.sleep patch,
    or the loop's exit condition never re-evaluates true and this hangs forever."""
    scheduler.STATE.task_registry["std"] = {
        "task_id": "std",
        "status": "active",
        "is_priority": False,
    }

    original_wait = scheduler.STATE.cond.wait

    def _wait_once(*args, **kwargs):
        scheduler.STATE.task_registry["std"]["status"] = "queued"
        return original_wait(*args, **kwargs)

    monkeypatch.setattr(scheduler.STATE.cond, "wait", _wait_once)

    try:
        assert scheduler._wait_for_pause_confirmation("CPU", expected_generation=99) is True
    finally:
        scheduler.STATE.task_registry.pop("std", None)


def test_cleanup_failed_task_removes_arrival_order_entries():
    """cleanup_failed_task should remove registry, logs, and FIFO arrival tracking."""
    task_id = "failed-task"
    thread_id = threading.get_ident()
    utils.THREAD_CONTEXT.task_id = task_id
    utils.THREAD_CONTEXT.registration_thread_id = thread_id

    scheduler.STATE.task_registry[task_id] = {"task_id": task_id, "status": "queued"}
    scheduler.STATE.task_arrival_order[task_id] = time.time()
    scheduler.STATE.task_arrival_order[thread_id] = time.time()
    scheduler.cleanup_failed_task()

    assert task_id not in scheduler.STATE.task_registry
    assert task_id not in scheduler.STATE.task_arrival_order
    assert thread_id not in scheduler.STATE.task_arrival_order


def test_scheduler_task_helpers_cleanup_branches():
    """Verify cleanup_failed_task handles task-id and thread-id lookups."""
    thread_id = threading.get_ident()
    utils.THREAD_CONTEXT.task_id = None
    utils.THREAD_CONTEXT.registration_thread_id = thread_id

    scheduler.STATE.task_registry[thread_id] = {"task_id": "thread-task"}
    logging_setup.TASK_LOGS[thread_id] = ["log1"]

    scheduler.cleanup_failed_task()

    task_id = "some-task-id"
    utils.THREAD_CONTEXT.task_id = task_id
    utils.THREAD_CONTEXT.registration_thread_id = None
    logging_setup.TASK_LOGS[task_id] = ["log2"]
    scheduler.cleanup_failed_task()

    assert (
        thread_id not in scheduler.STATE.task_registry,
        thread_id not in logging_setup.TASK_LOGS,
        task_id not in logging_setup.TASK_LOGS,
    ) == (True, True, True)
