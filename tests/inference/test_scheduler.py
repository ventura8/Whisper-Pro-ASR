"""Tests for the priority model access queue mechanism (Scheduler)."""

import threading
import time
from unittest import mock

import pytest

from modules.core import logging_setup, utils
from modules.inference import scheduler, scheduler_state_helpers


@pytest.fixture(autouse=True)
def reset_state():
    """Reset global state and threading primitives before each test."""
    with mock.patch("modules.core.config.HARDWARE_UNITS", [{"id": "CPU", "type": "CPU", "name": "CPU"}]):
        scheduler.STATE = scheduler.SchedulerState()
        # Reset thread context completely
        for attr in list(vars(utils.THREAD_CONTEXT).keys()):
            delattr(utils.THREAD_CONTEXT, attr)
        utils.THREAD_CONTEXT.is_priority = False
        utils.THREAD_CONTEXT.assigned_unit = None

        yield

        scheduler.STATE = scheduler.SchedulerState()


def simulate_confirmation():
    """Helper to auto-confirm pauses in tests."""

    def _target():
        while True:
            if scheduler.STATE.pause_requested.is_set() and not scheduler.STATE.pause_confirmed.is_set():
                scheduler.STATE.pause_confirmed.set()
            is_any_ptask = any(t.is_alive() for t in threading.enumerate() if t.name.startswith("p_task"))
            if scheduler.STATE.priority_requests == 0 and not is_any_ptask:
                break
            time.sleep(0.01)

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    return t


def test_wait_for_priority_sets_flags():
    """Test that wait_for_priority sets the correct flags."""
    scheduler.STATE.active_sessions = 2
    scheduler.STATE.accel_limit = 1

    # Run simulation in background
    with mock.patch("modules.inference.scheduler.logger"):

        def run_wait():
            scheduler.wait_for_priority()

        def simulate_confirm():
            # Wait for pause_requested to be set
            start = time.time()
            while not scheduler.STATE.pause_requested.is_set() and time.time() - start < 1.0:
                time.sleep(0.01)
            scheduler.STATE.pause_confirmed.set()

        threading.Thread(target=simulate_confirm).start()
        t = threading.Thread(target=run_wait)
        t.start()

        time.sleep(0.2)
        assert scheduler.STATE.priority_requests == 1
        assert scheduler.STATE.pause_requested.is_set()
        assert not scheduler.STATE.resume_event.is_set()

        # Cleanup
        scheduler.release_priority()
        t.join()


def test_release_priority_clears_flags():
    """Test that release_priority clears flags when counter reaches 0."""
    scheduler.STATE.priority_requests = 1
    scheduler.STATE.pause_requested.set()
    scheduler.STATE.resume_event.clear()

    utils.THREAD_CONTEXT.is_priority = True
    scheduler.release_priority()

    assert scheduler.STATE.priority_requests == 0
    assert not scheduler.STATE.pause_requested.is_set()
    assert scheduler.STATE.resume_event.is_set()


def test_release_priority_resumes_single_unit_when_no_queued_priority_remains():
    """Single-unit deployments must resume ASR immediately when detect-language backlog is empty."""
    scheduler.STATE.priority_requests = 1
    scheduler.STATE.pause_requested.set()
    scheduler.STATE.resume_event.clear()

    u_sync = scheduler.STATE.unit_sync["CPU"]
    u_sync["pause_requested"].set()
    u_sync["resume_event"].clear()
    scheduler.STATE.targeted_units.add("CPU")

    utils.THREAD_CONTEXT.is_priority = True
    utils.THREAD_CONTEXT.target_unit_id = "CPU"

    scheduler.release_priority()

    assert scheduler.STATE.priority_requests == 0
    assert not scheduler.STATE.pause_requested.is_set()
    assert scheduler.STATE.resume_event.is_set()
    assert not u_sync["pause_requested"].is_set()
    assert u_sync["resume_event"].is_set()


def test_wait_for_pause_confirmation_returns_when_target_has_no_active_standard():
    """Targeted confirmation should not block on unrelated active standard tasks."""
    from modules.inference.scheduler import SchedulerState

    with mock.patch(
        "modules.core.config.HARDWARE_UNITS",
        [
            {"id": "GPU.0", "type": "GPU", "name": "Intel GPU"},
            {"id": "NPU.0", "type": "NPU", "name": "Intel NPU"},
        ],
    ):
        scheduler.STATE = SchedulerState()

    scheduler.STATE.task_registry["other_active"] = {
        "task_id": "other_active",
        "is_priority": False,
        "status": "active",
        "unit_id": "NPU.0",
    }

    start = time.time()
    assert scheduler._wait_for_pause_confirmation(target_unit_id="GPU.0", expected_generation=1) is True
    assert time.time() - start < 0.2


def test_release_priority_keeps_pause_asserted_when_priority_tasks_queued():
    """Queued detect-language backlog should keep pause asserted."""
    scheduler.STATE.priority_requests = 1
    scheduler.STATE.pause_requested.set()
    scheduler.STATE.resume_event.clear()

    scheduler.STATE.task_registry["queued_prio"] = {
        "task_id": "queued_prio",
        "is_priority": True,
        "status": "queued",
        "unit_id": None,
    }

    utils.THREAD_CONTEXT.is_priority = True
    scheduler.release_priority()

    assert scheduler.STATE.priority_requests == 0
    assert scheduler.STATE.pause_requested.is_set()
    assert not scheduler.STATE.resume_event.is_set()


def test_release_priority_ignores_coalesced_priority_followers_for_pause():
    """Coalesced queued followers must not block ASR resume when active priority work is done."""
    scheduler.STATE.priority_requests = 1
    scheduler.STATE.pause_requested.set()
    scheduler.STATE.resume_event.clear()

    scheduler.STATE.task_registry["queued_coalesced"] = {
        "task_id": "queued_coalesced",
        "is_priority": True,
        "status": "queued",
        "unit_id": None,
        "coalesced": True,
    }

    utils.THREAD_CONTEXT.is_priority = True
    scheduler.release_priority()

    assert scheduler.STATE.priority_requests == 0
    assert not scheduler.STATE.pause_requested.is_set()
    assert scheduler.STATE.resume_event.is_set()


def test_release_priority_ignores_duplicate_queued_priority_same_source():
    """Queued detect-language retries for the same source must keep the system paused."""
    scheduler.STATE.priority_requests = 1
    scheduler.STATE.pause_requested.set()
    scheduler.STATE.resume_event.clear()

    current_task_id = "active_priority_ld"
    utils.THREAD_CONTEXT.task_id = current_task_id

    scheduler.STATE.task_registry[current_task_id] = {
        "task_id": current_task_id,
        "is_priority": True,
        "status": "active",
        "unit_id": "GPU",
        "source_path": "/tv/American Dad!/Specials/American Dad! - S00E05 - I Love Patrick Stewart SDTV.mp4",
    }
    scheduler.STATE.task_registry["queued_duplicate_ld"] = {
        "task_id": "queued_duplicate_ld",
        "is_priority": True,
        "status": "queued",
        "unit_id": None,
        "request_json": {
            "video_file": "/tv/American Dad!/Specials/American Dad! - S00E05 - I Love Patrick Stewart SDTV.mp4"
        },
    }

    utils.THREAD_CONTEXT.is_priority = True
    scheduler.release_priority()

    assert scheduler.STATE.priority_requests == 0
    assert scheduler.STATE.pause_requested.is_set()
    assert not scheduler.STATE.resume_event.is_set()


def test_release_priority_resumes_when_backlog_below_capacity_on_two_units():
    """On two units, one queued priority task should not keep ASR paused."""
    from modules.inference.scheduler import SchedulerState

    with mock.patch(
        "modules.core.config.HARDWARE_UNITS",
        [
            {"id": "GPU.0", "type": "GPU", "name": "Intel GPU"},
            {"id": "NPU.0", "type": "NPU", "name": "Intel NPU"},
        ],
    ):
        scheduler.STATE = SchedulerState()

    scheduler.STATE.priority_requests = 1
    scheduler.STATE.pause_requested.set()
    scheduler.STATE.resume_event.clear()

    scheduler.STATE.task_registry["queued_prio"] = {
        "task_id": "queued_prio",
        "is_priority": True,
        "status": "queued",
        "unit_id": None,
    }

    utils.THREAD_CONTEXT.is_priority = True
    scheduler.release_priority()

    assert scheduler.STATE.priority_requests == 0
    assert not scheduler.STATE.pause_requested.is_set()
    assert scheduler.STATE.resume_event.is_set()


def test_release_priority_resumes_only_released_unit_when_other_unit_priority_is_active():
    """Releasing priority on one unit must not keep that unit paused due to requests on another unit."""
    from modules.inference.scheduler import SchedulerState

    with mock.patch(
        "modules.core.config.HARDWARE_UNITS",
        [
            {"id": "GPU.0", "type": "GPU", "name": "Intel GPU"},
            {"id": "NPU.0", "type": "NPU", "name": "Intel NPU"},
        ],
    ):
        scheduler.STATE = SchedulerState()

    scheduler.STATE.priority_requests = 2
    scheduler.STATE.unit_priority_requests["GPU.0"] = 1
    scheduler.STATE.unit_priority_requests["NPU.0"] = 1

    gpu_sync = scheduler.STATE.unit_sync["GPU.0"]
    npu_sync = scheduler.STATE.unit_sync["NPU.0"]
    gpu_sync["pause_requested"].set()
    gpu_sync["resume_event"].clear()
    npu_sync["pause_requested"].set()
    npu_sync["resume_event"].clear()

    utils.THREAD_CONTEXT.is_priority = True
    utils.THREAD_CONTEXT.target_unit_id = "NPU.0"
    scheduler.release_priority()

    assert scheduler.STATE.priority_requests == 1
    assert scheduler.STATE.unit_priority_requests["NPU.0"] == 0
    assert scheduler.STATE.unit_priority_requests["GPU.0"] == 1
    assert not npu_sync["pause_requested"].is_set()
    assert npu_sync["resume_event"].is_set()
    assert gpu_sync["pause_requested"].is_set()
    assert not gpu_sync["resume_event"].is_set()


@pytest.mark.parametrize(
    ("unit_count", "queued_priority_count", "expect_paused"),
    [
        (1, 1, True),
        (2, 1, False),
        (2, 2, True),
        (3, 2, False),
        (3, 3, True),
        (4, 3, False),
        (4, 4, True),
    ],
)
def test_release_priority_respects_capacity_threshold(unit_count, queued_priority_count, expect_paused):
    """Pause should remain only when queued priority backlog saturates unit capacity."""
    from modules.inference.scheduler import SchedulerState

    hardware_units = [{"id": f"U{i}", "type": "CPU", "name": f"Unit {i}"} for i in range(unit_count)]

    with mock.patch("modules.core.config.HARDWARE_UNITS", hardware_units):
        scheduler.STATE = SchedulerState()

    scheduler.STATE.priority_requests = 1
    scheduler.STATE.pause_requested.set()
    scheduler.STATE.resume_event.clear()

    for idx in range(queued_priority_count):
        scheduler.STATE.task_registry[f"queued_prio_{idx}"] = {
            "task_id": f"queued_prio_{idx}",
            "is_priority": True,
            "status": "queued",
            "unit_id": None,
        }

    utils.THREAD_CONTEXT.is_priority = True
    scheduler.release_priority()

    assert scheduler.STATE.priority_requests == 0
    assert scheduler.STATE.pause_requested.is_set() is expect_paused
    assert scheduler.STATE.resume_event.is_set() is (not expect_paused)


def test_multiple_priority_requests_tracked():
    """Test that multiple priority requests are tracked correctly."""
    scheduler.STATE.active_sessions = 2
    scheduler.STATE.accel_limit = 1

    def p_task():
        scheduler.wait_for_priority()
        # Keep each priority request active long enough to observe concurrent tracking.
        time.sleep(0.25)
        scheduler.release_priority()

    # Auto-confirm thread
    def auto_confirm():
        while scheduler.STATE.priority_requests >= 0:
            if scheduler.STATE.pause_requested.is_set() and not scheduler.STATE.pause_confirmed.is_set():
                scheduler.STATE.pause_confirmed.set()
            time.sleep(0.01)
            if threading.active_count() <= 2:  # Only main and this thread
                break

    threading.Thread(target=auto_confirm, daemon=True).start()

    t1 = threading.Thread(target=p_task, name="p_task_1")
    t2 = threading.Thread(target=p_task, name="p_task_2")

    t1.start()
    t2.start()

    time.sleep(0.05)
    assert scheduler.STATE.priority_requests >= 1
    assert scheduler.STATE.pause_requested.is_set()

    t1.join()
    t2.join()

    assert scheduler.STATE.priority_requests == 0
    assert not scheduler.STATE.pause_requested.is_set()


def test_release_priority_doesnt_go_negative():
    """Test that release_priority doesn't make counter negative."""
    scheduler.STATE.priority_requests = 0
    utils.THREAD_CONTEXT.is_priority = True
    scheduler.release_priority()
    assert scheduler.STATE.priority_requests == 0


def test_increment_decrement_active_sessions():
    """Test session count tracking."""
    assert scheduler.STATE.active_sessions == 0
    scheduler.increment_active_session()
    assert scheduler.STATE.active_sessions == 1
    scheduler.decrement_active_session()
    assert scheduler.STATE.active_sessions == 0


def test_get_preemptible_unit():
    """Test finding a unit to preempt."""
    thread_id = threading.get_ident()
    scheduler.STATE.unit_ownership["CPU"] = thread_id
    # Not preemptible yet
    assert scheduler.get_preemptible_unit() is None

    # Mark as preemptible
    scheduler.mark_unit_preemptible("CPU")
    assert scheduler.get_preemptible_unit() == "CPU"


def test_priority_not_blocked_by_sequential_lock_state():
    """Priority task registration should not be serialized by priority_sequential_lock."""

    # 1. Manually acquire the lock to block the next task
    scheduler.STATE.priority_sequential_lock.acquire()

    results = []

    def p_task():
        with scheduler.early_task_registration(is_priority=True):
            scheduler.wait_for_priority()
            results.append("done")
            # release_priority is called automatically by the context finally block

    t = threading.Thread(target=p_task)
    t.start()
    time.sleep(0.1)
    assert len(results) == 1

    # Release the manually-acquired compatibility lock to avoid affecting other tests.
    scheduler.STATE.priority_sequential_lock.release()
    t.join(timeout=2.0)


def test_priority_skips_ffmpeg_drain_when_idle_unit_available():
    """Priority task proceeds immediately without waiting for FFmpeg when a free hardware unit exists."""
    # Active standard FFmpeg (simulating ASR in preprocessing on one unit)
    with utils.STANDARD_FFMPEG_COND:
        utils.STANDARD_FFMPEG_STATE["count"] = 1

    results = []
    completion = threading.Event()
    t = None

    try:

        def priority_task():
            scheduler.wait_for_priority()
            results.append("priority_started")
            scheduler.release_priority()
            completion.set()

        t = threading.Thread(target=priority_task)
        t.start()

        # With active_sessions=0 < accel_limit=1, no preemption is needed; priority must NOT block on FFmpeg.
        assert completion.wait(timeout=1.0), "Priority should proceed immediately without waiting for FFmpeg"
        assert results == ["priority_started"]
    finally:
        with utils.STANDARD_FFMPEG_COND:
            utils.STANDARD_FFMPEG_STATE["count"] = 0
            utils.STANDARD_FFMPEG_COND.notify_all()
        if t is not None:
            t.join(timeout=2.0)


def test_priority_does_not_wait_for_ffmpeg_drain_when_preemption_is_needed():
    """At capacity, priority should not be blocked by unrelated standard FFmpeg."""
    # Force at-capacity state so preemption is triggered
    scheduler.STATE.active_sessions = 2  # > accel_limit=1

    with utils.STANDARD_FFMPEG_COND:
        utils.STANDARD_FFMPEG_STATE["count"] = 1

    completion = threading.Event()
    t = None

    def _auto_confirm():
        """Auto-confirm pause events so the test does not deadlock."""
        start = time.time()
        while time.time() - start < 5.0 and not completion.is_set():
            if scheduler.STATE.pause_requested.is_set() and not scheduler.STATE.pause_confirmed.is_set():
                scheduler.STATE.pause_confirmed.set()
                for u_sync in scheduler.STATE.unit_sync.values():
                    if not u_sync["pause_confirmed"].is_set():
                        u_sync["pause_confirmed"].set()
            time.sleep(0.01)

    def _run_wait():
        scheduler.wait_for_priority()
        scheduler.release_priority()
        completion.set()

    try:
        threading.Thread(target=_auto_confirm, daemon=True).start()
        t = threading.Thread(target=_run_wait)
        t.start()

        assert completion.wait(timeout=3.0), "Priority should proceed without waiting for unrelated FFmpeg"
        t.join(timeout=3.0)
    finally:
        with utils.STANDARD_FFMPEG_COND:
            utils.STANDARD_FFMPEG_STATE["count"] = 0
            utils.STANDARD_FFMPEG_COND.notify_all()
        if t is not None:
            t.join(timeout=3.0)


def test_priority_does_not_wait_for_unrelated_ffmpeg_once_target_is_preemptible():
    """Priority task should not be blocked by unrelated FFmpeg count."""
    scheduler.STATE.active_sessions = 2  # force preemption path (accel_limit=1 in fixture)
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

    # Even with 2 units, only one priority permit should be available.
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

    # Earlier ASR already acquired a unit, so it should not block later ASR acquisition.
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

    # is_priority intentionally omitted to cover inference branch.
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
        except Exception as exc:  # pragma: no cover
            errors.append(exc)
        finally:
            scheduler.release_priority()
            completion.set()

    try:
        wait_thread = threading.Thread(target=_run_wait)
        wait_thread.start()
        # Priority should proceed immediately — no preemption needed (active_sessions=0 < accel_limit=1)
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


def test_wait_for_standard_task_to_activate_times_out():
    """The wait helper should return False when no standard task activates in time."""
    assert (
        scheduler_state_helpers.wait_for_standard_task_to_activate(scheduler.STATE, None, None, timeout=0.01) is False
    )


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

    assert fallback_generation == scheduler.STATE.pause_generation
    assert fallback_wait is False
    assert not scheduler.STATE.pause_requested.is_set()
    assert scheduler.STATE.resume_event.is_set()

    u_sync = scheduler.STATE.unit_sync["CPU"]
    u_sync["resume_event"].clear()
    already_pausing_generation, already_pausing_wait = scheduler._request_pause_for_target("CPU")

    assert already_pausing_generation == u_sync["pause_generation"]
    assert already_pausing_wait is False


def test_wait_for_pause_confirmation_returns_when_no_active_standard(monkeypatch):
    """Pause confirmation should not block indefinitely once no active standard task remains."""
    scheduler.STATE.task_registry["std"] = {
        "task_id": "std",
        "status": "active",
        "is_priority": False,
    }

    def _sleep_once(*_args, **_kwargs):
        scheduler.STATE.task_registry["std"]["status"] = "queued"

    monkeypatch.setattr(scheduler.time, "sleep", _sleep_once)

    assert scheduler._wait_for_pause_confirmation("CPU", expected_generation=99) is True


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


def test_scheduler_task_helpers_missing_coverage():
    """Verify registry mutation edge cases to cover all branches in task helpers."""
    # 1. Test cleanup_failed_task when task_id is absent but thread_id is present
    thread_id = threading.get_ident()
    utils.THREAD_CONTEXT.task_id = None
    utils.THREAD_CONTEXT.registration_thread_id = thread_id

    scheduler.STATE.task_registry[thread_id] = {"task_id": "thread-task"}
    # Add items to TASK_LOGS to test log deletion branches
    logging_setup.TASK_LOGS[thread_id] = ["log1"]

    scheduler.cleanup_failed_task()
    assert thread_id not in scheduler.STATE.task_registry
    assert thread_id not in logging_setup.TASK_LOGS

    # Now verify log deletion when task_id is set
    task_id = "some-task-id"
    utils.THREAD_CONTEXT.task_id = task_id
    utils.THREAD_CONTEXT.registration_thread_id = None
    logging_setup.TASK_LOGS[task_id] = ["log2"]
    scheduler.cleanup_failed_task()
    assert task_id not in logging_setup.TASK_LOGS

    # 2. Test update_task_progress TypeError branch with non-numeric values
    task_id = "test-progress-type-error"
    utils.THREAD_CONTEXT.task_id = task_id
    utils.THREAD_CONTEXT.registration_thread_id = None

    # Case A: String current progress, numeric new progress
    scheduler.STATE.task_registry[task_id] = {"progress": "not-started"}
    scheduler.update_task_progress(50, "stage1")
    assert scheduler.STATE.task_registry[task_id]["progress"] == 50
    assert scheduler.STATE.task_registry[task_id]["stage"] == "stage1"

    # Case B: Numeric current progress, string new progress
    scheduler.STATE.task_registry[task_id] = {"progress": 50}
    scheduler.update_task_progress("done", "stage2")
    assert scheduler.STATE.task_registry[task_id]["progress"] == "done"
    assert scheduler.STATE.task_registry[task_id]["stage"] == "stage2"

    # Cleanup task entry
    scheduler.cleanup_failed_task()


def test_release_priority_resumes_targeted_unit_when_queue_empty():
    """A targeted paused unit should resume once the final priority request is released."""
    scheduler.STATE.priority_requests = 1
    scheduler.STATE.pause_requested.set()
    scheduler.STATE.resume_event.clear()
    utils.THREAD_CONTEXT.is_priority = True
    utils.THREAD_CONTEXT.target_unit_id = "CPU"

    scheduler.STATE.unit_sync["CPU"] = {
        "pause_requested": scheduler.STATE.pause_requested,
        "resume_event": scheduler.STATE.resume_event,
        "pause_confirmed": scheduler.STATE.pause_confirmed,
        "confirmed_generation": None,
    }

    scheduler.release_priority()

    assert scheduler.STATE.priority_requests == 0
    assert scheduler.STATE.resume_event.is_set()
    assert utils.THREAD_CONTEXT.target_unit_id is None


def test_request_pause_for_target_sets_targeted_unit_state():
    """The targeted pause request should mark the unit sync entry and record the generation."""
    scheduler.STATE.unit_sync["CPU"] = {
        "pause_requested": threading.Event(),
        "resume_event": threading.Event(),
        "pause_confirmed": threading.Event(),
        "confirmed_generation": None,
    }
    scheduler.STATE.unit_sync["CPU"]["resume_event"].set()

    generation, should_wait = scheduler._request_pause_for_target("CPU")

    assert generation == 1
    assert should_wait is True
    assert scheduler.STATE.pause_generation == 1
    assert scheduler.STATE.unit_sync["CPU"]["pause_requested"].is_set()
    assert not scheduler.STATE.unit_sync["CPU"]["resume_event"].is_set()


def test_wait_for_priority_skips_duplicate_confirmation_wait_when_target_already_pausing(monkeypatch):
    """A follower priority request should not wait again for pause confirmation."""
    scheduler.STATE.active_sessions = 1
    scheduler.STATE.accel_limit = 1
    scheduler.STATE.task_registry["standard-task"] = {
        "task_id": "standard-task",
        "status": "active",
        "is_priority": False,
        "unit_id": "GPU",
    }
    with utils.STANDARD_FFMPEG_COND:
        utils.STANDARD_FFMPEG_STATE["count"] = 0

    monkeypatch.setattr(scheduler, "_get_standard_task_state", lambda *_args: (True, False))
    monkeypatch.setattr(scheduler, "_select_preemption_target_unit", lambda: "CPU")
    monkeypatch.setattr(scheduler_state_helpers, "has_preferred_idle_unit", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(scheduler, "_request_pause_for_target", lambda *_args: (7, False))

    wait_calls = {"count": 0}

    def _unexpected_wait(*_args, **_kwargs):
        wait_calls["count"] += 1
        return True

    monkeypatch.setattr(scheduler, "_wait_for_pause_confirmation", _unexpected_wait)

    scheduler.wait_for_priority()

    assert wait_calls["count"] == 0
    scheduler.release_priority()


def test_update_task_metadata_updates_live_text_for_existing_entry():
    """Live text updates should refresh an existing task entry in place."""
    task_id = "live-task"
    utils.THREAD_CONTEXT.task_id = task_id
    utils.THREAD_CONTEXT.registration_thread_id = threading.get_ident()

    scheduler.STATE.task_registry[task_id] = {
        "task_id": task_id,
        "status": "active",
        "stage": "Queued",
    }

    scheduler.update_task_metadata(live_text="Hello world")

    with scheduler.STATE.task_registry_lock:
        assert scheduler.STATE.task_registry[task_id]["live_text"] == "Hello world"


def test_update_task_metadata_and_progress_create_missing_entry():
    """Missing task updates should log a warning and not create a registry entry."""
    task_id = "missing-task"
    utils.THREAD_CONTEXT.task_id = task_id
    utils.THREAD_CONTEXT.registration_thread_id = threading.get_ident()

    scheduler.update_task_metadata(stage="Queued")
    scheduler.update_task_progress(42, stage="Processing")

    with scheduler.STATE.task_registry_lock:
        assert utils.THREAD_CONTEXT.registration_thread_id not in scheduler.STATE.task_registry
        assert task_id not in scheduler.STATE.task_registry


def test_update_task_progress_does_not_regress_existing_progress_or_stage():
    """Fallback flows must not rewind a task's progress or visible stage."""
    task_id = "ld-task"
    utils.THREAD_CONTEXT.task_id = task_id
    utils.THREAD_CONTEXT.registration_thread_id = threading.get_ident()
    scheduler.STATE.task_registry[task_id] = {
        "task_id": task_id,
        "status": "active",
        "is_priority": True,
        "progress": 60,
        "stage": "Inference",
    }

    scheduler.update_task_progress(5, stage="Vocal Separation")

    with scheduler.STATE.task_registry_lock:
        task_entry = scheduler.STATE.task_registry[task_id]
        assert task_entry["progress"] == 60
        # Stage IS updated even when numeric progress regresses — the anti-regression
        # guard only protects the numeric value, never the stage label.
        assert task_entry["stage"] == "Vocal Separation"


def test_wait_for_priority_waits_for_pause_confirmation_without_timeout(monkeypatch):
    """wait_for_priority should rely on cooperative confirmation without timeout failures."""
    scheduler.STATE.active_sessions = 1
    scheduler.STATE.accel_limit = 1
    scheduler.STATE.task_registry["standard-task"] = {
        "task_id": "standard-task",
        "status": "active",
        "is_priority": False,
        "unit_id": "GPU",
    }
    with utils.STANDARD_FFMPEG_COND:
        utils.STANDARD_FFMPEG_STATE["count"] = 0

    monkeypatch.setattr(scheduler, "_get_standard_task_state", lambda *_args: (True, False))
    monkeypatch.setattr(scheduler, "_select_preemption_target_unit", lambda: "CPU")
    monkeypatch.setattr(scheduler_state_helpers, "has_preferred_idle_unit", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(scheduler, "_request_pause_for_target", lambda *_args: (1, True))
    wait_calls = {"count": 0}

    def _wait(*_args, **_kwargs):
        wait_calls["count"] += 1
        return True

    monkeypatch.setattr(scheduler, "_wait_for_pause_confirmation", _wait)

    scheduler.wait_for_priority()

    assert wait_calls["count"] == 1
    assert scheduler.STATE.priority_requests == 1
    assert utils.THREAD_CONTEXT.is_priority is True
    scheduler.release_priority()


# ---------------------------------------------------------------------------
# should_skip_pause_confirmation branch coverage
# ---------------------------------------------------------------------------


def test_should_skip_pause_confirmation_returns_true_when_target_in_vocal_separation():
    """Skip confirmation when the targeted unit is doing vocal separation (long UVR chunk)."""
    state = scheduler.SchedulerState()
    state.task_registry["asr-1"] = {
        "task_id": "asr-1",
        "status": "active",
        "is_priority": False,
        "unit_id": "GPU.0",
        "stage": "Vocal Separation (Chunk 1/10 | 00:10:00 / 01:30:26)",
    }
    # Vocal separation = skip so priority task can do prep work while UVR chunk runs.
    assert scheduler_state_helpers.should_skip_pause_confirmation(state, "GPU.0") is True


def test_should_skip_pause_confirmation_returns_true_when_target_in_vocal_separation_base_stage():
    """Skip also works for the non-chunked 'Vocal Separation' stage label."""
    state = scheduler.SchedulerState()
    state.task_registry["asr-1"] = {
        "task_id": "asr-1",
        "status": "active",
        "is_priority": False,
        "unit_id": "NPU.0",
        "stage": "Vocal Separation",
    }
    assert scheduler_state_helpers.should_skip_pause_confirmation(state, "NPU.0") is True


def test_should_skip_pause_confirmation_waits_when_target_in_inference():
    """Do not skip confirmation when the targeted unit is in inference (fast yield points)."""
    state = scheduler.SchedulerState()
    state.task_registry["asr-1"] = {
        "task_id": "asr-1",
        "status": "active",
        "is_priority": False,
        "unit_id": "GPU.0",
        "stage": "Inference",
    }
    # Inference has frequent yield checkpoints — wait for proper confirmation.
    assert scheduler_state_helpers.should_skip_pause_confirmation(state, "GPU.0") is False


def test_should_skip_pause_confirmation_returns_true_when_no_active_standard_at_all():
    """Skip when there are zero active standard tasks registered."""
    state = scheduler.SchedulerState()
    assert scheduler_state_helpers.should_skip_pause_confirmation(state, "GPU.0") is True


def test_should_skip_pause_confirmation_returns_true_when_target_has_no_owner():
    """Skip when active standard tasks exist but none own the targeted unit."""
    state = scheduler.SchedulerState()
    state.task_registry["asr-1"] = {
        "task_id": "asr-1",
        "status": "active",
        "is_priority": False,
        "unit_id": "NPU.0",
    }
    # GPU.0 is the target but has no standard task running on it.
    assert scheduler_state_helpers.should_skip_pause_confirmation(state, "GPU.0") is True


def test_should_skip_pause_confirmation_waits_when_target_owns_active_standard():
    """Do not skip when the targeted unit has an active standard task running on it."""
    state = scheduler.SchedulerState()
    state.task_registry["asr-1"] = {
        "task_id": "asr-1",
        "status": "active",
        "is_priority": False,
        "unit_id": "GPU.0",
    }
    assert scheduler_state_helpers.should_skip_pause_confirmation(state, "GPU.0") is False


def test_should_skip_pause_confirmation_waits_when_unit_ownership_unknown():
    """Do not skip when there are active standard tasks with unknown unit assignment."""
    state = scheduler.SchedulerState()
    state.task_registry["asr-1"] = {
        "task_id": "asr-1",
        "status": "active",
        "is_priority": False,
        "unit_id": None,  # ownership not yet assigned
    }
    assert scheduler_state_helpers.should_skip_pause_confirmation(state, "GPU.0") is False


def test_should_skip_pause_confirmation_returns_true_when_unit_preemptible():
    """Skip when the targeted unit is already in the preemptible pool."""
    state = scheduler.SchedulerState()
    state.task_registry["asr-1"] = {
        "task_id": "asr-1",
        "status": "active",
        "is_priority": False,
        "unit_id": "GPU.0",
    }
    state.preemptible_units.add("GPU.0")
    # Even though GPU.0 owns an active standard task, it has already yielded (preemptible).
    assert scheduler_state_helpers.should_skip_pause_confirmation(state, "GPU.0") is True


def test_should_skip_pause_confirmation_no_target_returns_true_when_no_active_standard():
    """Untargeted path: skip when no active standard tasks remain."""
    state = scheduler.SchedulerState()
    state.task_registry["prio-1"] = {
        "task_id": "prio-1",
        "status": "active",
        "is_priority": True,
        "unit_id": "GPU.0",
    }
    # Only priority task is active — should skip.
    assert scheduler_state_helpers.should_skip_pause_confirmation(state, None) is True


def test_should_skip_pause_confirmation_no_target_waits_when_standard_still_active():
    """Untargeted path: do not skip while standard tasks remain active."""
    state = scheduler.SchedulerState()
    state.task_registry["asr-1"] = {
        "task_id": "asr-1",
        "status": "active",
        "is_priority": False,
        "unit_id": "CPU",
    }
    assert scheduler_state_helpers.should_skip_pause_confirmation(state, None) is False
