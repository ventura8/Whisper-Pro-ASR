"""Tests for the priority model access queue mechanism (Scheduler)."""

import threading
import time
from unittest import mock

import pytest

from modules.core import utils
from modules.inference import scheduler


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

    assert (
        scheduler.STATE.priority_requests,
        scheduler.STATE.pause_requested.is_set(),
        scheduler.STATE.resume_event.is_set(),
    ) == (0, False, True)
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
        "request_json": {"video_file": "/tv/American Dad!/Specials/American Dad! - S00E05 - I Love Patrick Stewart SDTV.mp4"},
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

    assert (
        scheduler.STATE.priority_requests,
        scheduler.STATE.unit_priority_requests["NPU.0"],
        scheduler.STATE.unit_priority_requests["GPU.0"],
        npu_sync["pause_requested"].is_set(),
        npu_sync["resume_event"].is_set(),
        gpu_sync["pause_requested"].is_set(),
        gpu_sync["resume_event"].is_set(),
    ) == (1, 0, 1, False, True, True, False)


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

    scheduler.STATE.task_registry.update(
        {
            f"queued_prio_{idx}": {
                "task_id": f"queued_prio_{idx}",
                "is_priority": True,
                "status": "queued",
                "unit_id": None,
            }
            for idx in range(queued_priority_count)
        }
    )

    utils.THREAD_CONTEXT.is_priority = True
    scheduler.release_priority()

    assert (
        scheduler.STATE.priority_requests,
        scheduler.STATE.pause_requested.is_set(),
        scheduler.STATE.resume_event.is_set(),
    ) == (0, expect_paused, not expect_paused)


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


def test_archive_registry_task_normalizes_history_hardware_fields():
    """Archived tasks should retain canonical history hardware metadata."""
    task_id = "task-hw-normalize"
    scheduler.STATE.task_registry[task_id] = {
        "task_id": task_id,
        "filename": "sample.mp4",
        "status": "post-processing",
        "progress": 95,
        "unit_id": None,
        "unit_type": None,
        "unit_name": None,
        "history_unit_id": "CPU",
    }

    archived = scheduler._archive_registry_task(task_id)

    assert archived is not None
    assert (
        archived["history_unit_id"],
        archived["history_unit_type"],
        archived["history_unit_name"],
        archived["unit_id"],
        archived["unit_type"],
        archived["unit_name"],
    ) == ("CPU", "CPU", "CPU", "CPU", "CPU", "CPU")


def test_early_task_registration_exception_records_failure_payload():
    """Uncaught worker exceptions should archive error payloads and logs."""
    with mock.patch("modules.inference.scheduler.history_manager.log_completed_task") as log_mock:
        with pytest.raises(RuntimeError, match="boom"):
            with scheduler.early_task_registration(task_type="Language Detection", filename="test.mkv", is_priority=True):
                raise RuntimeError("boom")

    log_mock.assert_called_once()
    task = log_mock.call_args.args[0]
    assert task["status"] == "failed"
    assert task["result"]["error"] == "boom"
    assert task["response_json"]["error"] == "boom"
    assert task["logs"]


def test_early_task_registration_skips_duplicate_failure_recording():
    """Explicit record_task_failure before re-raise must not overwrite the first failure."""
    with mock.patch("modules.inference.scheduler.history_manager.log_completed_task") as log_mock:
        with pytest.raises(RuntimeError, match="boom"):
            with scheduler.early_task_registration(task_type="ASR", filename="test.mkv"):
                scheduler.record_task_failure("init failed", 400, context="ASR")
                raise RuntimeError("boom")

    log_mock.assert_called_once()
    task = log_mock.call_args.args[0]
    assert task["result"]["error"] == "init failed"
    assert task["response_json"]["error"] == "init failed"


def test_archive_registry_task_ensure_failed_payload_when_missing():
    """Failed archived tasks without payloads should receive a synthesized error."""
    task_id = "failed-empty"
    scheduler.STATE.task_registry[task_id] = {
        "task_id": task_id,
        "filename": "test.mkv",
        "status": "failed",
        "stage": "Waiting for Priority Slot",
    }

    archived = scheduler._archive_registry_task(task_id)

    assert archived is not None
    assert archived["result"]["error"] == "Task failed"
    assert archived["response_json"]["error"] == "Task failed"


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
