"""Tests for the priority model access queue mechanism (Scheduler)."""
# pylint: disable=protected-access, unused-import
import threading
import time
import queue
from unittest import mock
import pytest
from modules.inference import scheduler, model_manager
from modules import utils, config


@pytest.fixture(autouse=True)
def reset_state():
    """Reset global state and threading primitives before each test."""
    from modules.inference.scheduler import SchedulerState
    with mock.patch("modules.config.HARDWARE_UNITS", [{"id": "CPU", "type": "CPU", "name": "CPU"}]):
        scheduler.STATE = SchedulerState()

    # Reset thread context
    utils.THREAD_CONTEXT.is_priority = False
    if hasattr(utils.THREAD_CONTEXT, 'assigned_unit'):
        utils.THREAD_CONTEXT.assigned_unit = None

    yield

    with mock.patch("modules.config.HARDWARE_UNITS", [{"id": "CPU", "type": "CPU", "name": "CPU"}]):
        scheduler.STATE = SchedulerState()


def simulate_confirmation():
    """Helper to auto-confirm pauses in tests."""
    def _target():
        while True:
            if scheduler.STATE.pause_requested.is_set() and not scheduler.STATE.pause_confirmed.is_set():
                scheduler.STATE.pause_confirmed.set()
            if scheduler.STATE.priority_requests == 0 and not any(t.is_alive() for t in threading.enumerate() if t.name.startswith("p_task")):
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


def test_multiple_priority_requests_tracked():
    """Test that multiple priority requests are tracked correctly."""
    scheduler.STATE.active_sessions = 2
    scheduler.STATE.accel_limit = 1

    def p_task():
        scheduler.wait_for_priority()
        time.sleep(0.1)
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

    time.sleep(0.1)
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


def test_priority_sequential_enforcement():
    """Test that priority tasks respect the sequential lock."""
    # 1. Manually acquire the lock to block the next task
    scheduler.STATE.priority_sequential_lock.acquire()

    results = []

    def p_task():
        # This will block at the 'with STATE.priority_sequential_lock' inside early_task_registration
        with scheduler.early_task_registration(is_priority=True):
            scheduler.wait_for_priority()
            results.append("done")
            # release_priority is called automatically by the context finally block

    t = threading.Thread(target=p_task)
    t.start()
    time.sleep(0.1)
    assert len(results) == 0  # Should be blocked

    # 2. Release manually to let it proceed
    scheduler.STATE.priority_sequential_lock.release()
    t.join(timeout=2.0)
    assert len(results) == 1
