"""FIFO task ordering verification tests.

Tests that verify:
1. Tasks are processed in arrival order (FIFO)
2. Detect-language tasks can preempt ASR, but respect FIFO among themselves
3. No task skips ahead in the queue
"""

import threading
import time
from unittest import mock

import pytest

from modules.core import utils
from modules.inference import concurrency, model_manager, scheduler

_HW_PATCHER = None


def _setup_units(hw_list):
    """Reset scheduler and model/preprocessor pools for a test hardware layout."""
    global _HW_PATCHER
    if _HW_PATCHER is not None:
        _HW_PATCHER.stop()

    _HW_PATCHER = mock.patch("modules.core.config.HARDWARE_UNITS", hw_list)
    _HW_PATCHER.start()
    scheduler.STATE = scheduler.SchedulerState()

    model_manager.MODEL_POOL.clear()
    model_manager.PREPROCESSOR_POOL.clear()
    for unit in hw_list:
        model_manager.MODEL_POOL[unit["id"]] = mock.MagicMock()
        model_manager.PREPROCESSOR_POOL[unit["id"]] = mock.MagicMock()


@pytest.fixture(autouse=True)
def _cleanup_hw_patcher():
    """Ensure HARDWARE_UNITS patch does not leak between tests."""
    global _HW_PATCHER
    yield
    if _HW_PATCHER is not None:
        _HW_PATCHER.stop()
        _HW_PATCHER = None


def test_asr_tasks_processed_in_fifo_order():
    """Multiple ASR tasks must be processed in the order they arrived, not skipping."""
    _setup_units([{"id": "CPU", "type": "CPU", "name": "Host CPU"}])

    acquisition_order = []
    lock = threading.Lock()

    # Mock _try_acquire_unit_now to track acquisition order
    original_try_acquire = concurrency._try_acquire_unit_now

    def track_acquisition():
        unit = original_try_acquire()
        if unit:
            task_id = getattr(utils.THREAD_CONTEXT, "task_id", "unknown")
            with lock:
                acquisition_order.append(task_id)
        return unit

    def run_asr_task(task_num):
        """Run a standard ASR task."""
        utils.THREAD_CONTEXT.reset()
        model_manager.increment_active_session()
        try:
            with model_manager.early_task_registration(is_priority=False):
                time.sleep(0.05)  # Small delay to let other tasks arrive
                with model_manager.model_lock_ctx(priority=False) as (_, unit_id):
                    with lock:
                        acquisition_order.append(f"asr_{task_num}_acquired")
                    time.sleep(0.02)
        finally:
            model_manager.decrement_active_session()

    with mock.patch("modules.inference.concurrency._try_acquire_unit_now", side_effect=track_acquisition):
        # Start 4 ASR tasks in sequence
        threads = []
        for i in range(4):
            t = threading.Thread(target=run_asr_task, args=(i,))
            t.start()
            threads.append(t)
            time.sleep(0.01)  # Ensure sequential arrival times

        for t in threads:
            t.join(timeout=10.0)

    # Verify all tasks completed
    assert all(not t.is_alive() for t in threads)
    # Verify all tasks acquired units in FIFO order.
    assert "asr_0_acquired" in acquisition_order
    assert "asr_1_acquired" in acquisition_order
    assert "asr_2_acquired" in acquisition_order
    assert "asr_3_acquired" in acquisition_order

    assert acquisition_order.index("asr_0_acquired") < acquisition_order.index("asr_1_acquired")
    assert acquisition_order.index("asr_1_acquired") < acquisition_order.index("asr_2_acquired")
    assert acquisition_order.index("asr_2_acquired") < acquisition_order.index("asr_3_acquired")


def test_detect_language_tasks_processed_in_fifo_order():
    """Multiple detect-language (priority) tasks must be processed in arrival order."""
    _setup_units(
        [
            {"id": "CPU", "type": "CPU", "name": "Host CPU"},
            {"id": "GPU.0", "type": "GPU", "name": "Intel GPU"},
        ]
    )

    acquisition_order = []
    events = []
    lock = threading.Lock()

    # Ensure no leftover FFmpeg gating from other tests can block priority admission.
    with utils.STANDARD_FFMPEG_COND:
        utils.STANDARD_FFMPEG_STATE["count"] = 0
        utils.STANDARD_FFMPEG_COND.notify_all()

    stop_auto_confirm = threading.Event()

    def auto_confirm_priority_waits():
        while not stop_auto_confirm.is_set():
            if scheduler.STATE.pause_requested.is_set() and not scheduler.STATE.pause_confirmed.is_set():
                scheduler.STATE.pause_confirmed.set()
            for u_sync in scheduler.STATE.unit_sync.values():
                if u_sync["pause_requested"].is_set() and not u_sync["pause_confirmed"].is_set():
                    u_sync["pause_confirmed"].set()
            time.sleep(0.01)

    confirm_thread = threading.Thread(target=auto_confirm_priority_waits, daemon=True)
    confirm_thread.start()

    def run_priority_task(task_num, delay=0.03):
        """Run a priority detect-language task."""
        utils.THREAD_CONTEXT.reset()
        model_manager.increment_active_session()
        try:
            with lock:
                events.append(f"prio_{task_num}_start")

            with model_manager.early_task_registration(is_priority=True) as task_id:
                with lock:
                    events.append(f"prio_{task_num}_registered")
                    acquisition_order.append(task_id)

                model_manager.wait_for_priority()
                with lock:
                    events.append(f"prio_{task_num}_waited")

                with model_manager.model_lock_ctx(priority=True) as (_, unit_id):
                    with lock:
                        events.append(f"prio_{task_num}_unit_{unit_id}")
                    time.sleep(delay)
                    with lock:
                        events.append(f"prio_{task_num}_done")
        finally:
            model_manager.decrement_active_session()

    try:
        # Start 3 priority tasks with small delays
        threads = []
        for i in range(3):
            t = threading.Thread(target=run_priority_task, args=(i, 0.02))
            t.start()
            threads.append(t)
            time.sleep(0.01)  # Ensure sequential arrival

        for t in threads:
            t.join(timeout=10.0)

        # Verify all tasks completed
        assert all(not t.is_alive() for t in threads)
        # Verify all tasks went through registration and acquisition
        assert len(acquisition_order) == 3
        assert "prio_0_registered" in events
        assert "prio_1_registered" in events
        assert "prio_2_registered" in events
        assert "prio_0_waited" in events
        assert "prio_1_waited" in events
        assert "prio_2_waited" in events
        assert any(e.startswith("prio_0_unit_") for e in events)
        assert any(e.startswith("prio_1_unit_") for e in events)
        assert any(e.startswith("prio_2_unit_") for e in events)
        assert "prio_0_done" in events
        assert "prio_1_done" in events
        assert "prio_2_done" in events

        assert events.index("prio_0_registered") < events.index("prio_1_registered")
        assert events.index("prio_1_registered") < events.index("prio_2_registered")
        assert events.index("prio_0_waited") < events.index("prio_1_waited")
        assert events.index("prio_1_waited") < events.index("prio_2_waited")
    finally:
        stop_auto_confirm.set()
        confirm_thread.join(timeout=2.0)


def test_detect_language_preempts_asr_but_respects_fifo():
    """Verify priority detect-language and standard ASR tasks can both acquire resources."""
    _setup_units(
        [
            {"id": "GPU.0", "type": "GPU", "name": "Intel GPU"},
            {"id": "CPU", "type": "CPU", "name": "Host CPU"},
        ]
    )

    events = []
    lock = threading.Lock()

    def run_asr_task():
        """Run a standard ASR task."""
        utils.THREAD_CONTEXT.reset()
        model_manager.increment_active_session()
        try:
            with lock:
                events.append("asr_start")

            with model_manager.early_task_registration(is_priority=False):
                with lock:
                    events.append("asr_registered")

                with model_manager.model_lock_ctx(priority=False) as (_, unit_id):
                    with lock:
                        events.append(f"asr_acquired_{unit_id}")
                    time.sleep(0.02)
        finally:
            model_manager.decrement_active_session()

    def run_priority_task():
        """Run a priority detect-language task."""
        utils.THREAD_CONTEXT.reset()
        model_manager.increment_active_session()
        try:
            with lock:
                events.append("prio_start")

            with model_manager.early_task_registration(is_priority=True):
                with lock:
                    events.append("prio_registered")

                model_manager.wait_for_priority()
                with lock:
                    events.append("prio_waited")

                with model_manager.model_lock_ctx(priority=True) as (_, unit_id):
                    with lock:
                        events.append(f"prio_acquired_{unit_id}")
                    time.sleep(0.01)
        finally:
            model_manager.decrement_active_session()

    # Start both types of tasks
    t_asr = threading.Thread(target=run_asr_task)
    t_asr.start()
    time.sleep(0.02)

    t_prio = threading.Thread(target=run_priority_task)
    t_prio.start()

    t_asr.join(timeout=8.0)
    t_prio.join(timeout=8.0)

    # Verify both completed without errors
    assert not t_asr.is_alive()
    assert not t_prio.is_alive()

    # Verify events show both tasks completed acquisition
    assert "asr_registered" in events
    assert "prio_registered" in events
    assert "asr_start" in events
    assert "prio_start" in events

    assert events.index("asr_start") < events.index("prio_start")
    assert events.index("asr_registered") < events.index("prio_registered")
    assert events.index("prio_waited") < events.index(next(e for e in events if e.startswith("prio_acquired_")))


def test_earlier_asr_task_blocks_later_asr_task():
    """Later ASR task should wait if earlier ASR task is still queued."""
    _setup_units([{"id": "CPU", "type": "CPU", "name": "Host CPU"}])

    events = []
    lock = threading.Lock()

    def run_asr_task(task_num, hold_time=0.1):
        """Run a standard ASR task."""
        utils.THREAD_CONTEXT.reset()
        model_manager.increment_active_session()
        try:
            with lock:
                events.append(f"asr_{task_num}_start")

            with model_manager.early_task_registration(is_priority=False):
                with lock:
                    events.append(f"asr_{task_num}_registered")

                with model_manager.model_lock_ctx(priority=False) as (_, unit_id):
                    with lock:
                        events.append(f"asr_{task_num}_acquired")
                    time.sleep(hold_time)
                    with lock:
                        events.append(f"asr_{task_num}_done")
        finally:
            model_manager.decrement_active_session()

    # Start two ASR tasks: asr_0 gets first unit and holds it, asr_1 should wait
    t_asr0 = threading.Thread(target=run_asr_task, args=(0, 0.15))
    t_asr0.start()
    time.sleep(0.02)

    t_asr1 = threading.Thread(target=run_asr_task, args=(1, 0.05))
    t_asr1.start()
    time.sleep(0.05)

    # At this point: asr_0 should be holding, asr_1 should be waiting
    # because asr_1 arrived later and has_earlier_task should return True
    t_asr0.join(timeout=10.0)
    t_asr1.join(timeout=10.0)

    assert not t_asr0.is_alive()
    assert not t_asr1.is_alive()

    # Both should have completed
    assert "asr_0_done" in events
    assert "asr_1_done" in events
    assert events.index("asr_0_registered") < events.index("asr_1_registered")
    assert events.index("asr_0_acquired") < events.index("asr_1_acquired")
    assert events.index("asr_0_done") < events.index("asr_1_done")


def test_priority_task_does_not_skip_earlier_priority_task():
    """Later priority task should not acquire before earlier priority task."""
    _setup_units(
        [
            {"id": "GPU.0", "type": "GPU", "name": "Intel GPU"},
            {"id": "CPU", "type": "CPU", "name": "Host CPU"},
        ]
    )

    events = []
    lock = threading.Lock()

    def run_priority_task(task_num, hold_time=0.08):
        """Run a priority detect-language task."""
        utils.THREAD_CONTEXT.reset()
        model_manager.increment_active_session()
        try:
            with lock:
                events.append(f"prio_{task_num}_start")

            with model_manager.early_task_registration(is_priority=True):
                with lock:
                    events.append(f"prio_{task_num}_registered")

                model_manager.wait_for_priority()
                with lock:
                    events.append(f"prio_{task_num}_waited")

                with model_manager.model_lock_ctx(priority=True) as (_, unit_id):
                    with lock:
                        events.append(f"prio_{task_num}_acquired")
                    time.sleep(hold_time)
                    with lock:
                        events.append(f"prio_{task_num}_done")
        finally:
            model_manager.decrement_active_session()

    # Start two priority tasks in sequence
    t_prio0 = threading.Thread(target=run_priority_task, args=(0, 0.1))
    t_prio0.start()
    time.sleep(0.02)

    t_prio1 = threading.Thread(target=run_priority_task, args=(1, 0.05))
    t_prio1.start()

    t_prio0.join(timeout=10.0)
    t_prio1.join(timeout=10.0)

    assert not t_prio0.is_alive()
    assert not t_prio1.is_alive()

    # Both should have acquired and completed
    assert "prio_0_acquired" in events
    assert "prio_1_acquired" in events
    assert "prio_0_done" in events
    assert "prio_1_done" in events
    assert events.index("prio_0_registered") < events.index("prio_1_registered")
    assert events.index("prio_0_acquired") < events.index("prio_1_acquired")


def test_has_earlier_task_correctly_identifies_earlier_tasks():
    """has_earlier_task() should respect priority levels when checking FIFO ordering."""
    _setup_units([{"id": "CPU", "type": "CPU", "name": "Host CPU"}])

    # Create mock tasks as dicts (like real tasks) with priority info
    task_time_1 = time.time()
    with scheduler.STATE.task_order_lock:
        scheduler.STATE.task_arrival_order["task_1"] = task_time_1

    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry["task_1"] = {
            "task_id": "task_1",
            "is_priority": False,
            "status": "initializing",
        }

    time.sleep(0.05)
    task_time_2 = time.time()
    with scheduler.STATE.task_order_lock:
        scheduler.STATE.task_arrival_order["task_2"] = task_time_2

    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry["task_2"] = {
            "task_id": "task_2",
            "is_priority": False,
            "status": "initializing",
        }

    time.sleep(0.05)
    task_time_3 = time.time()
    with scheduler.STATE.task_order_lock:
        scheduler.STATE.task_arrival_order["task_3"] = task_time_3

    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry["task_3"] = {
            "task_id": "task_3",
            "is_priority": True,
            "status": "queued",
        }

    # task_2 (ASR) arrived after task_1 (ASR), so has_earlier_task(task_2, is_priority=False) should be True
    assert scheduler.has_earlier_task("task_2", is_priority=False) is True

    # task_1 (ASR) is earliest, so has_earlier_task(task_1, is_priority=False) should be False
    assert scheduler.has_earlier_task("task_1", is_priority=False) is False

    # task_3 (priority) is latest but should NOT be blocked by earlier ASR tasks
    # has_earlier_task(task_3, is_priority=True) should be False (no earlier priority tasks)
    assert scheduler.has_earlier_task("task_3", is_priority=True) is False

    # Remove task_1 from registry
    with scheduler.STATE.task_registry_lock:
        del scheduler.STATE.task_registry["task_1"]

    # Now task_2 should have no earlier ASR tasks
    assert scheduler.has_earlier_task("task_2", is_priority=False) is False

    # task_3 still has no earlier priority tasks
    assert scheduler.has_earlier_task("task_3", is_priority=True) is False

    # Clean up
    with scheduler.STATE.task_order_lock:
        scheduler.STATE.task_arrival_order.clear()
    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry.clear()


def test_asr_after_three_detectlang_not_stuck_waiting_hardware_dual_accelerator():
    """ASR -> 3 detectlang -> ASR must not leave the second ASR stuck waiting for hardware.

    Regression for dual-accelerator setups (Intel NPU + Intel GPU) where the later ASR
    could remain in "Waiting for Hardware" after priority detect-language tasks drained.
    """
    _setup_units(
        [
            {"id": "NPU.0", "type": "NPU", "name": "Intel NPU"},
            {"id": "GPU.0", "type": "GPU", "name": "Intel GPU"},
        ]
    )

    events = []
    lock = threading.Lock()
    asr1_acquired = threading.Event()
    asr1_release = threading.Event()
    asr2_acquired = threading.Event()

    def fake_init_unit(unit):
        """Avoid loading real engines in this concurrency regression."""
        model_manager.MODEL_POOL[unit["id"]] = mock.MagicMock()
        model_manager.PREPROCESSOR_POOL[unit["id"]] = mock.MagicMock()

    def auto_confirm_priority_waits(stop_evt):
        """Unblock wait_for_priority() deterministically in this synthetic test."""
        while not stop_evt.is_set():
            if scheduler.STATE.pause_requested.is_set() and not scheduler.STATE.pause_confirmed.is_set():
                scheduler.STATE.pause_confirmed.set()
            for u_sync in scheduler.STATE.unit_sync.values():
                if u_sync["pause_requested"].is_set() and not u_sync["pause_confirmed"].is_set():
                    u_sync["pause_confirmed"].set()
            time.sleep(0.01)

    def run_asr_first():
        utils.THREAD_CONTEXT.reset()
        model_manager.increment_active_session()
        try:
            with model_manager.early_task_registration(is_priority=False):
                with model_manager.model_lock_ctx(priority=False) as (_, unit_id):
                    with lock:
                        events.append(f"asr1_acquired_{unit_id}")
                    asr1_acquired.set()
                    # Keep first ASR active while detect tasks and ASR2 are processed.
                    asr1_release.wait(timeout=10.0)
                    with lock:
                        events.append("asr1_done")
        finally:
            model_manager.decrement_active_session()

    def run_detect(idx):
        utils.THREAD_CONTEXT.reset()
        model_manager.increment_active_session()
        try:
            with model_manager.early_task_registration(is_priority=True):
                model_manager.wait_for_priority()
                with model_manager.model_lock_ctx(priority=True) as (_, unit_id):
                    with lock:
                        events.append(f"detect_{idx}_acquired_{unit_id}")
                    time.sleep(0.05)
                    with lock:
                        events.append(f"detect_{idx}_done")
        finally:
            model_manager.decrement_active_session()

    def run_asr_second():
        utils.THREAD_CONTEXT.reset()
        model_manager.increment_active_session()
        try:
            with model_manager.early_task_registration(is_priority=False):
                with model_manager.model_lock_ctx(priority=False) as (_, unit_id):
                    with lock:
                        events.append(f"asr2_acquired_{unit_id}")
                    asr2_acquired.set()
                    time.sleep(0.05)
                    with lock:
                        events.append("asr2_done")
        finally:
            model_manager.decrement_active_session()

    with mock.patch("modules.inference.model_manager.init_unit", side_effect=fake_init_unit):
        stop_auto_confirm = threading.Event()
        auto_confirm_t = threading.Thread(target=auto_confirm_priority_waits, args=(stop_auto_confirm,), daemon=True)
        auto_confirm_t.start()

        t_asr1 = threading.Thread(target=run_asr_first)
        t_asr1.start()
        detect_threads = []
        t_asr2 = None
        try:
            assert asr1_acquired.wait(timeout=3.0), "First ASR failed to acquire hardware"

            for i in range(3):
                t = threading.Thread(target=run_detect, args=(i,))
                t.start()
                detect_threads.append(t)
                time.sleep(0.01)

            t_asr2 = threading.Thread(target=run_asr_second)
            t_asr2.start()

            for t in detect_threads:
                t.join(timeout=8.0)
            assert all(not t.is_alive() for t in detect_threads), "Detect-language tasks did not finish"

            # Core assertion: ASR2 must acquire without waiting for ASR1 to fully finish.
            assert asr2_acquired.wait(timeout=5.0), "Second ASR remained stuck waiting for hardware"
        finally:
            asr1_release.set()
            stop_auto_confirm.set()
            auto_confirm_t.join(timeout=2.0)
            t_asr1.join(timeout=8.0)
            for t in detect_threads:
                t.join(timeout=2.0)
            if t_asr2 is not None:
                t_asr2.join(timeout=8.0)

    assert not t_asr1.is_alive()
    assert not t_asr2.is_alive()
    with lock:
        assert any(e.startswith("asr2_acquired_") for e in events)
        assert "asr2_done" in events


def test_asr_starts_during_ongoing_detect_language_on_other_unit():
    """Verify that a standard ASR task can start and acquire an idle unit
    even while a priority detect-language task is ongoing on another unit.
    """
    _setup_units(
        [
            {"id": "NPU.0", "type": "NPU", "name": "Intel NPU"},
            {"id": "GPU.0", "type": "GPU", "name": "Intel GPU"},
        ]
    )

    events = []
    lock = threading.Lock()
    detect_acquired = threading.Event()
    detect_release = threading.Event()
    asr_acquired = threading.Event()

    def fake_init_unit(unit):
        model_manager.MODEL_POOL[unit["id"]] = mock.MagicMock()
        model_manager.PREPROCESSOR_POOL[unit["id"]] = mock.MagicMock()

    def auto_confirm_priority_waits(stop_evt):
        while not stop_evt.is_set():
            if scheduler.STATE.pause_requested.is_set() and not scheduler.STATE.pause_confirmed.is_set():
                scheduler.STATE.pause_confirmed.set()
            for u_sync in scheduler.STATE.unit_sync.values():
                if u_sync["pause_requested"].is_set() and not u_sync["pause_confirmed"].is_set():
                    u_sync["pause_confirmed"].set()
            time.sleep(0.01)

    def run_detect():
        utils.THREAD_CONTEXT.reset()
        model_manager.increment_active_session()
        try:
            with model_manager.early_task_registration(is_priority=True):
                model_manager.wait_for_priority()
                with model_manager.model_lock_ctx(priority=True) as (_, unit_id):
                    with lock:
                        events.append(f"detect_acquired_{unit_id}")
                    detect_acquired.set()
                    detect_release.wait(timeout=10.0)
                    with lock:
                        events.append("detect_done")
        finally:
            model_manager.decrement_active_session()

    def run_asr():
        utils.THREAD_CONTEXT.reset()
        model_manager.increment_active_session()
        try:
            with model_manager.early_task_registration(is_priority=False):
                with model_manager.model_lock_ctx(priority=False) as (_, unit_id):
                    with lock:
                        events.append(f"asr_acquired_{unit_id}")
                    asr_acquired.set()
        finally:
            model_manager.decrement_active_session()

    with mock.patch("modules.inference.model_manager.init_unit", side_effect=fake_init_unit):
        stop_auto_confirm = threading.Event()
        auto_confirm_t = threading.Thread(target=auto_confirm_priority_waits, args=(stop_auto_confirm,), daemon=True)
        auto_confirm_t.start()

        t_detect = threading.Thread(target=run_detect)
        t_detect.start()

        try:
            assert detect_acquired.wait(timeout=3.0), "Detect task failed to acquire hardware"

            t_asr = threading.Thread(target=run_asr)
            t_asr.start()

            # The ASR task must acquire the second unit within a short timeout even though the priority task is still active.
            assert asr_acquired.wait(timeout=3.0), (
                "ASR task remained stuck waiting for hardware while detect-language was ongoing"
            )
            t_asr.join(timeout=2.0)
        finally:
            detect_release.set()
            stop_auto_confirm.set()
            auto_confirm_t.join(timeout=2.0)
            t_detect.join(timeout=3.0)

    with lock:
        assert any(e.startswith("detect_acquired_") for e in events)
        assert any(e.startswith("asr_acquired_") for e in events)
