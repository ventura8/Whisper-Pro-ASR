"""
Concurrency tests for Whisper Pro ASR scheduler preemption logic.
"""
from unittest import mock
import threading
import time
from modules.inference import scheduler, model_manager
from modules import utils

# pylint: disable=protected-access,import-outside-toplevel


def helper_run_transcription(events, name, steps=3, step_delay=0.3):
    """Simulates running a standard transcription task with preemption checks."""
    events.append(f"transcription_{name}_start")
    model_manager.increment_active_session()
    try:
        with model_manager.early_task_registration(is_priority=False):
            with model_manager.model_lock_ctx() as (_, unit_id):
                events.append(f"transcription_{name}_running_on_{unit_id}")
                # Simulate segment loops with periodic preemption checks
                for i in range(steps):
                    time.sleep(step_delay)
                    events.append(f"transcription_{name}_check_preempt_{i}")
                    model_manager._check_preemption()
                events.append(f"transcription_{name}_done")
    finally:
        model_manager.decrement_active_session()


def helper_run_priority(events, name, task_delay=0.1):
    """Simulates running a high-priority task."""
    events.append(f"priority_{name}_start")
    model_manager.increment_active_session()
    try:
        with model_manager.early_task_registration(is_priority=True):
            events.append(f"priority_{name}_waiting")
            model_manager.wait_for_priority()
            events.append(f"priority_{name}_waited")

            with model_manager.model_lock_ctx() as (_, unit_id):
                events.append(f"priority_{name}_running_on_{unit_id}")
                time.sleep(task_delay)
                events.append(f"priority_{name}_done")
    finally:
        model_manager.decrement_active_session()


def test_concurrency_zero_hardware_units():
    """Verify that 0 hardware units falls back to Host CPU on initialization."""
    from modules.inference.scheduler import SchedulerState
    with mock.patch("modules.config.HARDWARE_UNITS", []):
        state = SchedulerState()
        assert state.accel_limit == 1
        assert len(state.hw_pool.queue) == 1
        unit = state.hw_pool.get()
        assert unit["type"] == "CPU"
        assert unit["id"] == "CPU"


def test_concurrency_one_hardware_unit():
    """Verify preemption and resource sharing with 1 hardware unit (Intel NPU)."""
    from modules.inference.scheduler import SchedulerState

    hw_list = [
        {"id": "NPU.0", "type": "NPU", "name": "Intel NPU"}
    ]

    with mock.patch("modules.config.HARDWARE_UNITS", hw_list), \
            mock.patch("modules.inference.model_manager.unload_models"):

        # Reset scheduler state with the mocked hardware units
        scheduler.STATE = SchedulerState()

        # Populate model manager pools
        model_manager._MODEL_POOL.clear()
        model_manager._PREPROCESSOR_POOL.clear()
        for u in hw_list:
            model_manager._MODEL_POOL[u["id"]] = mock.MagicMock()
            model_manager._PREPROCESSOR_POOL[u["id"]] = mock.MagicMock()

        utils.THREAD_CONTEXT.is_priority = False
        events = []

        # Start transcription task (occupies the single unit)
        t_trans = threading.Thread(target=helper_run_transcription, args=(events, "1", 3, 0.3))
        t_trans.start()
        time.sleep(0.1)  # Let it start and acquire unit

        # Start priority tasks concurrently
        t_p1 = threading.Thread(target=helper_run_priority, args=(events, "P1", 0.1))
        t_p2 = threading.Thread(target=helper_run_priority, args=(events, "P2", 0.1))

        t_p1.start()
        time.sleep(0.05)  # Offset slightly
        t_p2.start()

        t_trans.join(timeout=10.0)
        t_p1.join(timeout=10.0)
        t_p2.join(timeout=10.0)

        print("1 Hardware Unit Events:", events)
        assert not t_trans.is_alive()
        assert not t_p1.is_alive()
        assert not t_p2.is_alive()

        # Ensure they ran sequentially
        assert "priority_P1_running_on_NPU.0" in events
        assert "priority_P2_running_on_NPU.0" in events
        idx_p1_done = events.index("priority_P1_done")
        idx_p2_running = events.index("priority_P2_running_on_NPU.0")
        assert idx_p1_done < idx_p2_running


def test_concurrency_two_hardware_units():
    """Verify preemption and resource sharing with 2 hardware units (Intel NPU and Intel GPU)."""
    from modules.inference.scheduler import SchedulerState

    hw_list = [
        {"id": "NPU.0", "type": "NPU", "name": "Intel NPU"},
        {"id": "GPU.0", "type": "GPU", "name": "Intel GPU"}
    ]

    with mock.patch("modules.config.HARDWARE_UNITS", hw_list), \
            mock.patch("modules.inference.model_manager.unload_models"):

        # Reset scheduler state with the mocked hardware units
        scheduler.STATE = SchedulerState()

        # Populate model manager pools
        model_manager._MODEL_POOL.clear()
        model_manager._PREPROCESSOR_POOL.clear()
        for u in hw_list:
            model_manager._MODEL_POOL[u["id"]] = mock.MagicMock()
            model_manager._PREPROCESSOR_POOL[u["id"]] = mock.MagicMock()

        utils.THREAD_CONTEXT.is_priority = False
        events = []

        # Start 2 transcription tasks (will consume both units)
        t_trans1 = threading.Thread(target=helper_run_transcription, args=(events, "1", 3, 0.3))
        t_trans2 = threading.Thread(target=helper_run_transcription, args=(events, "2", 3, 0.3))

        t_trans1.start()
        t_trans2.start()
        time.sleep(0.1)  # Let them start and acquire units

        # Start a priority task (will trigger preemption because all 2 units are busy)
        t_prio = threading.Thread(target=helper_run_priority, args=(events, "P", 0.1))
        t_prio.start()

        t_trans1.join(timeout=10.0)
        t_trans2.join(timeout=10.0)
        t_prio.join(timeout=10.0)

        print("2 Hardware Units Events:", events)
        assert not t_trans1.is_alive()
        assert not t_trans2.is_alive()
        assert not t_prio.is_alive()

        # Ensure priority task completed and at least one transcription task was paused
        assert "priority_P_running_on_NPU.0" in events or "priority_P_running_on_GPU.0" in events
        assert "priority_P_done" in events

        # Verify the priority task finished before at least one transcription task completed
        idx_prio_done = events.index("priority_P_done")
        idx_t1_done = events.index("transcription_1_done")
        idx_t2_done = events.index("transcription_2_done")
        assert idx_prio_done < idx_t1_done or idx_prio_done < idx_t2_done


def test_concurrency_three_hardware_units():
    """Verify preemption and resource sharing with 3 hardware units (Intel NPU, Intel GPU, and NVIDIA GPU)."""
    from modules.inference.scheduler import SchedulerState

    hw_list = [
        {"id": "NPU.0", "type": "NPU", "name": "Intel NPU"},
        {"id": "GPU.0", "type": "GPU", "name": "Intel GPU"},
        {"id": "CUDA.0", "type": "CUDA", "name": "NVIDIA GPU"}
    ]

    with mock.patch("modules.config.HARDWARE_UNITS", hw_list), \
            mock.patch("modules.inference.model_manager.unload_models"):

        # Reset scheduler state with the mocked hardware units
        scheduler.STATE = SchedulerState()

        # Populate model manager pools
        model_manager._MODEL_POOL.clear()
        model_manager._PREPROCESSOR_POOL.clear()
        for u in hw_list:
            model_manager._MODEL_POOL[u["id"]] = mock.MagicMock()
            model_manager._PREPROCESSOR_POOL[u["id"]] = mock.MagicMock()

        utils.THREAD_CONTEXT.is_priority = False
        events = []

        # Start 3 transcription tasks (will consume all 3 units)
        t_trans1 = threading.Thread(target=helper_run_transcription, args=(events, "1", 3, 0.3))
        t_trans2 = threading.Thread(target=helper_run_transcription, args=(events, "2", 3, 0.3))
        t_trans3 = threading.Thread(target=helper_run_transcription, args=(events, "3", 3, 0.3))

        t_trans1.start()
        t_trans2.start()
        t_trans3.start()
        time.sleep(0.1)  # Let them start and acquire units

        # Start a priority task (will trigger preemption because all 3 units are busy)
        t_prio = threading.Thread(target=helper_run_priority, args=(events, "P", 0.1))
        t_prio.start()

        t_trans1.join(timeout=10.0)
        t_trans2.join(timeout=10.0)
        t_trans3.join(timeout=10.0)
        t_prio.join(timeout=10.0)

        print("3 Hardware Units Events:", events)
        assert not t_trans1.is_alive()
        assert not t_trans2.is_alive()
        assert not t_trans3.is_alive()
        assert not t_prio.is_alive()

        # Ensure priority task completed and at least one transcription task was paused
        assert any(f"priority_P_running_on_{u['id']}" in events for u in hw_list)
        assert "priority_P_done" in events

        # Verify the priority task finished before at least one transcription task completed
        idx_prio_done = events.index("priority_P_done")
        idx_t1_done = events.index("transcription_1_done")
        idx_t2_done = events.index("transcription_2_done")
        idx_t3_done = events.index("transcription_3_done")
        assert idx_prio_done < idx_t1_done or idx_prio_done < idx_t2_done or idx_prio_done < idx_t3_done


def test_concurrency_zero_hardware_units_execution():
    """Verify preemption and execution with 0 hardware units (CPU fallback)."""
    from modules.inference.scheduler import SchedulerState

    with mock.patch("modules.config.HARDWARE_UNITS", []), \
            mock.patch("modules.inference.model_manager.unload_models"):

        # Reset scheduler state with empty hardware units, forcing Host CPU fallback
        scheduler.STATE = SchedulerState()

        # Check it fallback-registered the CPU unit
        assert len(scheduler.STATE.hw_pool.queue) == 1
        assert scheduler.STATE.accel_limit == 1

        # Populate model manager pools for the CPU fallback unit
        model_manager._MODEL_POOL.clear()
        model_manager._PREPROCESSOR_POOL.clear()
        model_manager._MODEL_POOL["CPU"] = mock.MagicMock()
        model_manager._PREPROCESSOR_POOL["CPU"] = mock.MagicMock()

        utils.THREAD_CONTEXT.is_priority = False
        events = []

        # Start transcription task (occupies the single CPU unit)
        t_trans = threading.Thread(target=helper_run_transcription, args=(events, "1", 3, 0.3))
        t_trans.start()
        time.sleep(0.1)  # Let it start and acquire unit

        # Start priority task
        t_p = threading.Thread(target=helper_run_priority, args=(events, "P", 0.1))
        t_p.start()

        t_trans.join(timeout=10.0)
        t_p.join(timeout=10.0)

        print("0 Hardware Units CPU Fallback Events:", events)
        assert not t_trans.is_alive()
        assert not t_p.is_alive()

        # Ensure priority task completed and ran on the Host CPU unit
        assert "priority_P_running_on_CPU" in events
        assert "priority_P_done" in events

        # Verify the priority task finished before transcription task completed
        idx_p_done = events.index("priority_P_done")
        idx_t_done = events.index("transcription_1_done")
        assert idx_p_done < idx_t_done


def test_concurrency_priority_non_preemptive():
    """Verify that priority task does not preempt if there is an idle unit."""
    from modules.inference.scheduler import SchedulerState

    hw_list = [
        {"id": "NPU.0", "type": "NPU", "name": "Intel NPU"},
        {"id": "GPU.0", "type": "GPU", "name": "Intel GPU"}
    ]

    with mock.patch("modules.config.HARDWARE_UNITS", hw_list), \
            mock.patch("modules.inference.model_manager.unload_models"):

        # Reset scheduler state with the mocked hardware units
        scheduler.STATE = SchedulerState()

        # Populate model manager pools
        model_manager._MODEL_POOL.clear()
        model_manager._PREPROCESSOR_POOL.clear()
        for u in hw_list:
            model_manager._MODEL_POOL[u["id"]] = mock.MagicMock()
            model_manager._PREPROCESSOR_POOL[u["id"]] = mock.MagicMock()

        utils.THREAD_CONTEXT.is_priority = False
        events = []

        # Start 1 transcription task (claims NPU.0, leaving GPU.0 idle)
        t_trans = threading.Thread(target=helper_run_transcription, args=(events, "1", 3, 0.3))
        t_trans.start()
        time.sleep(0.1)  # Let it start and acquire NPU.0

        # Start a priority task (should acquire idle GPU.0 without preemption)
        t_prio = threading.Thread(target=helper_run_priority, args=(events, "P", 0.1))
        t_prio.start()

        t_trans.join(timeout=10.0)
        t_prio.join(timeout=10.0)

        print("Non-preemptive priority Events:", events)
        assert not t_trans.is_alive()
        assert not t_prio.is_alive()

        # Ensure priority task completed and did not trigger pause_requested
        assert "priority_P_running_on_GPU.0" in events
        assert "priority_P_done" in events
        # None of the transcription steps should be preceded by pause/resume signals
        # (transcription runs straight through)
        assert not any("Paused for Priority Task" in str(e) for e in events)
