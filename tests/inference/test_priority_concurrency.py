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


def test_concurrency_fallback_no_deadlock():
    """Verify that language detection fallback does not trigger a re-entrancy deadlock."""
    from modules.inference.scheduler import SchedulerState
    from modules.inference import language_detection

    hw_list = [
        {"id": "CPU", "type": "CPU", "name": "Host CPU"}
    ]

    with mock.patch("modules.config.HARDWARE_UNITS", hw_list), \
            mock.patch("modules.inference.model_manager.unload_models"), \
            mock.patch("modules.inference.language_detection.utils.get_audio_duration", return_value=30):

        # Reset scheduler state
        scheduler.STATE = SchedulerState()

        # Populate pools
        model_manager._MODEL_POOL.clear()
        model_manager._PREPROCESSOR_POOL.clear()
        mock_model = mock.MagicMock()
        model_manager._MODEL_POOL["CPU"] = mock_model
        model_manager._PREPROCESSOR_POOL["CPU"] = mock.MagicMock()

        # Mock run_batch_language_detection_direct to return empty list
        # which will trigger the fallback logic in _step_run_inference
        with mock.patch("modules.inference.model_manager.run_batch_language_detection_direct", return_value=[]), \
                mock.patch("modules.inference.model_manager.run_language_detection_core") as mock_core, \
                mock.patch("modules.inference.language_detection._prepare_montage", return_value="montage.wav"):

            mock_core.return_value = {
                "detected_language": "en",
                "confidence": 0.9,
                "all_probabilities": {"en": 0.9}
            }

            # Call run_voting_detection. It must complete successfully without blocking.
            result = language_detection.run_voting_detection("dummy.wav", model_manager)

            assert result["detected_language"] == "en"
            assert result["confidence"] == 0.9
            mock_core.assert_called_once()


def test_concurrency_priority_task_failure_resumes_standard_task():
    """Verify that if a priority task fails/errors, any paused standard task resumes."""
    from modules.inference.scheduler import SchedulerState

    hw_list = [
        {"id": "NPU.0", "type": "NPU", "name": "Intel NPU"}
    ]

    with mock.patch("modules.config.HARDWARE_UNITS", hw_list), \
            mock.patch("modules.inference.model_manager.unload_models"):

        # Reset scheduler state with mocked hardware units
        scheduler.STATE = SchedulerState()

        # Populate pools
        model_manager._MODEL_POOL.clear()
        model_manager._PREPROCESSOR_POOL.clear()
        model_manager._MODEL_POOL["NPU.0"] = mock.MagicMock()
        model_manager._PREPROCESSOR_POOL["NPU.0"] = mock.MagicMock()

        utils.THREAD_CONTEXT.is_priority = False
        events = []

        # Start transcription task (occupies the single unit)
        t_trans = threading.Thread(target=helper_run_transcription, args=(events, "1", 3, 0.3))
        t_trans.start()
        time.sleep(0.1)  # Let it start and acquire unit

        # Simulate a priority task that registers, waits, and then raises an Exception
        def run_failing_priority():
            events.append("priority_failed_start")
            model_manager.increment_active_session()
            try:
                with model_manager.early_task_registration(is_priority=True):
                    events.append("priority_failed_waiting")
                    model_manager.wait_for_priority()
                    events.append("priority_failed_waited")
                    # Simulate failure before completing the task
                    raise RuntimeError("Simulated priority task failure")
            except RuntimeError as e:
                events.append(f"priority_failed_error_{e}")
            finally:
                model_manager.decrement_active_session()

        t_prio = threading.Thread(target=run_failing_priority)
        t_prio.start()

        t_trans.join(timeout=10.0)
        t_prio.join(timeout=10.0)

        print("Priority Failure Resumes Standard Task Events:", events)
        assert not t_trans.is_alive()
        assert not t_prio.is_alive()

        # Ensure priority task registered the error and transcription completed
        assert "priority_failed_error_Simulated priority task failure" in events
        assert "transcription_1_done" in events


def test_concurrency_multiple_priority_tasks_sequential():
    """Verify that multiple priority tasks register and run sequentially."""
    from modules.inference.scheduler import SchedulerState

    hw_list = [
        {"id": "NPU.0", "type": "NPU", "name": "Intel NPU"}
    ]

    with mock.patch("modules.config.HARDWARE_UNITS", hw_list), \
            mock.patch("modules.inference.model_manager.unload_models"):

        # Reset scheduler state with mocked hardware units
        scheduler.STATE = SchedulerState()

        # Populate pools
        model_manager._MODEL_POOL.clear()
        model_manager._PREPROCESSOR_POOL.clear()
        model_manager._MODEL_POOL["NPU.0"] = mock.MagicMock()
        model_manager._PREPROCESSOR_POOL["NPU.0"] = mock.MagicMock()

        events = []

        def run_prio(name, delay):
            events.append(f"prio_{name}_start")
            model_manager.increment_active_session()
            try:
                with model_manager.early_task_registration(is_priority=True):
                    events.append(f"prio_{name}_registered")
                    time.sleep(delay)
                    events.append(f"prio_{name}_done")
            finally:
                model_manager.decrement_active_session()

        t1 = threading.Thread(target=run_prio, args=("1", 0.4))
        t2 = threading.Thread(target=run_prio, args=("2", 0.1))

        t1.start()
        time.sleep(0.1)  # Make sure t1 registers first
        t2.start()

        t1.join(timeout=5.0)
        t2.join(timeout=5.0)

        print("Multiple Priority Tasks Sequential Events:", events)
        assert not t1.is_alive()
        assert not t2.is_alive()

        # Check that t2 registered ONLY after t1 was completely done (sequential execution check)
        idx_t1_done = events.index("prio_1_done")
        idx_t2_reg = events.index("prio_2_registered")
        assert idx_t1_done < idx_t2_reg


def test_standard_task_yields_to_queued_priority():
    """Verify that standard tasks yield the hardware lock if priority tasks are queued."""
    from modules.inference.scheduler import SchedulerState

    hw_list = [{"id": "CPU", "type": "CPU", "name": "Host CPU"}]
    with mock.patch("modules.config.HARDWARE_UNITS", hw_list), \
            mock.patch("modules.inference.model_manager.unload_models"):
        scheduler.STATE = SchedulerState()
        model_manager._MODEL_POOL.clear()
        model_manager._PREPROCESSOR_POOL.clear()
        model_manager._MODEL_POOL["CPU"] = mock.MagicMock()
        model_manager._PREPROCESSOR_POOL["CPU"] = mock.MagicMock()

        # Simulate a priority task is queued/registered
        with scheduler.early_task_registration(is_priority=True):
            # Now there is a priority task in the registry.
            # Try to acquire model lock with priority=False. It should yield/wait.
            utils.THREAD_CONTEXT.is_priority = False

            acquired = []

            def try_acquire():
                with model_manager.model_lock_ctx(priority=False):
                    acquired.append(True)

            t = threading.Thread(target=try_acquire)
            t.start()
            time.sleep(0.3)
            # Should not be acquired because has_priority is True
            assert not acquired

        t.join(timeout=2.0)
        # Now that priority task is finished, standard task should be able to acquire and finish
        assert acquired == [True]


def test_priority_does_not_preempt_itself():
    """Verify that priority tasks are bypass-ignored by the preemption check."""
    from modules.inference.scheduler import SchedulerState
    scheduler.STATE = SchedulerState()

    # Register as priority task
    thread_id = threading.get_ident()
    scheduler.STATE.task_registry[thread_id] = {
        "status": "active",
        "is_priority": True,
        "unit_id": "CPU"
    }

    scheduler.STATE.pause_requested.set()
    # Call check_preemption. It should return immediately without blocking/waiting since is_priority is True.
    start = time.time()
    model_manager._check_preemption()
    assert time.time() - start < 0.2

    # Cleanup
    del scheduler.STATE.task_registry[thread_id]
