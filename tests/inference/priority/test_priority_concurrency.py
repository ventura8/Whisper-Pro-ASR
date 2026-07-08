"""
Concurrency tests for Whisper Pro ASR scheduler preemption logic.
"""

import threading
import time
from unittest import mock

import pytest

from modules.core import utils
from modules.inference import model_manager, scheduler


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

    with mock.patch("modules.core.config.HARDWARE_UNITS", []):
        state = SchedulerState()
        assert state.accel_limit == 1
        assert len(state.hw_pool.queue) == 1
        unit = state.hw_pool.get()
        assert unit["type"] == "CPU"
        assert unit["id"] == "CPU"


def test_concurrency_one_hardware_unit():
    """Verify preemption and resource sharing with 1 hardware unit (Intel NPU)."""
    from modules.inference.scheduler import SchedulerState

    hw_list = [{"id": "NPU.0", "type": "NPU", "name": "Intel NPU"}]

    with (
        mock.patch("modules.core.config.HARDWARE_UNITS", hw_list),
        mock.patch("modules.inference.model_manager.unload_models"),
    ):
        # Reset scheduler state with the mocked hardware units
        scheduler.STATE = SchedulerState()

        # Populate model manager pools
        model_manager.MODEL_POOL.clear()
        model_manager.PREPROCESSOR_POOL.clear()
        for u in hw_list:
            model_manager.MODEL_POOL[u["id"]] = mock.MagicMock()
            model_manager.PREPROCESSOR_POOL[u["id"]] = mock.MagicMock()

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

    hw_list = [{"id": "NPU.0", "type": "NPU", "name": "Intel NPU"}, {"id": "GPU.0", "type": "GPU", "name": "Intel GPU"}]

    with (
        mock.patch("modules.core.config.HARDWARE_UNITS", hw_list),
        mock.patch("modules.inference.model_manager.unload_models"),
    ):
        # Reset scheduler state with the mocked hardware units
        scheduler.STATE = SchedulerState()

        # Populate model manager pools
        model_manager.MODEL_POOL.clear()
        model_manager.PREPROCESSOR_POOL.clear()
        for u in hw_list:
            model_manager.MODEL_POOL[u["id"]] = mock.MagicMock()
            model_manager.PREPROCESSOR_POOL[u["id"]] = mock.MagicMock()

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


def test_detectlang_uses_parallel_priority_units_when_two_asr_yielded():
    """With 2 busy ASR units and 4 detect-language requests, priority can run in parallel across borrowed units."""
    from modules.inference.scheduler import SchedulerState

    hw_list = [{"id": "NPU.0", "type": "NPU", "name": "Intel NPU"}, {"id": "GPU.0", "type": "GPU", "name": "Intel GPU"}]

    with (
        mock.patch("modules.core.config.HARDWARE_UNITS", hw_list),
        mock.patch("modules.inference.model_manager.unload_models"),
    ):
        scheduler.STATE = SchedulerState()

        model_manager.MODEL_POOL.clear()
        model_manager.PREPROCESSOR_POOL.clear()
        for unit in hw_list:
            model_manager.MODEL_POOL[unit["id"]] = mock.MagicMock()
            model_manager.PREPROCESSOR_POOL[unit["id"]] = mock.MagicMock()

        events = []
        lock = threading.Lock()
        running_priority = 0
        max_running_priority = 0

        def run_standard(name):
            helper_run_transcription(events, name, steps=5, step_delay=0.12)

        def run_priority(name):
            nonlocal running_priority, max_running_priority
            utils.THREAD_CONTEXT.reset()
            model_manager.increment_active_session()
            try:
                with model_manager.early_task_registration(is_priority=True):
                    model_manager.wait_for_priority()
                    events.append(f"priority_{name}_waited")
                    with model_manager.model_lock_ctx(priority=True) as (_, unit_id):
                        with lock:
                            running_priority += 1
                            max_running_priority = max(max_running_priority, running_priority)
                        events.append(f"priority_{name}_running_on_{unit_id}")
                        time.sleep(0.2)
                        events.append(f"priority_{name}_done")
                        with lock:
                            running_priority -= 1
            finally:
                model_manager.decrement_active_session()

        # Occupy both units with ASR work first.
        t_asr_1 = threading.Thread(target=run_standard, args=("A1",))
        t_asr_2 = threading.Thread(target=run_standard, args=("A2",))
        t_asr_1.start()
        t_asr_2.start()
        time.sleep(0.15)

        # Queue four detect-language tasks.
        prio_threads = []
        for idx in range(4):
            thread = threading.Thread(target=run_priority, args=(f"P{idx + 1}",))
            thread.start()
            prio_threads.append(thread)
            time.sleep(0.03)

        for thread in prio_threads:
            thread.join(timeout=12.0)
        t_asr_1.join(timeout=12.0)
        t_asr_2.join(timeout=12.0)

        assert all(not thread.is_alive() for thread in prio_threads)
        assert not t_asr_1.is_alive()
        assert not t_asr_2.is_alive()
        # With GPU + NPU available, at least two priority tasks should overlap.
        assert max_running_priority >= 2
        assert sum(1 for event in events if event.endswith("_done") and event.startswith("priority_")) == 4


def test_detectlang_parallel_with_npu_gpu_cuda_available():
    """With NPU+GPU+CUDA available, detect-language priority tasks should overlap on multiple units."""
    from modules.inference.scheduler import SchedulerState

    hw_list = [
        {"id": "NPU.0", "type": "NPU", "name": "Intel NPU"},
        {"id": "GPU.0", "type": "GPU", "name": "Intel GPU"},
        {"id": "CUDA.0", "type": "CUDA", "name": "NVIDIA GPU"},
    ]

    with (
        mock.patch("modules.core.config.HARDWARE_UNITS", hw_list),
        mock.patch("modules.inference.model_manager.unload_models"),
    ):
        scheduler.STATE = SchedulerState()

        model_manager.MODEL_POOL.clear()
        model_manager.PREPROCESSOR_POOL.clear()
        for unit in hw_list:
            model_manager.MODEL_POOL[unit["id"]] = mock.MagicMock()
            model_manager.PREPROCESSOR_POOL[unit["id"]] = mock.MagicMock()

        events = []
        lock = threading.Lock()
        running_priority = 0
        max_running_priority = 0
        units_used = set()

        def run_standard(name):
            helper_run_transcription(events, name, steps=5, step_delay=0.12)

        def run_priority(name):
            nonlocal running_priority, max_running_priority
            utils.THREAD_CONTEXT.reset()
            model_manager.increment_active_session()
            try:
                with model_manager.early_task_registration(is_priority=True):
                    model_manager.wait_for_priority()
                    events.append(f"priority_{name}_waited")
                    with model_manager.model_lock_ctx(priority=True) as (_, unit_id):
                        with lock:
                            running_priority += 1
                            max_running_priority = max(max_running_priority, running_priority)
                            units_used.add(unit_id)
                        events.append(f"priority_{name}_running_on_{unit_id}")
                        time.sleep(0.2)
                        events.append(f"priority_{name}_done")
                        with lock:
                            running_priority -= 1
            finally:
                model_manager.decrement_active_session()

        # Occupy all three units with standard work first.
        asr_threads = [threading.Thread(target=run_standard, args=(f"A{i}",)) for i in range(1, 4)]
        for thread in asr_threads:
            thread.start()
        time.sleep(0.15)

        # Burst three priority tasks; they should not be serialized to one-at-a-time.
        prio_threads = [threading.Thread(target=run_priority, args=(f"P{i}",)) for i in range(1, 4)]
        for thread in prio_threads:
            thread.start()
            time.sleep(0.03)

        for thread in prio_threads:
            thread.join(timeout=12.0)
        for thread in asr_threads:
            thread.join(timeout=12.0)

        assert all(not thread.is_alive() for thread in prio_threads)
        assert all(not thread.is_alive() for thread in asr_threads)
        # Core regression check: do not serialize detect-language to a single runner.
        assert max_running_priority >= 2
        # Ensure priority tasks actually used multiple borrowed accelerators.
        assert len(units_used) >= 2
        assert sum(1 for event in events if event.endswith("_done") and event.startswith("priority_")) == 3


def test_concurrency_three_hardware_units():
    """Verify preemption and resource sharing with 3 hardware units (Intel NPU, Intel GPU, and NVIDIA GPU)."""
    from modules.inference.scheduler import SchedulerState

    hw_list = [
        {"id": "NPU.0", "type": "NPU", "name": "Intel NPU"},
        {"id": "GPU.0", "type": "GPU", "name": "Intel GPU"},
        {"id": "CUDA.0", "type": "CUDA", "name": "NVIDIA GPU"},
    ]

    with (
        mock.patch("modules.core.config.HARDWARE_UNITS", hw_list),
        mock.patch("modules.inference.model_manager.unload_models"),
    ):
        # Reset scheduler state with the mocked hardware units
        scheduler.STATE = SchedulerState()

        # Populate model manager pools
        model_manager.MODEL_POOL.clear()
        model_manager.PREPROCESSOR_POOL.clear()
        for u in hw_list:
            model_manager.MODEL_POOL[u["id"]] = mock.MagicMock()
            model_manager.PREPROCESSOR_POOL[u["id"]] = mock.MagicMock()

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

    with (
        mock.patch("modules.core.config.HARDWARE_UNITS", []),
        mock.patch("modules.inference.model_manager.unload_models"),
    ):
        # Reset scheduler state with empty hardware units, forcing Host CPU fallback
        scheduler.STATE = SchedulerState()

        # Check it fallback-registered the CPU unit
        assert len(scheduler.STATE.hw_pool.queue) == 1
        assert scheduler.STATE.accel_limit == 1

        # Populate model manager pools for the CPU fallback unit
        model_manager.MODEL_POOL.clear()
        model_manager.PREPROCESSOR_POOL.clear()
        model_manager.MODEL_POOL["CPU"] = mock.MagicMock()
        model_manager.PREPROCESSOR_POOL["CPU"] = mock.MagicMock()

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

    hw_list = [{"id": "NPU.0", "type": "NPU", "name": "Intel NPU"}, {"id": "GPU.0", "type": "GPU", "name": "Intel GPU"}]

    with (
        mock.patch("modules.core.config.HARDWARE_UNITS", hw_list),
        mock.patch("modules.inference.model_manager.unload_models"),
    ):
        # Reset scheduler state with the mocked hardware units
        scheduler.STATE = SchedulerState()

        # Populate model manager pools
        model_manager.MODEL_POOL.clear()
        model_manager.PREPROCESSOR_POOL.clear()
        for u in hw_list:
            model_manager.MODEL_POOL[u["id"]] = mock.MagicMock()
            model_manager.PREPROCESSOR_POOL[u["id"]] = mock.MagicMock()

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
        # Verify that no pause requests were triggered (as there was an idle GPU.0)
        assert not scheduler.STATE.pause_requested.is_set()
        assert not scheduler.STATE.unit_sync["NPU.0"]["pause_requested"].is_set()
        assert not scheduler.STATE.unit_sync["GPU.0"]["pause_requested"].is_set()


def test_concurrency_fallback_no_deadlock():
    """Verify that language detection fallback does not trigger a re-entrancy deadlock."""
    from modules.inference import language_detection
    from modules.inference.scheduler import SchedulerState

    hw_list = [{"id": "CPU", "type": "CPU", "name": "Host CPU"}]

    with (
        mock.patch("modules.core.config.HARDWARE_UNITS", hw_list),
        mock.patch("modules.inference.model_manager.unload_models"),
        mock.patch("modules.inference.language_detection.utils.get_audio_duration", return_value=30),
    ):
        # Reset scheduler state
        scheduler.STATE = SchedulerState()

        # Populate pools
        model_manager.MODEL_POOL.clear()
        model_manager.PREPROCESSOR_POOL.clear()
        mock_model = mock.MagicMock()
        model_manager.MODEL_POOL["CPU"] = mock_model
        model_manager.PREPROCESSOR_POOL["CPU"] = mock.MagicMock()

        # Mock run_batch_language_detection_direct to return empty list
        # which will trigger the fallback logic in _step_run_inference
        with (
            mock.patch("modules.inference.model_manager.run_batch_language_detection_direct", return_value=[]),
            mock.patch("modules.inference.model_manager.run_language_detection_core") as mock_core,
            mock.patch("modules.inference.language_detection._prepare_montage", return_value="montage.wav"),
        ):
            mock_core.return_value = {"detected_language": "en", "confidence": 0.9, "all_probabilities": {"en": 0.9}}

            # Call run_voting_detection. It must complete successfully without blocking.
            result = language_detection.run_voting_detection("dummy.wav", model_manager)

            assert result["detected_language"] == "en"
            assert result["confidence"] == 0.9
            mock_core.assert_called_once()


def test_concurrency_priority_task_failure_resumes_standard_task():
    """Verify that if a priority task fails/errors, any paused standard task resumes."""
    from modules.inference.scheduler import SchedulerState

    hw_list = [{"id": "NPU.0", "type": "NPU", "name": "Intel NPU"}]

    with (
        mock.patch("modules.core.config.HARDWARE_UNITS", hw_list),
        mock.patch("modules.inference.model_manager.unload_models"),
    ):
        # Reset scheduler state with mocked hardware units
        scheduler.STATE = SchedulerState()

        # Populate pools
        model_manager.MODEL_POOL.clear()
        model_manager.PREPROCESSOR_POOL.clear()
        model_manager.MODEL_POOL["NPU.0"] = mock.MagicMock()
        model_manager.PREPROCESSOR_POOL["NPU.0"] = mock.MagicMock()

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


def test_concurrency_multiple_priority_tasks_allow_parallel_registration():
    """Verify that priority task registration is not serialized."""
    from modules.inference.scheduler import SchedulerState

    hw_list = [{"id": "NPU.0", "type": "NPU", "name": "Intel NPU"}]

    with (
        mock.patch("modules.core.config.HARDWARE_UNITS", hw_list),
        mock.patch("modules.inference.model_manager.unload_models"),
    ):
        # Reset scheduler state with mocked hardware units
        scheduler.STATE = SchedulerState()

        # Populate pools
        model_manager.MODEL_POOL.clear()
        model_manager.PREPROCESSOR_POOL.clear()
        model_manager.MODEL_POOL["NPU.0"] = mock.MagicMock()
        model_manager.PREPROCESSOR_POOL["NPU.0"] = mock.MagicMock()

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

        # Registration should no longer be serialized.
        idx_t1_done = events.index("prio_1_done")
        idx_t2_reg = events.index("prio_2_registered")
        assert idx_t2_reg < idx_t1_done


def test_standard_task_not_blocked_by_queued_priority_registration():
    """Queued priority registration alone should not gate standard acquisition."""
    from modules.inference.scheduler import SchedulerState

    hw_list = [{"id": "CPU", "type": "CPU", "name": "Host CPU"}]
    with (
        mock.patch("modules.core.config.HARDWARE_UNITS", hw_list),
        mock.patch("modules.inference.model_manager.unload_models"),
    ):
        scheduler.STATE = SchedulerState()
        model_manager.MODEL_POOL.clear()
        model_manager.PREPROCESSOR_POOL.clear()
        model_manager.MODEL_POOL["CPU"] = mock.MagicMock()
        model_manager.PREPROCESSOR_POOL["CPU"] = mock.MagicMock()

        # Simulate a priority task is queued/registered
        with scheduler.early_task_registration(is_priority=True):
            # Try to acquire model lock with priority=False. With unit-scoped gating,
            # this should not be blocked by queued priority metadata alone.
            utils.THREAD_CONTEXT.is_priority = False

            acquired = []

            def try_acquire():
                with model_manager.model_lock_ctx(priority=False):
                    acquired.append(True)

            t = threading.Thread(target=try_acquire)
            t.start()
            time.sleep(0.3)
            assert acquired == [True]

        t.join(timeout=2.0)
        # Task should have already completed.
        assert acquired == [True]


def test_priority_does_not_preempt_itself():
    """Verify that priority tasks are bypass-ignored by the preemption check."""
    from modules.inference.scheduler import SchedulerState

    scheduler.STATE = SchedulerState()

    # Register as priority task
    thread_id = threading.get_ident()
    scheduler.STATE.task_registry[thread_id] = {"status": "active", "is_priority": True, "unit_id": "CPU"}

    scheduler.STATE.pause_requested.set()
    # Call check_preemption. It should return immediately without blocking/waiting since is_priority is True.
    start = time.time()
    model_manager._check_preemption()
    assert time.time() - start < 0.2

    # Cleanup
    del scheduler.STATE.task_registry[thread_id]


def test_concurrency_targeted_preemption():
    """Verify that a priority task only preempts/pauses the targeted unit, leaving other units running."""
    from modules.inference.scheduler import SchedulerState

    hw_list = [{"id": "GPU.0", "type": "GPU", "name": "Intel GPU"}, {"id": "NPU.0", "type": "NPU", "name": "Intel NPU"}]

    with (
        mock.patch("modules.core.config.HARDWARE_UNITS", hw_list),
        mock.patch("modules.inference.model_manager.unload_models"),
    ):
        # Reset scheduler state with the mocked hardware units
        scheduler.STATE = SchedulerState()

        # Populate model manager pools
        model_manager.MODEL_POOL.clear()
        model_manager.PREPROCESSOR_POOL.clear()
        for u in hw_list:
            model_manager.MODEL_POOL[u["id"]] = mock.MagicMock()
            model_manager.PREPROCESSOR_POOL[u["id"]] = mock.MagicMock()

        events = []

        # We will manually simulate 2 standard tasks:
        # Task 1 on GPU.0
        # Task 2 on NPU.0
        # When a priority task comes in, it targets NPU.0.
        # Task 2 (NPU.0) should pause. Task 1 (GPU.0) should not pause.

        def run_standard_gpu():
            utils.THREAD_CONTEXT.reset()
            model_manager.increment_active_session()
            try:
                # Manually register Task 1
                with model_manager.early_task_registration(is_priority=False):
                    # Mock running on GPU.0
                    with model_manager.model_lock_ctx(priority=False) as (_, unit_id):
                        events.append("std_gpu_running")
                        # Check preemption. Since GPU.0 is not targeted, it should not pause.
                        model_manager._check_preemption()
                        events.append("std_gpu_checked_no_pause")
            finally:
                model_manager.decrement_active_session()

        def run_standard_npu():
            utils.THREAD_CONTEXT.reset()
            model_manager.increment_active_session()
            try:
                # Manually register Task 2
                with model_manager.early_task_registration(is_priority=False):
                    # Mock running on NPU.0
                    with model_manager.model_lock_ctx(priority=False) as (_, unit_id):
                        events.append("std_npu_running")
                        # First check preemption (not paused yet)
                        model_manager._check_preemption()
                        events.append("std_npu_checked_no_pause")

                        # Wait slightly for priority task to target us
                        time.sleep(0.3)

                        # Second check preemption. Since we are targeted, it should pause!
                        model_manager._check_preemption()
                        events.append("std_npu_resumed")
            finally:
                model_manager.decrement_active_session()

        def run_priority():
            utils.THREAD_CONTEXT.reset()
            model_manager.increment_active_session()
            try:
                # Register priority task targeting NPU.0
                with model_manager.early_task_registration(is_priority=True):
                    # Wait for priority slot. In a real scenario, initialize_task_context does this,
                    # which selects target unit. Let's call wait_for_priority.
                    # Since NPU.0 is busy with a standard task, wait_for_priority will select NPU.0 to preempt.
                    model_manager.wait_for_priority()
                    target = getattr(utils.THREAD_CONTEXT, "target_unit_id", "NPU.0")
                    events.append(f"priority_waited_{target}")

                    with model_manager.model_lock_ctx(priority=True) as (_, unit_id):
                        events.append(f"priority_running_on_{unit_id}")
                        time.sleep(0.1)
                        events.append("priority_done")
            finally:
                model_manager.decrement_active_session()

        # Start standard tasks
        t_gpu = threading.Thread(target=run_standard_gpu)
        t_npu = threading.Thread(target=run_standard_npu)

        t_gpu.start()
        t_npu.start()
        time.sleep(0.1)  # Let them start and acquire units

        # Start priority task
        t_prio = threading.Thread(target=run_priority)
        t_prio.start()

        t_gpu.join(timeout=5.0)
        t_npu.join(timeout=5.0)
        t_prio.join(timeout=5.0)

        print("Targeted Preemption Events:", events)
        # Verify that GPU task checked without pausing
        assert "std_gpu_checked_no_pause" in events
        # Verify NPU task paused and resumed
        assert "std_npu_resumed" in events
        assert "priority_running_on_NPU.0" in events

        # Verify priority task finished before NPU task resumed
        idx_p_done = events.index("priority_done")
        idx_npu_resume = events.index("std_npu_resumed")
        assert idx_p_done < idx_npu_resume


def test_concurrency_targeted_preemption_unit_resume_with_multiple_priorities():
    """Verify paused ASR work resumes only after queued priority tasks are drained."""
    from modules.inference.scheduler import SchedulerState

    hw_list = [
        {"id": "NPU.0", "type": "NPU", "name": "Intel NPU"},
        {"id": "GPU.0", "type": "GPU", "name": "Intel GPU"},
    ]

    with (
        mock.patch("modules.core.config.HARDWARE_UNITS", hw_list),
        mock.patch("modules.inference.model_manager.unload_models"),
    ):
        scheduler.STATE = SchedulerState()
        model_manager.MODEL_POOL.clear()
        model_manager.PREPROCESSOR_POOL.clear()
        for u in hw_list:
            model_manager.MODEL_POOL[u["id"]] = mock.MagicMock()
            model_manager.PREPROCESSOR_POOL[u["id"]] = mock.MagicMock()

        events = []

        def run_standard(unit_id):
            utils.THREAD_CONTEXT.reset()
            model_manager.increment_active_session()
            try:
                with model_manager.early_task_registration(is_priority=False):
                    with model_manager.model_lock_ctx(priority=False) as (_, acquired_unit):
                        events.append(f"std_{acquired_unit}_running")
                        # Trigger one preemption check to pause if requested
                        model_manager._check_preemption()
                        events.append(f"std_{acquired_unit}_paused")
                        # Wait until resumed via the unit-specific resume event
                        model_manager._check_preemption()
                        events.append(f"std_{acquired_unit}_resumed")
            finally:
                model_manager.decrement_active_session()

        def run_priority(name, target):
            utils.THREAD_CONTEXT.reset()
            model_manager.increment_active_session()
            try:
                with model_manager.early_task_registration(is_priority=True):
                    model_manager.wait_for_priority()
                    events.append(f"prio_{name}_waited")
                    with model_manager.model_lock_ctx(priority=True) as (_, acquired_unit):
                        events.append(f"prio_{name}_running_on_{acquired_unit}")
                        time.sleep(0.1)
                        events.append(f"prio_{name}_done")
            finally:
                model_manager.decrement_active_session()

        # Start two standard tasks on both units
        t_std_npu = threading.Thread(target=run_standard, args=("NPU.0",))
        t_std_gpu = threading.Thread(target=run_standard, args=("GPU.0",))
        t_std_npu.start()
        t_std_gpu.start()
        time.sleep(0.1)

        # Start two priority tasks, each of which should target a different unit.
        t_prio1 = threading.Thread(target=run_priority, args=("P1", "NPU.0"))
        t_prio2 = threading.Thread(target=run_priority, args=("P2", "GPU.0"))
        t_prio1.start()
        time.sleep(0.05)
        t_prio2.start()

        t_std_npu.join(timeout=10.0)
        t_std_gpu.join(timeout=10.0)
        t_prio1.join(timeout=10.0)
        t_prio2.join(timeout=10.0)

        assert "std_NPU.0_resumed" in events
        assert "std_GPU.0_resumed" in events
        assert "prio_P1_done" in events
        assert "prio_P2_done" in events


def test_preemption_resume_when_one_priority_remains():
    """Verify that when 2 ASR calls are paused by 2 priority calls, finishing one priority call resumes one ASR task."""
    from modules.inference.scheduler import SchedulerState

    hw_list = [
        {"id": "GPU.0", "type": "GPU", "name": "Intel GPU"},
        {"id": "NPU.0", "type": "NPU", "name": "Intel NPU"},
    ]

    with (
        mock.patch("modules.core.config.HARDWARE_UNITS", hw_list),
        mock.patch("modules.inference.model_manager.unload_models"),
    ):
        scheduler.STATE = SchedulerState()
        model_manager.MODEL_POOL.clear()
        model_manager.PREPROCESSOR_POOL.clear()
        for u in hw_list:
            model_manager.MODEL_POOL[u["id"]] = mock.MagicMock()
            model_manager.PREPROCESSOR_POOL[u["id"]] = mock.MagicMock()

        events = []
        barrier = threading.Barrier(4)

        def run_standard(unit_id):
            utils.THREAD_CONTEXT.reset()
            model_manager.increment_active_session()
            try:
                with model_manager.early_task_registration(is_priority=False):
                    with model_manager.model_lock_ctx(priority=False) as (_, acquired_unit):
                        events.append(f"std_{acquired_unit}_running")
                        barrier.wait()
                        # Trigger preemption check to pause
                        model_manager._check_preemption()
                        events.append(f"std_{acquired_unit}_resumed")
            finally:
                model_manager.decrement_active_session()

        def run_priority(name, target):
            utils.THREAD_CONTEXT.reset()
            model_manager.increment_active_session()
            try:
                with model_manager.early_task_registration(is_priority=True):
                    barrier.wait()
                    utils.THREAD_CONTEXT.target_unit_id = target
                    model_manager.wait_for_priority()
                    with model_manager.model_lock_ctx(priority=True) as (_, acquired_unit):
                        events.append(f"prio_{name}_running_on_{acquired_unit}")
                        time.sleep(0.2)
                        events.append(f"prio_{name}_done")
            finally:
                model_manager.decrement_active_session()

        t_std_gpu = threading.Thread(target=run_standard, args=("GPU.0",))
        t_std_npu = threading.Thread(target=run_standard, args=("NPU.0",))
        t_std_gpu.start()
        t_std_npu.start()

        t_prio1 = threading.Thread(target=run_priority, args=("P1", "GPU.0"))
        t_prio2 = threading.Thread(target=run_priority, args=("P2", "NPU.0"))
        t_prio1.start()
        t_prio2.start()

        time.sleep(0.5)
        t_prio1.join(timeout=2.0)
        t_prio2.join(timeout=2.0)
        t_std_gpu.join(timeout=2.0)
        t_std_npu.join(timeout=2.0)

        assert "prio_P1_done" in events
        assert "prio_P2_done" in events
        assert "std_GPU.0_resumed" in events
        assert "std_NPU.0_resumed" in events


def test_concurrency_no_priority_preemption_reset_deadlock():
    """Verify that multiple concurrent priority requests do not reset preemption flags and deadlock."""
    from modules.inference.scheduler import SchedulerState

    hw_list = [{"id": "NPU.0", "type": "NPU", "name": "Intel NPU"}]

    with (
        mock.patch("modules.core.config.HARDWARE_UNITS", hw_list),
        mock.patch("modules.inference.model_manager.unload_models"),
    ):
        scheduler.STATE = SchedulerState()
        model_manager.MODEL_POOL.clear()
        model_manager.PREPROCESSOR_POOL.clear()
        model_manager.MODEL_POOL["NPU.0"] = mock.MagicMock()
        model_manager.PREPROCESSOR_POOL["NPU.0"] = mock.MagicMock()

        events = []

        # Start 1 standard task on NPU
        def run_standard():
            utils.THREAD_CONTEXT.reset()
            model_manager.increment_active_session()
            try:
                with model_manager.early_task_registration(is_priority=False):
                    with model_manager.model_lock_ctx(priority=False):
                        events.append("std_running")
                        # First check preemption (doesn't pause)
                        model_manager._check_preemption()
                        time.sleep(0.3)
                        # Second check preemption (will pause because Priority 1 targets us)
                        model_manager._check_preemption()
                        events.append("std_resumed")
            finally:
                model_manager.decrement_active_session()

        def run_priority(name):
            utils.THREAD_CONTEXT.reset()
            model_manager.increment_active_session()
            try:
                with model_manager.early_task_registration(is_priority=True):
                    model_manager.wait_for_priority()
                    events.append(f"priority_{name}_waited")
                    with model_manager.model_lock_ctx(priority=True) as (_, unit_id):
                        events.append(f"priority_{name}_running")
                        time.sleep(0.1)
                        events.append(f"priority_{name}_done")
            finally:
                model_manager.decrement_active_session()

        t_std = threading.Thread(target=run_standard)
        t_std.start()
        time.sleep(0.1)

        t_p1 = threading.Thread(target=run_priority, args=("P1",))
        t_p2 = threading.Thread(target=run_priority, args=("P2",))

        t_p1.start()
        time.sleep(0.05)
        t_p2.start()

        # Both priority tasks should complete without waiting for the full 30s timeout
        start_time = time.time()
        t_std.join(timeout=10.0)
        t_p1.join(timeout=10.0)
        t_p2.join(timeout=10.0)
        elapsed = time.time() - start_time

        print("No preemption reset deadlock Events:", events)
        assert elapsed < 5.0  # Should be way under 30 seconds
        assert "priority_P1_done" in events
        assert "priority_P2_done" in events
        assert "std_resumed" in events


def test_concurrency_targeted_preemption_npu_gpu_cuda():
    """Verify targeted preemption on a 3-unit system (NPU, GPU, CUDA), ensuring only the targeted unit is preempted."""
    from modules.inference.scheduler import SchedulerState

    hw_list = [
        {"id": "NPU.0", "type": "NPU", "name": "Intel NPU"},
        {"id": "GPU.0", "type": "GPU", "name": "Intel GPU"},
        {"id": "CUDA.0", "type": "CUDA", "name": "NVIDIA GPU"},
    ]

    with (
        mock.patch("modules.core.config.HARDWARE_UNITS", hw_list),
        mock.patch("modules.inference.model_manager.unload_models"),
    ):
        scheduler.STATE = SchedulerState()

        model_manager.MODEL_POOL.clear()
        model_manager.PREPROCESSOR_POOL.clear()
        for u in hw_list:
            model_manager.MODEL_POOL[u["id"]] = mock.MagicMock()
            model_manager.PREPROCESSOR_POOL[u["id"]] = mock.MagicMock()

        events = []

        # Standard tasks running on NPU, GPU, and CUDA
        def run_std(unit_name):
            utils.THREAD_CONTEXT.reset()
            model_manager.increment_active_session()
            try:
                with model_manager.early_task_registration(is_priority=False):
                    with model_manager.model_lock_ctx(priority=False):
                        events.append(f"std_{unit_name}_running")
                        # Check preemption (none initially)
                        model_manager._check_preemption()
                        events.append(f"std_{unit_name}_checked_1")

                        time.sleep(0.3)

                        # Second check preemption.
                        model_manager._check_preemption()
                        events.append(f"std_{unit_name}_checked_2")
            finally:
                model_manager.decrement_active_session()

        def run_priority():
            utils.THREAD_CONTEXT.reset()
            model_manager.increment_active_session()
            try:
                # Register priority task
                with model_manager.early_task_registration(is_priority=True):
                    model_manager.wait_for_priority()
                    target = getattr(utils.THREAD_CONTEXT, "target_unit_id", "CUDA.0")
                    events.append(f"priority_waited_for_{target}")

                    with model_manager.model_lock_ctx(priority=True) as (_, unit_id):
                        events.append(f"priority_running_on_{unit_id}")
                        time.sleep(0.1)
                        events.append(f"priority_done_on_{unit_id}")
            finally:
                model_manager.decrement_active_session()

        t_npu = threading.Thread(target=run_std, args=("NPU.0",))
        t_gpu = threading.Thread(target=run_std, args=("GPU.0",))
        t_cuda = threading.Thread(target=run_std, args=("CUDA.0",))

        # Start NPU and GPU standard tasks first
        t_npu.start()
        t_gpu.start()
        time.sleep(0.05)
        # Start CUDA standard task last
        t_cuda.start()
        time.sleep(0.1)

        t_prio = threading.Thread(target=run_priority)
        t_prio.start()

        t_npu.join(timeout=5.0)
        t_gpu.join(timeout=5.0)
        t_cuda.join(timeout=5.0)
        t_prio.join(timeout=5.0)

        print("NPU/GPU/CUDA Targeted Preemption Events:", events)

        # Verify that all std tasks started running
        assert "std_NPU.0_running" in events
        assert "std_GPU.0_running" in events
        assert "std_CUDA.0_running" in events

        # Find which unit was targeted by the priority task
        prio_waited_event = [e for e in events if e.startswith("priority_waited_for_")][0]
        targeted_unit = prio_waited_event.split("priority_waited_for_")[1]

        # The targeted unit must have paused and resumed, so its second check event happened after priority was done
        prio_done_event = [e for e in events if e.startswith("priority_done_on_")][0]
        idx_prio_done = events.index(prio_done_event)

        idx_target_checked_2 = events.index(f"std_{targeted_unit}_checked_2")
        assert idx_prio_done < idx_target_checked_2

        # Other non-targeted units should have completed their second check
        for u in hw_list:
            u_id = u["id"]
            assert f"std_{u_id}_checked_2" in events


@pytest.mark.parametrize(
    "unit",
    [
        {"id": "CPU", "type": "CPU", "name": "Host CPU"},
        {"id": "NPU.0", "type": "NPU", "name": "Intel NPU"},
        {"id": "GPU.0", "type": "GPU", "name": "Intel GPU"},
        {"id": "CUDA.0", "type": "CUDA", "name": "NVIDIA GPU"},
    ],
)
def test_concurrency_single_unit_matrix(unit):
    """Verify preemption flow works for each single-unit hardware type."""
    from modules.inference.scheduler import SchedulerState

    hw_list = [unit]
    unit_id = unit["id"]

    with (
        mock.patch("modules.core.config.HARDWARE_UNITS", hw_list),
        mock.patch("modules.inference.model_manager.unload_models"),
    ):
        scheduler.STATE = SchedulerState()
        model_manager.MODEL_POOL.clear()
        model_manager.PREPROCESSOR_POOL.clear()
        model_manager.MODEL_POOL[unit_id] = mock.MagicMock()
        model_manager.PREPROCESSOR_POOL[unit_id] = mock.MagicMock()

        events = []
        t_std = threading.Thread(target=helper_run_transcription, args=(events, "matrix", 3, 0.2))
        t_pri = threading.Thread(target=helper_run_priority, args=(events, "matrix", 0.1))

        t_std.start()
        time.sleep(0.05)
        t_pri.start()

        t_std.join(timeout=10.0)
        t_pri.join(timeout=10.0)

        assert not t_std.is_alive()
        assert not t_pri.is_alive()
        assert f"priority_matrix_running_on_{unit_id}" in events
        assert "priority_matrix_done" in events
        assert "transcription_matrix_done" in events


@pytest.mark.parametrize(
    "hw_list",
    [
        [
            {"id": "NPU.0", "type": "NPU", "name": "Intel NPU"},
            {"id": "GPU.0", "type": "GPU", "name": "Intel GPU"},
        ],
        [
            {"id": "NPU.0", "type": "NPU", "name": "Intel NPU"},
            {"id": "CUDA.0", "type": "CUDA", "name": "NVIDIA GPU"},
        ],
        [
            {"id": "GPU.0", "type": "GPU", "name": "Intel GPU"},
            {"id": "CUDA.0", "type": "CUDA", "name": "NVIDIA GPU"},
        ],
    ],
)
def test_concurrency_dual_unit_matrix(hw_list):
    """Verify priority preemption/resume works across all dual-unit combinations."""
    from modules.inference.scheduler import SchedulerState

    with (
        mock.patch("modules.core.config.HARDWARE_UNITS", hw_list),
        mock.patch("modules.inference.model_manager.unload_models"),
    ):
        scheduler.STATE = SchedulerState()
        model_manager.MODEL_POOL.clear()
        model_manager.PREPROCESSOR_POOL.clear()
        for u in hw_list:
            model_manager.MODEL_POOL[u["id"]] = mock.MagicMock()
            model_manager.PREPROCESSOR_POOL[u["id"]] = mock.MagicMock()

        events = []
        t1 = threading.Thread(target=helper_run_transcription, args=(events, "a", 3, 0.2))
        t2 = threading.Thread(target=helper_run_transcription, args=(events, "b", 3, 0.2))
        tp = threading.Thread(target=helper_run_priority, args=(events, "dual", 0.1))

        t1.start()
        t2.start()
        time.sleep(0.08)
        tp.start()

        t1.join(timeout=10.0)
        t2.join(timeout=10.0)
        tp.join(timeout=10.0)

        assert not t1.is_alive()
        assert not t2.is_alive()
        assert not tp.is_alive()
        assert any(f"priority_dual_running_on_{u['id']}" in events for u in hw_list)
        assert "priority_dual_done" in events
        assert "transcription_a_done" in events
        assert "transcription_b_done" in events


def test_concurrency_priority_burst_no_livelock():
    """Verify repeated priority bursts complete sequentially without livelock."""
    from modules.inference.scheduler import SchedulerState

    hw_list = [{"id": "NPU.0", "type": "NPU", "name": "Intel NPU"}]
    with (
        mock.patch("modules.core.config.HARDWARE_UNITS", hw_list),
        mock.patch("modules.inference.model_manager.unload_models"),
    ):
        scheduler.STATE = SchedulerState()
        model_manager.MODEL_POOL.clear()
        model_manager.PREPROCESSOR_POOL.clear()
        model_manager.MODEL_POOL["NPU.0"] = mock.MagicMock()
        model_manager.PREPROCESSOR_POOL["NPU.0"] = mock.MagicMock()

        events = []

        def run_priority(name):
            utils.THREAD_CONTEXT.reset()
            model_manager.increment_active_session()
            try:
                with model_manager.early_task_registration(is_priority=True):
                    events.append(f"{name}_start")
                    model_manager.wait_for_priority()
                    with model_manager.model_lock_ctx(priority=True):
                        events.append(f"{name}_run")
                        time.sleep(0.08)
                        events.append(f"{name}_done")
            finally:
                model_manager.decrement_active_session()

        threads = []
        for name in ["p1", "p2", "p3", "p4"]:
            t = threading.Thread(target=run_priority, args=(name,))
            threads.append(t)
            t.start()
            time.sleep(0.02)

        for t in threads:
            t.join(timeout=10.0)

        assert all(not t.is_alive() for t in threads)
        for name in ["p1", "p2", "p3", "p4"]:
            assert f"{name}_done" in events

        run_order = [e for e in events if e.endswith("_run") or e.endswith("_done")]
        for i in range(0, len(run_order), 2):
            assert run_order[i].endswith("_run")
            assert run_order[i + 1].endswith("_done")


def test_model_lock_ctx_releases_unit_on_metadata_failure():
    """Verify unit/semaphore are released even if metadata update fails before yielding."""
    from modules.inference.scheduler import SchedulerState

    hw_list = [{"id": "CPU", "type": "CPU", "name": "Host CPU"}]
    with (
        mock.patch("modules.core.config.HARDWARE_UNITS", hw_list),
        mock.patch("modules.inference.model_manager.unload_models"),
    ):
        scheduler.STATE = SchedulerState()
        model_manager.MODEL_POOL.clear()
        model_manager.PREPROCESSOR_POOL.clear()
        model_manager.MODEL_POOL["CPU"] = mock.MagicMock()
        model_manager.PREPROCESSOR_POOL["CPU"] = mock.MagicMock()

        calls = {"count": 0}
        original_update = scheduler.update_task_metadata

        def flaky_update(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("Injected metadata failure")
            return original_update(*args, **kwargs)

        with mock.patch("modules.inference.concurrency.scheduler.update_task_metadata", side_effect=flaky_update):
            with pytest.raises(RuntimeError, match="Injected metadata failure"):
                with model_manager.model_lock_ctx(priority=False):
                    pass

        # Semaphore permit was returned and unit was put back into the pool.
        assert scheduler.STATE.model_lock.acquire(blocking=False)
        scheduler.STATE.model_lock.release()
        assert scheduler.STATE.hw_pool.qsize() == 1
