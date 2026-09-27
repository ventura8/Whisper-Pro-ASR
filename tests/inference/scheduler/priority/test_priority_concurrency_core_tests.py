"""Core priority concurrency tests split from test_priority_concurrency.py."""

import threading
import time
from unittest import mock

from modules.core import utils
from modules.inference import scheduler
from modules.inference.runtime import model_manager
from tests.inference.scheduler.priority.priority_concurrency_scenarios import (
    _exercise_parallel_detectlang_two_asr_yielded,
    _exercise_priority_non_preemptive,
    _exercise_zero_hardware_units_execution,
)
from tests.inference.scheduler.priority.test_priority_concurrency import (
    _exercise_one_hardware_unit,
    _exercise_three_hardware_units,
    _exercise_two_hardware_units,
    _run_resume_priority,
    _run_resume_standard,
    _run_standard_gpu_preemption,
    _run_standard_npu_preemption,
    _run_targeted_priority_preemption,
    _setup_priority_scheduler,
    helper_run_transcription,
)


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
    assert _exercise_one_hardware_unit() == {
        "threads_done": True,
        "p1_before_p2": True,
        "p1_running": True,
        "p2_running": True,
    }


def test_concurrency_two_hardware_units():
    """Verify preemption and resource sharing with 2 hardware units (Intel NPU and Intel GPU)."""
    assert _exercise_two_hardware_units() == {
        "threads_done": True,
        "priority_running": True,
        "priority_done_before_one_transcription": True,
    }


def test_detectlang_uses_parallel_priority_units_when_two_asr_yielded():
    """With 2 busy ASR units and 4 detect-language requests, priority can run in parallel across borrowed units."""
    result = _exercise_parallel_detectlang_two_asr_yielded()
    assert result == {
        "prio_threads_done": True,
        "asr_threads_done": True,
        "max_running_priority": 2,
        "priority_done_count": 4,
    }


def test_concurrency_three_hardware_units():
    """Verify preemption and resource sharing with 3 hardware units (Intel NPU, Intel GPU, and NVIDIA GPU)."""
    assert _exercise_three_hardware_units() == {
        "threads_done": True,
        "priority_running": True,
        "priority_done_before_one_transcription": True,
    }


def test_concurrency_zero_hardware_units_execution():
    """Verify preemption and execution with 0 hardware units (CPU fallback)."""
    assert _exercise_zero_hardware_units_execution() == {
        "threads_done": True,
        "priority_on_cpu": True,
        "priority_done_before_transcription": True,
    }


def test_concurrency_priority_non_preemptive():
    """Verify that priority task does not preempt if there is an idle unit."""
    assert _exercise_priority_non_preemptive() == {
        "threads_done": True,
        "priority_on_idle_unit": True,
        "no_pause_requested": True,
    }


def test_concurrency_fallback_no_deadlock(monkeypatch):
    """Verify that language detection fallback does not trigger a re-entrancy deadlock."""
    from modules.inference.pipeline import language_detection
    from modules.inference.scheduler import SchedulerState

    hw_list = [{"id": "CPU", "type": "CPU", "name": "Host CPU"}]

    with (
        mock.patch("modules.core.config.HARDWARE_UNITS", hw_list),
        mock.patch("modules.inference.runtime.model_manager.unload_models"),
        mock.patch("modules.inference.pipeline.language_detection.utils.get_audio_duration", return_value=30),
    ):
        monkeypatch.setattr(scheduler, "STATE", SchedulerState())

        model_manager.MODEL_POOL.clear()
        model_manager.PREPROCESSOR_POOL.clear()
        mock_model = mock.MagicMock()
        model_manager.MODEL_POOL["CPU"] = mock_model
        model_manager.PREPROCESSOR_POOL["CPU"] = mock.MagicMock()

        with (
            mock.patch("modules.inference.runtime.model_manager.run_batch_language_detection_direct", return_value=[]),
            mock.patch("modules.inference.runtime.model_manager.run_language_detection_core") as mock_core,
            mock.patch("modules.inference.pipeline.language_detection._prepare_montage", return_value="montage.wav"),
        ):
            mock_core.return_value = {"detected_language": "en", "confidence": 0.9, "all_probabilities": {"en": 0.9}}

            result = language_detection.run_voting_detection("dummy.wav", model_manager)

            assert result["detected_language"] == "en"
            assert result["confidence"] == 0.9
            mock_core.assert_called_once()


def test_concurrency_priority_task_failure_resumes_standard_task(monkeypatch):
    """Verify that if a priority task fails/errors, any paused standard task resumes."""
    from modules.inference.scheduler import SchedulerState

    hw_list = [{"id": "NPU.0", "type": "NPU", "name": "Intel NPU"}]

    with (
        mock.patch("modules.core.config.HARDWARE_UNITS", hw_list),
        mock.patch("modules.inference.runtime.model_manager.unload_models"),
    ):
        monkeypatch.setattr(scheduler, "STATE", SchedulerState())

        model_manager.MODEL_POOL.clear()
        model_manager.PREPROCESSOR_POOL.clear()
        model_manager.MODEL_POOL["NPU.0"] = mock.MagicMock()
        model_manager.PREPROCESSOR_POOL["NPU.0"] = mock.MagicMock()

        monkeypatch.setattr(utils.THREAD_CONTEXT, "is_priority", False, raising=False)
        events = []

        t_trans = threading.Thread(target=helper_run_transcription, args=(events, "1", 3, 0.3))
        t_trans.start()
        time.sleep(0.1)

        def run_failing_priority():
            events.append("priority_failed_start")
            model_manager.increment_active_session()
            try:
                with model_manager.early_task_registration(is_priority=True):
                    events.append("priority_failed_waiting")
                    model_manager.wait_for_priority()
                    events.append("priority_failed_waited")
                    raise RuntimeError("Simulated priority task failure")
            except RuntimeError as error:
                events.append(f"priority_failed_error_{error}")
            finally:
                model_manager.decrement_active_session()

        t_prio = threading.Thread(target=run_failing_priority)
        t_prio.start()

        t_trans.join(timeout=10.0)
        t_prio.join(timeout=10.0)

        assert not t_trans.is_alive()
        assert not t_prio.is_alive()
        assert "priority_failed_error_Simulated priority task failure" in events
        assert "transcription_1_done" in events


def test_concurrency_multiple_priority_tasks_allow_parallel_registration(monkeypatch):
    """Verify that priority task registration is not serialized."""
    from modules.inference.scheduler import SchedulerState

    hw_list = [{"id": "NPU.0", "type": "NPU", "name": "Intel NPU"}]

    with (
        mock.patch("modules.core.config.HARDWARE_UNITS", hw_list),
        mock.patch("modules.inference.runtime.model_manager.unload_models"),
    ):
        monkeypatch.setattr(scheduler, "STATE", SchedulerState())

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
        time.sleep(0.1)
        t2.start()

        t1.join(timeout=5.0)
        t2.join(timeout=5.0)

        assert not t1.is_alive()
        assert not t2.is_alive()
        assert events.index("prio_2_registered") < events.index("prio_1_done")


def test_standard_task_not_blocked_by_queued_priority_registration(monkeypatch):
    """Queued priority registration alone should not gate standard acquisition."""
    from modules.inference.scheduler import SchedulerState

    hw_list = [{"id": "CPU", "type": "CPU", "name": "Host CPU"}]
    with (
        mock.patch("modules.core.config.HARDWARE_UNITS", hw_list),
        mock.patch("modules.inference.runtime.model_manager.unload_models"),
    ):
        monkeypatch.setattr(scheduler, "STATE", SchedulerState())
        model_manager.MODEL_POOL.clear()
        model_manager.PREPROCESSOR_POOL.clear()
        model_manager.MODEL_POOL["CPU"] = mock.MagicMock()
        model_manager.PREPROCESSOR_POOL["CPU"] = mock.MagicMock()

        with scheduler.early_task_registration(is_priority=True):
            monkeypatch.setattr(utils.THREAD_CONTEXT, "is_priority", False, raising=False)

            acquired = []

            def try_acquire():
                with model_manager.model_lock_ctx(priority=False):
                    acquired.append(True)

            thread = threading.Thread(target=try_acquire)
            thread.start()
            time.sleep(0.3)
            assert acquired == [True]

        thread.join(timeout=2.0)
        assert acquired == [True]


def test_priority_does_not_preempt_itself(monkeypatch):
    """Verify that priority tasks are bypass-ignored by the preemption check."""
    from modules.inference.scheduler import SchedulerState

    monkeypatch.setattr(scheduler, "STATE", SchedulerState())
    thread_id = threading.get_ident()
    scheduler.STATE.task_registry[thread_id] = {"status": "active", "is_priority": True, "unit_id": "CPU"}

    scheduler.STATE.pause_requested.set()
    start = time.time()
    model_manager._check_preemption()
    assert time.time() - start < 0.2

    del scheduler.STATE.task_registry[thread_id]


def test_concurrency_targeted_preemption():
    """Verify that a priority task only preempts/pauses the targeted unit, leaving other units running."""
    hw_list = [{"id": "GPU.0", "type": "GPU", "name": "Intel GPU"}, {"id": "NPU.0", "type": "NPU", "name": "Intel NPU"}]

    with (
        mock.patch("modules.core.config.HARDWARE_UNITS", hw_list),
        mock.patch("modules.inference.runtime.model_manager.unload_models"),
    ):
        _setup_priority_scheduler(hw_list)

        events = []

        t_gpu = threading.Thread(target=_run_standard_gpu_preemption, args=(events,))
        t_npu = threading.Thread(target=_run_standard_npu_preemption, args=(events,))

        t_gpu.start()
        t_npu.start()
        time.sleep(0.1)

        t_prio = threading.Thread(target=_run_targeted_priority_preemption, args=(events,))
        t_prio.start()

        t_gpu.join(timeout=5.0)
        t_npu.join(timeout=5.0)
        t_prio.join(timeout=5.0)

        assert "std_gpu_checked_no_pause" in events
        assert "std_npu_resumed" in events
        assert "priority_running_on_NPU.0" in events
        assert events.index("priority_done") < events.index("std_npu_resumed")


def test_concurrency_targeted_preemption_unit_resume_with_multiple_priorities():
    """Verify paused ASR work resumes only after queued priority tasks are drained."""
    hw_list = [
        {"id": "NPU.0", "type": "NPU", "name": "Intel NPU"},
        {"id": "GPU.0", "type": "GPU", "name": "Intel GPU"},
    ]

    with (
        mock.patch("modules.core.config.HARDWARE_UNITS", hw_list),
        mock.patch("modules.inference.runtime.model_manager.unload_models"),
    ):
        _setup_priority_scheduler(hw_list)

        events = []

        barrier = threading.Barrier(4)
        t_std_npu = threading.Thread(target=_run_resume_standard, args=(events, barrier))
        t_std_gpu = threading.Thread(target=_run_resume_standard, args=(events, barrier))
        t_prio1 = threading.Thread(target=_run_resume_priority, args=(events, "P1", "NPU.0", barrier))
        t_prio2 = threading.Thread(target=_run_resume_priority, args=(events, "P2", "GPU.0", barrier))
        t_std_npu.start()
        t_std_gpu.start()
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
