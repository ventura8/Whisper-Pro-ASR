"""Extended priority concurrency tests split from test_priority_concurrency.py."""

import threading
import time
from unittest import mock

import pytest

from modules.inference import scheduler
from modules.inference.runtime import model_manager
from tests.inference.scheduler.priority.priority_concurrency_scenarios import (
    _exercise_dual_unit_matrix,
    _exercise_single_unit_matrix,
)
from tests.inference.scheduler.priority.test_priority_concurrency import (
    _exercise_priority_burst_no_livelock,
    _exercise_targeted_preemption_npu_gpu_cuda,
    _run_barrier_priority,
    _run_barrier_standard,
    _setup_priority_scheduler,
)


def test_preemption_resume_when_one_priority_remains():
    """Verify that when 2 ASR calls are paused by 2 priority calls, finishing one priority call resumes one ASR task."""
    hw_list = [
        {"id": "GPU.0", "type": "GPU", "name": "Intel GPU"},
        {"id": "NPU.0", "type": "NPU", "name": "Intel NPU"},
    ]

    with (
        mock.patch("modules.core.config.HARDWARE_UNITS", hw_list),
        mock.patch("modules.inference.runtime.model_manager.unload_models"),
    ):
        _setup_priority_scheduler(hw_list)

        events = []
        barrier = threading.Barrier(4)
        t_std_gpu = threading.Thread(target=_run_barrier_standard, args=(events, barrier))
        t_std_npu = threading.Thread(target=_run_barrier_standard, args=(events, barrier))
        t_std_gpu.start()
        t_std_npu.start()

        t_prio1 = threading.Thread(target=_run_barrier_priority, args=(events, "P1", "GPU.0", barrier))
        t_prio2 = threading.Thread(target=_run_barrier_priority, args=(events, "P2", "NPU.0", barrier))
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


def test_concurrency_no_priority_preemption_reset_deadlock(monkeypatch):
    """Verify that multiple concurrent priority requests do not reset preemption flags and deadlock."""
    from modules.core import utils
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

        def run_standard():
            utils.THREAD_CONTEXT.reset()
            model_manager.increment_active_session()
            try:
                with model_manager.early_task_registration(is_priority=False):
                    with model_manager.model_lock_ctx(priority=False):
                        events.append("std_running")
                        model_manager._check_preemption()
                        time.sleep(0.3)
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
                    with model_manager.model_lock_ctx(priority=True):
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

        start_time = time.time()
        t_std.join(timeout=10.0)
        t_p1.join(timeout=10.0)
        t_p2.join(timeout=10.0)
        elapsed = time.time() - start_time

        assert elapsed < 5.0
        assert "priority_P1_done" in events
        assert "priority_P2_done" in events
        assert "std_resumed" in events


def test_concurrency_targeted_preemption_npu_gpu_cuda():
    """Verify targeted preemption on a 3-unit system (NPU, GPU, CUDA), ensuring only the targeted unit is preempted."""
    result = _exercise_targeted_preemption_npu_gpu_cuda()
    assert result == {
        "threads_done": True,
        "all_standard_started": True,
        "targeted_unit": result["targeted_unit"],
        "prio_done_before_target_second_check": True,
        "all_units_completed_second_check": True,
    }


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
    assert _exercise_single_unit_matrix(unit) == {
        "threads_done": True,
        "priority_running": True,
        "priority_done": True,
        "transcription_done": True,
    }


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
    assert _exercise_dual_unit_matrix(hw_list) == {
        "threads_done": True,
        "priority_running": True,
        "priority_done": True,
        "transcriptions_done": True,
    }


def test_concurrency_priority_burst_no_livelock():
    """Verify repeated priority bursts complete sequentially without livelock."""
    result = _exercise_priority_burst_no_livelock()
    assert result == {
        "threads_done": True,
        "all_done_events": True,
        "run_order_pairs_valid": True,
    }


def test_model_lock_ctx_releases_unit_on_metadata_failure(monkeypatch):
    """Verify unit/semaphore are released even if metadata update fails before yielding."""
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

        calls = {"count": 0}
        original_update = scheduler.update_task_metadata

        def flaky_update(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("Injected metadata failure")
            return original_update(*args, **kwargs)

        with mock.patch("modules.inference.runtime.concurrency.scheduler.update_task_metadata", side_effect=flaky_update):
            with pytest.raises(RuntimeError, match="Injected metadata failure"):
                with model_manager.model_lock_ctx(priority=False):
                    pass

        assert scheduler.STATE.model_lock.acquire(blocking=False)
        scheduler.STATE.model_lock.release()
        assert scheduler.STATE.hw_pool.qsize() == 1
