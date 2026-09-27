"""Scheduler pause confirmation and metadata regression tests split from test_scheduler.py."""

import threading
from unittest import mock

import pytest

from modules.core import utils
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


def test_scheduler_task_helpers_update_progress_type_error(monkeypatch):
    """Verify update_task_progress tolerates mixed progress types."""
    task_id = "test-progress-type-error"
    monkeypatch.setattr(utils.THREAD_CONTEXT, "task_id", task_id, raising=False)
    monkeypatch.setattr(utils.THREAD_CONTEXT, "registration_thread_id", None, raising=False)

    scheduler.STATE.task_registry[task_id] = {"progress": "not-started"}
    scheduler.update_task_progress(50, "stage1")
    first_state = (scheduler.STATE.task_registry[task_id]["progress"], scheduler.STATE.task_registry[task_id]["stage"])

    scheduler.STATE.task_registry[task_id] = {"progress": 50}
    scheduler.update_task_progress("done", "stage2")
    second_state = (scheduler.STATE.task_registry[task_id]["progress"], scheduler.STATE.task_registry[task_id]["stage"])

    scheduler.cleanup_failed_task()
    assert (first_state, second_state) == (("not-started", "stage1"), (50, "stage2"))


def test_release_priority_resumes_targeted_unit_when_queue_empty(monkeypatch):
    """A targeted paused unit should resume once the final priority request is released."""
    monkeypatch.setattr(scheduler.STATE, "priority_requests", 1)
    scheduler.STATE.pause_requested.set()
    scheduler.STATE.resume_event.clear()
    monkeypatch.setattr(utils.THREAD_CONTEXT, "is_priority", True)
    monkeypatch.setattr(utils.THREAD_CONTEXT, "target_unit_id", "CPU", raising=False)

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

    assert (
        generation,
        should_wait,
        scheduler.STATE.pause_generation,
        scheduler.STATE.unit_sync["CPU"]["pause_requested"].is_set(),
        scheduler.STATE.unit_sync["CPU"]["resume_event"].is_set(),
    ) == (1, True, 1, True, False)


def test_wait_for_priority_skips_duplicate_confirmation_wait_when_target_already_pausing(monkeypatch):
    """A follower priority request should not wait again for pause confirmation."""
    monkeypatch.setattr(scheduler.STATE, "active_sessions", 1)
    monkeypatch.setattr(scheduler.STATE, "accel_limit", 1)
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


def test_select_preemption_target_prefers_configured_gpu_before_npu():
    """When both GPU and NPU are active, preemption should target the earlier configured GPU first."""
    hw_list = [
        {"id": "GPU.0", "type": "GPU", "name": "Intel GPU"},
        {"id": "NPU.0", "type": "NPU", "name": "Intel NPU"},
    ]

    with mock.patch("modules.core.config.HARDWARE_UNITS", hw_list):
        scheduler.STATE.task_registry["gpu-task"] = {
            "task_id": "gpu-task",
            "status": "active",
            "is_priority": False,
            "unit_id": "GPU.0",
        }
        scheduler.STATE.task_registry["npu-task"] = {
            "task_id": "npu-task",
            "status": "active",
            "is_priority": False,
            "unit_id": "NPU.0",
        }

        assert scheduler._select_preemption_target_unit() == "GPU.0"


def test_update_task_metadata_updates_live_text_for_existing_entry(monkeypatch):
    """Live text updates should refresh an existing task entry in place."""
    task_id = "live-task"
    monkeypatch.setattr(utils.THREAD_CONTEXT, "task_id", task_id, raising=False)
    monkeypatch.setattr(utils.THREAD_CONTEXT, "registration_thread_id", threading.get_ident(), raising=False)

    scheduler.STATE.task_registry[task_id] = {
        "task_id": task_id,
        "status": "active",
        "stage": "Queued",
    }

    scheduler.update_task_metadata(live_text="Hello world")

    with scheduler.STATE.task_registry_lock:
        assert scheduler.STATE.task_registry[task_id]["live_text"] == "Hello world"


def test_update_task_metadata_and_progress_create_missing_entry(monkeypatch):
    """Missing task updates should log a warning and not create a registry entry."""
    task_id = "missing-task"
    monkeypatch.setattr(utils.THREAD_CONTEXT, "task_id", task_id, raising=False)
    monkeypatch.setattr(utils.THREAD_CONTEXT, "registration_thread_id", threading.get_ident(), raising=False)

    scheduler.update_task_metadata(stage="Queued")
    scheduler.update_task_progress(42, stage="Processing")

    with scheduler.STATE.task_registry_lock:
        assert utils.THREAD_CONTEXT.registration_thread_id not in scheduler.STATE.task_registry
        assert task_id not in scheduler.STATE.task_registry


def test_update_task_progress_does_not_regress_existing_progress_or_stage(monkeypatch):
    """Fallback flows must not rewind a task's progress or visible stage."""
    task_id = "ld-task"
    monkeypatch.setattr(utils.THREAD_CONTEXT, "task_id", task_id, raising=False)
    monkeypatch.setattr(utils.THREAD_CONTEXT, "registration_thread_id", threading.get_ident(), raising=False)
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
        assert task_entry["stage"] == "Vocal Separation"


def test_wait_for_priority_waits_for_pause_confirmation_without_timeout(monkeypatch):
    """wait_for_priority should rely on cooperative confirmation without timeout failures."""
    monkeypatch.setattr(scheduler.STATE, "active_sessions", 1)
    monkeypatch.setattr(scheduler.STATE, "accel_limit", 1)
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
        "unit_id": None,
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
