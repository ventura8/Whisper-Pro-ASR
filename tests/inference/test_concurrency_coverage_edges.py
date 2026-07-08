"""Targeted coverage tests for concurrency helper edge branches."""

import queue
import threading
from unittest import mock

import pytest

from modules.core import utils
from modules.inference import concurrency, model_manager, scheduler


def test_try_borrow_preemptible_unit_without_target_covers_known_and_unknown_ids():
    """Cover non-targeted borrow branch for resolvable and unknown preemptible IDs."""
    utils.THREAD_CONTEXT.target_unit_id = None

    isolated_state = mock.Mock()
    isolated_state.task_registry_lock = threading.RLock()

    with (
        mock.patch.object(scheduler, "STATE", isolated_state),
        mock.patch(
            "modules.core.config.HARDWARE_UNITS",
            [
                {"id": "CPU", "type": "CPU", "name": "Host CPU"},
                {"id": "GPU.0", "type": "GPU", "name": "Intel GPU"},
            ],
        ),
    ):
        isolated_state.preemptible_units = {"CPU"}
        with mock.patch("modules.inference.concurrency._is_preprocessor_lock_available", return_value=True):
            unit = concurrency._try_borrow_preemptible_unit()
            assert unit is not None
            assert unit["id"] == "CPU"
            assert "CPU" not in isolated_state.preemptible_units

        isolated_state.preemptible_units = {"UNKNOWN"}
        with mock.patch("modules.inference.concurrency._is_preprocessor_lock_available", return_value=True):
            unit = concurrency._try_borrow_preemptible_unit()
            assert unit is None


def test_is_preprocessor_lock_available_returns_false_when_lock_held():
    """Lock probe should report unavailable when non-blocking acquire fails."""
    lock = mock.MagicMock()
    lock.acquire.return_value = False
    fake_model_manager = mock.Mock(PREPROCESSOR_POOL={"GPU.0": mock.Mock(lock=lock)})

    with mock.patch.dict(
        concurrency.sys.modules,
        {"modules.inference.model_manager": fake_model_manager},
        clear=False,
    ):
        assert concurrency._is_preprocessor_lock_available("GPU.0") is False
        lock.acquire.assert_called_once_with(blocking=False)
        lock.release.assert_not_called()


def test_is_preprocessor_lock_available_releases_lock_after_probe():
    """Successful lock probe must release immediately to avoid side effects."""
    lock = mock.MagicMock()
    lock.acquire.return_value = True
    fake_model_manager = mock.Mock(PREPROCESSOR_POOL={"GPU.0": mock.Mock(lock=lock)})

    with mock.patch.dict(
        concurrency.sys.modules,
        {"modules.inference.model_manager": fake_model_manager},
        clear=False,
    ):
        assert concurrency._is_preprocessor_lock_available("GPU.0") is True
        lock.acquire.assert_called_once_with(blocking=False)
        lock.release.assert_called_once_with()


def test_is_preprocessor_lock_available_allows_missing_model_manager_or_lock():
    """Defensive compatibility: missing model manager/preprocessor/lock remains borrow-eligible."""
    with mock.patch.dict(
        concurrency.sys.modules,
        {"modules.inference.model_manager": None},
        clear=False,
    ):
        assert concurrency._is_preprocessor_lock_available("GPU.0") is True

    fake_model_manager = mock.Mock(PREPROCESSOR_POOL={})
    with mock.patch.dict(
        concurrency.sys.modules,
        {"modules.inference.model_manager": fake_model_manager},
        clear=False,
    ):
        assert concurrency._is_preprocessor_lock_available("GPU.0") is True

    fake_model_manager = mock.Mock(PREPROCESSOR_POOL={"GPU.0": mock.Mock()})
    with mock.patch.dict(
        concurrency.sys.modules,
        {"modules.inference.model_manager": fake_model_manager},
        clear=False,
    ):
        assert concurrency._is_preprocessor_lock_available("GPU.0") is True


@mock.patch("modules.core.config.HARDWARE_UNITS", [{"id": "CPU", "type": "CPU", "name": "Host CPU"}])
def test_try_borrow_preemptible_unit_targeted_missing_returns_none():
    """Cover targeted borrow branch when target is not currently preemptible."""
    utils.THREAD_CONTEXT.target_unit_id = "CPU"
    scheduler.STATE.preemptible_units.clear()

    unit = concurrency._try_borrow_preemptible_unit()

    assert unit is None


def test_try_borrow_preemptible_unit_falls_back_to_other_preemptible_when_target_missing():
    """Priority task should borrow another available preemptible unit if target is not ready."""
    isolated_state = mock.Mock()
    isolated_state.task_registry_lock = threading.RLock()
    isolated_state.preemptible_units = {"GPU.0"}

    with (
        mock.patch.object(scheduler, "STATE", isolated_state),
        mock.patch(
            "modules.core.config.HARDWARE_UNITS",
            [
                {"id": "NPU.0", "type": "NPU", "name": "Intel NPU"},
                {"id": "GPU.0", "type": "GPU", "name": "Intel GPU"},
            ],
        ),
    ):
        utils.THREAD_CONTEXT.target_unit_id = "NPU.0"

        unit = concurrency._try_borrow_preemptible_unit()

        assert unit is not None
        assert unit["id"] == "GPU.0"


def test_try_borrow_preemptible_unit_skips_target_when_preprocessor_lock_is_held():
    """Targeted borrow should not remove a preemptible unit when preprocessor lock is unavailable."""
    isolated_state = mock.Mock()
    isolated_state.task_registry_lock = threading.RLock()
    isolated_state.preemptible_units = {"NPU.0"}

    with (
        mock.patch.object(scheduler, "STATE", isolated_state),
        mock.patch("modules.core.config.HARDWARE_UNITS", [{"id": "NPU.0", "type": "NPU", "name": "Intel NPU"}]),
        mock.patch("modules.inference.concurrency._is_preprocessor_lock_available", return_value=False),
    ):
        utils.THREAD_CONTEXT.target_unit_id = "NPU.0"
        unit = concurrency._try_borrow_preemptible_unit()

    assert unit is None
    assert "NPU.0" in isolated_state.preemptible_units


def test_try_borrow_preemptible_unit_fallback_skips_locked_candidate_and_uses_next():
    """Fallback borrow should skip locked candidates and use the next runnable unit."""
    isolated_state = mock.Mock()
    isolated_state.task_registry_lock = threading.RLock()
    isolated_state.preemptible_units = {"GPU.0", "GPU.1"}

    with (
        mock.patch.object(scheduler, "STATE", isolated_state),
        mock.patch(
            "modules.core.config.HARDWARE_UNITS",
            [
                {"id": "NPU.0", "type": "NPU", "name": "Intel NPU"},
                {"id": "GPU.0", "type": "GPU", "name": "Intel GPU 0"},
                {"id": "GPU.1", "type": "GPU", "name": "Intel GPU 1"},
            ],
        ),
        mock.patch(
            "modules.inference.concurrency._is_preprocessor_lock_available",
            side_effect=lambda uid: uid != "GPU.0",
        ),
    ):
        utils.THREAD_CONTEXT.target_unit_id = "NPU.0"
        unit = concurrency._try_borrow_preemptible_unit()

    assert unit is not None
    assert unit["id"] == "GPU.1"
    assert "GPU.0" in isolated_state.preemptible_units
    assert "GPU.1" not in isolated_state.preemptible_units


def test_try_take_idle_unit_handles_missing_qsize():
    """Cover qsize AttributeError fallback path."""
    with mock.patch.object(scheduler.STATE, "hw_pool", object()):
        assert concurrency._try_take_idle_unit() is None


def test_try_take_idle_unit_lock_not_acquired_returns_none():
    """Cover idle path when non-blocking lock acquisition fails."""
    mock_pool = mock.MagicMock()
    mock_pool.qsize.return_value = 1
    with (
        mock.patch.object(scheduler.STATE, "hw_pool", mock_pool),
        mock.patch.object(scheduler.STATE.model_lock, "acquire", return_value=False),
    ):
        assert concurrency._try_take_idle_unit() is None


def test_try_take_idle_unit_queue_empty_releases_permit():
    """Cover queue.Empty path and ensure semaphore release is called."""
    mock_pool = mock.MagicMock()
    mock_pool.qsize.return_value = 1
    mock_pool.get.side_effect = queue.Empty
    with (
        mock.patch.object(scheduler.STATE, "hw_pool", mock_pool),
        mock.patch.object(scheduler.STATE.model_lock, "acquire", return_value=True),
        mock.patch.object(scheduler.STATE.model_lock, "release") as mock_release,
    ):
        assert concurrency._try_take_idle_unit() is None
        mock_release.assert_called_once()


def test_priority_acquire_unit_fallback_marks_borrowed():
    """Cover fallback borrow path when no targeted unit is set."""
    utils.THREAD_CONTEXT.target_unit_id = None
    with (
        mock.patch("modules.inference.concurrency._try_take_idle_unit", return_value=None),
        mock.patch("modules.inference.concurrency._try_borrow_preemptible_unit", return_value={"id": "CPU"}),
    ):
        unit, borrowed = concurrency._priority_acquire_unit()
    assert unit == {"id": "CPU"}
    assert borrowed is True


def test_mark_task_queued_once_true_path():
    """Cover branch where task is already marked as queued."""
    assert concurrency._mark_task_queued_once(True) is True


def test_try_acquire_unit_now_queue_empty_path():
    """Cover queue.Empty release path in immediate acquisition helper."""
    mock_pool = mock.MagicMock()
    mock_pool.get.side_effect = queue.Empty
    with (
        mock.patch.object(scheduler.STATE, "hw_pool", mock_pool),
        mock.patch.object(scheduler.STATE.model_lock, "acquire", return_value=True),
        mock.patch.object(scheduler.STATE.model_lock, "release") as mock_release,
    ):
        assert concurrency._try_acquire_unit_now() is None
        mock_release.assert_called_once()


def test_acquire_unit_for_task_waits_without_timeout_when_no_progress_possible():
    """Acquisition loop for standard tasks should keep waiting until progress is possible."""
    with (
        mock.patch("modules.inference.concurrency._try_acquire_unit_now", return_value=None),
        mock.patch("modules.inference.concurrency.time.sleep", side_effect=RuntimeError("stop-loop")),
    ):
        with pytest.raises(RuntimeError, match="stop-loop"):
            concurrency._acquire_unit_for_task(is_priority=False)


def test_standard_task_acquires_idle_unit_while_priority_is_active_without_backlog():
    """Active priority work should not block standard acquisition when no queued priority exists."""
    utils.THREAD_CONTEXT.reset()

    with (
        mock.patch("modules.inference.scheduler.has_queued_priority_tasks", return_value=False),
        mock.patch("modules.inference.concurrency._try_acquire_unit_now", return_value={"id": "GPU.0"}),
        mock.patch("modules.inference.concurrency.time.sleep") as mock_sleep,
    ):
        unit, borrowed = concurrency._acquire_unit_for_task(is_priority=False)

    assert unit == {"id": "GPU.0"}
    assert borrowed is False
    mock_sleep.assert_not_called()


def test_standard_task_not_blocked_when_queued_priority_below_capacity(monkeypatch):
    """On two units, one queued priority task should not block standard acquisition."""
    scheduler.STATE.accel_limit = 2
    utils.THREAD_CONTEXT.reset()

    monkeypatch.setattr(scheduler, "get_queued_priority_count", lambda *_args, **_kwargs: 1)

    with (
        mock.patch("modules.inference.scheduler.has_earlier_task", return_value=False),
        mock.patch("modules.inference.concurrency._try_acquire_unit_now", return_value={"id": "GPU.0"}),
        mock.patch("modules.inference.concurrency.time.sleep") as mock_sleep,
    ):
        unit, borrowed = concurrency._acquire_unit_for_task(is_priority=False)

    assert unit == {"id": "GPU.0"}
    assert borrowed is False
    mock_sleep.assert_not_called()


@pytest.mark.parametrize(
    ("accel_limit", "queued_priority_count", "expect_blocked"),
    [
        (1, 1, False),
        (2, 1, False),
        (2, 2, False),
        (3, 2, False),
        (3, 3, False),
        (4, 3, False),
        (4, 4, False),
    ],
)
def test_standard_task_capacity_threshold_for_queued_priority(
    monkeypatch, accel_limit, queued_priority_count, expect_blocked
):
    """Standard acquisition should remain unit-driven and not block on queued-priority count alone."""
    scheduler.STATE.accel_limit = accel_limit
    utils.THREAD_CONTEXT.reset()

    monkeypatch.setattr(scheduler, "get_queued_priority_count", lambda *_args, **_kwargs: queued_priority_count)
    monkeypatch.setattr(scheduler, "has_earlier_task", lambda *_args, **_kwargs: False)

    if expect_blocked:
        with (
            mock.patch("modules.inference.concurrency._try_acquire_unit_now", return_value={"id": "GPU.0"}),
            mock.patch("modules.inference.concurrency.time.sleep", side_effect=RuntimeError("blocked-loop")),
        ):
            with pytest.raises(RuntimeError, match="blocked-loop"):
                concurrency._acquire_unit_for_task(is_priority=False)
    else:
        with (
            mock.patch("modules.inference.concurrency._try_acquire_unit_now", return_value={"id": "GPU.0"}),
            mock.patch("modules.inference.concurrency.time.sleep") as mock_sleep,
        ):
            unit, borrowed = concurrency._acquire_unit_for_task(is_priority=False)

        assert unit == {"id": "GPU.0"}
        assert borrowed is False
        mock_sleep.assert_not_called()


def test_check_preemption_ignores_shared_pause_when_unit_sync_missing():
    """Preemption should not trigger from shared pause flags when no unit-specific sync exists."""
    utils.THREAD_CONTEXT.reset()
    utils.THREAD_CONTEXT.task_id = "fallback-task"
    scheduler.STATE.task_registry["fallback-task"] = {
        "status": "active",
        "is_priority": False,
        "unit_id": "UNKNOWN.0",
        "progress": 0,
        "stage": "Inference",
    }
    scheduler.STATE.pause_requested.set()

    with mock.patch.object(model_manager, "update_task_metadata") as mock_update_meta:
        model_manager._check_preemption()
        mock_update_meta.assert_not_called()

    scheduler.STATE.pause_requested.clear()
    scheduler.STATE.task_registry.pop("fallback-task", None)


def test_determine_preemption_needed_includes_pause_generation():
    """Preemption detection should include the active pause generation token."""
    unit_id = "CPU"
    u_sync = scheduler.STATE.unit_sync[unit_id]
    u_sync["pause_generation"] = 11
    u_sync["pause_requested"].set()

    should_preempt, _, pause_req_evt, pause_generation = concurrency._determine_preemption_needed(unit_id)

    assert should_preempt is True
    assert pause_req_evt is u_sync["pause_requested"]
    assert pause_generation == 11

    u_sync["pause_requested"].clear()


def test_restore_task_state_promotes_stale_paused_queue_status_to_active():
    """Resumed tasks should not fabricate a stage when none is available."""
    utils.THREAD_CONTEXT.reset()
    utils.THREAD_CONTEXT.task_id = "resume-task"
    scheduler.STATE.task_registry["resume-task"] = {
        "status": "queued",
        "is_priority": False,
        "unit_id": "CPU",
        "progress": 55,
        "stage": "Paused for Priority Task",
    }

    concurrency._restore_task_state(
        task_id="resume-task",
        thread_id=threading.get_ident(),
        old_status="queued",
        old_stage="Paused for Priority Task",
        unit_id="CPU",
    )

    task = scheduler.STATE.task_registry["resume-task"]
    assert task["status"] == "active"
    assert task["stage"] == "Paused for Priority Task"
    assert task["progress"] == 55

    scheduler.STATE.task_registry.pop("resume-task", None)


def test_restore_task_state_restores_vocal_separation_segment_stage():
    """Resume should keep the last known vocal-separation segment stage."""
    utils.THREAD_CONTEXT.reset()
    utils.THREAD_CONTEXT.task_id = "resume-vocal-task"
    scheduler.STATE.task_registry["resume-vocal-task"] = {
        "status": "queued",
        "is_priority": False,
        "unit_id": "GPU",
        "progress": 6,
        "stage": "Paused for Priority Task",
    }

    concurrency._restore_task_state(
        task_id="resume-vocal-task",
        thread_id=threading.get_ident(),
        old_status="active",
        old_stage="Vocal Separation (2/10 segments | 00:20:00 / 01:30:26)",
        unit_id="GPU",
    )

    task = scheduler.STATE.task_registry["resume-vocal-task"]
    assert task["status"] == "active"
    assert task["stage"] == "Vocal Separation (2/10 segments | 00:20:00 / 01:30:26)"
    assert task["progress"] == 6

    scheduler.STATE.task_registry.pop("resume-vocal-task", None)


def test_handle_preemption_pause_resume_self_heals_stale_pause_gate_without_priority_work():
    """Paused standard tasks should not block forever when no priority work remains."""
    utils.THREAD_CONTEXT.reset()
    unit_id = "CPU"
    u_sync = scheduler.STATE.unit_sync[unit_id]
    pause_req_evt = u_sync["pause_requested"]

    pause_req_evt.set()
    u_sync["resume_event"].clear()
    scheduler.STATE.pause_requested.set()
    scheduler.STATE.resume_event.clear()
    scheduler.STATE.preemptible_units.add(unit_id)

    concurrency._handle_preemption_pause_resume(
        unit_id=unit_id,
        u_sync=u_sync,
        pause_req_evt=pause_req_evt,
        pause_generation=5,
    )

    assert u_sync["resume_event"].is_set()
    assert not pause_req_evt.is_set()
    assert unit_id not in scheduler.STATE.preemptible_units
