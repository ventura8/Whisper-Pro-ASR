"""
Concurrency and resource acquisition logic for Whisper Pro.
"""

import contextlib
import logging
import queue
import sys
import threading
import time
from typing import Optional

from modules.core import config, model_provisioning, utils
from modules.inference import scheduler
from modules.inference.pipeline import preprocessing
from modules.inference.scheduler import state_helpers as scheduler_state_helpers

logger = logging.getLogger(__name__)


PRIORITY_RETRY_DELAY_SEC = 0.05
STANDARD_RETRY_DELAY_SEC = 0.5


def _is_preprocessor_lock_available(unit_id):
    """Return True when a unit preprocessor lock is immediately available."""
    lock = _get_preprocessor_lock(unit_id)
    if lock is None:
        return True

    acquired = lock.acquire(blocking=False)
    if not acquired:
        return False

    lock.release()
    return True


def _get_preprocessor_lock(unit_id):
    model_manager = sys.modules.get("modules.inference.runtime.model_manager")
    if model_manager:
        pool = getattr(model_manager, "PREPROCESSOR_POOL", None)
        if isinstance(pool, dict):
            pm = pool.get(unit_id)
            if pm:
                return getattr(pm, "lock", None)
    return None


def _try_borrow_preemptible_unit():
    """Try to borrow a targeted preemptible unit for a priority task."""
    target_unit_id = getattr(utils.THREAD_CONTEXT, "target_unit_id", None)
    if target_unit_id:
        unit = _borrow_targeted_or_fallback_unit(target_unit_id)
        if unit:
            return unit
        return None

    # Fallback to borrowing any preemptible unit when no specific target is selected.
    return _borrow_any_preemptible_unit()


def _borrow_targeted_or_fallback_unit(target_unit_id: str):
    unit = _borrow_targeted_unit(target_unit_id)
    if unit:
        return unit
    return _borrow_fallback_unit(target_unit_id)


def _borrow_targeted_unit(target_unit_id: str):
    with scheduler.STATE.task_registry_lock:
        if target_unit_id in scheduler.STATE.preemptible_units and _is_preprocessor_lock_available(target_unit_id):
            unit = _find_unit_by_id(target_unit_id)
            if unit:
                scheduler.STATE.preemptible_units.remove(target_unit_id)
                logger.info("[Engine] Priority task borrowed targeted unit %s", unit["id"])
                return unit
    return None


def _borrow_fallback_unit(target_unit_id: str):
    with scheduler.STATE.task_registry_lock:
        for unit in config.HARDWARE_UNITS:
            candidate_id = unit.get("id")
            if candidate_id != target_unit_id and _is_candidate_preemptible(candidate_id):
                resolved = _find_unit_by_id(candidate_id)
                if resolved:
                    scheduler.STATE.preemptible_units.remove(candidate_id)
                    logger.info(
                        "[Engine] Priority task fallback-borrowed unit %s (target %s unavailable)",
                        candidate_id,
                        target_unit_id,
                    )
                    return resolved
    return None


def _find_unit_by_id(unit_id: str) -> Optional[dict]:
    for u in config.HARDWARE_UNITS:
        if u.get("id") == unit_id:
            return u
    return None


def _is_candidate_preemptible(candidate_id: str) -> bool:
    return candidate_id in scheduler.STATE.preemptible_units and _is_preprocessor_lock_available(candidate_id)


def _borrow_any_preemptible_unit():
    with scheduler.STATE.task_registry_lock:
        for unit in config.HARDWARE_UNITS:
            candidate_id = unit.get("id")
            if _is_candidate_preemptible(candidate_id):
                resolved = _find_unit_by_id(candidate_id)
                if resolved:
                    scheduler.STATE.preemptible_units.remove(candidate_id)
                    logger.info("[Engine] Priority task borrowed unit %s", candidate_id)
                    return resolved
    return None


def _try_take_idle_unit():
    """Try to take an idle unit using a non-blocking semaphore acquire."""
    try:
        idle_count = scheduler.STATE.hw_pool.qsize()
    except AttributeError:
        idle_count = 0

    if idle_count <= 0:
        return None

    @contextlib.contextmanager
    def _acquire_model_lock_nonblocking():
        acquired = scheduler.STATE.model_lock.acquire(blocking=False)
        try:
            yield acquired
        finally:
            # Intentionally do not release here; caller releases when unit is returned.
            pass

    with _acquire_model_lock_nonblocking() as acquired:
        if not acquired:
            return None
        try:
            return scheduler.STATE.hw_pool.get(block=False)
        except queue.Empty:
            scheduler.STATE.model_lock.release()
            return None


def _priority_acquire_unit():
    """Attempt to acquire a unit for a priority task.

    If this priority task has targeted a specific unit, prefer to borrow that
    unit when it becomes preemptible. If the targeted unit is not yet available,
    but an idle unit is free, use the idle unit instead to avoid deadlock.
    Otherwise, prefer idle hardware first before borrowing from the preemptible pool.

    Returns a tuple (unit, borrowed) where `unit` is the selected unit dict or None,
    and `borrowed` is True when the unit was borrowed from a preemptible pool.
    """
    unit = None
    borrowed = False
    target_unit_id = getattr(utils.THREAD_CONTEXT, "target_unit_id", None)

    if target_unit_id:
        unit = _try_borrow_preemptible_unit()
        if unit:
            borrowed = True
        else:
            # If the targeted unit is not yet preemptible, fall back to any idle hardware.
            unit = _try_take_idle_unit()
    else:
        unit = _try_take_idle_unit()
        if not unit:
            # Otherwise, try to borrow a preemptible/targeted unit
            unit = _try_borrow_preemptible_unit()
            borrowed = bool(unit)

    return unit, borrowed


def _has_priority_tasks_in_registry():
    """Return True when any priority task is currently registered."""
    with scheduler.STATE.task_registry_lock:
        return any(t.get("is_priority", False) for t in scheduler.STATE.task_registry.values())


def _mark_task_queued_once(queued_added, stage="Waiting for Hardware"):
    """Mark current task queued exactly once and return updated state."""
    if not queued_added:
        scheduler.update_task_metadata(status="queued")
        scheduler.update_task_progress(None, stage)
        scheduler.increment_queued_session()
        return True
    return queued_added


def _try_acquire_unit_now():
    """Try immediate semaphore+queue acquisition, returning unit or None."""

    @contextlib.contextmanager
    def _acquire_lock_nonblocking():
        acquired = scheduler.STATE.model_lock.acquire(blocking=False)
        try:
            yield acquired
        finally:
            # Intentionally not releasing here. The caller releases on successful handoff.
            pass

    with _acquire_lock_nonblocking() as acquired:
        if not acquired:
            return None
        try:
            return scheduler.STATE.hw_pool.get(block=False)
        except queue.Empty:
            scheduler.STATE.model_lock.release()
            return None


def _acquire_unit_for_task(is_priority):
    """Acquire a hardware unit for current task and return (unit, borrowed)."""
    unit = None
    borrowed = False
    queued_added = False
    task_id = getattr(utils.THREAD_CONTEXT, "task_id", None)

    with scheduler.STATE.cond:
        try:
            while unit is None:
                unit, borrowed, queued_added = _loop_step_acquire(task_id, is_priority, queued_added)
        finally:
            _finalize_queued_status(queued_added)
    return unit, borrowed


def _loop_step_acquire(task_id, is_priority, queued_added) -> tuple[Optional[dict], bool, bool]:
    if model_provisioning.should_gate_tasks():
        return None, False, _wait_for_model_download(queued_added)

    if _is_task_waiting_for_earlier_fifo(task_id, is_priority):
        return None, False, _wait_in_scheduler_queue(queued_added)

    unit, borrowed = _try_allocating_unit(is_priority)
    if unit is None:
        unit = _try_non_priority_acquire(is_priority)

    if unit is not None:
        return unit, borrowed, queued_added

    return None, False, _wait_in_scheduler_queue(queued_added)


def _is_task_waiting_for_earlier_fifo(task_id, is_priority) -> bool:
    return bool(task_id and scheduler.has_earlier_task(task_id, is_priority=is_priority))


def _try_non_priority_acquire(is_priority) -> Optional[dict]:
    if not _should_yield_to_priority_tasks(is_priority):
        return _try_acquire_unit_now()
    return None


def _finalize_queued_status(queued_added: bool):
    if queued_added:
        scheduler.decrement_queued_session()
        scheduler.STATE.cond.notify_all()


def _wait_for_model_download(queued_added: bool) -> bool:
    """Hold the task in the queue while startup provisioning downloads the models.

    Waiting here rather than inside init_unit keeps the task out of MODEL_POOL and the
    hardware pool, so it neither holds a unit nor reports "active" while it waits.
    """
    percent = model_provisioning.get_progress().get("percent", 0)
    stage = f"Downloading Model ({percent}%)"
    res = _mark_task_queued_once(queued_added, stage)
    # Refresh every iteration so the dashboard tracks download progress. Stage-only
    # updates pass progress=None because task progress is monotonic.
    scheduler.update_task_progress(None, stage)
    scheduler.STATE.cond.notify_all()
    scheduler.STATE.cond.wait(timeout=0.5)
    return res


def _wait_in_scheduler_queue(queued_added: bool) -> bool:
    res = _mark_task_queued_once(queued_added)
    scheduler.STATE.cond.notify_all()
    scheduler.STATE.cond.wait(timeout=0.1)
    time.sleep(0.001)
    return res


def _try_allocating_unit(is_priority: bool) -> tuple[Optional[dict], bool]:
    if is_priority:
        return _priority_acquire_unit()
    return None, False


def _should_yield_to_priority_tasks(is_priority: bool) -> bool:
    if is_priority:
        return False
    return scheduler.STATE.priority_requests > 0 and scheduler_state_helpers.has_queued_priority_tasks(scheduler.STATE)


@contextlib.contextmanager
def model_lock_ctx(priority=None):
    """Hardware resource acquisition context with priority borrowing support."""
    is_priority = priority if priority is not None else getattr(utils.THREAD_CONTEXT, "is_priority", False)
    unit, borrowed = _acquire_unit_for_task(is_priority)

    if is_priority:
        _handle_priority_unit_setup(unit)

    try:
        model = _initialize_unit_model_and_preprocessor(unit)
        yield model, unit["id"]
    finally:
        _cleanup_unit_after_task(unit, borrowed)


def _handle_priority_unit_setup(unit: dict):
    target_unit_id = getattr(utils.THREAD_CONTEXT, "target_unit_id", None)
    if target_unit_id and target_unit_id != unit["id"]:
        logger.info(
            "[Engine] Priority task fallback-borrowed unit %s (target %s unavailable). Releasing original target preemption hold.",
            unit["id"],
            target_unit_id,
        )
        scheduler.release_unit_preemption_hold(target_unit_id)
        target_unit_id = None

    if not target_unit_id:
        utils.THREAD_CONTEXT.target_unit_id = unit["id"]
        with scheduler.STATE.priority_lock:
            if not hasattr(scheduler.STATE, "unit_priority_requests"):
                scheduler.STATE.unit_priority_requests = {}
            scheduler.STATE.unit_priority_requests[unit["id"]] = scheduler.STATE.unit_priority_requests.get(unit["id"], 0) + 1


def _initialize_unit_model_and_preprocessor(unit: dict):
    scheduler.update_task_metadata(status="active", start_active=time.time(), unit_id=unit["id"])
    model_manager = sys.modules["modules.inference.runtime.model_manager"]

    model_pool = model_manager.MODEL_POOL
    preprocessor_pool = model_manager.PREPROCESSOR_POOL
    init_unit = model_manager.init_unit

    if unit["id"] not in model_pool:
        init_unit(unit)

    if unit["id"] not in preprocessor_pool:
        preprocessor_pool[unit["id"]] = preprocessing.create_manager(unit)

    model = model_pool.get(unit["id"])
    if model is None:
        # Prefer the real load failure; the empty pool is only its symptom.
        reason = getattr(model_manager, "LAST_INIT_ERROR", {}).get(unit["id"])
        raise RuntimeError(reason or f"Engine pool for {unit['id']} is empty after initialization.")

    scheduler.update_task_metadata(
        unit_id=unit["id"],
        unit_type=unit["type"],
        unit_name=unit["name"],
        history_unit_id=unit["id"],
        history_unit_type=unit["type"],
        history_unit_name=unit["name"],
        status="active",
    )
    return model


def _cleanup_unit_after_task(unit: dict, borrowed: bool):
    try:
        scheduler.update_task_metadata(unit_id=None, status="post-processing")
    except (KeyError, RuntimeError, ValueError, TypeError, AttributeError):
        logger.exception("[Engine] Failed to update post-processing metadata during cleanup.")

    if borrowed:
        # Return unit to preemptible pool so the original task can take it back
        scheduler.mark_unit_preemptible(unit["id"])
        logger.info("[Engine] Priority task finished with borrowed unit %s", unit["id"])
    else:
        scheduler.STATE.hw_pool.put(unit)
        scheduler.STATE.model_lock.release()


def _get_current_task_info():
    """Retrieve current task metadata from registry (unit_id, status, priority flag)."""
    task_id = getattr(utils.THREAD_CONTEXT, "task_id", None)
    thread_id = threading.get_ident()
    unit_id, old_status, is_priority, task = None, "active", False, None

    with scheduler.STATE.task_registry_lock:
        if task_id and task_id in scheduler.STATE.task_registry:
            task = scheduler.STATE.task_registry[task_id]
        elif thread_id in scheduler.STATE.task_registry:
            task = scheduler.STATE.task_registry[thread_id]
        if task:
            unit_id = task.get("unit_id")
            old_status = task.get("status", "active")
            is_priority = task.get("is_priority", False)

    return task_id, thread_id, unit_id, old_status, is_priority, task


def _determine_preemption_needed(unit_id):
    """Check if preemption is needed; return (should_preempt, u_sync, pause_req_evt, pause_generation)."""
    u_sync = scheduler.STATE.unit_sync.get(unit_id) if unit_id else None
    pause_req_evt = None
    pause_generation = None

    if u_sync:
        pause_req_evt = u_sync.get("pause_requested")
        pause_generation = u_sync.get("pause_generation")
        if pause_req_evt and pause_req_evt.is_set():
            return True, u_sync, pause_req_evt, pause_generation

    return False, u_sync, pause_req_evt, pause_generation


def _handle_preemption_pause_resume(unit_id, u_sync, pause_req_evt, pause_generation=None):
    """Handle preemption: pause confirmation, resume wait, and unit reclaim."""
    with scheduler.STATE.cond:
        _set_pause_confirmed(u_sync, pause_generation)
        try:
            if _is_resume_event_active(u_sync, pause_req_evt):
                _wait_for_resume_signal(unit_id, u_sync, pause_req_evt)
                _reclaim_preempted_unit(unit_id)
        finally:
            _clear_pause_confirmed(u_sync)


def _is_resume_event_active(u_sync, pause_req_evt) -> bool:
    if u_sync and pause_req_evt and pause_req_evt.is_set():
        return u_sync.get("resume_event") is not None
    return False


def _reclaim_preempted_unit(unit_id: str):
    if unit_id in scheduler.STATE.preemptible_units:
        scheduler.STATE.preemptible_units.remove(unit_id)


def _set_pause_confirmed(u_sync, pause_generation):
    if u_sync:
        pause_conf_evt = u_sync.get("pause_confirmed")
        if pause_conf_evt:
            u_sync["confirmed_generation"] = pause_generation
            pause_conf_evt.set()

    scheduler.STATE.confirmed_generation = pause_generation
    scheduler.STATE.pause_confirmed.set()
    scheduler.STATE.cond.notify_all()


def _clear_pause_confirmed(u_sync):
    if u_sync:
        pause_conf_evt = u_sync.get("pause_confirmed")
        if pause_conf_evt:
            pause_conf_evt.clear()
            u_sync["confirmed_generation"] = None
    scheduler.STATE.confirmed_generation = None
    scheduler.STATE.pause_confirmed.clear()
    scheduler.STATE.cond.notify_all()


def _wait_for_resume_signal(unit_id, u_sync, pause_req_evt):
    while not _can_resume_preempted_unit(unit_id, u_sync, pause_req_evt):
        scheduler.STATE.cond.wait(timeout=0.1)
        time.sleep(0.001)


def _can_resume_preempted_unit(unit_id, u_sync, pause_req_evt) -> bool:
    if not _has_queued_or_active_priority_tasks():
        _self_heal_stale_pause_state(u_sync, pause_req_evt)
        return True

    return _is_preempted_unit_ready(unit_id, u_sync)


def _has_queued_or_active_priority_tasks() -> bool:
    with scheduler.STATE.task_registry_lock:
        for task in scheduler.STATE.task_registry.values():
            if task.get("is_priority", False) and task.get("status") in {"initializing", "queued", "active"}:
                return True
    return False


def _self_heal_stale_pause_state(u_sync, pause_req_evt):
    if pause_req_evt is not None:
        pause_req_evt.clear()
    if u_sync:
        unit_resume_evt = u_sync.get("resume_event")
        if unit_resume_evt is not None:
            unit_resume_evt.set()
        u_sync["pause_requested"].clear()


def _is_preempted_unit_ready(unit_id, u_sync) -> bool:
    is_paused_cleared = u_sync and not u_sync["pause_requested"].is_set()
    is_unit_preemptible = unit_id in scheduler.STATE.preemptible_units
    return is_paused_cleared and is_unit_preemptible


def _restore_task_state(task_id, thread_id, old_status, old_stage, unit_id):
    """Restore task to original status and stage after preemption completes."""
    current_progress, current_stage = _get_current_task_progress_and_stage(task_id, thread_id)

    # A task that successfully reclaimed a preempted unit is actively running again.
    # Force status to active for stale queued snapshots captured during pause windows.
    restored_status = old_status if old_status in {"active", "post-processing", "completed", "failed"} else "active"
    restored_stage = _resolve_restored_stage(current_stage, old_stage)

    scheduler.update_task_metadata(status=restored_status)
    scheduler.update_task_progress(current_progress, restored_stage)
    logger.info(
        "[Engine] Resumed task on %s (status=%s, stage=%s, progress=%d%%)",
        unit_id,
        restored_status,
        restored_stage,
        current_progress or 0,
    )


def _get_current_task_progress_and_stage(task_id, thread_id) -> tuple[int, Optional[str]]:
    with scheduler.STATE.task_registry_lock:
        key = _find_active_registry_key(task_id, thread_id)
        if key:
            task = scheduler.STATE.task_registry[key]
            return task.get("progress", 0), task.get("stage")
    return 0, None


def _find_active_registry_key(task_id, thread_id) -> Optional[str]:
    if task_id and task_id in scheduler.STATE.task_registry:
        return task_id
    if thread_id in scheduler.STATE.task_registry:
        return thread_id
    return None


def _resolve_restored_stage(current_stage: Optional[str], old_stage: Optional[str]) -> Optional[str]:
    if current_stage and current_stage != "Paused for Priority Task":
        return current_stage
    if old_stage and old_stage != "Paused for Priority Task":
        return old_stage
    return None


def _check_preemption():
    """Yields execution if a priority task is waiting."""
    task_id, thread_id, unit_id, old_status, is_priority, task = _get_current_task_info()

    if is_priority:
        return

    should_preempt, u_sync, pause_req_evt, pause_generation = _determine_preemption_needed(unit_id)

    if should_preempt and unit_id:
        _execute_preemption_flow(
            task_id,
            thread_id,
            unit_id,
            old_status,
            task=task,
            u_sync=u_sync,
            pause_req_evt=pause_req_evt,
            pause_generation=pause_generation,
        )


def _execute_preemption_flow(
    task_id,
    thread_id,
    unit_id,
    old_status,
    *,
    task,
    u_sync,
    pause_req_evt,
    pause_generation,
):
    old_stage = task.get("stage") if task else None
    logger.info("[Engine] Preempting task on %s... (old_status=%s, old_stage=%s)", unit_id, old_status, old_stage)

    # Temporarily mark task as queued during preemption/pause
    scheduler.update_task_metadata(status="queued")
    progress = task.get("progress") if task else 0
    scheduler.update_task_progress(progress, "Paused for Priority Task")
    logger.debug("[Engine] Task marked as paused (status=queued, stage=Paused for Priority Task)")

    scheduler.mark_unit_preemptible(unit_id)

    _handle_preemption_pause_resume(unit_id, u_sync, pause_req_evt, pause_generation=pause_generation)

    _restore_task_state(task_id, thread_id, old_status, old_stage, unit_id)
