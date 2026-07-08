"""
Concurrency and resource acquisition logic for Whisper Pro.
"""

import contextlib
import logging
import queue
import sys
import threading
import time

from modules.core import config, utils
from modules.inference import preprocessing, scheduler, scheduler_state_helpers

logger = logging.getLogger(__name__)


PRIORITY_RETRY_DELAY_SEC = 0.05
STANDARD_RETRY_DELAY_SEC = 0.5


def _is_preprocessor_lock_available(unit_id):
    """Return True when a unit preprocessor lock is immediately available."""
    model_manager = sys.modules.get("modules.inference.model_manager")
    if model_manager is None:
        return True

    preprocessor_pool = getattr(model_manager, "PREPROCESSOR_POOL", None)
    if not isinstance(preprocessor_pool, dict):
        return True

    preprocessor = preprocessor_pool.get(unit_id)
    if preprocessor is None:
        return True

    lock = getattr(preprocessor, "lock", None)
    if lock is None:
        return True

    acquired = lock.acquire(blocking=False)
    if not acquired:
        return False

    lock.release()
    return True


def _try_borrow_preemptible_unit():
    """Try to borrow a targeted preemptible unit for a priority task."""
    target_unit_id = getattr(utils.THREAD_CONTEXT, "target_unit_id", None)
    if target_unit_id:
        with scheduler.STATE.task_registry_lock:
            if target_unit_id in scheduler.STATE.preemptible_units and _is_preprocessor_lock_available(target_unit_id):
                unit = next((u for u in config.HARDWARE_UNITS if u["id"] == target_unit_id), None)
                if unit:
                    scheduler.STATE.preemptible_units.remove(target_unit_id)
                    logger.info("[Engine] Priority task borrowed targeted unit %s", unit["id"])
                    return unit
        # If targeted unit is not yet preemptible, use another already-preemptible unit.
        # This prevents priority queue stalls when one paused unit is busy finishing work,
        # while another paused unit is already available to run detect-language tasks.
        with scheduler.STATE.task_registry_lock:
            for unit in config.HARDWARE_UNITS:
                candidate_id = unit.get("id")
                if candidate_id == target_unit_id:
                    continue
                if candidate_id in scheduler.STATE.preemptible_units and _is_preprocessor_lock_available(candidate_id):
                    resolved = next((u for u in config.HARDWARE_UNITS if u.get("id") == candidate_id), None)
                    if resolved:
                        scheduler.STATE.preemptible_units.remove(candidate_id)
                        logger.info(
                            "[Engine] Priority task fallback-borrowed unit %s (target %s unavailable)",
                            candidate_id,
                            target_unit_id,
                        )
                        return resolved
        return None

    # Fallback to borrowing any preemptible unit when no specific target is selected.
    with scheduler.STATE.task_registry_lock:
        for unit in config.HARDWARE_UNITS:
            candidate_id = unit.get("id")
            if candidate_id in scheduler.STATE.preemptible_units and _is_preprocessor_lock_available(candidate_id):
                resolved = next((u for u in config.HARDWARE_UNITS if u.get("id") == candidate_id), None)
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


def _mark_task_queued_once(queued_added):
    """Mark current task queued exactly once and return updated state."""
    if not queued_added:
        scheduler.update_task_metadata(status="queued")
        scheduler.update_task_progress(None, "Waiting for Hardware")
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
    """Acquire a hardware unit for current task and return (unit, borrowed).

    Respects FIFO ordering within priority levels: tasks of same priority are processed
    in arrival order. However, priority (detect-language) tasks can preempt standard (ASR)
    tasks even if standard tasks arrived earlier.
    """
    unit = None
    borrowed = False
    queued_added = False
    task_id = getattr(utils.THREAD_CONTEXT, "task_id", None)

    with scheduler.STATE.cond:
        try:
            while unit is None:
                # FIFO check: if earlier tasks of SAME priority are waiting, yield to them
                # Note: priority tasks are NOT blocked by earlier standard tasks (priority preemption)
                if task_id and scheduler.has_earlier_task(task_id, is_priority=is_priority):
                    queued_added = _mark_task_queued_once(queued_added)
                    scheduler.STATE.cond.notify_all()
                    scheduler.STATE.cond.wait(timeout=0.1)
                    time.sleep(0.001)
                    continue

                if is_priority:
                    unit, borrowed = _priority_acquire_unit()
                    if unit is not None:
                        break
                else:
                    if scheduler.STATE.priority_requests > 0 and scheduler_state_helpers.has_queued_priority_tasks(
                        scheduler.STATE
                    ):
                        queued_added = _mark_task_queued_once(queued_added)
                        scheduler.STATE.cond.notify_all()
                        scheduler.STATE.cond.wait(timeout=0.1)
                        time.sleep(0.001)
                        continue

                unit = _try_acquire_unit_now()
                if unit is not None:
                    break

                queued_added = _mark_task_queued_once(queued_added)
                scheduler.STATE.cond.notify_all()
                scheduler.STATE.cond.wait(timeout=0.1)
                time.sleep(0.001)
        finally:
            if queued_added:
                scheduler.decrement_queued_session()
                scheduler.STATE.cond.notify_all()
    return unit, borrowed


@contextlib.contextmanager
def model_lock_ctx(priority=None):
    """Hardware resource acquisition context with priority borrowing support."""
    is_priority = priority if priority is not None else getattr(utils.THREAD_CONTEXT, "is_priority", False)
    unit, borrowed = _acquire_unit_for_task(is_priority)

    if is_priority:
        target_unit_id = getattr(utils.THREAD_CONTEXT, "target_unit_id", None)
        if target_unit_id and target_unit_id != unit["id"]:
            logger.info(
                "[Engine] Priority task fallback-borrowed unit %s (target %s unavailable). "
                "Releasing original target preemption hold.",
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
                scheduler.STATE.unit_priority_requests[unit["id"]] = (
                    scheduler.STATE.unit_priority_requests.get(unit["id"], 0) + 1
                )

    try:
        scheduler.update_task_metadata(status="active", start_active=time.time(), unit_id=unit["id"])
        model_manager = sys.modules["modules.inference.model_manager"]

        model_pool = model_manager.MODEL_POOL
        preprocessor_pool = model_manager.PREPROCESSOR_POOL
        init_unit = model_manager.init_unit

        if unit["id"] not in model_pool:
            init_unit(unit)

        if unit["id"] not in preprocessor_pool:
            preprocessor_pool[unit["id"]] = preprocessing.PreprocessingManager(unit)

        model = model_pool.get(unit["id"])
        if model is None:
            raise RuntimeError(f"Engine pool for {unit['id']} is empty after initialization.")

        scheduler.update_task_metadata(
            unit_id=unit["id"], unit_type=unit["type"], unit_name=unit["name"], status="active"
        )
        yield model, unit["id"]
    finally:
        try:
            scheduler.update_task_metadata(unit_id=None, status="post-processing")
        except (KeyError, RuntimeError, ValueError, TypeError, AttributeError):  # pragma: no cover
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
    unit_id = None
    old_status = "active"
    is_priority = False
    task = None

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
        # Set pause confirmed for the target unit.
        if u_sync:
            pause_conf_evt = u_sync.get("pause_confirmed")
            if pause_conf_evt:
                u_sync["confirmed_generation"] = pause_generation
                pause_conf_evt.set()

        scheduler.STATE.confirmed_generation = pause_generation
        scheduler.STATE.pause_confirmed.set()

        # Notify priority task waiting for pause confirmation
        scheduler.STATE.cond.notify_all()

        # Preemption/resume gating is unit-scoped only.
        resume_evt = u_sync.get("resume_event") if (u_sync and pause_req_evt and pause_req_evt.is_set()) else None
        if resume_evt is None:
            return

        # Keep waiting while priority backlog exists; this enforces strict priority queue draining
        # before paused ASR work is resumed.
        def can_resume():
            has_priority_work = any(
                task.get("is_priority", False) and task.get("status") in {"initializing", "queued", "active"}
                for task in scheduler.STATE.task_registry.values()
            )
            if not has_priority_work:
                # Self-heal stale pause state so tasks cannot remain stuck waiting for resume.
                if pause_req_evt is not None:
                    pause_req_evt.clear()
                if u_sync:
                    unit_resume_evt = u_sync.get("resume_event")
                    if unit_resume_evt is not None:
                        unit_resume_evt.set()
                    u_sync["pause_requested"].clear()
                return True

            is_paused_cleared = u_sync and not u_sync["pause_requested"].is_set()
            is_unit_preemptible = unit_id in scheduler.STATE.preemptible_units
            return is_paused_cleared and is_unit_preemptible

        while not can_resume():
            scheduler.STATE.cond.wait(timeout=0.1)
            time.sleep(0.001)

        # Wait for unit reclaim (already confirmed preemptible, remove from preemptible pool)
        if unit_id in scheduler.STATE.preemptible_units:
            scheduler.STATE.preemptible_units.remove(unit_id)

        # Clear pause confirmed
        if u_sync:
            pause_conf_evt = u_sync.get("pause_confirmed")
            if pause_conf_evt:
                pause_conf_evt.clear()
                u_sync["confirmed_generation"] = None
        scheduler.STATE.confirmed_generation = None
        scheduler.STATE.pause_confirmed.clear()

        scheduler.STATE.cond.notify_all()


def _restore_task_state(task_id, thread_id, old_status, old_stage, unit_id):
    """Restore task to original status and stage after preemption completes."""
    current_progress = 0
    current_stage = None
    with scheduler.STATE.task_registry_lock:
        if task_id and task_id in scheduler.STATE.task_registry:
            current_progress = scheduler.STATE.task_registry[task_id].get("progress", 0)
            current_stage = scheduler.STATE.task_registry[task_id].get("stage")
        elif thread_id in scheduler.STATE.task_registry:
            current_progress = scheduler.STATE.task_registry[thread_id].get("progress", 0)
            current_stage = scheduler.STATE.task_registry[thread_id].get("stage")

    # A task that successfully reclaimed a preempted unit is actively running again.
    # Force status to active for stale queued snapshots captured during pause windows.
    restored_status = old_status if old_status in {"active", "post-processing", "completed", "failed"} else "active"
    if current_stage and current_stage != "Paused for Priority Task":
        restored_stage = current_stage
    elif old_stage and old_stage != "Paused for Priority Task":
        restored_stage = old_stage
    else:
        restored_stage = None

    scheduler.update_task_metadata(status=restored_status)
    scheduler.update_task_progress(current_progress, restored_stage)
    logger.info(
        "[Engine] Resumed task on %s (status=%s, stage=%s, progress=%d%%)",
        unit_id,
        restored_status,
        restored_stage,
        current_progress or 0,
    )


def _check_preemption():
    """Yields execution if a priority task is waiting."""
    task_id, thread_id, unit_id, old_status, is_priority, task = _get_current_task_info()

    if is_priority:
        # Priority tasks are never preempted
        return

    should_preempt, u_sync, pause_req_evt, pause_generation = _determine_preemption_needed(unit_id)

    if should_preempt and unit_id:
        old_stage = task.get("stage") if task else None
        logger.info("[Engine] Preempting task on %s... (old_status=%s, old_stage=%s)", unit_id, old_status, old_stage)

        # Temporarily mark task as queued during preemption/pause
        scheduler.update_task_metadata(status="queued")
        scheduler.update_task_progress(task.get("progress") if task else 0, "Paused for Priority Task")
        logger.debug("[Engine] Task marked as paused (status=queued, stage=Paused for Priority Task)")

        scheduler.mark_unit_preemptible(unit_id)

        _handle_preemption_pause_resume(unit_id, u_sync, pause_req_evt, pause_generation=pause_generation)

        _restore_task_state(task_id, thread_id, old_status, old_stage, unit_id)
