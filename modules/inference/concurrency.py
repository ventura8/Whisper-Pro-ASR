"""
Concurrency and resource acquisition logic for Whisper Pro.
"""
import contextlib
import sys
import logging
import threading
import time
from modules import config, utils
from modules.inference import scheduler, preprocessing

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def model_lock_ctx(priority=None):
    """Hardware resource acquisition context with priority borrowing support."""
    is_priority = priority if priority is not None else getattr(
        utils.THREAD_CONTEXT, "is_priority", False)
    unit = None
    borrowed = False

    # 1. Acquisition logic
    queued_added = False
    try:
        while not unit:
            if is_priority:
                borrowed_unit_id = scheduler.get_preemptible_unit()
                if borrowed_unit_id:
                    unit = next(
                        (u for u in config.HARDWARE_UNITS if u['id'] == borrowed_unit_id), None)
                    if unit:
                        logger.info("[Engine] Priority task borrowed unit %s", unit['id'])
                        borrowed = True
                        break

            # If there are any priority tasks queued or running, standard tasks must yield
            has_priority = False
            with scheduler.STATE.task_registry_lock:
                has_priority = any(t.get('is_priority', False) for t in scheduler.STATE.task_registry.values())

            if not is_priority and has_priority:
                if not queued_added:
                    scheduler.update_task_metadata(status="queued")
                    scheduler.update_task_progress(None, "Waiting for Hardware")
                    scheduler.increment_queued_session()
                    queued_added = True
                time.sleep(0.5)
                continue

            if scheduler.STATE.model_lock.acquire(blocking=False):
                unit = scheduler.STATE.hw_pool.get()
                break

            if not queued_added:
                scheduler.update_task_metadata(status="queued")
                scheduler.update_task_progress(None, "Waiting for Hardware")
                scheduler.increment_queued_session()
                queued_added = True

            if not is_priority:
                # Standard task: Wait on semaphore with a short timeout to check for new priority tasks
                if scheduler.STATE.model_lock.acquire(timeout=0.5):
                    unit = scheduler.STATE.hw_pool.get()
                    break
            else:
                # Priority task: Loop until a unit is preempted or a slot opens
                time.sleep(0.5)
    finally:
        if queued_added:
            scheduler.decrement_queued_session()

    scheduler.update_task_metadata(status="active", start_active=time.time(), unit_id=unit['id'])
    try:
        model_manager = sys.modules['modules.inference.model_manager']
        # pylint: disable=protected-access
        model_pool = model_manager._MODEL_POOL
        preprocessor_pool = model_manager._PREPROCESSOR_POOL
        init_unit = model_manager._init_unit

        if unit['id'] not in model_pool:
            init_unit(unit)

        if unit['id'] not in preprocessor_pool:
            preprocessor_pool[unit['id']] = preprocessing.PreprocessingManager(unit)

        model = model_pool.get(unit['id'])
        if model is None:
            raise RuntimeError(f"Engine pool for {unit['id']} is empty after initialization.")

        scheduler.update_task_metadata(
            unit_id=unit['id'],
            unit_type=unit['type'],
            unit_name=unit['name'],
            status="active"
        )
        yield model, unit['id']
    finally:
        if borrowed:
            # Return unit to preemptible pool so the original task can take it back
            scheduler.mark_unit_preemptible(unit['id'])
            logger.info("[Engine] Priority task finished with borrowed unit %s", unit['id'])
        else:
            scheduler.STATE.hw_pool.put(unit)
            scheduler.STATE.model_lock.release()


def _check_preemption():
    """Yields execution if a priority task is waiting."""
    if scheduler.STATE.pause_requested.is_set():
        thread_id = threading.get_ident()
        unit_id = None
        old_status = "active"
        is_priority = False
        with scheduler.STATE.task_registry_lock:
            task = scheduler.STATE.task_registry.get(thread_id)
            if task:
                unit_id = task.get('unit_id')
                old_status = task.get('status', 'active')
                is_priority = task.get('is_priority', False)

        if is_priority:
            # Priority tasks are never preempted
            return

        if unit_id:
            logger.info("[Engine] Preempting task on %s...", unit_id)
            old_stage = task.get('stage')

            # Temporarily mark task as queued during preemption/pause
            scheduler.update_task_metadata(status="queued")
            scheduler.update_task_progress(task.get('progress'), "Paused for Priority Task")

            scheduler.mark_unit_preemptible(unit_id)
            scheduler.STATE.pause_confirmed.set()
            scheduler.STATE.resume_event.wait()

            # Wait until our unit is no longer "borrowed"
            while True:
                with scheduler.STATE.task_registry_lock:
                    if unit_id in scheduler.STATE.preemptible_units:
                        # It's back in the pool, we can take it
                        scheduler.STATE.preemptible_units.remove(unit_id)
                        break
                time.sleep(0.5)

            scheduler.STATE.pause_confirmed.clear()
            scheduler.update_task_metadata(status=old_status)
            scheduler.update_task_progress(task.get('progress'), old_stage)
            logger.info("[Engine] Resuming task on %s", unit_id)
