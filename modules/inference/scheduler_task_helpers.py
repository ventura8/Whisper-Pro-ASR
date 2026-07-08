"""Task-registry mutation helpers used by the scheduler module."""

import threading

from modules.core import logging_setup, utils


def cleanup_failed_task(state):
    """Remove task/log/order entries for the current thread context on failure."""
    task_id = getattr(utils.THREAD_CONTEXT, "task_id", None)
    thread_id = getattr(utils.THREAD_CONTEXT, "registration_thread_id", None) or threading.get_ident()

    with state.task_registry_lock:
        if task_id and task_id in state.task_registry:
            del state.task_registry[task_id]
        elif thread_id in state.task_registry:
            del state.task_registry[thread_id]

        with logging_setup.TASK_LOGS_LOCK:
            if task_id and task_id in logging_setup.TASK_LOGS:
                del logging_setup.TASK_LOGS[task_id]
            if thread_id in logging_setup.TASK_LOGS:
                del logging_setup.TASK_LOGS[thread_id]

    with state.task_order_lock:
        if task_id and task_id in state.task_arrival_order:
            del state.task_arrival_order[task_id]
        if thread_id in state.task_arrival_order:
            del state.task_arrival_order[thread_id]


def update_task_metadata(state, logger, **kwargs):
    """Update metadata for the current task; create a minimal fallback entry if missing."""
    task_id = getattr(utils.THREAD_CONTEXT, "task_id", None)
    thread_id = getattr(utils.THREAD_CONTEXT, "registration_thread_id", None) or threading.get_ident()

    with state.task_registry_lock:
        target_key = None
        if task_id and task_id in state.task_registry:
            target_key = task_id
        elif thread_id in state.task_registry:
            target_key = thread_id

        if target_key:
            state.task_registry[target_key].update(kwargs)
            if "live_text" in kwargs:
                logger.debug(
                    "[Scheduler] Updated live_text for task %s", state.task_registry[target_key].get("task_id")
                )
            return

        logger.warning(
            "[Scheduler] Missing task registry entry for task_id=%s, thread_id=%s. Skipping metadata update: %s",
            task_id,
            thread_id,
            kwargs,
        )


def update_task_progress(state, progress, stage=None):
    """Update progress and optional stage for the current task."""
    task_id = getattr(utils.THREAD_CONTEXT, "task_id", None)
    thread_id = getattr(utils.THREAD_CONTEXT, "registration_thread_id", None) or threading.get_ident()

    with state.task_registry_lock:
        target_key = None
        if task_id and task_id in state.task_registry:
            target_key = task_id
        elif thread_id in state.task_registry:
            target_key = thread_id

        if target_key:
            current_progress = state.task_registry[target_key].get("progress")
            should_update_progress = progress is not None
            should_update_stage = bool(stage)

            if progress is not None and current_progress is not None:
                try:
                    if progress < current_progress:
                        should_update_progress = False
                        # Do NOT suppress stage: stage strings are always valid
                        # even when the numeric progress appears to regress (e.g.
                        # the first inference segment arrives while the UVR-set
                        # progress counter is still higher than the segment pct).
                except TypeError:
                    # Fall back to direct assignment when one side is non-numeric.
                    pass

            if should_update_progress:
                state.task_registry[target_key]["progress"] = progress
            if should_update_stage:
                state.task_registry[target_key]["stage"] = stage
