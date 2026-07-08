"""Read-only helper functions for scheduler state queries."""

import time

from modules.inference import scheduler_ordering


def has_earlier_task(state, current_task_id, is_priority=None):
    """Check if there are earlier same-priority tasks still waiting for hardware."""
    return scheduler_ordering.has_earlier_task(state, current_task_id, is_priority=is_priority)


def has_queued_priority_tasks(state, exclude_task_id=None):
    """Return True if any detect-language task is currently queued."""
    return scheduler_ordering.has_queued_priority_tasks(state, exclude_task_id=exclude_task_id)


def get_queued_priority_count(state, exclude_task_id=None):
    """Return count of queued detect-language tasks."""
    return scheduler_ordering.get_queued_priority_count(state, exclude_task_id=exclude_task_id)


def get_service_stats_minimal(state):
    """Lightweight status check for circular-safe metrics discovery."""
    with state.task_registry_lock:
        active = []
        for task in state.task_registry.values():
            if task.get("status") == "active":
                active.append(
                    {
                        "unit_type": task.get("unit_type"),
                        "unit_name": task.get("unit_name", ""),
                        "unit_id": task.get("unit_id"),
                        "stage": task.get("stage", ""),
                    }
                )
        return {"active_tasks": active}


def wait_for_standard_task_to_activate(state, task_id, thread_id, timeout=0.5):
    """Wait briefly for an initializing standard task to become active."""
    end_wait = time.time() + timeout
    while time.time() < end_wait:
        with state.task_registry_lock:
            for tid, task in state.task_registry.items():
                is_current = (tid == task_id) or (task.get("task_id") == task_id) or (tid == thread_id)
                if not is_current and task.get("status") == "active" and not task.get("is_priority", False):
                    return True
        time.sleep(0.01)
    return False


def has_preferred_idle_unit(state, hardware_units, target_unit_id):
    """Return True if an idle unit is preferred over the selected target unit."""
    unit_order = {unit["id"]: idx for idx, unit in enumerate(hardware_units)}
    target_rank = unit_order.get(target_unit_id, -1)

    try:
        with state.hw_pool.mutex:
            idle_units = [u.get("id") for u in list(state.hw_pool.queue) if isinstance(u, dict)]
    except AttributeError:
        idle_units = []

    if not idle_units:
        return False

    best_idle_rank = max((unit_order.get(uid, -1) for uid in idle_units), default=-1)
    return best_idle_rank > target_rank


def should_skip_pause_confirmation(state, target_unit_id):
    """Return True when waiting for pause confirmation is no longer necessary.

    Skips the wait when:
    - The targeted unit has no active standard task running on it (it already yielded).
    - A unit is already in the preemptible pool and ready to be borrowed by a priority task.
    - The targeted unit's task is in vocal separation: a long phase with coarse yield
      checkpoints between UVR chunks. The priority task proceeds to do preparatory work
      (montage extraction) while the standard task reaches its next yield point. The
      actual unit acquisition in _acquire_unit_for_task handles the remainder of the wait.
    """
    with state.task_registry_lock:
        if target_unit_id:
            active_standard_tasks = [
                task
                for task in state.task_registry.values()
                if task.get("status") == "active" and not task.get("is_priority", False)
            ]
            if not active_standard_tasks:
                return True
            target_has_active_standard = any(task.get("unit_id") == target_unit_id for task in active_standard_tasks)
            has_unknown_active_standard = any(task.get("unit_id") in (None, "") for task in active_standard_tasks)
            if not target_has_active_standard and not has_unknown_active_standard:
                return True
            # Vocal separation processes audio in chunks of several minutes. The yield_cb
            # (preemption checkpoint) only fires between chunks, so confirmation can take
            # up to ~70 seconds. Skip the blocking wait here and let _acquire_unit_for_task
            # spin at 50ms intervals instead — it unblocks the instant the chunk finishes.
            if target_has_active_standard:
                target_in_vocal_separation = any(
                    "Vocal Separation" in (task.get("stage") or "")
                    for task in active_standard_tasks
                    if task.get("unit_id") == target_unit_id
                )
                if target_in_vocal_separation:
                    return True
        else:
            still_active_standard = any(
                task.get("status") == "active" and not task.get("is_priority", False)
                for task in state.task_registry.values()
            )
            if not still_active_standard:
                return True
        if state.preemptible_units:
            return True
    return False


def is_engine_initialized(state):
    """Return whether any models are loaded."""
    return state.engine_initialized


def is_uvr_loaded(state):
    """Return whether UVR is loaded."""
    return state.uvr_loaded
