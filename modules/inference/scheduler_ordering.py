"""Ordering and priority backlog helpers for scheduler state."""


def has_queued_priority_tasks(state, exclude_task_id=None):
    """Return True if any detect-language task is currently queued."""
    return get_queued_priority_count(state, exclude_task_id=exclude_task_id) > 0


def get_queued_priority_count(state, exclude_task_id=None):
    """Return count of queued non-coalesced priority tasks."""
    with state.task_registry_lock:
        queued_count = 0
        for task_key, task in state.task_registry.items():
            if exclude_task_id and (task_key == exclude_task_id or task.get("task_id") == exclude_task_id):
                continue
            if task.get("coalesced", False):
                continue
            if task.get("is_priority", False) and task.get("status") == "queued":
                queued_count += 1
        return queued_count


def has_earlier_task(state, current_task_id, is_priority=None):
    """Check whether an earlier same-priority task is still waiting for hardware."""
    if is_priority is None:
        with state.task_registry_lock:
            task = state.task_registry.get(current_task_id)
            if task is None:
                return False
            is_priority = task.get("is_priority", False)

    with state.task_registry_lock:
        task_snapshot = {
            task_id: {
                "is_priority": task.get("is_priority", False),
                "status": task.get("status"),
                "unit_id": task.get("unit_id"),
            }
            for task_id, task in state.task_registry.items()
        }

    with state.task_order_lock:
        arrival_snapshot = dict(state.task_arrival_order)

    current_arrival_time = arrival_snapshot.get(current_task_id)
    if current_arrival_time is None:
        return False

    for task_id, arrival_time in arrival_snapshot.items():
        if task_id == current_task_id or arrival_time >= current_arrival_time:
            continue

        task = task_snapshot.get(task_id)
        if not task:
            continue

        if task.get("is_priority", False) != is_priority:
            continue

        status = task.get("status")
        unit_id = task.get("unit_id")
        waiting_for_hardware = status in {"initializing", "queued"} and not unit_id
        if waiting_for_hardware:
            return True

    return False
