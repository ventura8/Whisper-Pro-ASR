"""Task-registry mutation helpers used by the scheduler module."""

import logging
import threading
from typing import Any

from modules.core import logging_setup, utils


def cleanup_failed_task(state):
    """Remove task/log/order entries for the current thread context on failure."""
    task_id = getattr(utils.THREAD_CONTEXT, "task_id", None)
    thread_id = getattr(utils.THREAD_CONTEXT, "registration_thread_id", None) or threading.get_ident()

    cleanup_keys = _cleanup_keys(task_id, thread_id)
    with state.task_registry_lock:
        _remove_registry_entries(state.task_registry, cleanup_keys)
        with logging_setup.TASK_LOGS_LOCK:
            _remove_registry_entries(logging_setup.TASK_LOGS, cleanup_keys)
    with state.task_order_lock:
        _remove_registry_entries(state.task_arrival_order, cleanup_keys)


def _cleanup_keys(task_id: str | None, thread_id: int) -> list[str | int]:
    keys: list[str | int] = []
    if task_id:
        keys.append(task_id)
    keys.append(thread_id)
    return keys


def _remove_registry_entries(mapping: dict[Any, Any], keys: list[str | int]) -> None:
    for key in keys:
        if key in mapping:
            del mapping[key]


def update_task_metadata(state: Any, **kwargs: Any) -> None:
    """Update metadata for the current task; create a minimal fallback entry if missing."""
    task_id = getattr(utils.THREAD_CONTEXT, "task_id", None)
    thread_id = getattr(utils.THREAD_CONTEXT, "registration_thread_id", None) or threading.get_ident()

    with state.task_registry_lock:
        target_key = _find_target_key(state, task_id, thread_id)
        if target_key:
            _apply_metadata_update(state, target_key, kwargs)
            return
    _log_missing_metadata_target(task_id, thread_id, kwargs)


def _apply_metadata_update(state: Any, target_key: str | int, kwargs: dict[str, Any]) -> None:
    state.task_registry[target_key].update(kwargs)
    if "live_text" in kwargs:
        logger.debug("[Scheduler] Updated live_text for task %s", state.task_registry[target_key].get("task_id"))


def _log_missing_metadata_target(task_id: str | None, thread_id: int, kwargs: dict[str, Any]) -> None:
    logger.warning(
        "[Scheduler] Missing task registry entry for task_id=%s, thread_id=%s. Skipping metadata update: %s",
        task_id,
        thread_id,
        kwargs,
    )


def record_task_failure(state: Any, msg: str, code: int = 400, context: str = "Task") -> None:
    """Persist failure details to task history and execution logs."""
    error_payload = {"error": msg, "status_code": code}
    update_task_metadata(
        state,
        status="failed",
        stage="Failed",
        result=error_payload,
        response_json=error_payload,
    )
    logger.error("[%s] %s", context, msg)


def ensure_failed_task_payload(task: dict[str, Any]) -> None:
    """Guarantee failed archived tasks expose error payloads for dashboard history."""
    if task.get("status") != "failed":
        return
    result = task.get("result")
    response_json = task.get("response_json")
    if _has_error_payload(result) or _has_error_payload(response_json):
        _mirror_failed_payloads(task, result, response_json)
        return
    msg = _derive_failure_message(task)
    error_payload = {"error": msg, "status_code": 500}
    task["result"] = error_payload
    task["response_json"] = error_payload


def _has_error_payload(value: Any) -> bool:
    return isinstance(value, dict) and bool(value.get("error"))


def task_has_failure_metadata(state: Any, task_id: str) -> bool:
    """Return True when the registered task already persisted failure details."""
    with state.task_registry_lock:
        task = state.task_registry.get(task_id)
        if not task or task.get("status") != "failed":
            return False
        return _has_error_payload(task.get("result")) or _has_error_payload(task.get("response_json"))


def _mirror_failed_payloads(task: dict[str, Any], result: Any, response_json: Any) -> None:
    if _has_error_payload(result) and not _has_error_payload(response_json):
        task["response_json"] = result
    elif _has_error_payload(response_json) and not _has_error_payload(result):
        task["result"] = response_json


def _derive_failure_message(task: dict[str, Any]) -> str:
    logs = task.get("logs") or []
    if logs:
        return str(logs[-1])
    stage = task.get("stage")
    if stage and stage not in {"Failed", "Waiting for Priority Slot"}:
        return f"Task failed during {stage}"
    return "Task failed"


def update_task_progress(state: Any, progress: int | float | None, stage: str | None = None) -> None:
    """Update progress and optional stage for the current task."""
    task_id = getattr(utils.THREAD_CONTEXT, "task_id", None)
    thread_id = getattr(utils.THREAD_CONTEXT, "registration_thread_id", None) or threading.get_ident()

    with state.task_registry_lock:
        target_key = _find_target_key(state, task_id, thread_id)
        if target_key:
            _apply_progress_update(state, target_key, progress, stage)


def _find_target_key(state: Any, task_id: str | None, thread_id: int) -> str | int | None:
    if task_id and task_id in state.task_registry:
        return task_id
    if thread_id in state.task_registry:
        return thread_id
    return None


def _apply_progress_update(
    state: Any,
    target_key: str | int,
    progress: int | float | None,
    stage: str | None,
) -> None:
    current_progress = state.task_registry[target_key].get("progress")
    should_update_progress = progress is not None
    should_update_stage = bool(stage)

    if progress is not None and current_progress is not None:
        should_update_progress = _verify_progress_not_regressing(progress, current_progress)

    if should_update_progress:
        state.task_registry[target_key]["progress"] = progress
    if should_update_stage:
        state.task_registry[target_key]["stage"] = stage


def _verify_progress_not_regressing(progress: int | float, current_progress: int | float) -> bool:
    try:
        if progress < current_progress:
            return False
    except TypeError:
        return False
    return True


logger = logging.getLogger(__name__)


def increment_active_session(state):
    """Tracks active session count."""
    with state.task_registry_lock:
        state.active_sessions += 1


def decrement_active_session(state):
    """Tracks active session count."""
    with state.task_registry_lock:
        state.active_sessions = max(0, state.active_sessions - 1)


def increment_queued_session(state):
    """Tracks queued session count."""
    with state.task_registry_lock:
        state.queued_sessions += 1


def decrement_queued_session(state):
    """Tracks queued session count."""
    with state.task_registry_lock:
        state.queued_sessions = max(0, state.queued_sessions - 1)


def get_preemptible_unit(state):
    """Finds a unit that can be borrowed from a paused task."""
    with state.task_registry_lock:
        for unit_id in list(state.preemptible_units):
            state.preemptible_units.remove(unit_id)
            logger.info("[Scheduler] Borrowing unit %s for priority task", unit_id)
            return unit_id
    return None


def mark_unit_preemptible(state, unit_id):
    """Marks a unit as available for borrowing by priority tasks."""
    with state.task_registry_lock:
        state.preemptible_units.add(unit_id)


def unmark_unit_preemptible(state, unit_id):
    """Removes unit from preemptible pool."""
    with state.task_registry_lock:
        if unit_id in state.preemptible_units:
            state.preemptible_units.remove(unit_id)


def cleanup_task_logs(task_id: str):
    """Remove task log history for a finished task."""
    with logging_setup.TASK_LOGS_LOCK:
        if task_id in logging_setup.TASK_LOGS:
            del logging_setup.TASK_LOGS[task_id]


def cleanup_task_order(state, task_id: str):
    """Remove task arrival ordering tracking for a finished task."""
    with state.task_order_lock:
        if task_id in state.task_arrival_order:
            del state.task_arrival_order[task_id]
