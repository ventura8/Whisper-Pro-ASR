"""Hardware Scheduling and Task Registry for Whisper Pro ASR."""

import contextlib
import logging
import threading
import time
import uuid
from types import SimpleNamespace
from typing import Optional

from modules.core import config, logging_setup, utils
from modules.inference.scheduler import state_helpers as scheduler_state_helpers
from modules.inference.scheduler import task_helpers as scheduler_task_helpers
from modules.monitoring import history_manager

logger = logging.getLogger(__name__)


def _build_scheduler_state():
    """Build scheduler state container with stable attribute names."""
    return scheduler_state_helpers.build_scheduler_state(config)


class SchedulerState(SimpleNamespace):
    """Backward-compatible state factory used by tests and fixtures."""

    @staticmethod
    def build():
        """Build a fresh scheduler state instance."""
        return _build_scheduler_state()

    @staticmethod
    def create():
        """Alias for build() for external callers/tests."""
        return SchedulerState.build()

    def __new__(cls):
        return cls.build()


STATE = _build_scheduler_state()


def _select_preemption_target_unit():
    """Select best hardware unit to preempt."""
    return scheduler_state_helpers.select_preemption_target_unit(STATE)


def _get_standard_task_state(task_id, thread_id):
    """Return whether another standard task is active/initializing."""
    return scheduler_state_helpers.get_standard_task_state(STATE, task_id, thread_id)


def _request_pause_for_target(target_unit_id):
    """Request pause on a specific target unit using unit-scoped sync only."""
    return scheduler_state_helpers.request_pause_for_target(STATE, target_unit_id)


def _wait_for_pause_confirmation(target_unit_id, expected_generation):
    """Wait until pause confirmation for the requested generation is observed."""
    return scheduler_state_helpers.wait_for_pause_confirmation(STATE, target_unit_id, expected_generation)


def wait_for_priority():
    """Handles priority task synchronization (Request pause from others)."""
    utils.THREAD_CONTEXT.is_priority = True

    with STATE.priority_lock:
        STATE.priority_requests += 1

    task_id = getattr(utils.THREAD_CONTEXT, "task_id", None)
    thread_id = getattr(utils.THREAD_CONTEXT, "registration_thread_id", None) or threading.get_ident()

    _check_activation_wait(task_id, thread_id)

    with STATE.task_registry_lock:
        has_registered_tasks = bool(STATE.task_registry)

    with STATE.priority_lock:
        do_pause, target_unit_id, pause_gen, wait_confirm = _handle_preemption_pause(task_id, thread_id, has_registered_tasks)

    if do_pause:
        logger.debug(
            "[Scheduler] Priority preemption requested for %s; using per-unit pause confirmation only.",
            target_unit_id,
        )
        _wait_for_preemption_confirm(target_unit_id, pause_gen, wait_confirm)


def _check_activation_wait(task_id, thread_id):
    has_active_standard, has_initializing_standard = _get_standard_task_state(task_id, thread_id)
    if not has_active_standard and has_initializing_standard:
        scheduler_state_helpers.wait_for_standard_task_to_activate(STATE, task_id, thread_id)


def _handle_preemption_pause(task_id, thread_id, has_registered_tasks) -> tuple[bool, Optional[str], Optional[int], bool]:
    has_active_standard, _ = _get_standard_task_state(task_id, thread_id)
    if not _should_attempt_preemption(has_registered_tasks, has_active_standard):
        return False, None, None, False

    target_candidate = _select_preemption_target_unit()
    if _should_skip_preemption(has_registered_tasks, target_candidate):
        return False, None, None, False

    _update_preemption_target_state(target_candidate)

    pause_request = _request_pause_for_target(target_candidate)
    pause_gen, wait_confirm = _parse_pause_request(pause_request)

    utils.THREAD_CONTEXT.target_pause_generation = pause_gen
    return True, target_candidate, pause_gen, wait_confirm


def _should_attempt_preemption(has_registered_tasks: bool, has_active_standard: bool) -> bool:
    if STATE.active_sessions < STATE.accel_limit:
        return False
    if not has_active_standard:
        return False
    if has_registered_tasks and _has_priority_only_registry_tasks():
        return False
    return True


def _should_skip_preemption(has_registered_tasks: bool, target_candidate: str) -> bool:
    if has_registered_tasks:
        return scheduler_state_helpers.has_preferred_idle_unit(STATE, config.HARDWARE_UNITS, target_candidate)
    return False


def _has_priority_only_registry_tasks() -> bool:
    with STATE.task_registry_lock:
        if not STATE.task_registry:
            return False
        return not any(not task.get("is_priority", False) for task in STATE.task_registry.values())


def _update_preemption_target_state(target_candidate: str):
    utils.THREAD_CONTEXT.target_unit_id = target_candidate
    logger.info("[Scheduler] Priority task targeting unit %s for preemption", target_candidate)
    with STATE.cond:
        if not hasattr(STATE, "unit_priority_requests"):
            STATE.unit_priority_requests = {}
        STATE.unit_priority_requests[target_candidate] = STATE.unit_priority_requests.get(target_candidate, 0) + 1
        STATE.cond.notify_all()


def _parse_pause_request(pause_request) -> tuple[Optional[int], bool]:
    if isinstance(pause_request, tuple):
        return pause_request
    return pause_request, True


def _wait_for_preemption_confirm(target_unit_id, pause_generation, wait_for_confirmation):
    if not wait_for_confirmation:
        logger.debug(
            "[Scheduler] Skipping duplicate pause confirmation wait for unit %s (already pausing/paused).",
            target_unit_id,
        )
        return

    if not _has_active_standard_tasks():
        logger.debug(
            "[Scheduler] No active standard tasks remain. Skipping preemption confirmation wait for unit %s.",
            target_unit_id,
        )
        return

    logger.debug("[Scheduler] Waiting for preemption confirmation on unit %s...", target_unit_id)
    expected_generation = pause_generation if pause_generation is not None else STATE.pause_generation
    _wait_for_pause_confirmation(
        target_unit_id=target_unit_id,
        expected_generation=expected_generation,
    )


def _has_active_standard_tasks() -> bool:
    with STATE.task_registry_lock:
        for task in STATE.task_registry.values():
            if task.get("status") == "active" and not task.get("is_priority", False):
                return True
    return False


def release_unit_preemption_hold(unit_id):
    """Release preemption hold on a specific target unit (e.g. when fallback borrowing)."""
    with STATE.cond:
        u_sync = STATE.unit_sync.get(unit_id)
        if u_sync:
            if not hasattr(STATE, "unit_priority_requests"):
                STATE.unit_priority_requests = {}
            current_unit_requests = max(0, STATE.unit_priority_requests.get(unit_id, 0) - 1)
            STATE.unit_priority_requests[unit_id] = current_unit_requests

            # If no targeted requests remain for this unit, resume it
            if current_unit_requests == 0:
                logger.info("[Scheduler] Resuming unit %s due to fallback preemption release...", unit_id)
                u_sync["pause_requested"].clear()
                u_sync["resume_event"].set()
                u_sync["pause_confirmed"].clear()
                u_sync["confirmed_generation"] = None
                STATE.targeted_units.discard(unit_id)
        STATE.cond.notify_all()


def release_priority():
    """Releases priority hold and resumes paused tasks."""
    # Safety: only release if this thread actually holds a priority token.
    if not getattr(utils.THREAD_CONTEXT, "is_priority", False):
        return

    # Reset thread-local priority flag to avoid double-release.
    utils.THREAD_CONTEXT.is_priority = False

    with STATE.priority_lock:
        queued_priority_count = scheduler_state_helpers.get_queued_priority_count(
            STATE,
            exclude_task_id=getattr(utils.THREAD_CONTEXT, "task_id", None),
        )
        STATE.priority_requests = max(0, STATE.priority_requests - 1)
        keep_pause_for_backlog = queued_priority_count >= STATE.accel_limit

        with STATE.cond:
            _release_targeted_unit_hold(keep_pause_for_backlog, queued_priority_count)
            _reset_all_syncs_if_idle(keep_pause_for_backlog, queued_priority_count)
            STATE.cond.notify_all()


def _release_targeted_unit_hold(keep_pause_for_backlog: bool, queued_priority_count: int):
    target_unit_id = getattr(utils.THREAD_CONTEXT, "target_unit_id", None)
    if not target_unit_id:
        return
    u_sync = STATE.unit_sync.get(target_unit_id)
    if u_sync:
        _process_targeted_unit_resume(target_unit_id, u_sync, keep_pause_for_backlog, queued_priority_count)
    # Clear thread-local targeting metadata after release processing.
    utils.THREAD_CONTEXT.target_unit_id = None
    utils.THREAD_CONTEXT.target_pause_generation = None


def _process_targeted_unit_resume(target_unit_id: str, u_sync: dict, keep_pause_for_backlog: bool, queued_priority_count: int):
    if not hasattr(STATE, "unit_priority_requests"):
        STATE.unit_priority_requests = {}
    current_unit_requests = max(0, STATE.unit_priority_requests.get(target_unit_id, 0) - 1)
    STATE.unit_priority_requests[target_unit_id] = current_unit_requests

    if keep_pause_for_backlog or current_unit_requests > 0:
        _log_preemption_hold_continuation(target_unit_id, keep_pause_for_backlog, queued_priority_count, current_unit_requests)
    else:
        logger.info("[Scheduler] Resuming unit %s...", target_unit_id)
        u_sync["pause_requested"].clear()
        u_sync["resume_event"].set()
        u_sync["pause_confirmed"].clear()
        u_sync["confirmed_generation"] = None
        STATE.targeted_units.discard(target_unit_id)


def _log_preemption_hold_continuation(target_unit_id: str, keep_backlog: bool, backlog_count: int, unit_reqs: int):
    if keep_backlog:
        logger.info(
            "[Scheduler] Keeping unit %s paused: queued priority backlog (%d) saturates capacity (%d).",
            target_unit_id,
            backlog_count,
            STATE.accel_limit,
        )
    else:
        logger.info(
            "[Scheduler] Keeping unit %s paused: %d unit-targeted priority request(s) still active.",
            target_unit_id,
            unit_reqs,
        )


def _reset_all_syncs_if_idle(keep_pause_for_backlog: bool, queued_priority_count: int):
    if STATE.priority_requests != 0:
        return
    if keep_pause_for_backlog:
        logger.info(
            "[Scheduler] Priority active request completed, but queued priority tasks remain. "
            "Keeping targeted unit pauses while backlog (%d) saturates capacity (%d). "
            "Active: %d | Queued: %d",
            queued_priority_count,
            STATE.accel_limit,
            STATE.active_sessions,
            STATE.queued_sessions,
        )
    else:
        logger.info(
            "[Scheduler] Priority released. Active: %d | Queued: %d",
            STATE.active_sessions,
            STATE.queued_sessions,
        )
        STATE.pause_requested.clear()
        STATE.resume_event.set()
        STATE.pause_confirmed.clear()
        STATE.confirmed_generation = None
        # Reset all unit sync primitives when no priority workload remains.
        STATE.targeted_units.clear()
        for u_sync in STATE.unit_sync.values():
            u_sync["pause_requested"].clear()
            u_sync["resume_event"].set()
            u_sync["pause_confirmed"].clear()
            u_sync["confirmed_generation"] = None
        for unit_id in list(getattr(STATE, "unit_priority_requests", {}).keys()):
            STATE.unit_priority_requests[unit_id] = 0


@contextlib.contextmanager
def early_task_registration(task_type="ASR/LD", stage="Initializing", filename=None, is_priority=False):
    """
    Context manager to handle registration and cleanup of a task lifecycle,
    including UUID generation, registry binding, thread context assignment,
    and priority synchronization.
    """
    task_id = str(uuid.uuid4())
    utils.THREAD_CONTEXT.task_id = task_id
    thread_id = threading.get_ident()
    utils.THREAD_CONTEXT.registration_thread_id = thread_id
    display_name = filename or getattr(utils.THREAD_CONTEXT, "filename", "Unknown")

    _register_task_in_state(task_id, display_name, task_type, stage, is_priority)

    try:
        _handle_queued_session_state(is_priority)
        try:
            yield
        except Exception as exc:
            if not scheduler_task_helpers.task_has_failure_metadata(STATE, task_id):
                record_task_failure(str(exc), 500, context=task_type)
            raise
    finally:
        # Always release priority if this task entered priority flow.
        release_priority()
        _finalize_registered_task(task_id)


def _handle_queued_session_state(is_priority: bool):
    if is_priority:
        increment_queued_session()
        decrement_queued_session()


def _register_task_in_state(task_id, display_name, task_type, stage, is_priority):
    # Priority tasks start queued for immediate dashboard visibility.
    initial_status = "queued" if is_priority else "initializing"
    initial_stage = "Waiting for Priority Slot" if is_priority else stage
    arrival_time = time.time()
    with STATE.cond:
        with logging_setup.TASK_LOGS_LOCK:
            if task_id not in logging_setup.TASK_LOGS:
                logging_setup.TASK_LOGS[task_id] = []
        STATE.task_registry[task_id] = {
            "task_id": task_id,
            "filename": display_name,
            "start_time": arrival_time,
            "status": initial_status,
            "progress": 0,
            "stage": initial_stage,
            "type": task_type,
            "is_priority": is_priority,
            "endpoint": getattr(utils.THREAD_CONTEXT, "endpoint", ""),
            "video_duration": getattr(utils.THREAD_CONTEXT, "total_duration", 0),
            "caller_info": getattr(utils.THREAD_CONTEXT, "caller_info", {}),
            "request_json": getattr(utils.THREAD_CONTEXT, "request_json", {}),
            "live_text": "",
            "logs": [],
        }
        with STATE.task_order_lock:
            STATE.task_arrival_order[task_id] = arrival_time
        STATE.cond.notify_all()


def _finalize_registered_task(task_id):
    history_task = None
    with STATE.cond:
        history_task = _archive_registry_task(task_id)
        _cleanup_task_logs(task_id)
        _cleanup_task_order(task_id)
        STATE.cond.notify_all()

    if history_task is not None:
        try:
            history_manager.log_completed_task(history_task)
        except (RuntimeError, OSError, ValueError, TypeError, KeyError, AttributeError) as err:
            logger.error("[Scheduler] Failed to persist task history for %s: %s", task_id, err)


def finalize_stale_task(task_id):
    """Public entry point for external reapers (e.g. telemetry's stale-active-task
    detection) to run a task through the same archive/remove lifecycle normal task
    completion uses, without reaching into the private `_finalize_registered_task`.
    Idempotent: a no-op if the task was already finalized (e.g. by its own owning
    worker's `early_task_registration` finally-block, concurrently)."""
    _finalize_registered_task(task_id)


def _archive_registry_task(task_id: str) -> Optional[dict]:
    if task_id not in STATE.task_registry:
        return None
    task = STATE.task_registry[task_id]
    with logging_setup.TASK_LOGS_LOCK:
        task["logs"] = logging_setup.TASK_LOGS.get(task_id, [])
    task["status"] = "failed" if task.get("status") == "failed" else "completed"
    task["progress"] = 100
    _normalize_history_hardware_fields(task)
    scheduler_task_helpers.ensure_failed_task_payload(task)
    res = task.copy()
    del STATE.task_registry[task_id]
    return res


def _normalize_history_hardware_fields(task: dict) -> None:
    """Persist canonical hardware metadata for history rendering.

    Task cleanup can clear live `unit_id` before archival. Preserve any known unit
    values into stable `history_unit_*` fields and backfill missing live fields.
    """
    unit_id, unit_type, unit_name = _resolve_history_hardware_triplet(task)
    _write_history_hardware_fields(task, unit_id, unit_type, unit_name)


def _resolve_history_hardware_triplet(task: dict) -> tuple[Optional[str], Optional[str], Optional[str]]:
    unit_id = _task_history_or_live_value(task, "history_unit_id", "unit_id")
    unit_type = _task_history_or_live_value(task, "history_unit_type", "unit_type")
    unit_name = _task_history_or_live_value(task, "history_unit_name", "unit_name")
    if _needs_unit_meta_resolution(unit_id, unit_type, unit_name):
        resolved_type, resolved_name = _resolve_unit_meta_from_config(unit_id)
        unit_type = unit_type or resolved_type
        unit_name = unit_name or resolved_name
    return unit_id, unit_type, unit_name


def _task_history_or_live_value(task: dict, history_key: str, live_key: str) -> Optional[str]:
    value = task.get(history_key)
    if value:
        return value
    return task.get(live_key)


def _needs_unit_meta_resolution(unit_id: Optional[str], unit_type: Optional[str], unit_name: Optional[str]) -> bool:
    if not unit_id:
        return False
    if unit_type and unit_name:
        return False
    return True


def _resolve_unit_meta_from_config(unit_id: str) -> tuple[Optional[str], Optional[str]]:
    for unit in config.HARDWARE_UNITS:
        if str(unit.get("id")) == str(unit_id):
            return unit.get("type"), unit.get("name")
    return None, None


def _write_history_hardware_fields(task: dict, unit_id: Optional[str], unit_type: Optional[str], unit_name: Optional[str]) -> None:
    _set_if_present(task, "history_unit_id", unit_id)
    _set_if_present(task, "history_unit_type", unit_type)
    _set_if_present(task, "history_unit_name", unit_name)
    _set_if_missing(task, "unit_id", unit_id)
    _set_if_missing(task, "unit_type", unit_type)
    _set_if_missing(task, "unit_name", unit_name)


def _set_if_present(task: dict, key: str, value: Optional[str]) -> None:
    if value:
        task[key] = value


def _set_if_missing(task: dict, key: str, value: Optional[str]) -> None:
    if value and not task.get(key):
        task[key] = value


def _cleanup_task_logs(task_id: str):
    scheduler_task_helpers.cleanup_task_logs(task_id)


def _cleanup_task_order(task_id: str):
    scheduler_task_helpers.cleanup_task_order(STATE, task_id)


def cleanup_failed_task():
    """Removes task from registry on early failure."""
    scheduler_task_helpers.cleanup_failed_task(STATE)


def update_task_metadata(**kwargs):
    """Updates metadata for the current thread's task."""
    scheduler_task_helpers.update_task_metadata(STATE, **kwargs)


def record_task_failure(msg: str, code: int = 400, context: str = "Task") -> None:
    """Persist failure details to task history and execution logs."""
    scheduler_task_helpers.record_task_failure(STATE, msg, code, context)


def update_task_progress(progress, stage=None):
    """Updates progress percentage and stage."""
    scheduler_task_helpers.update_task_progress(STATE, progress, stage=stage)


def increment_active_session():
    """Tracks active session count."""
    scheduler_task_helpers.increment_active_session(STATE)


def decrement_active_session():
    """Tracks active session count."""
    scheduler_task_helpers.decrement_active_session(STATE)


def increment_queued_session():
    """Tracks queued session count."""
    scheduler_task_helpers.increment_queued_session(STATE)


def decrement_queued_session():
    """Tracks queued session count."""
    scheduler_task_helpers.decrement_queued_session(STATE)


def get_preemptible_unit():
    """Finds a unit that can be borrowed from a paused task."""
    return scheduler_task_helpers.get_preemptible_unit(STATE)


def mark_unit_preemptible(unit_id):
    """Marks a unit as available for borrowing by priority tasks."""
    scheduler_task_helpers.mark_unit_preemptible(STATE, unit_id)


def unmark_unit_preemptible(unit_id):
    """Removes unit from preemptible pool."""
    scheduler_task_helpers.unmark_unit_preemptible(STATE, unit_id)


def has_earlier_task(current_task_id, is_priority=None):
    """FIFO check delegate."""
    return scheduler_state_helpers.has_earlier_task(STATE, current_task_id, is_priority=is_priority)


def has_queued_priority_tasks(exclude_task_id=None):
    """Priority check delegate."""
    return scheduler_state_helpers.has_queued_priority_tasks(STATE, exclude_task_id=exclude_task_id)


def get_queued_priority_count(exclude_task_id=None):
    """Priority count delegate."""
    return scheduler_state_helpers.get_queued_priority_count(STATE, exclude_task_id=exclude_task_id)


def get_service_stats_minimal():
    """Stats delegate."""
    return scheduler_state_helpers.get_service_stats_minimal(STATE)


def is_engine_initialized():
    """Engine init check delegate."""
    return scheduler_state_helpers.is_engine_initialized(STATE)


def is_uvr_loaded():
    """UVR load check delegate."""
    return scheduler_state_helpers.is_uvr_loaded(STATE)
