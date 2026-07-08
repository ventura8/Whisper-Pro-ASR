"""Hardware Scheduling and Task Registry for Whisper Pro ASR."""

import contextlib
import logging
import queue
import threading
import time
import uuid
from types import SimpleNamespace

from modules.core import config, logging_setup, utils
from modules.inference import scheduler_state_helpers, scheduler_task_helpers
from modules.monitoring import history_manager

logger = logging.getLogger(__name__)


def _build_scheduler_state():
    """Build scheduler state container with stable attribute names."""
    if not config.HARDWARE_UNITS:
        logger.warning("[Scheduler] No hardware units configured. Falling back to Host CPU.")
        config.HARDWARE_UNITS.append({"type": "CPU", "id": "CPU", "name": "Host CPU"})

    hw_pool = queue.Queue()
    for unit_item in config.HARDWARE_UNITS:
        hw_pool.put(unit_item)

    accel_limit = len(config.HARDWARE_UNITS)
    model_lock = threading.Semaphore(accel_limit)
    # Legacy field kept for backward compatibility with fixtures/tests.
    priority_sequential_lock = threading.Semaphore(1)
    priority_lock = threading.Lock()
    pause_requested = threading.Event()
    pause_confirmed = threading.Event()
    resume_event = threading.Event()
    resume_event.set()

    unit_sync = {}
    unit_priority_requests = {}
    for unit_item in config.HARDWARE_UNITS:
        u_id = unit_item["id"]
        unit_sync[u_id] = {
            "pause_requested": threading.Event(),
            "pause_confirmed": threading.Event(),
            "resume_event": threading.Event(),
            "pause_generation": 0,
            "confirmed_generation": None,
        }
        unit_sync[u_id]["resume_event"].set()
        unit_priority_requests[u_id] = 0

    task_registry_lock = threading.RLock()
    cond = threading.Condition(task_registry_lock)

    return SimpleNamespace(
        hw_pool=hw_pool,
        accel_limit=accel_limit,
        model_lock=model_lock,
        priority_sequential_lock=priority_sequential_lock,
        priority_lock=priority_lock,
        pause_requested=pause_requested,
        pause_confirmed=pause_confirmed,
        resume_event=resume_event,
        unit_sync=unit_sync,
        unit_priority_requests=unit_priority_requests,
        active_sessions=0,
        queued_sessions=0,
        priority_requests=0,
        task_registry={},
        task_registry_lock=task_registry_lock,
        cond=cond,
        task_arrival_order={},
        task_order_lock=threading.Lock(),
        unit_ownership={},
        preemptible_units=set(),
        targeted_units=set(),
        pause_generation=0,
        confirmed_generation=None,
        engine_initialized=False,
        whisper_loaded=False,
        uvr_loaded=False,
    )


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
    with STATE.task_registry_lock:
        active_units = [
            task.get("unit_id")
            for task in STATE.task_registry.values()
            if task.get("status") == "active" and not task.get("is_priority", False) and task.get("unit_id")
        ]
        unit_order = {unit["id"]: idx for idx, unit in enumerate(config.HARDWARE_UNITS)}
        active_units.sort(key=lambda uid: unit_order.get(uid, -1), reverse=True)
        for uid in active_units:
            u_sync = STATE.unit_sync.get(uid)
            if u_sync and u_sync["resume_event"].is_set():
                return uid
        if active_units:
            return active_units[0]
        for unit in config.HARDWARE_UNITS:
            uid = unit["id"]
            u_sync = STATE.unit_sync.get(uid)
            if u_sync and u_sync["resume_event"].is_set():
                return uid
        return config.HARDWARE_UNITS[0]["id"]


def _get_standard_task_state(task_id, thread_id):
    """Return whether another standard task is active/initializing."""
    has_active_standard = False
    has_initializing_standard = False
    with STATE.task_registry_lock:
        if STATE.task_registry:
            for tid, task in STATE.task_registry.items():
                is_current = (tid == task_id) or (task.get("task_id") == task_id) or (tid == thread_id)
                if is_current or task.get("is_priority", False):
                    continue
                if task.get("status") == "active":
                    has_active_standard = True
                    break
                if task.get("status") == "initializing":
                    has_initializing_standard = True

        if not has_active_standard:
            has_active_standard = (STATE.active_sessions - STATE.priority_requests) > 0

    return has_active_standard, has_initializing_standard


def _request_pause_for_target(target_unit_id):
    """Request pause on a specific target unit using unit-scoped sync only."""
    generation = STATE.pause_generation + 1
    STATE.pause_generation = generation

    u_sync = STATE.unit_sync.get(target_unit_id)
    if u_sync:
        if u_sync["resume_event"].is_set():
            logger.info("[Scheduler] Priority request: Pausing unit %s...", target_unit_id)
            u_sync["pause_confirmed"].clear()
            u_sync["confirmed_generation"] = None
            u_sync["pause_generation"] = generation
            u_sync["resume_event"].clear()
            u_sync["pause_requested"].set()
            STATE.targeted_units.add(target_unit_id)
            STATE.confirmed_generation = None
            STATE.pause_requested.set()
            STATE.resume_event.clear()
            STATE.pause_confirmed.clear()
        else:
            logger.info("[Scheduler] Priority request: Unit %s is already pausing/paused.", target_unit_id)
            return u_sync.get("pause_generation", generation), False
        return u_sync.get("pause_generation", generation), True

    logger.warning("[Scheduler] Missing unit sync for target %s; skipping pause request.", target_unit_id)
    return generation, False


def _wait_for_pause_confirmation(target_unit_id, expected_generation):
    """Wait until pause confirmation for the requested generation is observed."""
    last_wait_log_at = 0.0
    with STATE.cond:
        while True:
            u_sync = STATE.unit_sync.get(target_unit_id) if target_unit_id else None
            if u_sync:
                if u_sync["pause_confirmed"].is_set():
                    confirmed_generation = u_sync.get("confirmed_generation")
                    if confirmed_generation in (None, expected_generation):
                        return True
            if scheduler_state_helpers.should_skip_pause_confirmation(STATE, target_unit_id):
                return True

            if time.time() - last_wait_log_at >= 30.0:
                logger.info(
                    "[Scheduler] Still waiting for pause confirmation (unit=%s, expected_generation=%s)",
                    target_unit_id,
                    expected_generation,
                )
                last_wait_log_at = time.time()
            STATE.cond.wait(timeout=0.1)


def wait_for_priority():
    """Handles priority task synchronization (Request pause from others)."""
    # Mark this thread as priority for the duration of the request lifecycle.
    utils.THREAD_CONTEXT.is_priority = True

    # Register a priority request and optionally request pause from standard workers.
    do_pause = False
    target_unit_id = None
    pause_generation = None
    wait_for_confirmation = True
    needs_activation_wait = False
    task_id = None
    thread_id = None

    with STATE.priority_lock:
        STATE.priority_requests += 1

    task_id = getattr(utils.THREAD_CONTEXT, "task_id", None)
    thread_id = getattr(utils.THREAD_CONTEXT, "registration_thread_id", None) or threading.get_ident()
    has_active_standard, has_initializing_standard = _get_standard_task_state(task_id, thread_id)
    if not has_active_standard and has_initializing_standard:
        needs_activation_wait = True

    if needs_activation_wait:
        has_active_standard = scheduler_state_helpers.wait_for_standard_task_to_activate(STATE, task_id, thread_id)

    with STATE.task_registry_lock:
        has_registered_tasks = bool(STATE.task_registry)

    with STATE.priority_lock:
        if not has_active_standard:
            has_active_standard, has_initializing_standard = _get_standard_task_state(task_id, thread_id)

        # Pause only when saturated by active standard work and no preferred idle unit exists.

        should_pause = STATE.active_sessions >= STATE.accel_limit and has_active_standard

        if should_pause:
            target_candidate = _select_preemption_target_unit()

            if has_registered_tasks and scheduler_state_helpers.has_preferred_idle_unit(
                STATE, config.HARDWARE_UNITS, target_candidate
            ):
                logger.info(
                    "[Scheduler] Skipping preemption for target %s because a preferred idle unit exists.",
                    target_candidate,
                )
            else:
                do_pause = True
                target_unit_id = target_candidate
                utils.THREAD_CONTEXT.target_unit_id = target_unit_id
                logger.info("[Scheduler] Priority task targeting unit %s for preemption", target_unit_id)
                if not hasattr(STATE, "unit_priority_requests"):
                    STATE.unit_priority_requests = {}
                STATE.unit_priority_requests[target_unit_id] = STATE.unit_priority_requests.get(target_unit_id, 0) + 1
                pause_request = _request_pause_for_target(target_unit_id)
                if isinstance(pause_request, tuple):
                    pause_generation, wait_for_confirmation = pause_request
                else:
                    pause_generation = pause_request
                    wait_for_confirmation = True
                utils.THREAD_CONTEXT.target_pause_generation = pause_generation

    if do_pause:
        logger.debug(
            "[Scheduler] Priority preemption requested for %s; using per-unit pause confirmation only.",
            target_unit_id,
        )

    if do_pause and wait_for_confirmation:
        with STATE.task_registry_lock:
            still_active_standard = any(
                task.get("status") == "active" and not task.get("is_priority", False)
                for task in STATE.task_registry.values()
            )
        if not still_active_standard:
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
    elif do_pause:
        logger.debug(
            "[Scheduler] Skipping duplicate pause confirmation wait for unit %s (already pausing/paused).",
            target_unit_id,
        )


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
            target_unit_id = getattr(utils.THREAD_CONTEXT, "target_unit_id", None)
            if target_unit_id:
                u_sync = STATE.unit_sync.get(target_unit_id)
                if u_sync:
                    if not hasattr(STATE, "unit_priority_requests"):
                        STATE.unit_priority_requests = {}
                    current_unit_requests = max(0, STATE.unit_priority_requests.get(target_unit_id, 0) - 1)
                    STATE.unit_priority_requests[target_unit_id] = current_unit_requests

                    keep_paused = keep_pause_for_backlog or current_unit_requests > 0
                    if keep_paused:
                        if keep_pause_for_backlog:
                            logger.info(
                                "[Scheduler] Keeping unit %s paused: queued priority backlog (%d) saturates capacity (%d).",
                                target_unit_id,
                                queued_priority_count,
                                STATE.accel_limit,
                            )
                        else:
                            logger.info(
                                "[Scheduler] Keeping unit %s paused: %d unit-targeted priority request(s) still active.",
                                target_unit_id,
                                current_unit_requests,
                            )
                    else:
                        logger.info("[Scheduler] Resuming unit %s...", target_unit_id)
                        u_sync["pause_requested"].clear()
                        u_sync["resume_event"].set()
                        u_sync["pause_confirmed"].clear()
                        u_sync["confirmed_generation"] = None
                        STATE.targeted_units.discard(target_unit_id)
                # Clear thread-local targeting metadata after release processing.
                utils.THREAD_CONTEXT.target_unit_id = None
                utils.THREAD_CONTEXT.target_pause_generation = None

            if STATE.priority_requests == 0:
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

            STATE.cond.notify_all()


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

    # Priority task ordering is enforced at acquisition time.
    lock_ctx = contextlib.nullcontext()

    try:
        if is_priority:
            increment_queued_session()
        with lock_ctx:
            if is_priority:
                decrement_queued_session()
            try:
                yield
            except Exception as e:
                with STATE.cond:
                    if task_id in STATE.task_registry:
                        STATE.task_registry[task_id]["status"] = "failed"
                    STATE.cond.notify_all()
                raise e
    finally:
        # Always release priority if this task entered priority flow.
        release_priority()

        history_task = None
        with STATE.cond:
            if task_id in STATE.task_registry:
                task = STATE.task_registry[task_id]
                with logging_setup.TASK_LOGS_LOCK:
                    task["logs"] = logging_setup.TASK_LOGS.get(task_id, [])
                task["status"] = "completed" if task.get("status") != "failed" else "failed"
                task["progress"] = 100
                history_task = task.copy()
                del STATE.task_registry[task_id]
            with logging_setup.TASK_LOGS_LOCK:
                if task_id in logging_setup.TASK_LOGS:
                    del logging_setup.TASK_LOGS[task_id]
            # Remove from FIFO arrival-order tracking.
            with STATE.task_order_lock:
                if task_id in STATE.task_arrival_order:
                    del STATE.task_arrival_order[task_id]
            STATE.cond.notify_all()

        if history_task is not None:
            history_manager.log_completed_task(history_task)


def cleanup_failed_task():
    """Removes task from registry on early failure."""
    scheduler_task_helpers.cleanup_failed_task(STATE)


def update_task_metadata(**kwargs):
    """Updates metadata for the current thread's task."""
    scheduler_task_helpers.update_task_metadata(STATE, logger, **kwargs)


def update_task_progress(progress, stage=None):
    """Updates progress percentage and stage."""
    scheduler_task_helpers.update_task_progress(STATE, progress, stage=stage)


def increment_active_session():
    """Tracks active session count."""
    with STATE.task_registry_lock:
        STATE.active_sessions += 1


def decrement_active_session():
    """Tracks active session count."""
    with STATE.task_registry_lock:
        STATE.active_sessions = max(0, STATE.active_sessions - 1)


def increment_queued_session():
    """Tracks queued session count."""
    with STATE.task_registry_lock:
        STATE.queued_sessions += 1


def decrement_queued_session():
    """Tracks queued session count."""
    with STATE.task_registry_lock:
        STATE.queued_sessions = max(0, STATE.queued_sessions - 1)


def get_preemptible_unit():
    """Finds a unit that can be borrowed from a paused task."""
    with STATE.task_registry_lock:
        for unit_id in list(STATE.preemptible_units):
            STATE.preemptible_units.remove(unit_id)
            logger.info("[Scheduler] Borrowing unit %s for priority task", unit_id)
            return unit_id
    return None


def mark_unit_preemptible(unit_id):
    """Marks a unit as available for borrowing by priority tasks."""
    with STATE.task_registry_lock:
        STATE.preemptible_units.add(unit_id)


def unmark_unit_preemptible(unit_id):
    """Removes unit from preemptible pool."""
    with STATE.task_registry_lock:
        if unit_id in STATE.preemptible_units:
            STATE.preemptible_units.remove(unit_id)


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
