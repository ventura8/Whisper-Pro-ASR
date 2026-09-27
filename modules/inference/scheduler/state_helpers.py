"""Read-only helper functions for scheduler state queries."""

import logging
import queue
import threading
import time
from collections.abc import Iterable, Mapping
from types import SimpleNamespace
from typing import Any, Optional

from modules.core import config
from modules.inference.scheduler import ordering as scheduler_ordering

TaskKey = str | int
TaskRecord = dict[str, Any]
HardwareUnit = Mapping[str, Any]


def has_earlier_task(state: Any, current_task_id: TaskKey, is_priority: Optional[bool] = None) -> bool:
    """Check if there are earlier same-priority tasks still waiting for hardware."""
    return scheduler_ordering.has_earlier_task(state, current_task_id, is_priority=is_priority)


def has_queued_priority_tasks(state: Any, exclude_task_id: Optional[TaskKey] = None) -> bool:
    """Return True if any detect-language task is currently queued."""
    return scheduler_ordering.has_queued_priority_tasks(state, exclude_task_id=exclude_task_id)


def get_queued_priority_count(state: Any, exclude_task_id: Optional[TaskKey] = None) -> int:
    """Return count of queued detect-language tasks."""
    return scheduler_ordering.get_queued_priority_count(state, exclude_task_id=exclude_task_id)


def get_service_stats_minimal(state: Any) -> dict[str, list[dict[str, Any]]]:
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


def wait_for_standard_task_to_activate(state: Any, task_id: Optional[str], thread_id: int, timeout: float = 0.5) -> bool:
    """Wait briefly for an initializing standard task to become active."""
    end_wait = time.time() + timeout
    while time.time() < end_wait:
        with state.task_registry_lock:
            for tid, task in state.task_registry.items():
                if _is_other_active_standard_task(tid, task, task_id, thread_id):
                    return True
        time.sleep(0.01)
    return False


def _is_other_active_standard_task(tid: TaskKey, task: TaskRecord, task_id: Optional[str], thread_id: int) -> bool:
    is_current = (tid == task_id) or (task.get("task_id") == task_id) or (tid == thread_id)
    return not is_current and task.get("status") == "active" and not task.get("is_priority", False)


def has_preferred_idle_unit(state: Any, hardware_units: list[HardwareUnit], target_unit_id: str) -> bool:
    """Return True if an idle unit is preferred over the selected target unit."""
    unit_order = _unit_order_map(hardware_units)
    target_rank = unit_order.get(target_unit_id, -1)
    idle_units = _idle_unit_ids(state)
    if not idle_units:
        return False
    best_idle_rank = max((unit_order.get(uid, -1) for uid in idle_units), default=-1)
    return best_idle_rank > target_rank


def _unit_order_map(hardware_units: list[HardwareUnit]) -> dict[str, int]:
    return {unit["id"]: idx for idx, unit in enumerate(hardware_units)}


def _idle_unit_ids(state: Any) -> list[str | None]:
    try:
        with state.hw_pool.mutex:
            # Iterated in place: every queue.Queue mutation holds this same mutex.
            return [u.get("id") for u in state.hw_pool.queue if isinstance(u, dict)]
    except AttributeError:
        return []


def should_skip_pause_confirmation(state: Any, target_unit_id: Optional[str]) -> bool:
    """Return True when waiting for pause confirmation is no longer necessary.

    Skips the wait when:
    - The targeted unit has no active standard task running on it (it already yielded).
    - A unit is already in the preemptible pool and ready to be borrowed by a priority task.
    - The targeted unit's task is in vocal separation.
    """
    with state.task_registry_lock:
        if target_unit_id:
            if _should_skip_for_unit(state, target_unit_id):
                return True
        else:
            if not _has_active_standard_tasks(state.task_registry.values()):
                return True
        if state.preemptible_units:
            return True
    return False


def _should_skip_for_unit(state: Any, target_unit_id: str) -> bool:
    active_standard_tasks = [task for task in state.task_registry.values() if _is_active_standard_task(task)]
    if _should_skip_without_active_tasks(active_standard_tasks):
        return True
    if _should_skip_without_target_or_unknown_unit(active_standard_tasks, target_unit_id):
        return True
    return _should_skip_for_vocal_separation(active_standard_tasks, target_unit_id)


def _should_skip_without_active_tasks(active_standard_tasks: list[TaskRecord]) -> bool:
    return not active_standard_tasks


def _should_skip_without_target_or_unknown_unit(active_standard_tasks: list[TaskRecord], target_unit_id: str) -> bool:
    target_has_active_standard = _has_target_active_standard(active_standard_tasks, target_unit_id)
    has_unknown_active_standard = _has_unknown_unit_active_standard(active_standard_tasks)
    return not target_has_active_standard and not has_unknown_active_standard


def _should_skip_for_vocal_separation(active_standard_tasks: list[TaskRecord], target_unit_id: str) -> bool:
    if not _has_target_active_standard(active_standard_tasks, target_unit_id):
        return False
    return _target_in_vocal_separation(active_standard_tasks, target_unit_id)


def _is_active_standard_task(task: TaskRecord) -> bool:
    return task.get("status") == "active" and not task.get("is_priority", False)


def _has_active_standard_tasks(tasks: Iterable[TaskRecord]) -> bool:
    return any(_is_active_standard_task(task) for task in tasks)


def _has_target_active_standard(active_standard_tasks: list[TaskRecord], target_unit_id: str) -> bool:
    return any(task.get("unit_id") == target_unit_id for task in active_standard_tasks)


def _has_unknown_unit_active_standard(active_standard_tasks: list[TaskRecord]) -> bool:
    return any(task.get("unit_id") in (None, "") for task in active_standard_tasks)


def _target_in_vocal_separation(active_standard_tasks: list[TaskRecord], target_unit_id: str) -> bool:
    return any("Vocal Separation" in (task.get("stage") or "") for task in active_standard_tasks if task.get("unit_id") == target_unit_id)


def is_engine_initialized(state: Any) -> bool:
    """Return whether any models are loaded."""
    return state.engine_initialized


def is_uvr_loaded(state: Any) -> bool:
    """Return whether UVR is loaded."""
    return state.uvr_loaded


logger = logging.getLogger(__name__)


def select_preemption_target_unit(state: Any) -> Optional[str]:
    """Select best hardware unit to preempt."""
    with state.task_registry_lock:
        active_units = get_sorted_active_standard_units(state)
        for uid in active_units:
            if is_unit_resumed(state, uid):
                return uid
        if active_units:
            return active_units[0]
        return find_first_resumed_or_default_unit(state)


def get_sorted_active_standard_units(state: Any) -> list[str]:
    """Retrieve all active standard units sorted by priority."""
    active_units = [task.get("unit_id") for task in state.task_registry.values() if is_active_standard_unit(task)]
    unit_order = {}
    for idx, unit in enumerate(config.HARDWARE_UNITS):
        unit_order[unit["id"]] = idx
    active_units.sort(key=lambda uid: unit_order.get(uid, -1))
    return active_units


def is_active_standard_unit(task: TaskRecord) -> bool:
    """Return whether the task is currently active on a standard hardware unit."""
    return task.get("status") == "active" and not task.get("is_priority", False) and bool(task.get("unit_id"))


def is_unit_resumed(state, uid: str) -> bool:
    """Check if the specified hardware unit is resumed."""
    u_sync = state.unit_sync.get(uid)
    return u_sync is not None and u_sync["resume_event"].is_set()


def find_first_resumed_or_default_unit(state: Any) -> str:
    """Find first resumed unit or fall back to default unit ID."""
    for unit in config.HARDWARE_UNITS:
        uid = unit["id"]
        if is_unit_resumed(state, uid):
            return uid
    return config.HARDWARE_UNITS[0]["id"]


def get_standard_task_state(state: Any, task_id: Optional[str], thread_id: int) -> tuple[bool, bool]:
    """Return whether another standard task is active/initializing."""
    has_active_standard, has_init = scan_task_registry_states(state, task_id, thread_id)
    if not has_active_standard and _should_use_session_fallback(state):
        has_active_standard = (state.active_sessions - state.priority_requests) > 0
    return has_active_standard, has_init


def _should_use_session_fallback(state: Any) -> bool:
    """Allow legacy session fallback when registry is empty or includes standard tasks.

    Do not infer standard activity from counters while registry has only priority tasks,
    otherwise priority-only bursts can spuriously trigger preemption/pause flows.
    """
    with state.task_registry_lock:
        if not state.task_registry:
            return True
        return any(not task.get("is_priority", False) for task in state.task_registry.values())


def _has_registered_standard_task(state: Any) -> bool:
    with state.task_registry_lock:
        return any(not task.get("is_priority", False) for task in state.task_registry.values())


def is_other_standard_task(tid: TaskKey, task: TaskRecord, task_id: Optional[str], thread_id: int) -> bool:
    """Check if a task in registry is a different standard task."""
    is_current = (tid == task_id) or (task.get("task_id") == task_id) or (tid == thread_id)
    return not is_current and not task.get("is_priority", False)


def scan_task_registry_states(state: Any, task_id: Optional[str], thread_id: int) -> tuple[bool, bool]:
    """Scan registry to see if there are other active/initializing standard tasks."""
    has_active = False
    has_init = False
    with state.task_registry_lock:
        for tid, task in state.task_registry.items():
            if not is_other_standard_task(tid, task, task_id, thread_id):
                continue
            if task.get("status") == "active":
                has_active = True
                break
            if task.get("status") == "initializing":
                has_init = True
    return has_active, has_init


def request_pause_for_target(state: Any, target_unit_id: str) -> tuple[int, bool]:
    """Request pause on a specific target unit using unit-scoped sync only."""
    with state.cond:
        generation = state.pause_generation + 1
        state.pause_generation = generation

        u_sync = state.unit_sync.get(target_unit_id)
        if u_sync:
            if u_sync["resume_event"].is_set():
                logger.info("[Scheduler] Priority request: Pausing unit %s...", target_unit_id)
                u_sync["pause_confirmed"].clear()
                u_sync["confirmed_generation"] = None
                u_sync["pause_generation"] = generation
                u_sync["resume_event"].clear()
                u_sync["pause_requested"].set()
                state.targeted_units.add(target_unit_id)
                state.confirmed_generation = None
                state.pause_requested.set()
                state.resume_event.clear()
                state.pause_confirmed.clear()
            else:
                logger.info("[Scheduler] Priority request: Unit %s is already pausing/paused.", target_unit_id)
                state.cond.notify_all()
                return u_sync.get("pause_generation", generation), False
            state.cond.notify_all()
            return u_sync.get("pause_generation", generation), True

        logger.warning("[Scheduler] Missing unit sync for target %s; skipping pause request.", target_unit_id)
        state.cond.notify_all()
        return generation, False


def wait_for_pause_confirmation(state: Any, target_unit_id: Optional[str], expected_generation: int) -> bool:
    """Wait until pause confirmation for the requested generation is observed."""
    last_wait_log_at = 0.0
    with state.cond:
        # Indefinite by design (priority wait semantics): the loop ends only once the pause is
        # confirmed or no longer needed, so the result is always True.
        while not _pause_wait_over(state, target_unit_id, expected_generation):
            last_wait_log_at = log_waiting_status_periodically(target_unit_id, expected_generation, last_wait_log_at)
            state.cond.wait(timeout=0.1)
    return True


def _pause_wait_over(state: Any, target_unit_id: Optional[str], expected_generation: int) -> bool:
    """Whether the pause is confirmed for this generation, or waiting for it is moot."""
    return is_pause_confirmed(state, target_unit_id, expected_generation) or should_skip_pause_confirmation(state, target_unit_id)


def is_pause_confirmed(state, target_unit_id: Optional[str], expected_gen: int) -> bool:
    """Return whether the pause is confirmed for the targeted unit."""
    if not target_unit_id:
        return False
    u_sync = state.unit_sync.get(target_unit_id)
    if u_sync and u_sync["pause_confirmed"].is_set():
        confirmed_gen = u_sync.get("confirmed_generation")
        return confirmed_gen in (None, expected_gen)
    return False


def log_waiting_status_periodically(target_unit_id: Optional[str], expected_gen: int, last_log_time: float) -> float:
    """Log waiting status for pause confirmation periodically."""
    now = time.time()
    if now - last_log_time >= 30.0:
        logger.info(
            "[Scheduler] Still waiting for pause confirmation (unit=%s, expected_generation=%s)",
            target_unit_id,
            expected_gen,
        )
        return now
    return last_log_time


def build_scheduler_state(cfg: Any) -> SimpleNamespace:
    """Build scheduler state container with stable attribute names."""
    if not cfg.HARDWARE_UNITS:
        logger.warning("[Scheduler] No hardware units configured. Falling back to Host CPU.")
        cfg.HARDWARE_UNITS.append({"type": "CPU", "id": "CPU", "name": "Host CPU"})

    hw_pool = queue.Queue()
    for unit_item in cfg.HARDWARE_UNITS:
        hw_pool.put(unit_item)

    accel_limit = len(cfg.HARDWARE_UNITS)
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
    for unit_item in cfg.HARDWARE_UNITS:
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
