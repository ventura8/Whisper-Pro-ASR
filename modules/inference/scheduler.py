"""
Hardware Scheduling and Task Registry for Whisper Pro ASR
"""
import logging
import queue
import threading
import time
import contextlib
import uuid
from modules import config
from modules import utils
from modules import logging_setup
from modules.monitoring import history_manager

logger = logging.getLogger(__name__)


class SchedulerState:  # pylint: disable=too-few-public-methods
    """Encapsulates global scheduler state to avoid global statements."""

    # pylint: disable=too-many-instance-attributes
    def __init__(self):
        self.hw_pool = queue.Queue()
        for unit_item in config.HARDWARE_UNITS:
            self.hw_pool.put(unit_item)

        self.accel_limit = len(config.HARDWARE_UNITS)
        self.model_lock = threading.Semaphore(self.accel_limit)
        self.priority_sequential_lock = threading.Semaphore(self.accel_limit)
        self.priority_lock = threading.Lock()
        self.pause_requested = threading.Event()
        self.pause_confirmed = threading.Event()
        self.resume_event = threading.Event()
        self.resume_event.set()

        self.session_stats = {
            "active": 0,
            "queued": 0,
            "priority": 0
        }

        self.task_registry = {}
        self.task_registry_lock = threading.Lock()

        self.unit_ownership = {}
        self.preemptible_units = set()

        self.engine_flags = {
            "initialized": False,
            "whisper_loaded": False,
            "uvr_loaded": False
        }

    @property
    def active_sessions(self):
        """Returns the number of active sessions."""
        return self.session_stats["active"]

    @active_sessions.setter
    def active_sessions(self, val):
        """Setter for active sessions."""
        self.session_stats["active"] = val

    @property
    def queued_sessions(self):
        """Returns the number of queued sessions."""
        return self.session_stats["queued"]

    @queued_sessions.setter
    def queued_sessions(self, val):
        """Setter for queued sessions."""
        self.session_stats["queued"] = val

    @property
    def priority_requests(self):
        """Returns the number of priority requests."""
        return self.session_stats["priority"]

    @priority_requests.setter
    def priority_requests(self, val):
        """Setter for priority requests."""
        self.session_stats["priority"] = val

    @property
    def engine_initialized(self):
        """Returns engine initialization status."""
        return self.engine_flags["initialized"]

    @engine_initialized.setter
    def engine_initialized(self, val):
        """Setter for engine initialization flag."""
        self.engine_flags["initialized"] = val

    @property
    def whisper_loaded(self):
        """Returns whisper loading status."""
        return self.engine_flags["whisper_loaded"]

    @whisper_loaded.setter
    def whisper_loaded(self, val):
        """Setter for whisper loaded flag."""
        self.engine_flags["whisper_loaded"] = val

    @property
    def uvr_loaded(self):
        """Returns UVR loading status."""
        return self.engine_flags["uvr_loaded"]

    @uvr_loaded.setter
    def uvr_loaded(self, val):
        """Setter for UVR loaded flag."""
        self.engine_flags["uvr_loaded"] = val


STATE = SchedulerState()


def wait_for_priority():
    """Handles priority task synchronization (Request pause from others)."""
    # Mark this thread as a priority thread for the duration of the task
    utils.THREAD_CONTEXT.is_priority = True

    # 1. Hardware-aware enforcement: Limit concurrent priority tasks to available units
    # The lock is now acquired by the caller via early_task_registration context manager
    # to satisfy Pylint's structured 'with' block requirements.

    do_pause = False
    with STATE.priority_lock:
        STATE.priority_requests += 1
        # Only pause others if we are actually at capacity (no free units)
        if STATE.active_sessions > STATE.accel_limit:
            do_pause = True
            logger.info("[Scheduler] Priority request: Pausing other tasks...")
            STATE.pause_requested.set()
            STATE.resume_event.clear()

    if do_pause:
        # Wait for others to confirm they are paused
        logger.debug("[Scheduler] Waiting for preemption confirmation...")
        # If it takes > 30s, we proceed anyway to avoid service deadlock,
        # but in normal operation, tasks will signal via pause_confirmed.
        STATE.pause_confirmed.wait(timeout=30)


def release_priority():
    """Releases priority hold and resumes paused tasks."""
    # Safety: Only proceed if this thread actually holds a priority token
    if not getattr(utils.THREAD_CONTEXT, "is_priority", False):
        return

    # Reset priority flag to prevent double-release
    utils.THREAD_CONTEXT.is_priority = False

    try:
        with STATE.priority_lock:
            STATE.priority_requests = max(0, STATE.priority_requests - 1)
            if STATE.priority_requests == 0:
                logger.info("[Scheduler] Priority released. Active: %d | Queued: %d",
                            STATE.active_sessions, STATE.queued_sessions)
                STATE.pause_requested.clear()
                STATE.resume_event.set()
                STATE.pause_confirmed.clear()
    finally:
        # Note: priority_sequential_lock is now released by the early_task_registration
        # context manager via a 'with' block.
        pass


@contextlib.contextmanager
def early_task_registration(task_type="ASR/LD", stage="Initializing", filename=None, is_priority=False):
    """Registers a task immediately for dashboard visibility."""
    # 1. Hardware-aware enforcement for priority tasks
    # Using 'with' satisfies Pylint R1732 (consider-using-with)
    lock_ctx = STATE.priority_sequential_lock if is_priority else contextlib.nullcontext()

    with lock_ctx:
        thread_id = threading.get_ident()
    task_id = str(uuid.uuid4())
    display_name = filename or getattr(utils.THREAD_CONTEXT, "filename", "Unknown")

    # Determine initial status
    initial_status = "initializing"

    with STATE.task_registry_lock:
        if thread_id not in logging_setup.TASK_LOGS:
            logging_setup.TASK_LOGS[thread_id] = []
        STATE.task_registry[thread_id] = {
            "task_id": task_id,
            "filename": display_name,
            "start_time": time.time(),
            "status": initial_status,
            "progress": 0,
            "stage": stage,
            "type": task_type,
            "video_duration": getattr(utils.THREAD_CONTEXT, "total_duration", 0),
            "caller_info": getattr(utils.THREAD_CONTEXT, "caller_info", {}),
            "request_json": getattr(utils.THREAD_CONTEXT, "request_json", {}),
            "live_text": "",
            "logs": []
        }
    try:
        yield
    finally:
        # Ensure priority is released if this task was a priority task
        # Since initialize_task_context calls wait_for_priority, we should always try to release
        release_priority()

        with STATE.task_registry_lock:
            if thread_id in STATE.task_registry:
                task = STATE.task_registry[thread_id]
                task['logs'] = logging_setup.TASK_LOGS.get(thread_id, [])
                task['status'] = 'completed' if task.get('status') != 'failed' else 'failed'
                task['progress'] = 100
                history_manager.log_completed_task(task.copy())
                del STATE.task_registry[thread_id]
            if thread_id in logging_setup.TASK_LOGS:
                del logging_setup.TASK_LOGS[thread_id]


def cleanup_failed_task():
    """Removes task from registry on early failure."""
    thread_id = threading.get_ident()
    with STATE.task_registry_lock:
        if thread_id in STATE.task_registry:
            del STATE.task_registry[thread_id]
        if thread_id in logging_setup.TASK_LOGS:
            del logging_setup.TASK_LOGS[thread_id]


def update_task_metadata(**kwargs):
    """Updates metadata for the current thread's task."""
    thread_id = threading.get_ident()
    with STATE.task_registry_lock:
        if thread_id in STATE.task_registry:
            STATE.task_registry[thread_id].update(kwargs)
            if 'live_text' in kwargs:
                logger.debug("[Scheduler] Updated live_text for task %s",
                             STATE.task_registry[thread_id].get('task_id'))
        else:
            logger.warning("[Scheduler] Attempted to update metadata for unknown thread %s. Keys: %s",
                           thread_id, list(kwargs.keys()))


def update_task_progress(progress, stage=None):
    """Updates progress percentage and stage."""
    thread_id = threading.get_ident()
    with STATE.task_registry_lock:
        if thread_id in STATE.task_registry:
            STATE.task_registry[thread_id]["progress"] = progress
            if stage:
                STATE.task_registry[thread_id]["stage"] = stage


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
            # For now, any preemptible unit works as they all have Whisper loaded
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


def get_service_stats_minimal():
    """Lightweight status check for circular-safe metrics discovery."""
    with STATE.task_registry_lock:
        active = []
        for task in STATE.task_registry.values():
            if task.get('status') == 'active':
                active.append({
                    "unit_type": task.get('unit_type'),
                    "unit_name": task.get('unit_name', ''),
                    "unit_id": task.get('unit_id')
                })
        return {"active_tasks": active}


def is_engine_initialized():
    """Checks if any models are loaded (Placeholder, will be set by model_manager)."""
    return STATE.engine_initialized


def is_uvr_loaded():
    """Checks if UVR is loaded (Placeholder, will be set by model_manager)."""
    return STATE.uvr_loaded
