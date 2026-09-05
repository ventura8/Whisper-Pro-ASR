"""
Telemetry and Statistics Collection for Whisper Pro ASR
"""

import logging
import math
import threading
import time
from collections import deque
from typing import Any

from modules.core import config, logging_setup, model_provisioning, utils
from modules.inference import scheduler
from modules.inference.runtime import model_manager
from modules.monitoring import history_manager, metrics_discovery

logger = logging.getLogger(__name__)
SERVICE_START_TIME: float = time.time()
_STOP_EVENT: threading.Event = threading.Event()
# deque, not list: at the high end of the exposed retention range (720h/30 days,
# sampled every 2s = up to 1,296,000 entries), list.pop(0) is O(n) per trim --
# deque.popleft() is O(1). maxlen isn't fixed here because retention_hours (and
# so max_points) is read fresh from config each time _telemetry_worker() starts.
TELEMETRY_HISTORY: deque[dict[str, Any]] = deque()
_TELEMETRY_LOCK: threading.Lock = threading.Lock()

_DISPLAYABLE_STATUSES: set[str] = {
    "initializing",
    "queued",
    "active",
    "post-processing",
    "completed",
    "failed",
}

# Track which task IDs have already emitted a stale-task warning so repeated
# dashboard polls do not produce duplicate log entries for the same orphan.
_STALE_TASK_WARNED: set[str] = set()


def _clear_stale_task_warned() -> None:
    """Clear the stale-task deduplication set. Intended for test teardown."""
    _STALE_TASK_WARNED.clear()


def _normalize_status_value(status: Any) -> str:
    """Return a dashboard-safe status that never uses placeholder values."""
    status_key = str(status or "").strip().lower()
    if status_key in _DISPLAYABLE_STATUSES:
        return status_key
    return "initializing"


def _is_placeholder_stage(stage_text: Any) -> bool:
    """Return True when stage text is missing or looks like placeholder content."""
    normalized = str(stage_text or "").strip().lower()
    if not normalized:
        return True
    return _is_placeholder_token(normalized)


def _is_placeholder_token(normalized: str) -> bool:
    return (
        _is_sentinel_stage(normalized)
        or _is_ratio_placeholder(normalized)
        or "placeholder" in normalized
        or normalized in {"resume", "resuming"}
    )


def _is_sentinel_stage(normalized: str) -> bool:
    return normalized in {"none", "null", "undefined", "unknown", "na", "n/a"}


def _is_ratio_placeholder(normalized: str) -> bool:
    ratio_candidate = normalized.replace("(", "").replace(")", "").replace(" ", "")
    return ratio_candidate == "0/0"


def _default_stage_for_status(status: Any) -> str:
    """Return a deterministic dashboard stage label from task status."""
    status_key = _normalize_status_value(status)
    mapping = {
        "initializing": "Initializing",
        "queued": "Queued",
        "active": "Active",
        "post-processing": "Post-Processing",
        "completed": "Completed",
        "failed": "Failed",
    }
    return mapping.get(status_key, "Initializing")


def _normalize_stage_value(stage: Any, status: Any) -> str:
    """Ensure stage is always a concrete, non-placeholder dashboard label."""
    if stage is not None:
        normalized = str(stage).strip()
        if not _is_placeholder_stage(normalized):
            return normalized
    return _default_stage_for_status(status)


#: Substrings of the stage strings that mean the ASR engine is still holding its hardware
#: unit. "detect" and the diarization stages belong here as much as transcription does:
#: language detection runs the Whisper model, and diarization runs inside the same claimed
#: unit before the task releases it. Omitting them made /status report the engine as
#: "loaded" (idle) for the whole of a diarizing request, so the dashboard showed an idle
#: engine while the model lock was held and later requests queued behind it.
#: Matched as substrings, so "Detection" and "Language Detection" both hit "detect", and
#: "Diarizing Speakers" / "Loading Diarization Model" both hit "diariz".
_WHISPER_ACTIVE_STAGE_TOKENS = (
    "transcrib",
    "inference",
    "translat",
    "detect",
    "diariz",
    "assigning speakers",
    "aligning",
    "alignment",
)


def _is_whisper_active_stage(stage_text: Any) -> bool:
    """Return True when a stage indicates Whisper is still doing ASR work."""
    normalized = str(stage_text or "").lower()
    return any(token in normalized for token in _WHISPER_ACTIVE_STAGE_TOKENS)


def start_telemetry_loop() -> threading.Event:
    """Spawns the background telemetry collection thread."""
    thread = threading.Thread(target=_telemetry_worker, daemon=True)
    thread.start()
    return _STOP_EVENT


def _telemetry_worker() -> None:
    """Background worker for system metrics."""
    retention_hours = int(config.TELEMETRY_RETENTION_HOURS)
    max_points = (retention_hours * 3600) // 2

    while not _STOP_EVENT.is_set():
        try:
            metrics = utils.get_system_telemetry()
            with _TELEMETRY_LOCK:
                TELEMETRY_HISTORY.append(
                    {
                        "timestamp": time.time(),
                        "system": metrics,
                        "telemetry": {
                            "nvidia": metrics_discovery.get_nvidia_metrics(),
                            "intel_gpu_load": metrics_discovery.get_intel_gpu_load(),
                            "npu_load": metrics_discovery.get_npu_load(),
                            "hardware_util": metrics_discovery.get_all_hardware_utilization(),
                        },
                    }
                )
                if len(TELEMETRY_HISTORY) > max_points:
                    TELEMETRY_HISTORY.popleft()
        except (OSError, ValueError, AttributeError, KeyError, TypeError, RuntimeError) as e:
            logger.debug("[Telemetry] Worker cycle failed: %s", e)
        time.sleep(2)


def get_service_stats() -> dict[str, Any]:
    """Consolidates service state for the dashboard."""
    tasks = _get_dashboard_tasks_snapshot()
    tasks.sort(key=_task_sort_key)
    history_stats = history_manager.get_history_stats()
    whisper_status, uvr_status = _resolve_engine_statuses(tasks)
    telemetry_snap = _get_telemetry_snapshot()
    latest_telemetry = _get_latest_telemetry(telemetry_snap)
    actual_active, actual_queued = _count_task_statuses(tasks)
    hw_units_with_status = _build_hardware_unit_statuses(tasks)

    return {
        "version": config.VERSION,
        "edition": config.IMAGE_EDITION,
        "version_display": config.VERSION_DISPLAY,
        "uptime_sec": time.time() - SERVICE_START_TIME,
        "scheduler": {"active": actual_active, "queued": actual_queued},
        "active_sessions": actual_active,
        "queued_sessions": actual_queued,
        "tasks": tasks,
        "telemetry_history": telemetry_snap,
        "hardware_units": hw_units_with_status,
        "history": history_stats[0],
        "history_stats": history_stats[1],
        "telemetry": latest_telemetry,
        "engines": {
            "whisper": {
                "status": whisper_status,
                "model": utils.get_pretty_model_name(config.MODEL_ID),
                "device": config.DEVICE,
                "compute_type": config.COMPUTE_TYPE,
            },
            "uvr": {"status": uvr_status, "model": utils.get_pretty_model_name(config.VOCAL_SEPARATION_MODEL)},
        },
    }


# Safety net so a crashed/killed worker doesn't leave a task registry entry stuck
# reporting "active"/"running" forever on the dashboard. Normal task completion always
# removes the entry via scheduler._finalize_registered_task, so any "active" entry that
# outlives this window with no owning worker is treated as a stale ghost for display.
_STALE_ACTIVE_TASK_TIMEOUT_SEC = 6 * 3600
# Below this, a start_active/start_time value is assumed to be a relative/synthetic
# offset (e.g. in tests) rather than a real wall-clock epoch timestamp, so staleness
# is not evaluated against it. Mirrors the epoch-vs-relative guard used client-side in
# active_tasks.js (_areComparableTimestamps).
_EPOCH_TIMESTAMP_THRESHOLD = 100_000_000


def _resolve_epoch_timestamp(value: Any) -> float | None:
    """Return `value` as a float epoch timestamp, or None if it is missing,
    non-numeric, non-finite (NaN/inf), or too small to be a real wall-clock epoch
    (i.e. it looks like a relative/synthetic offset rather than an actual epoch
    timestamp)."""
    if not value:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    if value < _EPOCH_TIMESTAMP_THRESHOLD:
        return None
    return value


def _resolve_epoch_start_active(task: dict[str, Any], now: float) -> float | None:
    """Return the task's start timestamp as a float epoch value, or None if
    neither field resolves to one. `start_active` and `start_time` are resolved
    independently (not via `or`), so an invalid/non-finite `start_active` cannot
    mask an otherwise-valid `start_time`. A value later than `now` (clock skew,
    a bad write) is rejected the same way `_resolve_last_liveness_signal` rejects
    a future `last_progress_at`, since it would otherwise make `now - last_signal`
    negative and permanently hide staleness."""
    resolved = _resolve_epoch_timestamp(task.get("start_active"))
    if resolved is not None and resolved <= now:
        return resolved
    resolved = _resolve_epoch_timestamp(task.get("start_time"))
    if resolved is not None and resolved <= now:
        return resolved
    return None


def _resolve_last_liveness_signal(task: dict[str, Any], now: float) -> float | None:
    """Return the most recent evidence the owning worker is still alive: the last
    progress/stage update (`last_progress_at`, written on every `update_task_progress`
    call) if present, otherwise the task's start timestamp. A task whose worker keeps
    making progress keeps refreshing this value and is never considered stale,
    regardless of total elapsed run time. A `last_progress_at` later than `now`
    (clock skew, a bad write) is rejected rather than trusted, since it would
    otherwise make `now - last_signal` negative and permanently hide staleness;
    the task's start timestamp is used instead in that case."""
    last_progress_at = _resolve_epoch_timestamp(task.get("last_progress_at"))
    if last_progress_at is not None and last_progress_at <= now:
        return last_progress_at
    return _resolve_epoch_start_active(task, now)


def _is_stale_active_task(task: dict[str, Any], now: float) -> bool:
    """Return True when a task has been reported 'active' with no liveness signal
    (no progress/stage update, or none ever recorded past start) for far longer than
    any real run should go silent -- i.e. it looks orphaned, not merely long-running."""
    if task.get("status") != "active":
        return False
    last_signal = _resolve_last_liveness_signal(task, now)
    if last_signal is None:
        return False
    return (now - last_signal) > _STALE_ACTIVE_TASK_TIMEOUT_SEC


def _mark_task_copy_stale(task_copy: dict[str, Any], tid: Any) -> None:
    """Relabel a display copy as stale and emit a one-time dedup warning for it."""
    task_copy["status"] = "failed"
    task_copy["stage"] = "Stale (worker did not report completion)"
    task_id_str = str(tid)
    if task_id_str not in _STALE_TASK_WARNED:
        _STALE_TASK_WARNED.add(task_id_str)
        logger.warning(
            "[Telemetry] Stale active task detected (no liveness signal for > %ss): task_id=%s",
            _STALE_ACTIVE_TASK_TIMEOUT_SEC,
            task_id_str,
        )


def _build_task_copy(tid: Any, task: dict[str, Any], now: float, stale_task_ids: list[Any]) -> dict[str, Any]:
    """Build one task's display copy, relabeling it (and recording it in
    `stale_task_ids` for later revalidation/finalization) if it's stale. Does NOT
    mutate the live registry entry -- that only happens in `_finalize_one_stale_task`,
    under a fresh revalidation, to avoid a TOCTOU race where a heartbeat arriving
    between this snapshot and finalization would otherwise be silently overridden."""
    task_copy = task.copy()
    if _is_stale_active_task(task_copy, now):
        stale_task_ids.append(tid)
        _mark_task_copy_stale(task_copy, tid)
    else:
        task_copy["status"] = _normalize_status_value(task_copy.get("status"))
        task_copy["stage"] = _normalize_stage_value(task_copy.get("stage"), task_copy.get("status"))
    task_copy["logs"] = logging_setup.TASK_LOGS.get(tid, [])
    return task_copy


def _finalize_one_stale_task(tid: Any) -> None:
    """Revalidate, mark, and finalize a single stale task atomically. Reacquires
    task_registry_lock and re-runs the staleness check fresh (not the earlier
    snapshot) immediately before marking the live entry failed and handing it to
    the scheduler's archive/remove lifecycle -- this closes the race window
    between the initial staleness snapshot (in _build_task_copy) and finalization:
    if the owning worker reported a fresh heartbeat in between, the task is
    genuinely alive and must not be archived/removed. If it's still stale, marking
    the live entry failed happens under the same lock acquisition as the
    revalidation, so `_archive_registry_task` (invoked by scheduler.finalize_stale_task
    right after, outside this lock) always sees the correct status."""
    with scheduler.STATE.task_registry_lock:
        task = scheduler.STATE.task_registry.get(tid)
        if task is None:
            return  # Already finalized concurrently (e.g. the owning worker's own path).
        if not _is_stale_active_task(task, time.time()):
            return  # A fresh liveness signal arrived since the snapshot -- not stale anymore.
        task["status"] = "failed"
        task["stage"] = "Stale (worker did not report completion)"

    scheduler.finalize_stale_task(tid)
    task_id_str = str(tid)
    if task_id_str in _STALE_TASK_WARNED and tid not in scheduler.STATE.task_registry:
        _STALE_TASK_WARNED.discard(task_id_str)


def _finalize_stale_tasks(stale_task_ids: list[Any]) -> None:
    """Finalize each confirmed-stale task through the scheduler's own lifecycle (the
    same archive/remove path normal completion uses), so a ghost entry doesn't just
    look reaped in the display copy while staying "active" forever in
    scheduler.STATE.task_registry itself (e.g. get_minimal_stats() counts
    active_sessions directly off the live registry, bypassing the display-layer
    logic entirely, and would otherwise keep counting the ghost). Idempotent: if the
    "crashed" worker is actually still alive and eventually reaches its own
    finally-block finalize call, that call no-ops on an already-removed task_id
    rather than erroring. Must be called after the outer task_registry_lock (held
    while building display copies) is released."""
    for tid in stale_task_ids:
        _finalize_one_stale_task(tid)


def _get_dashboard_tasks_snapshot() -> list[dict[str, Any]]:
    now = time.time()
    stale_task_ids: list[Any] = []
    with scheduler.STATE.task_registry_lock:
        tasks = [_build_task_copy(tid, task, now, stale_task_ids) for tid, task in scheduler.STATE.task_registry.items()]

    _finalize_stale_tasks(stale_task_ids)
    return tasks


def _task_sort_key(task: dict[str, Any]) -> tuple[int, float, str]:
    status = task.get("status", "unknown")
    start_time = float(task.get("start_time", 0.0) or 0.0)
    task_id = str(task.get("task_id", ""))
    tier = _task_status_tier(status)
    return (tier, start_time, task_id)


def _task_status_tier(status: str) -> int:
    if status == "active":
        return 0
    return 1


def _is_uvr_active_stage(stage_text: Any) -> bool:
    return any(token in str(stage_text or "").lower() for token in ["isolation", "separation", "uvr"])


def _resolve_engine_statuses(tasks: list[dict[str, Any]]) -> tuple[str, str]:
    return _resolve_whisper_status(tasks), _resolve_uvr_status(tasks)


def _has_active_whisper_task(tasks: list[dict[str, Any]]) -> bool:
    return any(t.get("status") == "active" and _is_whisper_active_stage(t.get("stage")) for t in tasks)


def _resolve_whisper_status(tasks: list[dict[str, Any]]) -> str:
    if model_provisioning.should_gate_tasks():
        return "downloading"
    if _has_active_whisper_task(tasks):
        return "busy"
    return "loaded" if model_manager.is_engine_actually_loaded() else "ready"


def _resolve_uvr_status(tasks: list[dict[str, Any]]) -> str:
    if any(t.get("status") == "active" and _is_uvr_active_stage(t.get("stage")) for t in tasks):
        return "busy"
    return "loaded" if model_manager.is_uvr_actually_loaded() else "ready"


def _get_telemetry_snapshot() -> list[dict[str, Any]]:
    with _TELEMETRY_LOCK:
        return _downsample_telemetry(TELEMETRY_HISTORY)


def _get_latest_telemetry(telemetry_snap: list[dict[str, Any]]) -> dict[str, Any]:
    if telemetry_snap:
        return telemetry_snap[-1].get("telemetry", {})
    return {
        "nvidia": [],
        "intel_gpu_load": 0,
        "npu_load": 0,
        "hardware_util": {},
    }


def _downsample_telemetry(telemetry_snap: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(telemetry_snap) <= 300:
        return telemetry_snap
    sampled = [telemetry_snap[int(i * len(telemetry_snap) / 299.0)] for i in range(299)]
    sampled.append(telemetry_snap[-1])
    return sampled


def _count_task_statuses(tasks: list[dict[str, Any]]) -> tuple[int, int]:
    active = sum(1 for t in tasks if t.get("status") in ["active", "initializing"])
    queued = sum(1 for t in tasks if t.get("status") == "queued")
    return active, queued


def _build_hardware_unit_statuses(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    units = []
    for unit in config.HARDWARE_UNITS:
        unit_copy = unit.copy()
        unit_id = unit["id"]
        unit_copy["whisper_status"] = _resolve_whisper_unit_status(tasks, unit_id)
        unit_copy["uvr_status"] = _resolve_uvr_unit_status(tasks, unit_id)
        units.append(unit_copy)
    return units


def _resolve_whisper_unit_status(tasks: list[dict[str, Any]], unit_id: Any) -> str:
    if _is_whisper_unit_active(tasks, unit_id):
        return "busy"
    if _is_whisper_model_loaded(unit_id):
        return "loaded"
    return "ready"


def _is_whisper_unit_active(tasks: list[dict[str, Any]], unit_id: Any) -> bool:
    return any(
        t.get("status") == "active" and str(t.get("unit_id")) == str(unit_id) and _is_whisper_active_stage(t.get("stage")) for t in tasks
    )


def _is_whisper_model_loaded(unit_id: Any) -> bool:
    return bool(unit_id in model_manager.MODEL_POOL)


def _resolve_uvr_unit_status(tasks: list[dict[str, Any]], unit_id: Any) -> str:
    if _is_uvr_unit_active(tasks, unit_id):
        return "busy"
    if _is_uvr_model_loaded(unit_id):
        return "loaded"
    return "ready"


def _is_uvr_unit_active(tasks: list[dict[str, Any]], unit_id: Any) -> bool:
    return any(
        t.get("status") == "active" and str(t.get("unit_id")) == str(unit_id) and _is_uvr_active_stage(t.get("stage")) for t in tasks
    )


def _is_uvr_model_loaded(unit_id: Any) -> bool:
    return bool(unit_id in model_manager.PREPROCESSOR_POOL and model_manager.PREPROCESSOR_POOL[unit_id].separator is not None)


def get_minimal_stats() -> dict[str, Any]:
    """Fast health check stats."""
    with scheduler.STATE.task_registry_lock:
        active = sum(1 for t in scheduler.STATE.task_registry.values() if t.get("status") in ["active", "initializing"])
        queued = sum(1 for t in scheduler.STATE.task_registry.values() if t.get("status") == "queued")

    return {"status": "healthy", "active": active, "queued": queued}


# Alias for backward compatibility with tests
get_summary = get_service_stats
