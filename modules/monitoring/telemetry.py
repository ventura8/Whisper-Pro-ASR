"""
Telemetry and Statistics Collection for Whisper Pro ASR
"""

import logging
import threading
import time

from modules.core import config, logging_setup, utils
from modules.inference import model_manager, scheduler
from modules.monitoring import history_manager, metrics_discovery

logger = logging.getLogger(__name__)
SERVICE_START_TIME = time.time()
_STOP_EVENT = threading.Event()
TELEMETRY_HISTORY = []
_TELEMETRY_LOCK = threading.Lock()

_DISPLAYABLE_STATUSES = {
    "initializing",
    "queued",
    "active",
    "post-processing",
    "completed",
    "failed",
}


def _normalize_status_value(status):
    """Return a dashboard-safe status that never uses placeholder values."""
    status_key = str(status or "").strip().lower()
    if status_key in _DISPLAYABLE_STATUSES:
        return status_key
    return "initializing"


def _is_placeholder_stage(stage_text):
    """Return True when stage text is missing or looks like placeholder content."""
    normalized = str(stage_text or "").strip().lower()
    if not normalized:
        return True

    # Canonical empty/sentinel spellings.
    if normalized in {"none", "null", "undefined", "unknown", "na", "n/a"}:
        return True

    # Ratio-like placeholder e.g. (0/0), 0/0
    ratio_candidate = normalized.replace("(", "").replace(")", "").replace(" ", "")
    if ratio_candidate == "0/0":
        return True

    if "placeholder" in normalized:
        return True

    # Generic resume placeholders without blocking valid phrases like "Resumed Inference".
    if normalized in {"resume", "resuming"}:
        return True

    return False


def _default_stage_for_status(status):
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


def _normalize_stage_value(stage, status):
    """Ensure stage is always a concrete, non-placeholder dashboard label."""
    if stage is not None:
        normalized = str(stage).strip()
        if not _is_placeholder_stage(normalized):
            return normalized
    return _default_stage_for_status(status)


def _is_whisper_active_stage(stage_text):
    """Return True when a stage indicates Whisper is still doing ASR work."""
    normalized = str(stage_text or "").lower()
    return any(token in normalized for token in ("transcrib", "inference", "translat"))


def start_telemetry_loop():
    """Spawns the background telemetry collection thread."""
    thread = threading.Thread(target=_telemetry_worker, daemon=True)
    thread.start()
    return _STOP_EVENT


def _telemetry_worker():
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
                    TELEMETRY_HISTORY.pop(0)
        except (OSError, ValueError, AttributeError, KeyError, TypeError, RuntimeError) as e:
            logger.debug("[Telemetry] Worker cycle failed: %s", e)
        time.sleep(2)


def get_service_stats():
    """Consolidates service state for the dashboard."""
    with scheduler.STATE.task_registry_lock:
        tasks = []
        for tid, task in scheduler.STATE.task_registry.items():
            task_copy = task.copy()
            task_copy["status"] = _normalize_status_value(task_copy.get("status"))
            task_copy["stage"] = _normalize_stage_value(task_copy.get("stage"), task_copy.get("status"))
            task_copy["logs"] = logging_setup.TASK_LOGS.get(tid, [])
            tasks.append(task_copy)

    # Enforce deterministic ordering per task_status_display_specification_skill:
    # 1. Active tasks first (all status='active'), sorted by start_time ascending, tie-break by task_id
    # 2. Priority queued tasks (status='queued' with is_priority=true), sorted by start_time ascending
    # 3. Standard queued tasks (status='queued' with is_priority=false), sorted by start_time ascending
    # This three-tier ordering ensures the dashboard displays tasks with correct precedence visibility.
    def sort_key(t):
        status = t.get("status", "unknown")
        start_time = float(t.get("start_time", 0.0) or 0.0)
        task_id = str(t.get("task_id", ""))
        is_priority = bool(t.get("is_priority", False))

        # Tier 1: active (status_tier=0)
        if status == "active":
            return (0, start_time, task_id)
        # Tier 2: priority queued (status_tier=1)
        if status == "queued" and is_priority:
            return (1, start_time, task_id)
        # Tier 3: standard queued (status_tier=2)
        if status == "queued" and not is_priority:
            return (2, start_time, task_id)
        # Other statuses (completed, failed, unknown, etc.) go last
        return (3, start_time, task_id)

    tasks.sort(key=sort_key)

    history_stats = history_manager.get_history_stats()

    # Engine status helpers
    # Whisper is 'busy' only when a task is actively in an ASR sub-stage.
    # UVR/vocal-separation stays separate, so whisper correctly shows 'loaded' there.
    whisper_active = any(t.get("status") == "active" and _is_whisper_active_stage(t.get("stage")) for t in tasks)
    whisper_status = "busy" if whisper_active else ("loaded" if model_manager.is_engine_actually_loaded() else "ready")

    uvr_active = any(
        t.get("status") == "active" and any(s in t.get("stage", "").lower() for s in ["isolation", "separation", "uvr"])
        for t in tasks
    )
    uvr_status = "busy" if uvr_active else ("loaded" if model_manager.is_uvr_actually_loaded() else "ready")

    with _TELEMETRY_LOCK:
        telemetry_snap = TELEMETRY_HISTORY[:]

    latest_telemetry = (
        telemetry_snap[-1].get("telemetry", {})
        if telemetry_snap
        else {
            "nvidia": [],
            "intel_gpu_load": 0,
            "npu_load": 0,
            "hardware_util": {},
        }
    )

    # Downsample telemetry history to a maximum of 300 points for dashboard rendering performance
    if len(telemetry_snap) > 300:
        telemetry_snap = [telemetry_snap[int(i * len(telemetry_snap) / 299.0)] for i in range(299)] + [
            telemetry_snap[-1]
        ]

    # Precise counters based on task registry status
    actual_active = sum(1 for t in tasks if t.get("status") in ["active", "initializing"])
    actual_queued = sum(1 for t in tasks if t.get("status") == "queued")

    # Dynamic hardware units status resolution
    hw_units_with_status = []
    for u in config.HARDWARE_UNITS:
        u_copy = u.copy()
        unit_id = u["id"]

        # Whisper is 'busy' on this unit only when the task is in the transcription/inference sub-stage.
        whisper_unit_active = any(
            t.get("status") == "active"
            and str(t.get("unit_id")) == str(unit_id)
            and _is_whisper_active_stage(t.get("stage"))
            for t in tasks
        )

        uvr_unit_active = any(
            t.get("status") == "active"
            and str(t.get("unit_id")) == str(unit_id)
            and any(s in t.get("stage", "").lower() for s in ["isolation", "separation", "uvr"])
            for t in tasks
        )

        if whisper_unit_active:
            u_copy["whisper_status"] = "busy"
        elif unit_id in model_manager.MODEL_POOL:
            u_copy["whisper_status"] = "loaded"
        else:
            u_copy["whisper_status"] = "ready"

        if uvr_unit_active:
            u_copy["uvr_status"] = "busy"
        elif (
            unit_id in model_manager.PREPROCESSOR_POOL
            and model_manager.PREPROCESSOR_POOL[unit_id].separator is not None
        ):
            u_copy["uvr_status"] = "loaded"
        else:
            u_copy["uvr_status"] = "ready"

        hw_units_with_status.append(u_copy)

    return {
        "version": config.VERSION,
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


def get_minimal_stats():
    """Fast health check stats."""
    with scheduler.STATE.task_registry_lock:
        active = sum(1 for t in scheduler.STATE.task_registry.values() if t.get("status") in ["active", "initializing"])
        queued = sum(1 for t in scheduler.STATE.task_registry.values() if t.get("status") == "queued")

    return {"status": "healthy", "active": active, "queued": queued}


# Alias for backward compatibility with tests
get_summary = get_service_stats
