"""
Telemetry and Statistics Collection for Whisper Pro ASR
"""
import logging
import threading
import time
from modules import config, utils, logging_setup
from modules.inference import scheduler, model_manager
from modules.monitoring import history_manager, metrics_discovery

logger = logging.getLogger(__name__)
SERVICE_START_TIME = time.time()
_STOP_EVENT = threading.Event()
TELEMETRY_HISTORY = []
_TELEMETRY_LOCK = threading.Lock()


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
                TELEMETRY_HISTORY.append({
                    "timestamp": time.time(),
                    "system": metrics,
                    "telemetry": {
                        "nvidia": metrics_discovery.get_nvidia_metrics(),
                        "intel_gpu_load": metrics_discovery.get_intel_gpu_load(),
                        "npu_load": metrics_discovery.get_npu_load()
                    }
                })
                if len(TELEMETRY_HISTORY) > max_points:
                    TELEMETRY_HISTORY.pop(0)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.debug("[Telemetry] Worker cycle failed: %s", e)
        time.sleep(2)


def get_service_stats():
    """Consolidates service state for the dashboard."""
    with scheduler.STATE.task_registry_lock:
        tasks = []
        for tid, task in scheduler.STATE.task_registry.items():
            task_copy = task.copy()
            task_copy['logs'] = logging_setup.TASK_LOGS.get(tid, [])
            tasks.append(task_copy)

    history, history_stats = history_manager.get_history_stats()

    # Engine status helpers
    whisper_active = any(t.get('status') == 'active' and
                         any(s in t.get('stage', '').lower() for s in ['transcrib', 'inference'])
                         for t in tasks)
    whisper_status = "busy" if whisper_active else (
        "loaded" if model_manager.is_engine_actually_loaded() else "ready")

    uvr_active = any(t.get('status') == 'active' and any(s in t.get('stage', '').lower()
                     for s in ['isolation', 'separation', 'uvr']) for t in tasks)
    uvr_status = "busy" if uvr_active else (
        "loaded" if model_manager.is_uvr_actually_loaded() else "ready")

    with _TELEMETRY_LOCK:
        telemetry_snap = TELEMETRY_HISTORY[:]

    # Precise counters based on task registry status
    actual_active = sum(1 for t in tasks if t.get('status') in ['active', 'initializing'])
    actual_queued = sum(1 for t in tasks if t.get('status') == 'queued')

    return {
        "version": config.VERSION,
        "uptime_sec": time.time() - SERVICE_START_TIME,
        "scheduler": {
            "active": actual_active,
            "queued": actual_queued
        },
        "active_sessions": actual_active,
        "queued_sessions": actual_queued,
        "tasks": tasks,
        "telemetry_history": telemetry_snap,
        "hardware_units": config.HARDWARE_UNITS,
        "history": history,
        "history_stats": history_stats,
        "telemetry": {
            "nvidia": metrics_discovery.get_nvidia_metrics(),
            "intel_gpu_load": metrics_discovery.get_intel_gpu_load(),
            "npu_load": metrics_discovery.get_npu_load()
        },
        "engines": {
            "whisper": {
                "status": whisper_status,
                "model": utils.get_pretty_model_name(config.MODEL_ID),
                "device": config.DEVICE,
                "compute_type": config.COMPUTE_TYPE
            },
            "uvr": {
                "status": uvr_status,
                "model": utils.get_pretty_model_name(config.VOCAL_SEPARATION_MODEL)
            }
        }
    }


def get_minimal_stats():
    """Fast health check stats."""
    with scheduler.STATE.task_registry_lock:
        active = sum(1 for t in scheduler.STATE.task_registry.values()
                     if t.get('status') in ['active', 'initializing'])
        queued = sum(1 for t in scheduler.STATE.task_registry.values()
                     if t.get('status') == 'queued')

    return {
        "status": "healthy",
        "active": active,
        "queued": queued
    }


# Alias for backward compatibility with tests
get_summary = get_service_stats
