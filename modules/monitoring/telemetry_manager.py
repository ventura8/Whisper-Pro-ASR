"""
Persistent Telemetry History Manager
Records system resource utilization over time.
"""

import json
import logging
import os
import tempfile
import threading
import time
from typing import Any

from modules.core import config
from modules.monitoring.io_utils import load_json_list_file

logger = logging.getLogger(__name__)
_FILE_LOCK: threading.Lock = threading.Lock()
TELEMETRY_FILE: str = os.path.join(config.STATE_DIR, "telemetry_history.json")


def get_telemetry_history() -> list[dict[str, Any]]:
    """Retrieves the list of recorded telemetry snapshots."""
    loaded = _load_telemetry_entries()
    if loaded is None:
        return []
    return [entry for entry in loaded if _is_valid_telemetry_entry(entry)]


def _load_telemetry_entries() -> list[Any] | None:
    if not os.path.exists(TELEMETRY_FILE):
        return None
    return load_json_list_file(TELEMETRY_FILE)


def _is_valid_telemetry_entry(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    return isinstance(entry.get("timestamp"), (int, float))


def record_snapshot(stats: dict[str, Any]) -> None:
    """
    Appends a new resource snapshot to the history and prunes old data.
    'stats' should contain 'system' and 'telemetry' from model_manager.get_service_stats()
    """
    os.makedirs(config.STATE_DIR, exist_ok=True)
    with _FILE_LOCK:
        try:
            history = get_telemetry_history()
            history.append(_build_snapshot(stats))
            history = _prune_history(history)
            _atomic_write_history(history)
        except (OSError, TypeError, ValueError) as e:
            logger.exception("Failed to record telemetry atomically: %s", e)
            raise


def _build_snapshot(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp": int(time.time()),
        "cpu_sys": stats["system"]["cpu_percent"],
        "cpu_app": stats["system"]["app_cpu_percent"],
        "mem_sys": stats["system"]["memory_percent"],
        "mem_sys_gb": stats["system"].get("memory_used_gb", 0.0),
        "mem_app_gb": stats["system"]["app_memory_gb"],
        "nvidia_util": [g["util"] for g in stats["telemetry"].get("nvidia", [])],
        "intel_util": stats["telemetry"].get("intel_gpu_load", 0),
        "npu_util": stats["telemetry"].get("npu_load", 0),
        "hardware_util": stats["telemetry"].get("hardware_util", {}),
    }


def _prune_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    retention_hours = int(os.environ.get("TELEMETRY_RETENTION_HOURS", 24))
    cutoff = int(time.time()) - (retention_hours * 3600)
    kept = [s for s in history if s["timestamp"] > cutoff]
    return kept[-2000:] if len(kept) > 2000 else kept


def _atomic_write_history(history: list[dict[str, Any]]) -> None:
    fd, temp_path = tempfile.mkstemp(dir=config.STATE_DIR, prefix="telemetry_", suffix=".json")
    t_path: str | None = temp_path
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            json.dump(history, tmp_file)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(temp_path, TELEMETRY_FILE)
        t_path = None
    finally:
        _cleanup_temp_file(t_path)


def _cleanup_temp_file(temp_path: str | None) -> None:
    if temp_path and os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except OSError as cleanup_error:
            logger.warning("Failed to cleanup telemetry temp file %s: %s", temp_path, cleanup_error)


def update_retention(telemetry_hours: int | None = None, log_days: int | None = None) -> None:
    """Updates retention periods in the environment."""
    if telemetry_hours is not None:
        os.environ["TELEMETRY_RETENTION_HOURS"] = str(telemetry_hours)
    if log_days is not None:
        os.environ["LOG_RETENTION_DAYS"] = str(log_days)


def clear_telemetry_history() -> None:
    """Purge all recorded telemetry snapshots."""
    with _FILE_LOCK:
        try:
            if os.path.exists(TELEMETRY_FILE):
                os.remove(TELEMETRY_FILE)
        except OSError as e:
            logger.exception("Failed to clear telemetry history: %s", e)
            raise
