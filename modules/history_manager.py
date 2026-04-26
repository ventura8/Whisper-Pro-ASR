"""
Persistent Task History Manager
"""
import os
import json
import time
from datetime import datetime
from . import config

HISTORY_FILE = os.path.join(config.STATE_DIR, "task_history.json")
MAX_HISTORY = 1000

# --- [DEFERRED PERSISTENCE ENGINE] ---
_HISTORY_CACHE = []
_UNSAVED_COUNT = 0
_LAST_SYNC = time.time()
PERSISTENT_FILE = os.path.join(config.OV_CACHE_DIR, "task_history.json")


def _ensure_loaded():
    """Lazy load history from SSD into RAM cache."""
    global _HISTORY_CACHE
    if not _HISTORY_CACHE:
        if os.path.exists(PERSISTENT_FILE):
            try:
                with open(PERSISTENT_FILE, 'r', encoding='utf-8') as f:
                    _HISTORY_CACHE = json.load(f)
            except Exception:
                _HISTORY_CACHE = []
        elif os.path.exists(HISTORY_FILE):
            # Fallback for migration from tmpfs
            try:
                with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                    _HISTORY_CACHE = json.load(f)
            except Exception:
                _HISTORY_CACHE = []


def log_completed_task(task_data):
    """Appends a task to RAM cache and periodically flushes to SSD."""
    global _HISTORY_CACHE
    try:
        _ensure_loaded()

        # Add completion metadata
        task_data["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        task_data["total_elapsed_sec"] = round(
            time.time() - task_data.get("start_time", time.time()), 2)
        if "logs" in task_data:
            task_data["log_count"] = len(task_data["logs"])

        _HISTORY_CACHE.insert(0, task_data)
        _HISTORY_CACHE = _HISTORY_CACHE[:MAX_HISTORY]

        # Immediate persistence as requested by user
        flush_history()

    except Exception as e:
        print(f"Failed to log task history: {e}")


def flush_history():
    """Synchronizes the RAM cache to the physical SSD."""
    global _UNSAVED_COUNT, _LAST_SYNC
    try:
        os.makedirs(config.OV_CACHE_DIR, exist_ok=True)
        with open(PERSISTENT_FILE, 'w', encoding='utf-8') as f:
            json.dump(_HISTORY_CACHE, f, indent=2)
        _UNSAVED_COUNT = 0
        _LAST_SYNC = time.time()
    except Exception as e:
        print(f"SSD Sync Failed: {e}")


def get_history():
    """Retrieves history from RAM cache (fast)."""
    _ensure_loaded()
    return _HISTORY_CACHE


def get_history_stats():
    """Retrieves history and calculates complex aggregate metrics."""
    history = get_history()

    stats = {
        "all_time": 0.0,
        "today": 0.0,
        "this_month": 0.0,
        "this_year": 0.0,
        "count_all_time": 0,
        "count_today": 0
    }

    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    month_str = now.strftime("%Y-%m")
    year_str = now.strftime("%Y")

    for item in history:
        dur = item.get("video_duration", 0)
        c_at = item.get("completed_at", "")  # Format: 2026-04-27 20:31:17

        stats["all_time"] += dur
        stats["count_all_time"] += 1

        if c_at.startswith(today_str):
            stats["today"] += dur
            stats["count_today"] += 1
        if c_at.startswith(month_str):
            stats["this_month"] += dur
        if c_at.startswith(year_str):
            stats["this_year"] += dur

    return history, stats
