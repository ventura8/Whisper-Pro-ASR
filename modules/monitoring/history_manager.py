"""
Persistent Task History Manager

This module handles the storage and retrieval of task history, providing
persistent storage on disk and a RAM cache for fast access.
"""
import os
import json
import time
from datetime import datetime
import logging
from typing import List, Dict, Any, Tuple, Optional
from modules import config

logger = logging.getLogger(__name__)

HISTORY_FILE = os.path.join(config.STATE_DIR, "task_history.json")
MAX_HISTORY_DISK = 1000  # Persistent storage limit
MAX_HISTORY_RAM = 100    # RAM cache limit (match disk limit for accurate stats)

# --- [DEFERRED PERSISTENCE ENGINE] ---
_HISTORY_CACHE: List[Dict[str, Any]] = []
_UNSAVED_COUNT = 0
_LAST_SYNC = time.time()
_STATS_CACHE: Optional[Dict[str, Any]] = None


def _ensure_loaded() -> None:
    """
    Lazy load history from SSD into RAM cache.
    """
    global _HISTORY_CACHE  # pylint: disable=global-statement
    if not _HISTORY_CACHE:
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                _HISTORY_CACHE = data
            except (IOError, json.JSONDecodeError) as e:
                logger.warning("[History] Failed to load history file: %s", e)
                _HISTORY_CACHE = []


def log_completed_task(task_data: Dict[str, Any]) -> None:
    """
    Appends a task to RAM cache and defers flushing to SSD for performance.

    Parameters:
        task_data: Dictionary containing task details and results.
    """
    global _HISTORY_CACHE, _UNSAVED_COUNT, _STATS_CACHE  # pylint: disable=global-statement
    try:
        _ensure_loaded()

        # Add completion metadata
        task_data["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        task_data["total_elapsed_sec"] = round(
            time.time() - task_data.get("start_time", time.time()), 2)
        if "logs" in task_data:
            task_data["log_count"] = len(task_data["logs"])

        if "result" in task_data:
            res_keys = list(task_data["result"].keys())
            text_len = len(str(task_data["result"].get("text", "")))
            logger.info("[History] Saving task with result keys: %s (Text len: %d)",
                        res_keys, text_len)
        else:
            logger.warning("[History] Saving task WITHOUT result field! Task: %s",
                           task_data.get("task_id"))

        # Memory Protection: Keep full data as requested by user
        _HISTORY_CACHE.insert(0, task_data.copy())
        _HISTORY_CACHE = _HISTORY_CACHE[:MAX_HISTORY_DISK]

        # Invalidate stats cache so it's recalculated on next request
        _STATS_CACHE = None
        _UNSAVED_COUNT += 1

        # Immediate persistence: Save to SSD after every task completion as requested
        flush_history()

    except (KeyError, ValueError, TypeError) as e:
        logger.error("[History] Failed to log task history: %s", e)


def flush_history() -> None:
    """
    Synchronizes the RAM cache to the physical SSD.
    Maintains up to 1000 items on disk.
    """
    global _UNSAVED_COUNT, _LAST_SYNC  # pylint: disable=global-statement
    try:
        os.makedirs(config.STATE_DIR, exist_ok=True)

        # 1. Load existing disk history to merge
        disk_history = []
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                    disk_history = json.load(f)
            except (IOError, json.JSONDecodeError):
                disk_history = []

        # 2. Merge with RAM cache (deduplicate by task_id)
        # RAM cache tasks are the newest.
        seen_ids = {t.get("task_id") for t in _HISTORY_CACHE if t.get("task_id")}
        merged = list(_HISTORY_CACHE)
        for t in disk_history:
            tid = t.get("task_id")
            if tid and tid not in seen_ids:
                merged.append(t)
                seen_ids.add(tid)

        # 3. Limit to MAX_HISTORY_DISK (1000)
        data_to_save = merged[:MAX_HISTORY_DISK]

        # 4. Atomic write
        tmp_file = f"{HISTORY_FILE}.tmp"
        with open(tmp_file, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2)
        os.replace(tmp_file, HISTORY_FILE)

        _UNSAVED_COUNT = 0
        _LAST_SYNC = time.time()
    except (IOError, OSError) as e:
        logger.error("[History] SSD Sync Failed: %s", e)


def get_history() -> List[Dict[str, Any]]:
    """
    Retrieves history from RAM cache.

    Returns:
        List of historical task dictionaries.
    """
    _ensure_loaded()
    # Filter out corrupted or legacy entries that don't match the task schema
    return [t for t in _HISTORY_CACHE if isinstance(t, dict) and "task_id" in t]


def get_history_stats() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Retrieves history and calculates complex aggregate metrics with caching.

    Returns:
        Tuple of (history_list, stats_dict).
    """
    global _STATS_CACHE  # pylint: disable=global-statement
    history = get_history()

    if _STATS_CACHE:
        return history, _STATS_CACHE

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

    _STATS_CACHE = stats
    return history, stats


def clear_history() -> None:
    """
    Purges all history from RAM cache and SSD.
    """
    global _HISTORY_CACHE, _STATS_CACHE, _UNSAVED_COUNT  # pylint: disable=global-statement
    _HISTORY_CACHE = []
    _STATS_CACHE = None
    _UNSAVED_COUNT = 0
    if os.path.exists(HISTORY_FILE):
        try:
            os.remove(HISTORY_FILE)
            logger.info("[History] History file purged on disk.")
        except OSError as e:
            logger.error("[History] Failed to purge history file: %s", e)
