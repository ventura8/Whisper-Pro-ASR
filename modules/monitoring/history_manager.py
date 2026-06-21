"""
Persistent Task History Manager

This module handles the storage and retrieval of task history, providing
persistent storage on disk and a RAM cache for fast access.
"""
import os
import json
import time
import sys
from datetime import datetime
import logging
from typing import List, Dict, Any, Tuple, Optional
from modules import config

logger = logging.getLogger(__name__)

HISTORY_FILE = os.path.join(config.STATE_DIR, "task_history.json")
ANALYTICS_FILE = os.path.join(config.STATE_DIR, "analytics_stats.json")
MAX_HISTORY_DISK = 1000  # Persistent storage limit
MAX_HISTORY_RAM = 20    # RAM cache limit (match disk limit for accurate stats)

# --- [DEFERRED PERSISTENCE ENGINE] ---
HISTORY_CACHE: List[Dict[str, Any]] = []
ANALYTICS_CACHE: Optional[Dict[str, Dict[str, Any]]] = None
UNSAVED_COUNT = 0
LAST_SYNC = time.time()
STATS_CACHE: Optional[Dict[str, Any]] = None
STATS_CACHE_DATE: Optional[str] = None


def ensure_loaded() -> None:
    """
    Lazy load history from SSD into RAM cache.
    """
    module = sys.modules[__name__]
    if not module.HISTORY_CACHE:
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                module.HISTORY_CACHE = data
            except (IOError, json.JSONDecodeError) as e:
                logger.warning("[History] Failed to load history file: %s", e)
                module.HISTORY_CACHE = []


def ensure_analytics_loaded() -> None:
    """
    Lazy load analytics stats from SSD into RAM cache.
    """
    module = sys.modules[__name__]
    if module.ANALYTICS_CACHE is None:
        if os.path.exists(ANALYTICS_FILE):
            try:
                with open(ANALYTICS_FILE, 'r', encoding='utf-8') as f:
                    module.ANALYTICS_CACHE = json.load(f)
            except (IOError, json.JSONDecodeError) as e:
                logger.warning("[Analytics] Failed to load analytics file: %s", e)
                module.ANALYTICS_CACHE = {}
        else:
            module.ANALYTICS_CACHE = {}


def update_analytics(task_data: Dict[str, Any]) -> None:
    """
    Updates the persistent analytics stats with a completed task's duration.
    """
    try:
        ensure_analytics_loaded()
        module = sys.modules[__name__]
        dur = float(task_data.get("video_duration", 0.0))

        # Get completion date or current date
        c_at = task_data.get("completed_at", "")
        if c_at:
            # Format: 2026-05-26 11:43:36 -> date: 2026-05-26
            date_str = c_at.split(" ")[0]
        else:
            date_str = datetime.now().strftime("%Y-%m-%d")

        if date_str not in module.ANALYTICS_CACHE:
            module.ANALYTICS_CACHE[date_str] = {"count": 0, "duration": 0.0}

        module.ANALYTICS_CACHE[date_str]["count"] += 1
        module.ANALYTICS_CACHE[date_str]["duration"] += dur

        # Save to disk atomically
        os.makedirs(config.STATE_DIR, exist_ok=True)
        tmp_file = f"{ANALYTICS_FILE}.tmp"
        with open(tmp_file, 'w', encoding='utf-8') as f:
            json.dump(module.ANALYTICS_CACHE, f, indent=2)
        os.replace(tmp_file, ANALYTICS_FILE)
    except (IOError, OSError, ValueError, TypeError) as e:
        logger.error("[Analytics] Failed to update analytics: %s", e)


def log_completed_task(task_data: Dict[str, Any]) -> None:
    """
    Appends a task to RAM cache and defers flushing to SSD for performance.

    Parameters:
        task_data: Dictionary containing task details and results.
    """
    module = sys.modules[__name__]
    try:
        ensure_loaded()

        # Add completion metadata
        if "completed_at" not in task_data:
            task_data["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        start_time = task_data.get("start_time", time.time())
        start_active = task_data.get("start_active")

        # Calculate total elapsed time
        total_elapsed = round(time.time() - start_time, 2)
        task_data["total_elapsed_sec"] = total_elapsed

        # Try to extract precise queue time from the performance stats
        perf = task_data.get("result", {}).get("performance", {}) or task_data.get("response_json", {}).get("performance", {})
        perf_queue = perf.get("queue_sec")

        if perf_queue is not None:
            queue_elapsed = round(float(perf_queue), 2)
            task_data["queue_elapsed_sec"] = queue_elapsed
            task_data["active_elapsed_sec"] = round(max(0.0, total_elapsed - queue_elapsed), 2)
        elif start_active is not None:
            queue_elapsed = round(start_active - start_time, 2)
            task_data["queue_elapsed_sec"] = queue_elapsed
            task_data["active_elapsed_sec"] = round(max(0.0, total_elapsed - queue_elapsed), 2)
        else:
            # Task never got active (e.g. failed/aborted in queue)
            task_data["queue_elapsed_sec"] = total_elapsed
            task_data["active_elapsed_sec"] = 0.0

        if "logs" in task_data:
            task_data["log_count"] = len(task_data["logs"])

        result = task_data.get("result", {}) or {}
        task_type = task_data.get("type", "")
        if task_type in ["Transcription", "Translation"]:
            segments = result.get("segments", []) or []
            task_data["segments_processed"] = len(segments)
        elif task_type == "Language Detection":
            task_data["segments_processed"] = result.get("segments_processed", 1)
        else:
            task_data["segments_processed"] = 0

        if "result" in task_data:
            res_keys = list(task_data["result"].keys())
            text_len = len(str(task_data["result"].get("text", "")))
            logger.info("[History] Saving task with result keys: %s (Text len: %d)",
                        res_keys, text_len)
        else:
            logger.warning("[History] Saving task WITHOUT result field! Task: %s",
                           task_data.get("task_id"))

        # Memory Protection: Truncate very large segment lists before persisting.
        # A 15h+ movie can produce 10K+ segments (2–5 MB per task entry).
        # We keep the first 100 for history/dashboard preview; the client
        # already received the full output from the HTTP response.
        # Note: result.text (full SRT) is intentionally preserved.
        if "result" in task_data:
            res = task_data["result"]
            segs = res.get("segments")
            if segs and len(segs) > 100:
                res["segments_total_count"] = len(segs)
                res["segments_truncated"] = True
                res["segments"] = segs[:100]

        module.HISTORY_CACHE.insert(0, task_data.copy())
        module.HISTORY_CACHE = module.HISTORY_CACHE[:MAX_HISTORY_DISK]

        # Invalidate stats cache so it's recalculated on next request
        module.STATS_CACHE = None
        module.UNSAVED_COUNT += 1

        # Immediate persistence: Save to SSD after every task completion as requested
        flush_history()

        # Update analytics stats
        update_analytics(task_data)

    except (KeyError, ValueError, TypeError) as e:
        logger.error("[History] Failed to log task history: %s", e)


def flush_history() -> None:
    """
    Synchronizes the RAM cache to the physical SSD.
    Maintains up to 1000 items on disk.
    """
    module = sys.modules[__name__]
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
        seen_ids = {t.get("task_id") for t in module.HISTORY_CACHE if t.get("task_id")}
        merged = list(module.HISTORY_CACHE)
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

        module.UNSAVED_COUNT = 0
        module.LAST_SYNC = time.time()
    except (IOError, OSError) as e:
        logger.error("[History] SSD Sync Failed: %s", e)


def get_history() -> List[Dict[str, Any]]:
    """
    Retrieves history from RAM cache.

    Returns:
        List of historical task dictionaries.
    """
    ensure_loaded()
    module = sys.modules[__name__]
    # Filter out corrupted or legacy entries that don't match the task schema
    valid_tasks = [t for t in module.HISTORY_CACHE if isinstance(t, dict) and "task_id" in t]
    # Return only the most recent tasks for the dashboard
    return valid_tasks[:MAX_HISTORY_RAM]


def get_history_stats() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Retrieves history and calculates complex aggregate metrics with caching.

    Returns:
        Tuple of (history_list, stats_dict).
    """
    module = sys.modules[__name__]
    history = get_history()

    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")

    if module.STATS_CACHE and module.STATS_CACHE_DATE == today_str:
        return history, module.STATS_CACHE

    ensure_analytics_loaded()

    stats = {
        "all_time": 0.0,
        "today": 0.0,
        "this_month": 0.0,
        "this_year": 0.0,
        "count_all_time": 0,
        "count_today": 0
    }

    month_str = now.strftime("%Y-%m")
    year_str = now.strftime("%Y")

    if module.ANALYTICS_CACHE:
        for date_str, daily_data in module.ANALYTICS_CACHE.items():
            dur = daily_data.get("duration", 0.0)
            cnt = daily_data.get("count", 0)

            stats["all_time"] += dur
            stats["count_all_time"] += cnt

            if date_str == today_str:
                stats["today"] += dur
                stats["count_today"] += cnt
            if date_str.startswith(month_str):
                stats["this_month"] += dur
            if date_str.startswith(year_str):
                stats["this_year"] += dur

    module.STATS_CACHE = stats
    module.STATS_CACHE_DATE = today_str
    return history, stats


def get_analytics_data() -> Dict[str, Any]:
    """
    Retrieves the detailed daily analytics and cumulative summary stats.
    """
    ensure_analytics_loaded()
    module = sys.modules[__name__]
    _, stats = get_history_stats()
    return {
        "cumulative": stats,
        "daily": module.ANALYTICS_CACHE or {}
    }


def clear_history() -> None:
    """
    Purges all history from RAM cache and SSD.
    """
    module = sys.modules[__name__]
    module.HISTORY_CACHE = []
    module.STATS_CACHE = None
    module.STATS_CACHE_DATE = None
    module.UNSAVED_COUNT = 0
    if os.path.exists(HISTORY_FILE):
        try:
            os.remove(HISTORY_FILE)
            logger.info("[History] History file purged on disk.")
        except OSError as e:
            logger.error("[History] Failed to purge history file: %s", e)
