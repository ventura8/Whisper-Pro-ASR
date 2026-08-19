"""
Persistent Task History Manager

This module handles the storage and retrieval of task history, providing
persistent storage on disk and a RAM cache for fast access.
"""

import json
import logging
import os
import sys
import threading
import time
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from modules.core import config
from modules.monitoring.history_helpers import (
    accumulate_stats,
    backfill_single_task_filename,
    backfill_single_task_request_json,
    backfill_task_filenames,
    iter_unique_legacy_paths,
    merge_legacy_analytics,
    new_stats_payload,
)
from modules.monitoring.io_utils import load_json_list_file

logger = logging.getLogger(__name__)

HISTORY_FILE = os.path.join(config.STATE_DIR, "task_history.json")
ANALYTICS_FILE = os.path.join(config.STATE_DIR, "analytics_stats.json")
LEGACY_STATE_DIR = os.environ.get("WHISPER_LEGACY_STATE_DIR", "/app/state-legacy")
LEGACY_HISTORY_FILES = [
    os.path.join(LEGACY_STATE_DIR, "task_history.json"),
    os.path.join("/app/state", "task_history.json"),
    os.path.join("/app/data-legacy", "task_history.json"),
    os.path.join(os.path.abspath("state"), "task_history.json"),
    os.path.join(os.path.abspath("data"), "task_history.json"),
]
LEGACY_ANALYTICS_FILES = [
    os.path.join(LEGACY_STATE_DIR, "analytics_stats.json"),
    os.path.join("/app/state", "analytics_stats.json"),
    os.path.join("/app/data-legacy", "analytics_stats.json"),
    os.path.join(os.path.abspath("state"), "analytics_stats.json"),
    os.path.join(os.path.abspath("data"), "analytics_stats.json"),
]
MAX_HISTORY_DISK = 1000  # Persistent storage limit
MAX_HISTORY_RAM = 60  # RAM cache limit for fast dashboard reads (disk persistence retains up to MAX_HISTORY_DISK)

# --- [DEFERRED PERSISTENCE ENGINE] ---
HISTORY_CACHE: List[Dict[str, Any]] = []
ANALYTICS_CACHE: Optional[Dict[str, Dict[str, Any]]] = None
UNSAVED_COUNT = 0
LAST_SYNC = time.time()
STATS_CACHE: Optional[Dict[str, Any]] = None
STATS_CACHE_DATE: Optional[str] = None
ANALYTICS_LOCK = threading.RLock()
ANALYTICS_SCHEMA_VERSION = 2
ANALYTICS_SCHEMA_KEY = "__schema_version__"


def ensure_loaded() -> None:
    """
    Lazy load history from SSD into RAM cache.
    """
    module = sys.modules[__name__]
    if not module.HISTORY_CACHE:
        data, imported_from_legacy = _load_history_cache_from_disk()
        module.HISTORY_CACHE = data
        backfill_task_filenames(module.HISTORY_CACHE)
        if imported_from_legacy:
            try:
                os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
                _persist_history_data(module.HISTORY_CACHE[:MAX_HISTORY_DISK])
            except (IOError, OSError) as e:
                logger.warning("[History] Failed to persist imported legacy history: %s", e)


def _load_history_cache_from_disk() -> tuple[List[Dict[str, Any]], bool]:
    primary_entries = _load_primary_history_entries()
    if primary_entries is not None:
        return primary_entries, False

    legacy_loaded = _load_legacy_history_entries()
    if legacy_loaded is None:
        return [], False

    legacy_path, legacy_entries = legacy_loaded
    if legacy_entries:
        logger.info("[History] Importing %d legacy tasks from %s", len(legacy_entries), legacy_path)
    return legacy_entries, True


def _load_primary_history_entries() -> Optional[List[Dict[str, Any]]]:
    if not os.path.exists(HISTORY_FILE):
        return None
    loaded = _read_history_file_entries()
    if loaded is None:
        return []
    return [entry for entry in loaded if isinstance(entry, dict)]


def _load_legacy_history_entries() -> Optional[Tuple[str, List[Dict[str, Any]]]]:
    legacy_path = _find_legacy_file(LEGACY_HISTORY_FILES, HISTORY_FILE)
    if not legacy_path:
        return None

    legacy_loaded = load_json_list_file(legacy_path)
    if legacy_loaded is None:
        return None

    legacy_entries = [entry for entry in legacy_loaded if isinstance(entry, dict)]
    return legacy_path, legacy_entries


def _find_legacy_file(candidates: List[str], current_file: str) -> Optional[str]:
    for candidate_abs in iter_unique_legacy_paths(candidates, current_file):
        if os.path.exists(candidate_abs):
            return candidate_abs
    return None


def ensure_analytics_loaded() -> None:
    """
    Lazy load analytics stats from SSD into RAM cache.
    """
    module = sys.modules[__name__]
    with ANALYTICS_LOCK:
        if module.ANALYTICS_CACHE is not None:
            return
        module.ANALYTICS_CACHE = _load_analytics_cache_from_disk()
        if _analytics_cache_needs_rebuild(module.ANALYTICS_CACHE):
            _rebuild_and_persist_analytics(module)


def _load_analytics_cache_from_disk() -> Dict[str, Any]:
    loaded, imported_from_path = _read_analytics_cache_file()
    if loaded is None:
        return _default_analytics_cache()
    sanitized = _sanitize_analytics_cache(loaded)
    if imported_from_path is not None and sanitized:
        logger.info("[Analytics] Imported legacy analytics from %s", imported_from_path)
        _persist_analytics_cache(sanitized)
    return sanitized


def _default_analytics_cache() -> Dict[str, Any]:
    return {ANALYTICS_SCHEMA_KEY: ANALYTICS_SCHEMA_VERSION}


def _read_analytics_cache_file() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not os.path.exists(ANALYTICS_FILE):
        legacy_path = _find_legacy_file(LEGACY_ANALYTICS_FILES, ANALYTICS_FILE)
        if legacy_path:
            return _read_analytics_json_file(legacy_path), legacy_path
        return None, None
    loaded = _read_analytics_json_file(ANALYTICS_FILE)
    return loaded, None


def _read_analytics_json_file(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        logger.warning("[Analytics] Failed to load analytics file %s: %s", path, e)
        return None
    if not isinstance(loaded, dict):
        return None
    return loaded


def _sanitize_analytics_cache(loaded: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    schema_value = loaded.get(ANALYTICS_SCHEMA_KEY)
    if isinstance(schema_value, int):
        sanitized[ANALYTICS_SCHEMA_KEY] = schema_value
    for key, value in loaded.items():
        if key == ANALYTICS_SCHEMA_KEY:
            continue
        if isinstance(value, dict):
            sanitized[key] = value
    return sanitized


def _analytics_cache_needs_rebuild(cache: Dict[str, Any]) -> bool:
    if not isinstance(cache, dict):
        return True
    if cache.get(ANALYTICS_SCHEMA_KEY) != ANALYTICS_SCHEMA_VERSION:
        return True
    return any(_invalid_analytics_bucket(key, value) for key, value in cache.items())


def _invalid_analytics_bucket(key: str, value: Any) -> bool:
    if key == ANALYTICS_SCHEMA_KEY:
        return False
    return not isinstance(value, dict) or "asr" not in value


def _rebuild_and_persist_analytics(module: Any) -> None:
    logger.info("[Analytics] Rebuilding analytics stats from task history...")
    old_cache = dict(module.ANALYTICS_CACHE)
    rebuild_analytics_from_history()
    merge_legacy_analytics(old_cache, module.ANALYTICS_CACHE)
    _persist_analytics_cache(module.ANALYTICS_CACHE)


def _persist_analytics_cache(cache: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(ANALYTICS_FILE), exist_ok=True)
        tmp_file = f"{ANALYTICS_FILE}.tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
        os.replace(tmp_file, ANALYTICS_FILE)
    except (IOError, OSError) as e:
        logger.error("[Analytics] Failed to save rebuilt analytics: %s", e)


def categorize_task(task_data: Dict[str, Any]) -> str:
    """
    Categorize task as 'asr', 'detectlang', or 'audio'.
    """
    for matcher, category in _category_matchers():
        if matcher(task_data):
            return category
    return "asr"


def _category_matchers() -> list[tuple[Any, str]]:
    return [
        (_is_asr_endpoint, "asr"),
        (_is_detect_endpoint, "detectlang"),
        (_is_audio_endpoint, "audio"),
        (_is_language_detection_type, "detectlang"),
        (_is_audio_category_by_task_fields, "audio"),
    ]


def _is_asr_endpoint(task_data: Dict[str, Any]) -> bool:
    endpoint = task_data.get("endpoint", "")
    return endpoint in {"/asr", "/v1/audio/transcriptions", "/v1/audio/translations"}


def _is_detect_endpoint(task_data: Dict[str, Any]) -> bool:
    endpoint = task_data.get("endpoint", "")
    return bool(endpoint and "detect" in endpoint)


def _is_audio_endpoint(task_data: Dict[str, Any]) -> bool:
    endpoint = task_data.get("endpoint", "")
    return bool(endpoint and endpoint.startswith("/v1/audio/") and endpoint not in {"/v1/audio/transcriptions", "/v1/audio/translations"})


def _is_language_detection_type(task_data: Dict[str, Any]) -> bool:
    return task_data.get("type", "") == "Language Detection"


def _is_audio_category_by_task_fields(task_data: Dict[str, Any]) -> bool:
    task_type = task_data.get("type", "")
    req_json = task_data.get("request_json", {}) or {}
    return task_type == "Translation" or "response_format" in req_json


def rebuild_analytics_from_history() -> None:
    """
    Rebuilds the analytics stats cache from the history cache.
    """
    module = sys.modules[__name__]
    ensure_loaded()
    with ANALYTICS_LOCK:
        new_cache = {ANALYTICS_SCHEMA_KEY: ANALYTICS_SCHEMA_VERSION}
        for task in module.HISTORY_CACHE:
            _accumulate_task_into_daily_cache(new_cache, task)

        module.ANALYTICS_CACHE = new_cache
        module.STATS_CACHE = None
        module.STATS_CACHE_DATE = None


def _accumulate_task_into_daily_cache(new_cache: Dict[str, Any], task: Dict[str, Any]) -> None:
    if task.get("status") != "completed":
        return
    duration = float(task.get("video_duration", 0.0))
    date_str = _extract_task_date(task)
    category = categorize_task(task)
    day_data = _ensure_daily_bucket(new_cache, date_str)
    _ensure_daily_categories(day_data)
    day_data["count"] += 1
    day_data["duration"] += duration
    day_data[category]["count"] += 1
    day_data[category]["duration"] += duration


def _extract_task_date(task: Dict[str, Any]) -> str:
    completed_at = task.get("completed_at", "")
    if completed_at:
        return completed_at.split(" ")[0]
    return datetime.fromtimestamp(task.get("start_time", time.time())).strftime("%Y-%m-%d")


def _ensure_daily_bucket(cache: Dict[str, Any], date_str: str) -> Dict[str, Any]:
    if date_str not in cache:
        cache[date_str] = {
            "count": 0,
            "duration": 0.0,
            "asr": {"count": 0, "duration": 0.0},
            "detectlang": {"count": 0, "duration": 0.0},
            "audio": {"count": 0, "duration": 0.0},
        }
    return cache[date_str]


def _ensure_daily_categories(day_data: Dict[str, Any]) -> None:
    for cat in ["asr", "detectlang", "audio"]:
        if cat not in day_data:
            day_data[cat] = {"count": 0, "duration": 0.0}


def update_analytics(task_data: Dict[str, Any]) -> None:
    """
    Updates the persistent analytics stats with a completed task's duration.
    """
    try:
        ensure_analytics_loaded()
        module = sys.modules[__name__]
        dur = float(task_data.get("video_duration", 0.0))
        date_str = _analytics_date_for_task(task_data)

        with ANALYTICS_LOCK:
            day_data = _ensure_daily_bucket(module.ANALYTICS_CACHE, date_str)
            _ensure_daily_categories(day_data)
            category = categorize_task(task_data)
            day_data["count"] += 1
            day_data["duration"] += dur
            day_data[category]["count"] += 1
            day_data[category]["duration"] += dur

            _persist_analytics_cache(module.ANALYTICS_CACHE)

        # Invalidate stats cache
        module.STATS_CACHE = None
        module.STATS_CACHE_DATE = None
    except (IOError, OSError, ValueError, TypeError) as e:
        logger.error("[Analytics] Failed to update analytics: %s", e)


def _analytics_date_for_task(task_data: Dict[str, Any]) -> str:
    completed_at = task_data.get("completed_at", "")
    if completed_at:
        return completed_at.split(" ")[0]
    return datetime.now().strftime("%Y-%m-%d")


def log_completed_task(task_data: Dict[str, Any]) -> None:
    """
    Appends a task to RAM cache and defers flushing to SSD for performance.

    Parameters:
        task_data: Dictionary containing task details and results.
    """
    module = sys.modules[__name__]
    try:
        ensure_loaded()
        _ensure_completion_timestamp(task_data)
        _compute_elapsed_fields(task_data)
        backfill_single_task_filename(task_data)
        backfill_single_task_request_json(task_data)
        _update_log_count(task_data)
        _update_segments_processed(task_data)
        _log_result_shape(task_data)
        _truncate_large_segments(task_data)

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


def _ensure_completion_timestamp(task_data: Dict[str, Any]) -> None:
    if "completed_at" not in task_data:
        task_data["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _compute_elapsed_fields(task_data: Dict[str, Any]) -> None:
    start_time = task_data.get("start_time", time.time())
    start_active = task_data.get("start_active")
    total_elapsed = round(time.time() - start_time, 2)
    task_data["total_elapsed_sec"] = total_elapsed

    perf_queue = _extract_perf_queue_seconds(task_data)
    if perf_queue is not None:
        _set_elapsed_from_queue(task_data, total_elapsed, perf_queue)
        return
    if start_active is not None:
        _set_elapsed_from_start_active(task_data, start_time, start_active, total_elapsed)
        return
    task_data["queue_elapsed_sec"] = total_elapsed
    task_data["active_elapsed_sec"] = 0.0


def _extract_perf_queue_seconds(task_data: Dict[str, Any]) -> Optional[float]:
    perf = (task_data.get("result") or {}).get("performance", {}) or (task_data.get("response_json") or {}).get("performance", {})
    value = perf.get("queue_sec")
    if value is None:
        return None
    return float(value)


def _set_elapsed_from_queue(task_data: Dict[str, Any], total_elapsed: float, perf_queue: float) -> None:
    queue_elapsed = round(perf_queue, 2)
    task_data["queue_elapsed_sec"] = queue_elapsed
    task_data["active_elapsed_sec"] = round(max(0.0, total_elapsed - queue_elapsed), 2)


def _set_elapsed_from_start_active(task_data: Dict[str, Any], start_time: float, start_active: float, total_elapsed: float) -> None:
    queue_elapsed = round(start_active - start_time, 2)
    task_data["queue_elapsed_sec"] = queue_elapsed
    task_data["active_elapsed_sec"] = round(max(0.0, total_elapsed - queue_elapsed), 2)


def _update_log_count(task_data: Dict[str, Any]) -> None:
    if "logs" in task_data:
        task_data["log_count"] = len(task_data["logs"])


def _update_segments_processed(task_data: Dict[str, Any]) -> None:
    result = task_data.get("result", {}) or {}
    task_type = task_data.get("type", "")
    if task_type in ["Transcription", "Translation"]:
        task_data["segments_processed"] = len(result.get("segments", []) or [])
    elif task_type == "Language Detection":
        task_data["segments_processed"] = result.get("segments_processed", 1)
    else:
        task_data["segments_processed"] = 0


def _log_result_shape(task_data: Dict[str, Any]) -> None:
    if "result" in task_data:
        res_keys = list(task_data["result"].keys())
        text_len = len(str(task_data["result"].get("text", "")))
        logger.info("[History] Saving task with result keys: %s (Text len: %d)", res_keys, text_len)
        return
    logger.warning("[History] Saving task WITHOUT result field! Task: %s", task_data.get("task_id"))


def _truncate_large_segments(task_data: Dict[str, Any]) -> None:
    if "result" not in task_data:
        return
    result = task_data["result"]
    segments = result.get("segments")
    if not segments or len(segments) <= 100:
        return
    result["segments_total_count"] = len(segments)
    result["segments_truncated"] = True
    result["segments"] = segments[:100]


def flush_history() -> None:
    """
    Synchronizes the RAM cache to the physical SSD.
    Maintains up to 1000 items on disk.
    """
    module = sys.modules[__name__]
    try:
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        disk_history = _load_disk_history()
        merged = _merge_history_cache(module.HISTORY_CACHE, disk_history)
        _persist_history_data(merged[:MAX_HISTORY_DISK])

        module.UNSAVED_COUNT = 0
        module.LAST_SYNC = time.time()
    except (IOError, OSError) as e:
        logger.error("[History] SSD Sync Failed: %s", e)


def _load_disk_history() -> List[Dict[str, Any]]:
    loaded = _read_history_file_entries()
    if loaded is None:
        return []
    return [entry for entry in loaded if isinstance(entry, dict)]


def _read_history_file_entries() -> Optional[List[Any]]:
    if not os.path.exists(HISTORY_FILE):
        return None
    return load_json_list_file(HISTORY_FILE)


def _merge_history_cache(ram_history: List[Dict[str, Any]], disk_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen_ids = {t.get("task_id") for t in ram_history if t.get("task_id")}
    merged = list(ram_history)
    for task in disk_history:
        _append_history_task_if_new(task, seen_ids, merged)
    return merged


def _append_history_task_if_new(task: Dict[str, Any], seen_ids: set, merged: List[Dict[str, Any]]) -> None:
    task_id = task.get("task_id")
    if not task_id:
        return
    if task_id in seen_ids:
        return
    merged.append(task)
    seen_ids.add(task_id)


def _persist_history_data(data_to_save: List[Dict[str, Any]]) -> None:
    tmp_file = f"{HISTORY_FILE}.tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=2)
    os.replace(tmp_file, HISTORY_FILE)


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
    for task in valid_tasks:
        backfill_single_task_filename(task)
        backfill_single_task_request_json(task)
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
    stats = new_stats_payload()
    month_str = now.strftime("%Y-%m")
    year_str = now.strftime("%Y")
    with ANALYTICS_LOCK:
        analytics_snapshot = deepcopy(module.ANALYTICS_CACHE) if module.ANALYTICS_CACHE else {}
    accumulate_stats(stats, analytics_snapshot, today_str, month_str, year_str)

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
    with ANALYTICS_LOCK:
        daily_snapshot = deepcopy(module.ANALYTICS_CACHE) if module.ANALYTICS_CACHE else {}
    return {"cumulative": stats, "daily": daily_snapshot}


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
