"""Helper functions for history and analytics processing."""

import ntpath
import os
from typing import Any, Dict, Optional

from modules.api.support.local_path import extract_path_from_mapping_keys, normalize_bazarr_request_params


def backfill_task_filenames(data: Any) -> None:
    """Helper to resolve and clean generic task filenames in a flat list of history tasks."""
    if not isinstance(data, list):
        return
    for task in data:
        backfill_single_task_filename(task)


def resolve_display_filename(task: Any) -> Optional[str]:
    """Return the best dashboard filename for a task/history entry."""
    if not isinstance(task, dict):
        return None
    current = task.get("filename")
    if not is_generic_filename(current):
        return current
    req_json = _get_request_json_dict(task)
    return _extract_best_filename(req_json) or current


def backfill_single_task_filename(task: Any) -> None:
    """Resolve and clean a single task filename if generic."""
    if not isinstance(task, dict):
        return
    resolved = resolve_display_filename(task)
    if resolved and is_generic_filename(task.get("filename")):
        task["filename"] = resolved


def backfill_single_task_request_json(task: Any) -> None:
    """Normalize Bazarr path-as-key payloads in stored request metadata."""
    if not isinstance(task, dict):
        return
    req_json = _get_request_json_dict(task)
    if not req_json:
        return
    task["request_json"] = normalize_bazarr_request_params(req_json)


def _get_request_json_dict(task: dict) -> dict:
    req_json = task.get("request_json") or {}
    return req_json if isinstance(req_json, dict) else {}


def _extract_best_filename(req_json: dict) -> Optional[str]:
    path_from_keys = extract_path_from_mapping_keys(req_json)
    if path_from_keys:
        base = _clean_candidate_filename(path_from_keys)
        if base:
            return base
    candidates = [
        req_json.get("video_file"),
        req_json.get("local_path"),
        req_json.get("file_path"),
        req_json.get("original_path"),
        req_json.get("file"),
        req_json.get("audio_file"),
    ]
    for val in candidates:
        base = _clean_candidate_filename(val)
        if base:
            return base
    return None


def _clean_candidate_filename(value: Any) -> Optional[str]:
    if not (value and isinstance(value, str) and value.strip()):
        return None
    clean_val = value.strip().strip('"').strip("'")
    base = ntpath.basename(clean_val)
    return None if is_generic_filename(base) else base


def is_generic_filename(value: Optional[str]) -> bool:
    """Check if a filename is generic or placeholder."""
    return value in [None, "", "audio_file", "file", "blob", "Unknown", "Unknown Media"]


def merge_legacy_analytics(old_cache: Dict[str, Any], new_cache: Dict[str, Any]) -> None:
    """Helper to merge legacy cache items that aren't in history anymore."""
    for date_str, daily_data in old_cache.items():
        _merge_single_legacy_day(date_str, daily_data, new_cache)


def _merge_single_legacy_day(date_str: str, daily_data: Any, new_cache: Dict[str, Any]) -> None:
    if not isinstance(daily_data, dict):
        return
    if date_str in new_cache:
        _merge_overlapping_legacy_day(daily_data, new_cache[date_str])
        return
    if _has_all_category_keys(daily_data):
        new_cache[date_str] = daily_data
        return
    new_cache[date_str] = _backfill_legacy_day_as_asr(daily_data)


def _has_all_category_keys(daily_data: dict) -> bool:
    return all(cat in daily_data for cat in ["asr", "detectlang", "audio"])


def _merge_overlapping_legacy_day(old_day: dict, rebuilt_day: dict) -> None:
    """Reconcile a stored day with the one rebuilt from history.

    The stored file is the fuller record: history is capped, so a rebuild only sees the
    tasks that survived the cap. Totals therefore take the larger of the two.

    How the categories are reconciled depends on what the stored day knows. A day that
    already carries a full breakdown keeps it -- it was accumulated from real tasks, and
    overwriting it with the rebuild's view discarded that. Measured on a day holding
    40 ASR and 10 language-detection tasks: the merge reported 50 ASR and 0 detections,
    silently rewriting history rather than preserving it.

    A legacy day with no breakdown has nothing to preserve, so the difference is still
    attributed to ASR -- the original heuristic, and the only guess available there.
    """
    old_count = old_day.get("count", 0)
    old_dur = old_day.get("duration", 0.0)
    rebuilt_count = rebuilt_day.get("count", 0)
    rebuilt_dur = rebuilt_day.get("duration", 0.0)

    rebuilt_day["count"] = max(old_count, rebuilt_count)
    rebuilt_day["duration"] = max(old_dur, rebuilt_dur)

    if _has_all_category_keys(old_day):
        for category in ("asr", "detectlang", "audio"):
            rebuilt_day[category] = dict(old_day[category])
        # The totals above took the max of the two views, so when the rebuild saw *more*
        # than the stored file (a day whose stored copy predates some tasks), keeping the
        # stored breakdown verbatim leaves the categories summing to less than the day's
        # own count -- the analytics page then shows a total its own breakdown cannot
        # account for. Attribute only the non-overlapping remainder, which is the same
        # ASR-shaped guess the legacy branch below makes, and only when there is one.
        _add_uncategorised_remainder(rebuilt_day, count=rebuilt_count - old_count, duration=rebuilt_dur - old_dur)
        return

    _add_uncategorised_remainder(rebuilt_day, count=old_count - rebuilt_count, duration=old_dur - rebuilt_dur)


def _add_uncategorised_remainder(day: dict, *, count: int, duration: float) -> None:
    """Attribute a count/duration remainder to ASR, the only guess available.

    Negative or zero remainders are no-ops: the two views already agree, or the side that
    is being kept is the larger one and there is nothing unaccounted for.
    """
    diff_count = max(0, count)
    diff_dur = max(0.0, duration)
    if not diff_count and not diff_dur:
        return
    if "asr" not in day:
        day["asr"] = {"count": 0, "duration": 0.0}
    day["asr"]["count"] += diff_count
    day["asr"]["duration"] += diff_dur


def _backfill_legacy_day_as_asr(daily_data: dict) -> dict:
    day = dict(daily_data)
    day["asr"] = {"count": day.get("count", 0), "duration": day.get("duration", 0.0)}
    day["detectlang"] = {"count": 0, "duration": 0.0}
    day["audio"] = {"count": 0, "duration": 0.0}
    return day


def new_stats_payload() -> Dict[str, Any]:
    """Return a clean default history stats dictionary payload."""
    return {
        "all_time": 0.0,
        "today": 0.0,
        "this_month": 0.0,
        "this_year": 0.0,
        "count_all_time": 0,
        "count_today": 0,
        "asr": {"count": 0, "duration": 0.0},
        "detectlang": {"count": 0, "duration": 0.0},
        "audio": {"count": 0, "duration": 0.0},
    }


def accumulate_stats(stats: Dict[str, Any], analytics_snapshot: Dict[str, Any], today_str: str, month_str: str, year_str: str) -> None:
    """Accumulate total duration and counts from daily analytics snapshot into stats dict."""
    for date_str, daily_data in analytics_snapshot.items():
        if not isinstance(daily_data, dict):
            continue
        _accumulate_daily_totals(
            stats,
            date_str,
            daily_data,
            today_str=today_str,
            month_str=month_str,
            year_str=year_str,
        )
        _accumulate_daily_categories(stats, daily_data)


def _accumulate_daily_totals(
    stats: Dict[str, Any],
    date_str: str,
    daily_data: Dict[str, Any],
    *,
    today_str: str,
    month_str: str,
    year_str: str,
) -> None:
    duration = daily_data.get("duration", 0.0)
    count = daily_data.get("count", 0)
    stats["all_time"] += duration
    stats["count_all_time"] += count
    if date_str == today_str:
        stats["today"] += duration
        stats["count_today"] += count
    if date_str.startswith(month_str):
        stats["this_month"] += duration
    if date_str.startswith(year_str):
        stats["this_year"] += duration


def _accumulate_daily_categories(stats: Dict[str, Any], daily_data: Dict[str, Any]) -> None:
    for cat in ["asr", "detectlang", "audio"]:
        cat_data = daily_data.get(cat, {})
        stats[cat]["count"] += cat_data.get("count", 0)
        stats[cat]["duration"] += cat_data.get("duration", 0.0)


def iter_unique_legacy_paths(candidates: list[str], current_file: str):
    """Yield unique normalized legacy file candidates excluding current file path."""
    current_abs = os.path.abspath(current_file)
    seen = set()
    for candidate in candidates:
        candidate_abs = normalize_legacy_candidate(candidate)
        if not candidate_abs or candidate_abs == current_abs or candidate_abs in seen:
            continue
        seen.add(candidate_abs)
        yield candidate_abs


def normalize_legacy_candidate(candidate: str) -> Optional[str]:
    """Normalize a legacy path candidate to absolute path when valid."""
    if not candidate:
        return None
    return os.path.abspath(candidate)


#: Segments kept in the history record. The full transcript still goes to the caller.
MAX_HISTORY_SEGMENTS = 100


def truncate_large_segments(task_data: dict) -> None:
    """Shrink the history copy of a large transcript, leaving the caller's untouched."""
    if "result" not in task_data:
        return
    result = task_data["result"]
    segments = result.get("segments")
    if not segments or len(segments) <= MAX_HISTORY_SEGMENTS:
        return
    # Replace the result dict rather than mutating it. ``task_data["result"]`` is the same
    # object the request handler returns to the caller, and ``log_completed_task`` only
    # takes a shallow ``task_data.copy()`` afterwards -- so truncating in place silently
    # clipped the client's transcript to keep the *history file* small. Measured on a
    # 20-minute clip: 169 real segments reaching 1202s were delivered as 100 segments
    # ending at 765s, a loss long recorded as an ASR "coverage" defect.
    task_data["result"] = {
        **result,
        "segments": segments[:MAX_HISTORY_SEGMENTS],
        "segments_total_count": len(segments),
        "segments_truncated": True,
    }
