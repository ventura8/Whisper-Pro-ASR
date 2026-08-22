"""Tests for modules/monitoring/telemetry.py's stale-task/ghost-reaping and
liveness-heartbeat logic. Split out of test_telemetry_loop.py (past the repo's
500-line-per-file limit) to keep each module focused and under that limit.
"""

import threading
import time
from typing import Any
from unittest import mock

from modules.monitoring import telemetry
from tests.monitoring._telemetry_test_helpers import get_service_stats_with_common_patches


def _seed_ghost_and_fresh_active_tasks(now: float) -> None:
    """Seed the registry with a stale crashed-worker ghost task ('crashed_ghost',
    marked active hours ago) and a genuinely fresh active task ('fresh_active'),
    for the ghost-reaping test below."""
    from modules.inference import scheduler

    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry.clear()
        # Simulated crash: task was marked active hours ago and the owning worker
        # thread/process died before the normal completion path ever ran, so the
        # entry was never archived/removed from the registry.
        scheduler.STATE.task_registry["crashed_ghost"] = {
            "task_id": "crashed_ghost",
            "status": "active",
            "stage": "Inference",
            "start_time": now - 999999,
            "start_active": now - 999999,
            "is_priority": False,
            "type": "Transcription",
            "unit_id": "NPU.0",
        }
        # A genuinely fresh active task must NOT be affected by the ghost-reaping logic.
        scheduler.STATE.task_registry["fresh_active"] = {
            "task_id": "fresh_active",
            "status": "active",
            "stage": "Inference",
            "start_time": now,
            "start_active": now,
            "is_priority": False,
            "type": "Transcription",
            "unit_id": "GPU.0",
        }


def _assert_ghost_task_reaped(tasks_by_id: dict[str, dict[str, Any]]) -> None:
    """Assert the stale ghost task was reported reaped: failed status, with the
    stale-stage message rather than perpetually 'active'."""
    assert tasks_by_id["crashed_ghost"]["status"] == "failed"
    assert tasks_by_id["crashed_ghost"]["stage"] == "Stale (worker did not report completion)"


def _assert_ghost_reaped_and_fresh_active_untouched(stats: dict[str, Any]) -> None:
    """Assert the stale ghost task was reaped while the genuinely fresh active
    task is unaffected, and the ghost isn't counted as an active session."""
    tasks_by_id = {t.get("task_id"): t for t in stats["tasks"]}
    _assert_ghost_task_reaped(tasks_by_id)
    assert tasks_by_id["fresh_active"]["status"] == "active"
    assert stats["active_sessions"] == 1


def test_get_service_stats_reaps_stale_ghost_task_after_simulated_crash(clean_telemetry: None) -> None:
    """Gap 7.6: a worker that crashes/restarts without going through normal task
    finalization (scheduler._finalize_registered_task) leaves its registry entry
    stuck at status='active' forever, since nothing else ever removes it. Verify
    get_service_stats() does not report such a ghost as perpetually 'active'/'running'.
    """
    from modules.inference import scheduler

    _seed_ghost_and_fresh_active_tasks(time.time())
    try:
        stats = get_service_stats_with_common_patches()
        _assert_ghost_reaped_and_fresh_active_untouched(stats)
    finally:
        # Restore the registry to a clean state regardless of assertion outcome,
        # so a failing assertion here cannot leak dirty entries into later tests.
        with scheduler.STATE.task_registry_lock:
            scheduler.STATE.task_registry.clear()


def test_get_service_stats_removes_stale_ghost_from_live_registry_not_just_display(clean_telemetry: None) -> None:
    """The stale-ghost reaping must mutate `scheduler.STATE.task_registry` itself
    (via the scheduler's own finalization lifecycle), not just relabel a display
    copy -- otherwise a code path that reads the live registry directly (e.g.
    `get_minimal_stats()`, which never goes through `get_service_stats()`'s
    display-normalization logic) would keep counting the ghost as active forever.
    """
    from modules.inference import scheduler

    _seed_ghost_and_fresh_active_tasks(time.time())
    try:
        stats = get_service_stats_with_common_patches()
        _assert_ghost_reaped_and_fresh_active_untouched(stats)

        # The ghost's entry must be gone from the actual live registry, not merely
        # relabeled in the returned display copy.
        with scheduler.STATE.task_registry_lock:
            assert "crashed_ghost" not in scheduler.STATE.task_registry
            assert "fresh_active" in scheduler.STATE.task_registry

        # A code path that reads the live registry directly must also no longer
        # count the reaped ghost as active.
        minimal = telemetry.get_minimal_stats()
        assert minimal["active"] == 1

        # The dedup-warning set must not grow unbounded once the entry is actually
        # removed from the registry -- it should have been discarded on reap.
        assert "crashed_ghost" not in telemetry._STALE_TASK_WARNED
    finally:
        with scheduler.STATE.task_registry_lock:
            scheduler.STATE.task_registry.clear()
        telemetry._clear_stale_task_warned()


def test_get_service_stats_persists_reaped_ghost_to_history_as_failed(clean_telemetry: None) -> None:
    """The reaped ghost must be archived into task history as 'failed', not
    'completed' at 100% progress. `_archive_registry_task` derives the archived
    status from the LIVE registry entry's status, not the telemetry display copy
    -- if only the display copy were marked failed (leaving the live entry
    'active'), the history record would wrongly say the crashed task completed
    successfully, hiding the real failure the reaper just detected.
    """
    from modules.inference import scheduler

    _seed_ghost_and_fresh_active_tasks(time.time())
    logged_tasks: list[dict[str, Any]] = []
    try:
        with mock.patch(
            "modules.monitoring.history_manager.log_completed_task",
            side_effect=logged_tasks.append,
        ):
            get_service_stats_with_common_patches()

        ghost_history_entries = [t for t in logged_tasks if t.get("task_id") == "crashed_ghost"]
        assert len(ghost_history_entries) == 1
        assert ghost_history_entries[0]["status"] == "failed"
    finally:
        with scheduler.STATE.task_registry_lock:
            scheduler.STATE.task_registry.clear()
        telemetry._clear_stale_task_warned()


def test_finalize_one_stale_task_skips_when_heartbeat_arrived_since_snapshot(clean_telemetry: None) -> None:
    """Race guard: a task judged stale by an earlier snapshot (e.g. the
    `_build_task_copy` pass inside `get_service_stats()`) must NOT be archived/removed
    if the owning worker reported a fresh heartbeat before `_finalize_one_stale_task`
    actually runs -- the window between the initial staleness decision and finalization.
    `_finalize_one_stale_task` revalidates staleness fresh, under task_registry_lock,
    immediately before marking failed and finalizing, so a task that is current live
    (recent `last_progress_at`) at that moment must be left completely untouched.
    """
    from modules.inference import scheduler

    now = time.time()
    task_id = "recovered-before-finalize"
    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry.clear()
        scheduler.STATE.task_registry[task_id] = {
            "task_id": task_id,
            "status": "active",
            "stage": "Inference",
            "start_time": now,
            "start_active": now,
            "last_progress_at": now,  # fresh -- not stale by the time finalize runs
            "is_priority": False,
            "type": "Transcription",
            "unit_id": "NPU.0",
        }
    try:
        telemetry._finalize_one_stale_task(task_id)

        with scheduler.STATE.task_registry_lock:
            entry = scheduler.STATE.task_registry.get(task_id)
        assert entry is not None, "a task that's still live at finalize time must not be removed"
        assert entry["status"] == "active", "status must not be overwritten to failed for a live task"
    finally:
        with scheduler.STATE.task_registry_lock:
            scheduler.STATE.task_registry.clear()


def _seed_old_but_alive_task(now: float) -> None:
    """Seed a task that started long ago (past _STALE_ACTIVE_TASK_TIMEOUT_SEC) but
    whose worker is still alive and progressing, evidenced by a recent
    `last_progress_at` heartbeat -- the counterpart to the crashed-ghost case."""
    from modules.inference import scheduler

    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry.clear()
        scheduler.STATE.task_registry["long_running_alive"] = {
            "task_id": "long_running_alive",
            "status": "active",
            "stage": "Inference",
            "start_time": now - 999999,
            "start_active": now - 999999,
            "last_progress_at": now,
            "is_priority": False,
            "type": "Transcription",
            "unit_id": "NPU.0",
        }


def test_get_service_stats_keeps_long_running_task_with_recent_progress_active(clean_telemetry: None) -> None:
    """Gap 7.6 (counterpart): age alone must not reap a task. A task started well
    past _STALE_ACTIVE_TASK_TIMEOUT_SEC ago but with a recent `last_progress_at`
    heartbeat (proving its worker is still alive and making progress, e.g. a very
    large media file that legitimately takes many hours) must stay 'active' and be
    counted in active_sessions, not be reaped solely for being old.
    """
    from modules.inference import scheduler

    _seed_old_but_alive_task(time.time())
    try:
        stats = get_service_stats_with_common_patches()
        tasks_by_id = {t.get("task_id"): t for t in stats["tasks"]}
        assert tasks_by_id["long_running_alive"]["status"] == "active"
        assert stats["active_sessions"] == 1
    finally:
        with scheduler.STATE.task_registry_lock:
            scheduler.STATE.task_registry.clear()


def _seed_recently_started_task_with_stale_progress(now: float) -> None:
    """Seed a task whose `start_active` is recent (well inside
    _STALE_ACTIVE_TASK_TIMEOUT_SEC) but whose `last_progress_at` heartbeat is
    itself older than _STALE_ACTIVE_TASK_TIMEOUT_SEC -- simulating a worker
    that reported one early heartbeat then hung/stopped reporting. Exercises
    the branch in `_resolve_last_liveness_signal` where a present-but-stale
    `last_progress_at` is used directly (not masked by a recent start time)."""
    from modules.inference import scheduler

    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry.clear()
        scheduler.STATE.task_registry["stale_progress_recent_start"] = {
            "task_id": "stale_progress_recent_start",
            "status": "active",
            "stage": "Inference",
            "start_time": now - 60,
            "start_active": now - 60,
            "last_progress_at": now - 999999,
            "is_priority": False,
            "type": "Transcription",
            "unit_id": "NPU.0",
        }


def test_get_service_stats_reaps_task_with_stale_progress_despite_recent_start(clean_telemetry: None) -> None:
    """Gap 7.6 (further counterpart): a recent `start_active` must not mask a
    stale `last_progress_at`. A task that started recently but whose last
    reported progress heartbeat is far older than
    _STALE_ACTIVE_TASK_TIMEOUT_SEC (worker reported once, then hung) must
    still be reaped -- the reaping logic keys off the age of the most recent
    liveness signal, not the age of the task's start time.
    """
    from modules.inference import scheduler

    _seed_recently_started_task_with_stale_progress(time.time())
    try:
        stats = get_service_stats_with_common_patches()
        tasks_by_id = {t.get("task_id"): t for t in stats["tasks"]}
        assert tasks_by_id["stale_progress_recent_start"]["status"] == "failed"
        assert stats["active_sessions"] == 0
    finally:
        with scheduler.STATE.task_registry_lock:
            scheduler.STATE.task_registry.clear()


def _register_bare_task_for_current_thread(task_id: str) -> int:
    """Register a minimal 'active' task keyed by the current thread's id (the
    fallback key `update_task_metadata`/`update_task_progress` use when
    `THREAD_CONTEXT.task_id` is unset), and clear the registry first. Returns
    the thread id used as the registry key."""
    from modules.inference import scheduler

    thread_id = threading.get_ident()
    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry.clear()
        scheduler.STATE.task_registry[thread_id] = {"task_id": task_id, "status": "active"}
    return thread_id


def _assert_recent_heartbeat_stamped(entry: dict[str, Any], before: float, after: float) -> None:
    """Assert `entry['last_progress_at']` was stamped within [before, after] and is
    the exact value `_resolve_last_liveness_signal` consumes as the liveness signal."""
    assert entry.get("last_progress_at") is not None
    assert before <= entry["last_progress_at"] <= after
    assert telemetry._resolve_last_liveness_signal(entry, after) == entry["last_progress_at"]


def test_update_task_metadata_stamps_last_progress_at_heartbeat() -> None:
    """`update_task_metadata()` (e.g. called with `current_position=...` during
    transcription/preprocessing) must itself stamp `last_progress_at`, not only
    `update_task_progress()` -- otherwise a real progress signal reported purely
    through metadata would never refresh liveness and could be wrongly reaped by
    `_resolve_last_liveness_signal` after `_STALE_ACTIVE_TASK_TIMEOUT_SEC`."""
    from modules.core import utils
    from modules.inference import scheduler

    thread_id = _register_bare_task_for_current_thread("meta-heartbeat-task")
    # Save the prior THREAD_CONTEXT dict so it can be restored even if assertions fail.
    prior_context = utils.THREAD_CONTEXT._cv.get()
    try:
        utils.THREAD_CONTEXT.reset()
        before = time.time()
        scheduler.update_task_metadata(current_position=42.0)
        after = time.time()

        with scheduler.STATE.task_registry_lock:
            entry = dict(scheduler.STATE.task_registry[thread_id])

        assert entry.get("current_position") == 42.0
        _assert_recent_heartbeat_stamped(entry, before, after)
    finally:
        # Restore the prior THREAD_CONTEXT state so later tests see a clean slate.
        utils.THREAD_CONTEXT._cv.set(prior_context)
        with scheduler.STATE.task_registry_lock:
            scheduler.STATE.task_registry.clear()


def test_update_task_progress_stamps_last_progress_at_heartbeat() -> None:
    """`update_task_progress()` must stamp `last_progress_at` on a genuine
    progress/stage update, the same heartbeat `_resolve_last_liveness_signal`
    consumes to decide whether an 'active' task is stale."""
    from modules.core import utils
    from modules.inference import scheduler

    thread_id = _register_bare_task_for_current_thread("progress-heartbeat-task")
    # Save the prior THREAD_CONTEXT dict so it can be restored even if assertions fail.
    prior_context = utils.THREAD_CONTEXT._cv.get()
    try:
        utils.THREAD_CONTEXT.reset()
        before = time.time()
        scheduler.update_task_progress(50, "Inference")
        after = time.time()

        with scheduler.STATE.task_registry_lock:
            entry = dict(scheduler.STATE.task_registry[thread_id])

        assert entry.get("progress") == 50
        assert entry.get("stage") == "Inference"
        _assert_recent_heartbeat_stamped(entry, before, after)
    finally:
        # Restore the prior THREAD_CONTEXT state so later tests see a clean slate.
        utils.THREAD_CONTEXT._cv.set(prior_context)
        with scheduler.STATE.task_registry_lock:
            scheduler.STATE.task_registry.clear()
