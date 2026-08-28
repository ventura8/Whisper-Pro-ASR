"""End-to-end tests for advanced concurrency edge cases (idle stealing, exception recovery, coalescing, file hygiene)."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import time
from typing import Any
from unittest import mock

from modules.api.routes import detect_coalescing
from modules.core import utils
from modules.inference import scheduler
from modules.inference.runtime import model_manager
from tests.integration.concurrency.concurrency_fixtures import (
    HW_TOPOLOGY_2_DUAL,
    assert_all_responses_successful,
    dispatch_single_request,
    execute_concurrent_workload,
    run_concurrency_test_harness,
    setup_concurrency_harness,
)


def _worker_npu_asr(events: list[str], asr_running_event: threading.Event) -> None:
    utils.THREAD_CONTEXT.reset()
    utils.THREAD_CONTEXT.target_unit_id = "NPU.0"
    model_manager.increment_active_session()
    events.append("init_worker_npu_asr")
    try:
        with model_manager.early_task_registration(task_type="Standard Edge ASR", is_priority=False):
            with model_manager.model_lock_ctx(priority=False):
                events.append("asr_running_npu")
                asr_running_event.set()
                time.sleep(0.1)
                model_manager.check_preemption()
                events.append("asr_done_npu")
    finally:
        model_manager.decrement_active_session()


def _worker_prio_idle(events: list[str]) -> None:
    utils.THREAD_CONTEXT.reset()
    model_manager.increment_active_session()
    events.append("init_worker_prio_idle")
    try:
        with model_manager.early_task_registration(task_type="Priority Edge LD", is_priority=True):
            model_manager.wait_for_priority()
            with model_manager.model_lock_ctx(priority=True) as (_, acquired_unit):
                npu_paused = scheduler.STATE.unit_sync["NPU.0"]["pause_requested"].is_set()
                events.append(f"npu_pause_state_during_prio_{npu_paused}")
                events.append(f"prio_running_{acquired_unit}")
                time.sleep(0.02)
                events.append("prio_done")
    finally:
        model_manager.decrement_active_session()


def _join_threads(*threads: threading.Thread, timeout: float = 5.0) -> None:
    for thread in threads:
        thread.join(timeout=timeout)
        assert not thread.is_alive()


def _assert_idle_stealing_events(events: list[str]) -> None:
    assert {
        "asr_running_npu",
        "prio_done",
        "asr_done_npu",
        "prio_running_GPU.0",
        "npu_pause_state_during_prio_False",
    }.issubset(events)
    assert not scheduler.STATE.unit_sync["NPU.0"]["pause_requested"].is_set()


def test_edge_idle_unit_stealing_without_preemption():
    """Verify priority tasks prefer idle units and do not trigger pause on busy units."""
    patcher, _ = setup_concurrency_harness(HW_TOPOLOGY_2_DUAL)
    events: list[str] = []
    asr_running = threading.Event()

    try:
        t_asr = threading.Thread(target=_worker_npu_asr, args=(events, asr_running))
        t_asr.start()
        assert asr_running.wait(timeout=5.0)

        t_prio = threading.Thread(target=_worker_prio_idle, args=(events,))
        t_prio.start()

        _join_threads(t_asr, t_prio)
        _assert_idle_stealing_events(events)
    finally:
        patcher.stop()


def _worker_standard_task(events: list[str], started_event: threading.Event) -> None:
    utils.THREAD_CONTEXT.reset()
    model_manager.increment_active_session()
    events.append("init_worker_standard_task")
    try:
        with model_manager.early_task_registration(task_type="Standard Failing Test Task", is_priority=False):
            with model_manager.model_lock_ctx(priority=False):
                events.append("std_start")
                started_event.set()
                time.sleep(0.05)
                model_manager.check_preemption()
                events.append("std_resumed_done")
    finally:
        model_manager.decrement_active_session()


def _worker_failing_prio(events: list[str]) -> None:
    utils.THREAD_CONTEXT.reset()
    model_manager.increment_active_session()
    events.append("init_worker_failing_prio")
    try:
        with model_manager.early_task_registration(is_priority=True):
            model_manager.wait_for_priority()
            events.append("prio_failing_waited")
            raise RuntimeError("Simulated priority crash")
    except RuntimeError as exc:
        events.append(f"prio_error_{exc}")
    finally:
        model_manager.decrement_active_session()


def _assert_priority_exception_recovery(events: list[str]) -> None:
    assert "prio_error_Simulated priority crash" in events
    assert "std_resumed_done" in events


def test_edge_priority_task_exception_resumes_standard_task():
    """Verify that if a priority task raises an exception, paused standard task unblocks and resumes."""
    patcher, _ = setup_concurrency_harness(HW_TOPOLOGY_2_DUAL)
    events: list[str] = []
    std_started = threading.Event()

    try:
        # Wait for the standard worker to acquire its model lock before starting priority.
        t_std = threading.Thread(target=_worker_standard_task, args=(events, std_started))
        t_std.start()
        assert std_started.wait(timeout=5.0)

        t_prio = threading.Thread(target=_worker_failing_prio, args=(events,))
        t_prio.start()

        _join_threads(t_std, t_prio, timeout=8.0)
        _assert_priority_exception_recovery(events)
    finally:
        patcher.stop()


def test_edge_request_coalescing_with_preemption(sample_wav: str):
    """Verify 10 duplicate LD calls coalesce into 1 leader + 9 followers without lock contention."""
    leader_count = 0
    follower_count = 0
    count_lock = threading.Lock()
    followers_entered = threading.Event()

    orig_leader = getattr(detect_coalescing, "_run_leader_detection")
    orig_follower = getattr(detect_coalescing, "_await_shared_result_with_dashboard_task")

    async def wrapped_leader(
        shared_future: concurrent.futures.Future[Any],
        dedupe_key: str,
        request_context: dict,
        *,
        run_detection_internal: Any,
    ) -> Any:
        nonlocal leader_count
        is_first_leader = False
        with count_lock:
            leader_count += 1
            is_first_leader = leader_count == 1

        # Synchronize: block the first leader until all nine followers have
        # entered wrapped_follower (incremented follower_count).
        if is_first_leader:
            expected_followers = 9
            deadline = time.monotonic() + 5.0
            while follower_count < expected_followers and not followers_entered.is_set() and time.monotonic() < deadline:
                await asyncio.sleep(0.01)

        return await orig_leader(shared_future, dedupe_key, request_context, run_detection_internal=run_detection_internal)

    async def wrapped_follower(
        shared_future: concurrent.futures.Future[Any],
        dedupe_key: str,
        filename: str,
        *,
        worker_context: dict,
    ) -> Any:
        nonlocal follower_count
        with count_lock:
            follower_count += 1
            if follower_count == 9:
                followers_entered.set()
        return await orig_follower(shared_future, dedupe_key, filename, worker_context=worker_context)

    with (
        mock.patch("modules.api.routes.detect_coalescing._run_leader_detection", side_effect=wrapped_leader),
        mock.patch("modules.api.routes.detect_coalescing._await_shared_result_with_dashboard_task", side_effect=wrapped_follower),
        run_concurrency_test_harness(HW_TOPOLOGY_2_DUAL, confidence=0.99) as client,
    ):
        specs = [{"endpoint": "/detect-language", "local_path": sample_wav} for _ in range(10)]
        responses = execute_concurrent_workload(client, specs, max_workers=10)
        assert_all_responses_successful(responses)
        assert leader_count == 1
        assert follower_count == 9


def _dispatch_and_sample_tracked_files(
    client: Any, spec: dict[str, Any], tracked_acc: list[list[str]], lock: threading.Lock
) -> dict[str, Any]:
    res = dispatch_single_request(client, spec)
    with lock:
        tracked_acc.append(list(utils.get_tracked_files()))
    return res


def _execute_temp_file_workload(
    client: Any, specs: list[dict[str, Any]], worker_tracked: list[list[str]], lock: threading.Lock
) -> list[dict[str, Any]]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(_dispatch_and_sample_tracked_files, client, s, worker_tracked, lock) for s in specs]
        return [f.result() for f in concurrent.futures.as_completed(futures)]


def test_edge_temp_file_hygiene(sample_wav: str):
    """Verify 100% temporary file tracking hygiene after concurrent workload execution."""
    utils.THREAD_CONTEXT.reset()
    try:
        worker_tracked: list[list[str]] = []
        track_lock = threading.Lock()

        specs = [{"endpoint": "/asr" if i % 2 == 0 else "/detect-language", "local_path": sample_wav} for i in range(10)]
        with run_concurrency_test_harness(HW_TOPOLOGY_2_DUAL) as client:
            responses = _execute_temp_file_workload(client, specs, worker_tracked, track_lock)
            assert_all_responses_successful(responses)
            assert all(len(tracked) == 0 for tracked in worker_tracked)
            assert len(utils.get_tracked_files()) == 0
    finally:
        utils.THREAD_CONTEXT.reset()
