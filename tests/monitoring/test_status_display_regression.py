"""Regression tests for task status display correctness per task_status_display_specification_skill."""

import threading
import time
from unittest import mock

import pytest

from modules.core import utils
from modules.inference import scheduler
from modules.monitoring import telemetry


@pytest.fixture
def clean_scheduler():
    """Reset scheduler state and task registry."""
    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry.clear()
    with scheduler.STATE.task_order_lock:
        scheduler.STATE.task_arrival_order.clear()
    yield
    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry.clear()
    with scheduler.STATE.task_order_lock:
        scheduler.STATE.task_arrival_order.clear()


@pytest.fixture
def clean_telemetry():
    """Reset telemetry history."""
    telemetry._STOP_EVENT.set()
    telemetry.TELEMETRY_HISTORY.clear()
    telemetry._STOP_EVENT.clear()
    yield
    telemetry._STOP_EVENT.set()


def test_unknown_status_is_normalized_before_payload(clean_scheduler, clean_telemetry):
    """Verify unknown/invalid status is normalized before dashboard payload exposure."""
    status_values = ["initializing", "queued", "active", "post-processing", "completed", "failed", "unknown"]

    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry.clear()
        for i, status in enumerate(status_values):
            scheduler.STATE.task_registry[f"task_{status}"] = {
                "task_id": f"task_{status}",
                "status": status,
                "stage": f"Stage for {status}",
                "start_time": 100.0 + i,
                "is_priority": False,
                "type": "Test",
            }

    with mock.patch("modules.monitoring.history_manager.get_history_stats", return_value=([], {})):
        with mock.patch("modules.inference.model_manager.is_engine_actually_loaded", return_value=True):
            with mock.patch("modules.inference.model_manager.is_uvr_actually_loaded", return_value=True):
                stats = telemetry.get_service_stats()

    # Collect all statuses from payload
    payload_statuses = {t.get("status") for t in stats["tasks"]}

    # Display-facing payload must never leak unknown placeholder status.
    assert "unknown" not in payload_statuses
    assert payload_statuses.issubset({"initializing", "queued", "active", "post-processing", "completed", "failed"})


def test_queued_paused_vs_waiting_distinction_by_stage(clean_scheduler, clean_telemetry):
    """Verify queued tasks distinguish paused-for-priority vs waiting-for-hardware via stage field."""
    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry.clear()
        scheduler.STATE.task_registry["paused_task"] = {
            "task_id": "paused_task",
            "status": "queued",
            "stage": "Paused for Priority Task",  # This distinguishes it as paused
            "start_time": 100.0,
            "is_priority": False,
            "type": "Transcription",
        }
        scheduler.STATE.task_registry["waiting_task"] = {
            "task_id": "waiting_task",
            "status": "queued",
            "stage": "Initializing",  # This indicates waiting for hardware
            "start_time": 110.0,
            "is_priority": False,
            "type": "Transcription",
        }

    with mock.patch("modules.monitoring.history_manager.get_history_stats", return_value=([], {})):
        with mock.patch("modules.inference.model_manager.is_engine_actually_loaded", return_value=True):
            with mock.patch("modules.inference.model_manager.is_uvr_actually_loaded", return_value=True):
                stats = telemetry.get_service_stats()

    tasks_by_id = {t.get("task_id"): t for t in stats["tasks"]}

    paused = tasks_by_id.get("paused_task")
    waiting = tasks_by_id.get("waiting_task")

    assert paused is not None, "Paused task missing from payload"
    assert waiting is not None, "Waiting task missing from payload"

    # Both have status='queued', but stages differ
    assert paused["status"] == "queued"
    assert waiting["status"] == "queued"
    assert "Paused for Priority Task" in paused["stage"]
    assert "Paused for Priority Task" not in waiting["stage"]


def test_ordering_active_first_then_priority_queued_then_standard_queued(clean_scheduler, clean_telemetry):
    """Verify deterministic three-tier ordering: active, priority-queued, standard-queued."""
    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry.clear()
        # Mix in different order to verify sorting
        tasks_to_add = [
            ("sq3", "queued", 300.0, False),  # Standard queued, late
            ("a2", "active", 200.0, False),  # Active, late
            ("pq1", "queued", 150.0, True),  # Priority queued, early
            ("a1", "active", 100.0, False),  # Active, early
            ("sq1", "queued", 250.0, False),  # Standard queued, mid
        ]

        for tid, status, start, is_prio in tasks_to_add:
            scheduler.STATE.task_registry[tid] = {
                "task_id": tid,
                "status": status,
                "stage": "Stage",
                "start_time": start,
                "is_priority": is_prio,
                "type": "Test",
            }

    with mock.patch("modules.monitoring.history_manager.get_history_stats", return_value=([], {})):
        with mock.patch("modules.inference.model_manager.is_engine_actually_loaded", return_value=True):
            with mock.patch("modules.inference.model_manager.is_uvr_actually_loaded", return_value=True):
                stats = telemetry.get_service_stats()

    task_order = [t.get("task_id") for t in stats["tasks"]]

    # Expected: active (100, 200) → priority queued (150) → standard queued (250, 300)
    assert task_order == ["a1", "a2", "pq1", "sq1", "sq3"]


def test_task_arrival_order_tracking_for_determinism(clean_scheduler):
    """Verify task_arrival_order registry tracks arrival times for deterministic ordering."""
    task_ids = []
    barrier = threading.Barrier(4)
    start_events = [threading.Event() for _ in range(3)]

    def run_reg(index, filename):
        start_events[index].wait()
        with scheduler.early_task_registration(filename=filename):
            task_ids.append(utils.THREAD_CONTEXT.task_id)
            barrier.wait(timeout=5)
            barrier.wait(timeout=5)

    threads = [threading.Thread(target=run_reg, args=(i, f"task{i + 1}")) for i in range(3)]
    for t in threads:
        t.start()

    # Deterministically start and register each thread one by one
    for i in range(3):
        start_events[i].set()
        timeout = 5.0
        start_time = time.time()
        while True:
            with scheduler.STATE.task_order_lock:
                if len(scheduler.STATE.task_arrival_order) == i + 1:
                    break
            if time.time() - start_time > timeout:
                raise AssertionError(f"Timeout waiting for task {i + 1} to register in task_arrival_order")
            time.sleep(0.001)

    barrier.wait(timeout=5)  # Wait for all threads to register and block

    try:
        # Verify arrival order is tracked and correct
        with scheduler.STATE.task_order_lock:
            for tid in task_ids:
                assert tid in scheduler.STATE.task_arrival_order

            ordered = sorted(scheduler.STATE.task_arrival_order.items(), key=lambda x: x[1])
            ordered_ids = [tid for tid, _ in ordered]
            assert ordered_ids == task_ids
    finally:
        barrier.wait(timeout=5)  # Release threads to clean up
        for t in threads:
            t.join(timeout=2.0)


def test_status_transition_preemption_cycle(clean_scheduler, clean_telemetry):
    """Verify standard task status transitions correctly during preemption pause/resume cycle."""
    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry.clear()

        # Simulate active standard task that gets preempted
        scheduler.STATE.task_registry["standard_asr"] = {
            "task_id": "standard_asr",
            "status": "active",
            "stage": "Inference",
            "start_time": 100.0,
            "is_priority": False,
            "type": "Transcription",
        }

    # Before preemption: status = active
    with mock.patch("modules.monitoring.history_manager.get_history_stats", return_value=([], {})):
        with mock.patch("modules.inference.model_manager.is_engine_actually_loaded", return_value=True):
            with mock.patch("modules.inference.model_manager.is_uvr_actually_loaded", return_value=True):
                stats_before = telemetry.get_service_stats()

    task_before = next((t for t in stats_before["tasks"] if t.get("task_id") == "standard_asr"), None)
    assert task_before is not None
    assert task_before["status"] == "active"

    # Simulate preemption: status transitions to queued with paused stage
    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry["standard_asr"]["status"] = "queued"
        scheduler.STATE.task_registry["standard_asr"]["stage"] = "Paused for Priority Task"

    # After preemption: status = queued with paused stage
    with mock.patch("modules.monitoring.history_manager.get_history_stats", return_value=([], {})):
        with mock.patch("modules.inference.model_manager.is_engine_actually_loaded", return_value=True):
            with mock.patch("modules.inference.model_manager.is_uvr_actually_loaded", return_value=True):
                stats_paused = telemetry.get_service_stats()

    task_paused = next((t for t in stats_paused["tasks"] if t.get("task_id") == "standard_asr"), None)
    assert task_paused is not None
    assert task_paused["status"] == "queued"
    assert "Paused for Priority Task" in task_paused["stage"]

    # Simulate resumption: status transitions back to active
    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry["standard_asr"]["status"] = "active"
        scheduler.STATE.task_registry["standard_asr"]["stage"] = "Inference"

    # After resumption: status = active again
    with mock.patch("modules.monitoring.history_manager.get_history_stats", return_value=([], {})):
        with mock.patch("modules.inference.model_manager.is_engine_actually_loaded", return_value=True):
            with mock.patch("modules.inference.model_manager.is_uvr_actually_loaded", return_value=True):
                stats_resumed = telemetry.get_service_stats()

    task_resumed = next((t for t in stats_resumed["tasks"] if t.get("task_id") == "standard_asr"), None)
    assert task_resumed is not None
    assert task_resumed["status"] == "active"


def test_hardware_units_show_busy_for_translating_and_inference(clean_scheduler, clean_telemetry):
    """Verify active Whisper ASR work keeps every occupied hardware unit marked busy."""
    from modules.inference import model_manager

    mock_units = [
        {"id": "GPU.0", "type": "GPU", "name": "Intel Arc"},
        {"id": "NPU.0", "type": "NPU", "name": "Intel NPU"},
    ]

    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry.clear()
        scheduler.STATE.task_registry["translate_task"] = {
            "task_id": "translate_task",
            "status": "active",
            "stage": "Translating",
            "start_time": 100.0,
            "is_priority": False,
            "type": "Transcription",
            "unit_id": "GPU.0",
        }
        scheduler.STATE.task_registry["infer_task"] = {
            "task_id": "infer_task",
            "status": "active",
            "stage": "Inference",
            "start_time": 101.0,
            "is_priority": False,
            "type": "Transcription",
            "unit_id": "NPU.0",
        }

    with mock.patch("modules.core.config.HARDWARE_UNITS", mock_units):
        with mock.patch.dict(model_manager.MODEL_POOL, {"GPU.0": mock.MagicMock(), "NPU.0": mock.MagicMock()}):
            with mock.patch("modules.monitoring.history_manager.get_history_stats", return_value=([], {})):
                with mock.patch("modules.inference.model_manager.is_engine_actually_loaded", return_value=True):
                    with mock.patch("modules.inference.model_manager.is_uvr_actually_loaded", return_value=False):
                        stats = telemetry.get_service_stats()

    hardware_by_id = {unit["id"]: unit for unit in stats["hardware_units"]}
    assert hardware_by_id["GPU.0"]["whisper_status"] == "busy"
    assert hardware_by_id["NPU.0"]["whisper_status"] == "busy"


def test_no_unknown_status_leakage_in_normal_operation(clean_scheduler, clean_telemetry):
    """Verify unknown status does not leak to dashboard under normal conditions."""
    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry.clear()
        # Only add tasks with valid canonical statuses
        for status in ["initializing", "queued", "active", "completed"]:
            scheduler.STATE.task_registry[f"task_{status}"] = {
                "task_id": f"task_{status}",
                "status": status,
                "stage": "Valid Stage",
                "start_time": 100.0,
                "is_priority": False,
            }

    with mock.patch("modules.monitoring.history_manager.get_history_stats", return_value=([], {})):
        with mock.patch("modules.inference.model_manager.is_engine_actually_loaded", return_value=True):
            with mock.patch("modules.inference.model_manager.is_uvr_actually_loaded", return_value=True):
                stats = telemetry.get_service_stats()

    unknown_tasks = [t for t in stats["tasks"] if t.get("status") == "unknown"]
    assert len(unknown_tasks) == 0, f"Unknown status leaked: {unknown_tasks}"


def test_concurrent_task_arrivals_deterministic_ordering(clean_scheduler, clean_telemetry):
    """Verify that when 5 tasks arrive concurrently with same start_time, they maintain consistent ordering across multiple get_service_stats() calls."""
    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry.clear()
        scheduler.STATE.task_registry["active_1"] = {
            "task_id": "active_1",
            "status": "active",
            "stage": "Inference",
            "start_time": 100.0,
            "is_priority": False,
            "type": "Transcription",
        }
        scheduler.STATE.task_registry["active_2"] = {
            "task_id": "active_2",
            "status": "active",
            "stage": "Inference",
            "start_time": 100.0,
            "is_priority": False,
            "type": "Transcription",
        }
        scheduler.STATE.task_registry["pq_1"] = {
            "task_id": "pq_1",
            "status": "queued",
            "stage": "Waiting",
            "start_time": 100.0,
            "is_priority": True,
            "type": "Transcription",
        }
        scheduler.STATE.task_registry["pq_2"] = {
            "task_id": "pq_2",
            "status": "queued",
            "stage": "Waiting",
            "start_time": 100.0,
            "is_priority": True,
            "type": "Transcription",
        }
        scheduler.STATE.task_registry["sq_1"] = {
            "task_id": "sq_1",
            "status": "queued",
            "stage": "Waiting",
            "start_time": 100.0,
            "is_priority": False,
            "type": "Transcription",
        }

    orderings = []
    for call_num in range(3):
        with mock.patch("modules.monitoring.history_manager.get_history_stats", return_value=([], {})):
            with mock.patch("modules.inference.model_manager.is_engine_actually_loaded", return_value=True):
                with mock.patch("modules.inference.model_manager.is_uvr_actually_loaded", return_value=True):
                    stats = telemetry.get_service_stats()
        task_order = [t.get("task_id") for t in stats["tasks"]]
        orderings.append(task_order)

    assert orderings[0] == orderings[1], f"Call 1 and 2 differ: {orderings[0]} vs {orderings[1]}"
    assert orderings[1] == orderings[2], f"Call 2 and 3 differ: {orderings[1]} vs {orderings[2]}"

    final_order = orderings[0]
    active_indices = [i for i, tid in enumerate(final_order) if tid.startswith("active_")]
    pq_indices = [i for i, tid in enumerate(final_order) if tid.startswith("pq_")]
    sq_indices = [i for i, tid in enumerate(final_order) if tid.startswith("sq_")]

    assert len(active_indices) == 2
    assert len(pq_indices) == 2
    assert len(sq_indices) == 1
    if active_indices and pq_indices:
        assert max(active_indices) < min(pq_indices)
    if pq_indices and sq_indices:
        assert max(pq_indices) < min(sq_indices)


def test_mixed_all_seven_statuses_with_priority_flags(clean_scheduler, clean_telemetry):
    """Verify that when all 7 statuses are present with mixed priority flags, ordering still respects three-tier rules."""
    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry.clear()
        tasks_to_add = [
            ("init_std", "initializing", 100.0, False),
            ("init_prio", "initializing", 101.0, True),
            ("queue_std1", "queued", 102.0, False),
            ("queue_prio", "queued", 103.0, True),
            ("active_std", "active", 104.0, False),
            ("active_prio", "active", 105.0, True),
            ("postproc", "post-processing", 106.0, False),
            ("completed", "completed", 107.0, False),
            ("failed", "failed", 108.0, False),
        ]
        for tid, status, start, is_prio in tasks_to_add:
            scheduler.STATE.task_registry[tid] = {
                "task_id": tid,
                "status": status,
                "stage": "Test Stage",
                "start_time": start,
                "is_priority": is_prio,
                "type": "Transcription",
            }

    with mock.patch("modules.monitoring.history_manager.get_history_stats", return_value=([], {})):
        with mock.patch("modules.inference.model_manager.is_engine_actually_loaded", return_value=True):
            with mock.patch("modules.inference.model_manager.is_uvr_actually_loaded", return_value=True):
                stats = telemetry.get_service_stats()

    task_order = [t.get("task_id") for t in stats["tasks"]]
    tier1_active = [tid for tid in task_order if tid in ["active_std", "active_prio"]]
    tier2_prio_queued = [tid for tid in task_order if tid == "queue_prio"]
    tier3_std_queued = [tid for tid in task_order if tid == "queue_std1"]
    tier4_others = [tid for tid in task_order if tid in ["init_std", "init_prio", "postproc", "completed", "failed"]]

    tier1_idx = [task_order.index(tid) for tid in tier1_active] if tier1_active else []
    tier2_idx = [task_order.index(tid) for tid in tier2_prio_queued] if tier2_prio_queued else []
    tier3_idx = [task_order.index(tid) for tid in tier3_std_queued] if tier3_std_queued else []
    tier4_idx = [task_order.index(tid) for tid in tier4_others] if tier4_others else []

    if tier1_idx and tier2_idx:
        assert max(tier1_idx) < min(tier2_idx)
    if tier2_idx and tier3_idx:
        assert max(tier2_idx) < min(tier3_idx)
    if tier3_idx and tier4_idx:
        assert max(tier3_idx) < min(tier4_idx)


def test_task_id_lexicographic_tiebreaker_same_start_time(clean_scheduler, clean_telemetry):
    """Verify that when multiple tasks have the same start_time and status, they are ordered lexicographically by task_id."""
    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry.clear()
        task_ids = ["task_c", "task_a", "task_d", "task_b"]
        for tid in task_ids:
            scheduler.STATE.task_registry[tid] = {
                "task_id": tid,
                "status": "queued",
                "stage": "Waiting",
                "start_time": 100.0,
                "is_priority": False,
                "type": "Transcription",
            }

    with mock.patch("modules.monitoring.history_manager.get_history_stats", return_value=([], {})):
        with mock.patch("modules.inference.model_manager.is_engine_actually_loaded", return_value=True):
            with mock.patch("modules.inference.model_manager.is_uvr_actually_loaded", return_value=True):
                stats = telemetry.get_service_stats()

    task_order = [t.get("task_id") for t in stats["tasks"]]
    expected_order = ["task_a", "task_b", "task_c", "task_d"]
    assert task_order == expected_order, f"Expected lexicographic order {expected_order}, but got {task_order}"


def test_stage_based_paused_task_filtering_mixed_load(clean_scheduler, clean_telemetry):
    """Verify that paused tasks (stage='Paused for Priority Task') are correctly distinguished from waiting tasks in heavy mixed load."""
    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry.clear()
        for i in range(5):
            scheduler.STATE.task_registry[f"active_{i}"] = {
                "task_id": f"active_{i}",
                "status": "active",
                "stage": "Inference",
                "start_time": 100.0 + i,
                "is_priority": False,
                "type": "Transcription",
            }
        for i in range(3):
            scheduler.STATE.task_registry[f"paused_prio_{i}"] = {
                "task_id": f"paused_prio_{i}",
                "status": "queued",
                "stage": "Paused for Priority Task",
                "start_time": 200.0 + i,
                "is_priority": False,
                "type": "Transcription",
            }
        for i in range(2):
            scheduler.STATE.task_registry[f"waiting_std_{i}"] = {
                "task_id": f"waiting_std_{i}",
                "status": "queued",
                "stage": "Initializing",
                "start_time": 300.0 + i,
                "is_priority": False,
                "type": "Transcription",
            }

    with mock.patch("modules.monitoring.history_manager.get_history_stats", return_value=([], {})):
        with mock.patch("modules.inference.model_manager.is_engine_actually_loaded", return_value=True):
            with mock.patch("modules.inference.model_manager.is_uvr_actually_loaded", return_value=True):
                stats = telemetry.get_service_stats()

    tasks_by_id = {t.get("task_id"): t for t in stats["tasks"]}
    assert len(stats["tasks"]) == 10, f"Expected 10 tasks total, got {len(stats['tasks'])}"

    for i in range(3):
        paused = tasks_by_id.get(f"paused_prio_{i}")
        assert paused is not None
        assert paused["status"] == "queued"
        assert paused["stage"] == "Paused for Priority Task"

    for i in range(2):
        waiting = tasks_by_id.get(f"waiting_std_{i}")
        assert waiting is not None
        assert waiting["status"] == "queued"
        assert waiting["stage"] == "Initializing"
        assert waiting["stage"] != "Paused for Priority Task"

    for i in range(5):
        active = tasks_by_id.get(f"active_{i}")
        assert active is not None
        assert active["status"] == "active"
        assert active["stage"] == "Inference"
