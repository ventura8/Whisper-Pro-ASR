"""End-to-end concurrency tests across hardware topologies (0, 1, 2, 3, 4 HW units)."""

from __future__ import annotations

import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any
from unittest import mock

from modules.inference.runtime import model_manager
from tests.integration.concurrency.concurrency_fixtures import (
    HW_TOPOLOGY_0_CPU,
    HW_TOPOLOGY_1_NPU,
    HW_TOPOLOGY_2_DUAL,
    HW_TOPOLOGY_3_TRIPLE,
    HW_TOPOLOGY_4_QUAD,
    assert_all_responses_successful,
    execute_concurrent_workload,
    run_concurrency_test_harness,
)


@contextmanager
def _record_scheduler_events() -> Generator[list[dict[str, Any]], None, None]:
    events: list[dict[str, Any]] = []
    lock = threading.Lock()
    orig_lock_ctx = model_manager.model_lock_ctx

    @contextmanager
    def instrumented_lock_ctx(*args: Any, **kwargs: Any) -> Generator[tuple[Any, str], None, None]:
        is_prio = kwargs.get("priority", False) or (args[0] if args else False)
        with orig_lock_ctx(*args, **kwargs) as (model, unit_id):
            with lock:
                events.append(
                    {
                        "priority": is_prio,
                        "unit_id": unit_id,
                        "time": time.time(),
                    }
                )
            yield model, unit_id

    with mock.patch("modules.inference.runtime.model_manager.model_lock_ctx", side_effect=instrumented_lock_ctx):
        yield events


def _run_hw_matrix_case(
    hw_topology: list[dict[str, str]],
    specs: list[dict[str, str]],
    confidence: float,
    expected_units: set[str],
) -> None:
    """Run a hardware-matrix scenario with deterministic scheduler lock accounting."""
    with (
        _record_scheduler_events() as events,
        mock.patch("modules.core.config.ENABLE_LD_REQUEST_COALESCING", False),
        run_concurrency_test_harness(hw_topology, confidence=confidence) as client,
    ):
        responses = execute_concurrent_workload(client, specs)
        assert_all_responses_successful(responses)
        assert len(events) == len(specs)
        for event in events:
            assert event["unit_id"] in expected_units


def test_e2e_0_hw_cpu_fallback_concurrency(sample_wav: str):
    """Verify that 0 hardware units fallback to Host CPU and process concurrent requests."""
    specs = [
        {"endpoint": "/asr", "local_path": sample_wav},
        {"endpoint": "/detect-language", "local_path": sample_wav},
        {"endpoint": "/v1/audio/transcriptions", "local_path": sample_wav},
    ]
    _run_hw_matrix_case(HW_TOPOLOGY_0_CPU, specs, 0.9, {"CPU"})


def test_e2e_1_hw_unit_preemption(sample_wav: str):
    """Verify concurrency and preemption on 1 hardware unit (Intel NPU)."""
    specs = [
        {"endpoint": "/asr", "local_path": sample_wav},
        {"endpoint": "/detect-language", "local_path": sample_wav},
        {"endpoint": "/detectlang", "local_path": sample_wav},
    ]
    _run_hw_matrix_case(HW_TOPOLOGY_1_NPU, specs, 0.9, {"NPU.0"})


def test_e2e_2_hw_units_parallel_and_targeted(sample_wav: str):
    """Verify dual hardware units allow parallel execution and targeted borrowing."""
    specs = [
        {"endpoint": "/asr", "local_path": sample_wav},
        {"endpoint": "/asr", "local_path": sample_wav},
        {"endpoint": "/detect-language", "local_path": sample_wav},
        {"endpoint": "/detectlang", "local_path": sample_wav},
    ]
    _run_hw_matrix_case(HW_TOPOLOGY_2_DUAL, specs, 0.9, {"NPU.0", "GPU.0"})


def test_e2e_3_hw_units_heterogeneous_matrix(sample_wav: str):
    """Verify triple heterogeneous hardware units (NPU, GPU, CUDA) handle mixed traffic."""
    specs = [
        {"endpoint": "/asr", "local_path": sample_wav},
        {"endpoint": "/v1/audio/transcriptions", "local_path": sample_wav},
        {"endpoint": "/v1/audio/translations", "local_path": sample_wav},
        {"endpoint": "/detect-language", "local_path": sample_wav},
        {"endpoint": "/detectlang", "local_path": sample_wav},
    ]
    _run_hw_matrix_case(HW_TOPOLOGY_3_TRIPLE, specs, 0.95, {"NPU.0", "GPU.0", "CUDA.0"})


def test_e2e_4_hw_units_full_cluster(sample_wav: str):
    """Verify quad hardware units (NPU, GPU, CUDA, CPU) handle multi-request bursts."""
    specs: list[dict[str, str]] = []
    for _ in range(4):
        specs.append({"endpoint": "/asr", "local_path": sample_wav})
        specs.append({"endpoint": "/detect-language", "local_path": sample_wav})

    _run_hw_matrix_case(HW_TOPOLOGY_4_QUAD, specs, 0.99, {"NPU.0", "GPU.0", "CUDA.0", "CPU"})
