"""End-to-end tests for idle timeout resource release, cancellation, and thread-safe lazy reloading."""

from __future__ import annotations

from unittest import mock

from modules.inference.runtime import model_manager
from tests.integration.concurrency.concurrency_fixtures import (
    HW_TOPOLOGY_2_DUAL,
    assert_all_responses_successful,
    execute_concurrent_workload,
    run_concurrency_test_harness,
    setup_concurrency_harness,
)


def test_idle_timeout_scheduling_and_cancellation():
    """Verify idle cleanup timer is scheduled when active_sessions==0 and cancelled when new task arrives."""
    patcher, _ = setup_concurrency_harness(HW_TOPOLOGY_2_DUAL)
    try:
        with mock.patch("modules.core.config.MODEL_IDLE_TIMEOUT", 10):
            model_manager.cancel_idle_cleanup()
            assert model_manager.CLEANER_STATE["timer"] is None

            model_manager.increment_active_session()
            assert model_manager.CLEANER_STATE["timer"] is None

            model_manager.decrement_active_session()
            assert model_manager.CLEANER_STATE["timer"] is not None

            model_manager.increment_active_session()
            assert model_manager.CLEANER_STATE["timer"] is None
            model_manager.decrement_active_session()
    finally:
        model_manager.cancel_idle_cleanup()
        patcher.stop()


def _assert_all_model_pools_empty() -> None:
    is_empty = (
        len(model_manager.MODEL_POOL) == 0
        and len(model_manager.PREPROCESSOR_POOL) == 0
        and len(model_manager.DIARIZE_POOL) == 0
        and len(model_manager.ALIGN_POOL) == 0
    )
    assert is_empty


def test_idle_timeout_purges_pools_and_reclaims_memory():
    """Verify idle cleanup callback unloads all model pools when timer fires."""
    patcher, _ = setup_concurrency_harness(HW_TOPOLOGY_2_DUAL)
    try:
        model_manager.MODEL_POOL["NPU.0"] = mock.MagicMock()
        model_manager.PREPROCESSOR_POOL["NPU.0"] = mock.MagicMock()
        assert len(model_manager.MODEL_POOL) > 0

        with mock.patch("modules.core.config.MODEL_IDLE_TIMEOUT", 10):
            model_manager.cancel_idle_cleanup()
            schedule_idle_cleanup_fn = getattr(model_manager, "_schedule_idle_cleanup")
            schedule_idle_cleanup_fn()
            timer_obj = model_manager.CLEANER_STATE["timer"]
            assert timer_obj is not None
            # Deterministically invoke scheduled cleanup callback
            timer_obj.cancel()
            timer_obj.function(*timer_obj.args, **timer_obj.kwargs)
            _assert_all_model_pools_empty()
            assert model_manager.CLEANER_STATE["timer"] is None
    finally:
        model_manager.cancel_idle_cleanup()
        patcher.stop()


def _build_burst_specs(test_wav: str, count: int = 10) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    for i in range(count):
        endpoint = "/asr" if i % 2 == 0 else "/detect-language"
        specs.append({"endpoint": endpoint, "local_path": test_wav})
    return specs


def test_lazy_reload_under_concurrent_traffic_burst(sample_wav: str):
    """Verify that a burst of 10 concurrent requests reloads unloaded pools safely without races."""
    with run_concurrency_test_harness(HW_TOPOLOGY_2_DUAL, confidence=0.95) as client:
        model_manager.unload_models()
        assert len(model_manager.MODEL_POOL) == 0

        responses = execute_concurrent_workload(client, _build_burst_specs(sample_wav, 10), max_workers=10)
        assert_all_responses_successful(responses)


def test_priority_preemption_during_on_demand_reload(sample_wav: str):
    """Verify priority LD task can acquire hardware while standard task triggers on-demand reload."""
    with run_concurrency_test_harness(HW_TOPOLOGY_2_DUAL, text="reloaded", confidence=0.99) as client:
        model_manager.unload_models()

        specs = [
            {"endpoint": "/asr", "local_path": sample_wav},
            {"endpoint": "/detect-language", "local_path": sample_wav},
        ]
        responses = execute_concurrent_workload(client, specs)
        assert_all_responses_successful(responses)
