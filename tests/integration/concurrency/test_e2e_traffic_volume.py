"""End-to-end traffic volume tests (1, 5, 10 LD, ASR, and v1 calls, mixed bursts)."""

from __future__ import annotations

import pytest

from tests.integration.concurrency.concurrency_fixtures import (
    HW_TOPOLOGY_2_DUAL,
    HW_TOPOLOGY_4_QUAD,
    assert_all_responses_successful,
    execute_concurrent_workload,
    run_concurrency_test_harness,
)


@pytest.mark.parametrize("volume", [1, 5, 10], ids=["1-call", "5-calls", "10-calls"])
def test_e2e_pure_ld_volume_tiers(sample_wav: str, volume: int):
    """Verify 1, 5, and 10 concurrent priority LD calls complete successfully."""
    with run_concurrency_test_harness(HW_TOPOLOGY_2_DUAL) as client:
        specs = [{"endpoint": "/detect-language" if i % 2 == 0 else "/detectlang", "local_path": sample_wav} for i in range(volume)]
        responses = execute_concurrent_workload(client, specs)
        assert_all_responses_successful(responses)


@pytest.mark.parametrize("volume", [1, 5, 10], ids=["1-call", "5-calls", "10-calls"])
def test_e2e_pure_asr_volume_tiers(sample_wav: str, volume: int):
    """Verify 1, 5, and 10 concurrent standard ASR calls complete successfully."""
    with run_concurrency_test_harness(HW_TOPOLOGY_2_DUAL) as client:
        specs = [{"endpoint": "/asr", "local_path": sample_wav} for _ in range(volume)]
        responses = execute_concurrent_workload(client, specs)
        assert_all_responses_successful(responses)


@pytest.mark.parametrize("volume", [1, 5, 10], ids=["1-call", "5-calls", "10-calls"])
def test_e2e_pure_v1_volume_tiers(sample_wav: str, volume: int):
    """Verify 1, 5, and 10 concurrent v1 transcription & translation calls complete successfully."""
    with run_concurrency_test_harness(HW_TOPOLOGY_2_DUAL) as client:
        specs = []
        for i in range(volume):
            endpoint = "/v1/audio/transcriptions" if i % 2 == 0 else "/v1/audio/translations"
            specs.append({"endpoint": endpoint, "local_path": sample_wav})

        responses = execute_concurrent_workload(client, specs)
        assert_all_responses_successful(responses)


def test_e2e_heavy_mixed_endpoint_burst(sample_wav: str):
    """Verify heavy 25-request mixed endpoint burst (ASR + v1 transcriptions + v1 translations + LD)."""
    with run_concurrency_test_harness(HW_TOPOLOGY_4_QUAD, confidence=0.99) as client:
        specs = []
        for _ in range(5):
            specs.append({"endpoint": "/asr", "local_path": sample_wav})
            specs.append({"endpoint": "/v1/audio/transcriptions", "local_path": sample_wav})
            specs.append({"endpoint": "/v1/audio/translations", "local_path": sample_wav})
            specs.append({"endpoint": "/detect-language", "local_path": sample_wav})
            specs.append({"endpoint": "/detectlang", "local_path": sample_wav})

        assert len(specs) == 25
        responses = execute_concurrent_workload(client, specs, max_workers=25)
        assert_all_responses_successful(responses)
