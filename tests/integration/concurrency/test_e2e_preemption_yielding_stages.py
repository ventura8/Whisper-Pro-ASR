"""End-to-end tests for all preemption & yielding checkpoints across all pipeline stages."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Generator
from typing import Any
from unittest import mock

from modules.core import utils
from modules.inference.runtime import model_manager
from modules.inference.runtime.model_segment_processing import consume_transcription_segments
from tests.integration.concurrency.concurrency_fixtures import (
    HW_TOPOLOGY_1_NPU,
    setup_concurrency_harness,
)
from tests.thread_join_helpers import join_and_assert_terminated


def _run_priority_detectlang(events: list[str], name: str, delay: float = 0.04) -> None:
    """Run a priority detect-language task recording lifecycle events."""
    utils.THREAD_CONTEXT.reset()
    events.append(f"prio_{name}_start")
    model_manager.increment_active_session()
    try:
        with model_manager.early_task_registration(task_type="Language Detection", is_priority=True):
            events.append(f"prio_{name}_wait")
            model_manager.wait_for_priority()
            events.append(f"prio_{name}_waited")
            with model_manager.model_lock_ctx(priority=True) as (_, unit_id):
                events.append(f"prio_{name}_unit_{unit_id}")
                time.sleep(delay)
                events.append(f"prio_{name}_done")
    finally:
        model_manager.decrement_active_session()


def _worker_uvr_task(test_wav: str, events: list[str]) -> None:
    utils.THREAD_CONTEXT.reset()
    model_manager.increment_active_session()
    try:
        with model_manager.early_task_registration(is_priority=False):
            res = model_manager.run_vocal_isolation_direct(test_wav, "NPU.0")
            assert res == "clean_isolated.wav"
            events.append("asr_uvr_finished")
    finally:
        model_manager.decrement_active_session()


def _execute_uvr_chunk_loop(events: list[str], uvr_started: threading.Event, yield_cb: Callable[[], None] | None) -> None:
    events.append("uvr_start")
    uvr_started.set()
    for chunk_idx in range(4):
        time.sleep(0.04)
        if yield_cb:
            yield_cb()
        events.append(f"uvr_chunk_{chunk_idx}")
    events.append("uvr_done")


def _assert_event_before(events: list[str], earlier: str, later: str) -> None:
    assert earlier in events
    assert later in events
    assert events.index(earlier) < events.index(later)


def _assert_stage_order(events: list[str], first: str, middle: str, last: str) -> None:
    _assert_event_before(events, first, middle)
    _assert_event_before(events, middle, last)


def _assert_uvr_stage_events(t_asr: threading.Thread, t_prio: threading.Thread, events: list[str]) -> None:
    assert not t_asr.is_alive()
    assert not t_prio.is_alive()
    assert "asr_uvr_finished" in events
    _assert_stage_order(events, "uvr_start", "prio_mid_uvr_done", "uvr_done")


def test_stage_mid_uvr_multi_chunk_preemption(sample_wav: str):
    """Mid-UVR multi-chunk separation: priority task interrupts mid-UVR, executes, and UVR resumes."""
    patcher, _ = setup_concurrency_harness(HW_TOPOLOGY_1_NPU)
    events: list[str] = []
    uvr_started = threading.Event()

    def mock_preprocess(audio_path: str, force: bool = False, yield_cb: Any = None) -> str:
        del audio_path, force
        _execute_uvr_chunk_loop(events, uvr_started, yield_cb)
        return "clean_isolated.wav"

    # Force a non-accelerated PREPROCESS_DEVICE so _resolve_preprocessor_for_unit
    # takes its direct unit_id lookup (PREPROCESSOR_POOL.get(unit_id)) instead of
    # its accelerated-device_type-matching branch, which would otherwise ignore
    # the "NPU.0"-keyed mock configured above whenever the machine running this
    # test happens to have real accelerator hardware (making config.PREPROCESS_DEVICE
    # resolve to e.g. "GPU").
    device_patcher = mock.patch("modules.core.config.PREPROCESS_DEVICE", "CPU")
    device_patcher.start()
    t_asr: threading.Thread | None = None
    t_prio: threading.Thread | None = None
    try:
        model_manager.PREPROCESSOR_POOL["NPU.0"].preprocess_audio = mock_preprocess

        t_asr = threading.Thread(target=_worker_uvr_task, args=(sample_wav, events))
        t_asr.start()
        assert uvr_started.wait(timeout=2.0)

        t_prio = threading.Thread(target=_run_priority_detectlang, args=(events, "mid_uvr", 0.03))
        t_prio.start()

        t_asr.join(timeout=10.0)
        t_prio.join(timeout=10.0)

        _assert_uvr_stage_events(t_asr, t_prio, events)
    finally:
        # Unconditional (not just the try-path joins above, which are skipped
        # entirely if _assert_uvr_stage_events raises): both threads must actually
        # terminate before the patched globals are restored, otherwise a still-running
        # straggler thread would keep executing against unpatched config/hooks.
        try:
            join_and_assert_terminated(t_asr, 30.0, "t_asr")
            join_and_assert_terminated(t_prio, 30.0, "t_prio")
        finally:
            device_patcher.stop()
            patcher.stop()


def _mock_segment_generator(events: list[str], infer_started: threading.Event) -> Generator[mock.MagicMock, None, None]:
    for seg_idx in range(5):
        if seg_idx == 0:
            infer_started.set()
            events.append("infer_start")
        time.sleep(0.04)
        events.append(f"infer_seg_{seg_idx}")
        yield mock.MagicMock(start=float(seg_idx), end=float(seg_idx + 1), text=f"seg-{seg_idx}", words=None)


def _worker_inference_task(events: list[str], infer_started: threading.Event) -> None:
    utils.THREAD_CONTEXT.reset()
    model_manager.increment_active_session()
    mock_info = mock.MagicMock(duration=5.0)
    try:
        with model_manager.early_task_registration(is_priority=False):
            results = consume_transcription_segments(
                _mock_segment_generator(events, infer_started),
                mock_info,
                task="transcribe",
                unit_id="NPU.0",
                processed_path="sample.wav",
                preemption_check=model_manager.check_preemption,
                diarize=False,
                min_speakers=None,
                max_speakers=None,
                hf_token=None,
            )
            assert len(results) == 5
            events.append("asr_infer_finished")
    finally:
        model_manager.decrement_active_session()


def _assert_infer_stage_events(t_asr: threading.Thread, t_prio: threading.Thread, events: list[str]) -> None:
    assert not t_asr.is_alive()
    assert not t_prio.is_alive()
    assert "asr_infer_finished" in events
    _assert_stage_order(events, "infer_start", "prio_mid_infer_done", "asr_infer_finished")


def test_stage_mid_whisper_inference_generator_preemption():
    """Mid-Whisper inference generator: segment generator yields per segment and allows preemption."""
    patcher, _ = setup_concurrency_harness(HW_TOPOLOGY_1_NPU)
    events: list[str] = []
    infer_started = threading.Event()

    try:
        t_asr = threading.Thread(target=_worker_inference_task, args=(events, infer_started))
        t_asr.start()
        assert infer_started.wait(timeout=2.0)

        t_prio = threading.Thread(target=_run_priority_detectlang, args=(events, "mid_infer", 0.03))
        t_prio.start()

        t_asr.join(timeout=10.0)
        t_prio.join(timeout=10.0)

        _assert_infer_stage_events(t_asr, t_prio, events)
    finally:
        patcher.stop()


def _worker_multi_stage_asr(events: list[str], stage1_ready: threading.Event, stage2_ready: threading.Event) -> None:
    utils.THREAD_CONTEXT.reset()
    model_manager.increment_active_session()
    try:
        with model_manager.early_task_registration(is_priority=False):
            with model_manager.model_lock_ctx(priority=False):
                events.append("asr_stage1_start")
                for i in range(3):
                    time.sleep(0.05)
                    model_manager.check_preemption()
                    events.append(f"asr_stage1_tick_{i}")
                    if i == 0:
                        stage1_ready.set()

                events.append("asr_stage2_start")
                for j in range(3):
                    time.sleep(0.05)
                    model_manager.check_preemption()
                    events.append(f"asr_stage2_tick_{j}")
                    if j == 0:
                        stage2_ready.set()

                events.append("asr_completed")
    finally:
        model_manager.decrement_active_session()


def _run_successive_prio_threads(
    events: list[str], stage1_ready: threading.Event, stage2_ready: threading.Event
) -> tuple[threading.Thread, threading.Thread]:
    assert stage1_ready.wait(timeout=5.0)
    t1 = threading.Thread(target=_run_priority_detectlang, args=(events, "first", 0.04))
    t1.start()
    t1.join(timeout=5.0)

    assert stage2_ready.wait(timeout=5.0)
    t2 = threading.Thread(target=_run_priority_detectlang, args=(events, "second", 0.04))
    t2.start()
    t2.join(timeout=5.0)

    return t1, t2


def test_stage_successive_multiple_preemptions():
    """Verify an ASR task can be preempted multiple times (at UVR and at inference) and finish cleanly."""
    patcher, _ = setup_concurrency_harness(HW_TOPOLOGY_1_NPU)
    events: list[str] = []
    stage1_ready = threading.Event()
    stage2_ready = threading.Event()

    try:
        t_asr = threading.Thread(target=_worker_multi_stage_asr, args=(events, stage1_ready, stage2_ready))
        t_asr.start()
        t1, t2 = _run_successive_prio_threads(events, stage1_ready, stage2_ready)
        assert not t1.is_alive()
        assert not t2.is_alive()
        t_asr.join(timeout=10.0)

        assert not t_asr.is_alive()
        _assert_stage_order(events, "prio_first_done", "prio_second_done", "asr_completed")
    finally:
        patcher.stop()


def _worker_standard_asr_loop(events: list[str]) -> None:
    utils.THREAD_CONTEXT.reset()
    model_manager.increment_active_session()
    try:
        with model_manager.early_task_registration(is_priority=False):
            with model_manager.model_lock_ctx(priority=False):
                events.append("asr_start")
                for i in range(6):
                    time.sleep(0.04)
                    model_manager.check_preemption()
                    events.append(f"asr_tick_{i}")
                events.append("asr_done")
    finally:
        model_manager.decrement_active_session()


def _run_backlog_prio_threads(events: list[str], count: int = 3) -> None:
    prio_threads = [threading.Thread(target=_run_priority_detectlang, args=(events, f"backlog_{i}", 0.03)) for i in range(count)]
    for t in prio_threads:
        t.start()
        time.sleep(0.01)

    for t in prio_threads:
        t.join(timeout=10.0)


def test_stage_cascading_priority_backlog_hold():
    """Verify paused ASR remains paused while priority backlog tasks remain queued/running."""
    patcher, _ = setup_concurrency_harness(HW_TOPOLOGY_1_NPU)
    events: list[str] = []

    try:
        t_asr = threading.Thread(target=_worker_standard_asr_loop, args=(events,))
        t_asr.start()
        time.sleep(0.03)

        _run_backlog_prio_threads(events, 3)
        t_asr.join(timeout=10.0)

        assert not t_asr.is_alive()
        _assert_stage_order(events, "prio_backlog_0_done", "prio_backlog_2_done", "asr_done")
    finally:
        patcher.stop()
