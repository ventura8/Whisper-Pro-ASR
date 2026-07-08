"""Deterministic stage-aware priority preemption coverage tests."""

import threading
import time
from unittest import mock

import pytest

from modules.api import routes_asr
from modules.core import utils
from modules.inference import concurrency, model_manager, scheduler

_HW_PATCHER = None


def _setup_units(hw_list):
    """Reset scheduler and model/preprocessor pools for a test hardware layout."""
    global _HW_PATCHER
    if _HW_PATCHER is not None:
        _HW_PATCHER.stop()

    _HW_PATCHER = mock.patch("modules.core.config.HARDWARE_UNITS", hw_list)
    _HW_PATCHER.start()
    scheduler.STATE = scheduler.SchedulerState()

    model_manager.MODEL_POOL.clear()
    model_manager.PREPROCESSOR_POOL.clear()
    for unit in hw_list:
        model_manager.MODEL_POOL[unit["id"]] = mock.MagicMock()
        model_manager.PREPROCESSOR_POOL[unit["id"]] = mock.MagicMock()


@pytest.fixture(autouse=True)
def _cleanup_hw_patcher():
    """Ensure HARDWARE_UNITS patch does not leak between tests."""
    global _HW_PATCHER
    yield
    if _HW_PATCHER is not None:
        _HW_PATCHER.stop()
        _HW_PATCHER = None
    scheduler.STATE = scheduler.SchedulerState()
    model_manager.MODEL_POOL.clear()
    model_manager.PREPROCESSOR_POOL.clear()


def _run_priority_detectlang(events, name, delay=0.03):
    """Run a priority detect-language task and record progression events."""
    utils.THREAD_CONTEXT.reset()
    events.append(f"prio_{name}_start")
    model_manager.increment_active_session()
    try:
        with model_manager.early_task_registration(is_priority=True):
            events.append(f"prio_{name}_wait")
            model_manager.wait_for_priority()
            events.append(f"prio_{name}_waited")
            with model_manager.model_lock_ctx(priority=True) as (_, unit_id):
                events.append(f"prio_{name}_unit_{unit_id}")
                time.sleep(delay)
                events.append(f"prio_{name}_done")
    finally:
        model_manager.decrement_active_session()


def _run_standard_hold(events, name, loops=3, delay=0.12):
    """Run a standard ASR-like hold task with periodic cooperative preemption checks."""
    utils.THREAD_CONTEXT.reset()
    events.append(f"asr_{name}_start")
    model_manager.increment_active_session()
    try:
        with model_manager.early_task_registration(is_priority=False):
            with model_manager.model_lock_ctx(priority=False) as (_, unit_id):
                events.append(f"asr_{name}_unit_{unit_id}")
                for i in range(loops):
                    time.sleep(delay)
                    model_manager._check_preemption()
                    events.append(f"asr_{name}_tick_{i}")
                events.append(f"asr_{name}_done")
    finally:
        model_manager.decrement_active_session()


def test_ffmpeg_stage_priority_does_not_wait_for_standard_ffmpeg():
    """Priority detect-language should proceed without waiting on tracked standard FFmpeg counters."""
    _setup_units([{"id": "CPU", "type": "CPU", "name": "Host CPU"}])
    events = []

    # Simulate at-capacity state so wait_for_priority triggers preemption and FFmpeg drain.
    scheduler.STATE.active_sessions = 2  # > accel_limit=1

    with utils.STANDARD_FFMPEG_COND:
        utils.STANDARD_FFMPEG_STATE["count"] = 1

    # Auto-confirm pause so the test can complete after FFmpeg drains.
    def _auto_confirm():
        start = time.time()
        while time.time() - start < 5.0 and "prio_ffmpeg_done" not in events:
            if scheduler.STATE.pause_requested.is_set() and not scheduler.STATE.pause_confirmed.is_set():
                scheduler.STATE.pause_confirmed.set()
                for u_sync in scheduler.STATE.unit_sync.values():
                    if not u_sync["pause_confirmed"].is_set():
                        u_sync["pause_confirmed"].set()
            time.sleep(0.01)

    confirm_thread = threading.Thread(target=_auto_confirm, daemon=True)
    confirm_thread.start()

    try:
        t_prio = threading.Thread(target=_run_priority_detectlang, args=(events, "ffmpeg", 0.02))
        t_prio.start()
        time.sleep(0.1)

        assert "prio_ffmpeg_wait" in events

        # Priority path should complete even while standard FFmpeg counters are non-zero.
        t_prio.join(timeout=8.0)

        assert not t_prio.is_alive()
        assert "prio_ffmpeg_waited" in events
        assert "prio_ffmpeg_done" in events
    finally:
        with utils.STANDARD_FFMPEG_COND:
            utils.STANDARD_FFMPEG_STATE["count"] = 0
            utils.STANDARD_FFMPEG_COND.notify_all()
        if t_prio.is_alive():
            t_prio.join(timeout=2.0)
        if "prio_ffmpeg_done" not in events:
            events.append("prio_ffmpeg_done")
        confirm_thread.join(timeout=2.0)


def test_vocal_separation_stage_invokes_yield_hook():
    """Vocal separation must invoke the yield callback and allow concurrent priority completion."""
    _setup_units(
        [
            {"id": "NPU.0", "type": "NPU", "name": "Intel NPU"},
            {"id": "GPU.0", "type": "GPU", "name": "Intel GPU"},
        ]
    )
    events = []
    hook_calls = []
    vocal_started = threading.Event()

    def fake_check_preemption():
        hook_calls.append(1)

    def mock_preprocess(_audio_path, force=False, yield_cb=None):
        _ = force
        events.append("vocal_start")
        vocal_started.set()
        for i in range(4):
            time.sleep(0.04)
            if yield_cb:
                yield_cb()
            events.append(f"vocal_tick_{i}")
        events.append("vocal_done")
        return "isolated.wav"

    model_manager.PREPROCESSOR_POOL["NPU.0"].preprocess_audio = mock_preprocess

    def run_vocal_stage():
        utils.THREAD_CONTEXT.reset()
        model_manager.increment_active_session()
        try:
            with model_manager.early_task_registration(is_priority=False):
                result = model_manager.run_vocal_isolation_direct("audio.wav", "NPU.0")
                assert result == "isolated.wav"
                events.append("asr_vocal_done")
        finally:
            model_manager.decrement_active_session()

    with mock.patch("modules.inference.model_manager._check_preemption", side_effect=fake_check_preemption):
        t_std = threading.Thread(target=run_vocal_stage)
        t_std.start()
        assert vocal_started.wait(timeout=2.0)

        t_prio = threading.Thread(target=_run_priority_detectlang, args=(events, "vocal", 0.03))
        t_prio.start()

        t_std.join(timeout=8.0)
        t_prio.join(timeout=8.0)

    assert not t_std.is_alive()
    assert not t_prio.is_alive()
    assert "asr_vocal_done" in events
    assert "prio_vocal_done" in events
    assert len(hook_calls) >= 3


def test_inference_stage_invokes_yield_hook():
    """Inference segment consumption must invoke preemption checks per segment."""
    _setup_units(
        [
            {"id": "GPU.0", "type": "GPU", "name": "Intel GPU"},
            {"id": "CPU", "type": "CPU", "name": "Host CPU"},
        ]
    )
    events = []
    hook_calls = []
    inference_started = threading.Event()
    info = mock.MagicMock(duration=5.0)

    def fake_check_preemption():
        hook_calls.append(1)

    def segments():
        for i in range(5):
            if i == 0:
                inference_started.set()
                events.append("infer_start")
            time.sleep(0.03)
            yield mock.MagicMock(start=float(i), end=float(i + 1), text=f"seg-{i}", words=None)

    def run_inference_stage():
        utils.THREAD_CONTEXT.reset()
        model_manager.increment_active_session()
        try:
            with model_manager.early_task_registration(is_priority=False):
                results = model_manager._consume_segments(
                    segments(),
                    info,
                    "transcribe",
                    diarize=False,
                    min_speakers=None,
                    max_speakers=None,
                    hf_token=None,
                    unit_id="GPU.0",
                    processed_path="audio.wav",
                )
                assert len(results) == 5
                events.append("asr_infer_done")
        finally:
            model_manager.decrement_active_session()

    with mock.patch("modules.inference.model_manager._check_preemption", side_effect=fake_check_preemption):
        t_std = threading.Thread(target=run_inference_stage)
        t_std.start()
        assert inference_started.wait(timeout=2.0)

        t_prio = threading.Thread(target=_run_priority_detectlang, args=(events, "infer", 0.03))
        t_prio.start()

        t_std.join(timeout=8.0)
        t_prio.join(timeout=8.0)

    assert not t_std.is_alive()
    assert not t_prio.is_alive()
    assert "asr_infer_done" in events
    assert "prio_infer_done" in events
    assert len(hook_calls) == 5


@pytest.mark.parametrize(
    ("hw_list", "std_task_count"),
    [
        (
            [
                {"id": "NPU.0", "type": "NPU", "name": "Intel NPU"},
                {"id": "GPU.0", "type": "GPU", "name": "Intel GPU"},
                {"id": "CPU", "type": "CPU", "name": "Host CPU"},
            ],
            2,
        ),
        (
            [
                {"id": "NPU.0", "type": "NPU", "name": "Intel NPU"},
                {"id": "CUDA.0", "type": "CUDA", "name": "NVIDIA GPU"},
                {"id": "CPU", "type": "CPU", "name": "Host CPU"},
            ],
            2,
        ),
        (
            [
                {"id": "GPU.0", "type": "GPU", "name": "Intel GPU"},
                {"id": "CUDA.0", "type": "CUDA", "name": "NVIDIA GPU"},
                {"id": "CPU", "type": "CPU", "name": "Host CPU"},
            ],
            2,
        ),
        (
            [
                {"id": "NPU.0", "type": "NPU", "name": "Intel NPU"},
                {"id": "GPU.0", "type": "GPU", "name": "Intel GPU"},
                {"id": "CUDA.0", "type": "CUDA", "name": "NVIDIA GPU"},
            ],
            1,
        ),
    ],
    ids=["CPU+NPU+GPU", "CPU+NPU+CUDA", "CPU+GPU+CUDA", "NPU+GPU+CUDA"],
)
def test_triple_unit_matrix_priority_liveness(hw_list, std_task_count):
    """All triple-unit combinations should complete standard+priority workloads without deadlock."""
    _setup_units(hw_list)
    events = []

    # Keep contention bounded per matrix entry to avoid non-deterministic starvation.
    std_threads = [
        threading.Thread(target=_run_standard_hold, args=(events, f"s{i}", 3, 0.12)) for i in range(std_task_count)
    ]
    for thread in std_threads:
        thread.start()

    time.sleep(0.08)
    t_prio = threading.Thread(target=_run_priority_detectlang, args=(events, "matrix", 0.03))
    t_prio.start()

    for thread in std_threads:
        thread.join(timeout=12.0)
    t_prio.join(timeout=12.0)

    assert all(not thread.is_alive() for thread in std_threads)
    assert not t_prio.is_alive()
    assert "prio_matrix_done" in events
    for idx in range(std_task_count):
        assert f"asr_s{idx}_done" in events


def test_task_state_restoration_after_preemption_resume():
    """Task status and stage must be properly restored after preemption completes.

    This test verifies the fix for the bug where ASR tasks showed as "queued" in the
    dashboard even though they were actively working in the inference stage after
    resuming from preemption by priority detect-language tasks.
    """
    _setup_units([{"id": "CPU", "type": "CPU", "name": "Host CPU"}])

    # Setup: Register a task as "active" in "Whisper Inference" stage
    task_id = "test_task_123"
    thread_id = threading.get_ident()

    # Manually register the task
    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry[thread_id] = {
            "task_id": task_id,
            "status": "active",
            "stage": "Whisper Inference",
            "progress": 45,
            "unit_id": "CPU",
            "is_priority": False,
        }

    # Set thread context
    utils.THREAD_CONTEXT.task_id = task_id
    utils.THREAD_CONTEXT.is_priority = False

    # Simulate preemption by setting pause_requested
    scheduler.STATE.pause_requested.set()

    # Prepare to resume in a separate thread
    def trigger_resume():
        time.sleep(0.1)
        scheduler.STATE.pause_requested.clear()
        scheduler.STATE.pause_confirmed.wait(timeout=2.0)
        scheduler.STATE.resume_event.set()

    resume_thread = threading.Thread(target=trigger_resume, daemon=True)
    resume_thread.start()

    # Call _check_preemption which should:
    # 1. Detect preemption is needed
    # 2. Mark task as "queued" with stage "Paused for Priority Task"
    # 3. Wait for resume event
    # 4. Restore task to original status and stage
    concurrency._check_preemption()

    # Verify task state is restored correctly
    with scheduler.STATE.task_registry_lock:
        task = scheduler.STATE.task_registry.get(thread_id)
        assert task is not None
        # After resumption, status should be back to "active"
        assert task["status"] == "active", f"Expected status='active', got {task['status']}"
        # After resumption, stage should be back to "Whisper Inference"
        assert task["stage"] == "Whisper Inference", f"Expected stage='Whisper Inference', got {task['stage']}"
        # Progress should be preserved
        assert task["progress"] == 45, f"Expected progress=45, got {task['progress']}"

    resume_thread.join(timeout=5.0)


def test_asr_yields_to_detect_language_before_vocal_separation():
    """ASR tasks invoke preemption check before vocal separation stage."""
    params = {
        "task": "transcribe",
        "language": "en",
        "diarize": False,
        "min_speakers": None,
        "max_speakers": None,
        "hf_token": None,
        "initial_prompt": None,
        "vad_filter": True,
        "word_timestamps": False,
    }

    events = []

    def fake_preemption():
        events.append("check_preemption")

    def fake_run_transcription(*_args, **_kwargs):
        events.append("run_transcription")
        return {"text": "ok", "segments": []}

    with (
        mock.patch("modules.api.routes_utils.initialize_task_context", return_value=("/tmp/in.wav", None, None)),
        mock.patch("modules.api.routes_asr._detect_lang_if_needed", return_value="en"),
        mock.patch("modules.api.routes_asr._check_preemption", side_effect=fake_preemption),
        mock.patch("modules.api.routes_asr.model_manager.run_transcription", side_effect=fake_run_transcription),
        mock.patch("modules.api.routes_asr.model_manager.update_task_progress"),
        mock.patch("modules.api.routes_asr.model_manager.update_task_metadata"),
        mock.patch("modules.api.routes_asr.config.ENABLE_VOCAL_SEPARATION", True),
        mock.patch("modules.api.routes_utils.get_clean_wav_or_error", return_value=("/tmp/clean.wav", None)),
    ):
        result, source_path, err = routes_asr._perform_transcription_task(
            params,
            "Transcription",
            "/tmp/input.mp3",
            None,
        )

    assert err is None
    assert source_path == "/tmp/in.wav"
    assert result["text"] == "ok"
    # Check that preemption check is invoked before vocal separation / run_transcription
    assert "check_preemption" in events
    assert "run_transcription" in events
    assert events.index("check_preemption") < events.index("run_transcription")


def test_asr_yields_again_immediately_before_inference():
    """ASR route must perform a preemption check immediately before run_transcription."""
    params = {
        "task": "transcribe",
        "language": "en",
        "diarize": False,
        "min_speakers": None,
        "max_speakers": None,
        "hf_token": None,
        "initial_prompt": None,
        "vad_filter": True,
        "word_timestamps": False,
    }

    events = []

    def fake_preemption():
        events.append("check_preemption")

    def fake_run_transcription(*_args, **_kwargs):
        events.append("run_transcription")
        return {"text": "ok", "segments": []}

    with (
        mock.patch("modules.api.routes_utils.initialize_task_context", return_value=("/tmp/in.wav", None, None)),
        mock.patch("modules.api.routes_asr._detect_lang_if_needed", return_value="en"),
        mock.patch("modules.api.routes_asr._check_preemption", side_effect=fake_preemption),
        mock.patch("modules.api.routes_asr.model_manager.run_transcription", side_effect=fake_run_transcription),
        mock.patch("modules.api.routes_asr.model_manager.update_task_progress"),
        mock.patch("modules.api.routes_asr.model_manager.update_task_metadata"),
        mock.patch("modules.api.routes_asr.config.ENABLE_VOCAL_SEPARATION", False),
        mock.patch("modules.api.routes_utils.get_clean_wav_or_error", return_value=("/tmp/clean.wav", None)),
    ):
        result, source_path, err = routes_asr._perform_transcription_task(
            params,
            "Transcription",
            "/tmp/input.mp3",
            None,
        )

    assert err is None
    assert source_path == "/tmp/in.wav"
    assert result["text"] == "ok"
    assert events.count("check_preemption") >= 2
    assert events[-2:] == ["check_preemption", "run_transcription"]
