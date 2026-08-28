"""Tests for model_manager resource lifecycle: unload/reclamation, preemption
and priority handling, vocal-isolation wrapper paths, and idle-timeout reclamation.

Split out from test_model_manager.py to keep that file under the project's
module-line-count threshold.
"""

import contextlib
import threading
import time
from unittest import mock

from modules.core import utils
from modules.inference import scheduler
from modules.inference.runtime import model_manager
from tests.thread_join_helpers import join_and_assert_terminated


class TestResourceManagement:
    """Tests for resource unloading and sessions."""

    def test_format_reclamation_memory_includes_cuda_vram_when_available(self):
        """Reclaim logs should include CUDA VRAM when NVIDIA metrics are available."""
        text = model_manager._format_reclamation_memory({"app_memory_gb": 3.39, "cuda_vram_mb": 4180})
        assert text == "RAM(RSS)=3.39 GB, CUDA VRAM=4180 MB"

    def test_format_reclamation_delta_includes_ram_and_cuda(self):
        """Reclaim delta should report both RSS and VRAM changes."""
        delta = model_manager._format_reclamation_delta(
            {"app_memory_gb": 3.39, "cuda_vram_mb": 4180},
            {"app_memory_gb": 2.64, "cuda_vram_mb": 910},
        )
        assert delta == "RAM(RSS)=+0.75 GB, CUDA VRAM=+3270 MB"

    def test_decrement_active_session_triggers_unload(self):
        """Test that idle state triggers unload when aggressive offload is on."""
        pm = mock.MagicMock()
        model_manager.PREPROCESSOR_POOL["CPU"] = pm
        model_manager.MODEL_POOL["CPU"] = mock.MagicMock()
        scheduler.STATE.active_sessions = 1

        with (
            mock.patch("modules.core.config.AGGRESSIVE_OFFLOAD", True),
            mock.patch("modules.core.config.MODEL_IDLE_TIMEOUT", 0),
            mock.patch("modules.inference.runtime.model_manager.utils.get_system_telemetry", return_value={}),
        ):
            model_manager.decrement_active_session()
            assert scheduler.STATE.active_sessions == 0
            assert len(model_manager.MODEL_POOL) == 0
            pm.unload_model.assert_called_once()

    def test_unload_models(self):
        """Test explicit model purging."""
        model_manager.MODEL_POOL["CPU"] = mock.MagicMock()
        pm = mock.MagicMock()
        model_manager.PREPROCESSOR_POOL["CPU"] = pm

        with mock.patch("modules.inference.runtime.model_manager.utils.get_system_telemetry", return_value={}):
            model_manager.unload_models()
            assert len(model_manager.MODEL_POOL) == 0
            pm.unload_model.assert_called_once()

    def test_unload_models_clears_multi_cuda_units(self):
        """Explicit purge should clear all per-unit CUDA models and preprocessors."""
        model_manager.MODEL_POOL["cuda:0"] = mock.MagicMock()
        model_manager.MODEL_POOL["cuda:1"] = mock.MagicMock()
        pm0 = mock.MagicMock()
        pm1 = mock.MagicMock()
        model_manager.PREPROCESSOR_POOL["cuda:0"] = pm0
        model_manager.PREPROCESSOR_POOL["cuda:1"] = pm1

        with mock.patch("modules.inference.runtime.model_manager.utils.get_system_telemetry", return_value={}):
            model_manager.unload_models()

        assert len(model_manager.MODEL_POOL) == 0
        assert len(model_manager.PREPROCESSOR_POOL) == 0
        pm0.unload_model.assert_called_once()
        pm1.unload_model.assert_called_once()


class TestPreemptionAndPriority:
    """Tests for priority and preemption logic."""

    def test_wait_for_priority(self):
        """Test priority registration."""
        model_manager.wait_for_priority()
        assert utils.THREAD_CONTEXT.is_priority is True

    def test_run_vocal_isolation_direct_passes_preemption_callback(self):
        """Test that run_vocal_isolation_direct passes check_preemption callback to preprocess_audio."""
        pm = mock.MagicMock()
        model_manager.PREPROCESSOR_POOL["CPU"] = pm

        # Force CPU preprocessing device selection so _resolve_preprocessor_for_unit
        # deterministically picks the "CPU"-keyed pooled mock above, regardless of
        # what accelerator hardware the machine running the test happens to have.
        with mock.patch("modules.core.config.PREPROCESS_DEVICE", "CPU"):
            model_manager.run_vocal_isolation_direct("test.wav", "CPU")

        # Verify preprocess_audio was called with yield_cb=check_preemption
        pm.preprocess_audio.assert_called_once_with("test.wav", force=False, yield_cb=model_manager.check_preemption)

    def test_run_vocal_isolation_uses_preferred_preprocess_device(self):
        """When preprocess device is NPU, UVR should use NPU preprocessor even for CPU ASR units."""
        cpu_pm = mock.MagicMock()
        cpu_pm.device_type = "CPU"
        npu_pm = mock.MagicMock()
        npu_pm.device_type = "NPU"

        model_manager.PREPROCESSOR_POOL["CPU"] = cpu_pm
        model_manager.PREPROCESSOR_POOL["NPU"] = npu_pm

        with mock.patch("modules.core.config.PREPROCESS_DEVICE", "NPU"):
            model_manager.run_vocal_isolation_direct("test.wav", "CPU")

        npu_pm.preprocess_audio.assert_called_once_with("test.wav", force=False, yield_cb=model_manager.check_preemption)
        cpu_pm.preprocess_audio.assert_not_called()

    def test_run_vocal_isolation_uses_assigned_accelerator_preprocessor_per_unit(self):
        """Accelerator-assigned tasks should use their own unit preprocessors (GPU and NPU) in parallel."""
        gpu_pm = mock.MagicMock()
        gpu_pm.device_type = "GPU"
        npu_pm = mock.MagicMock()
        npu_pm.device_type = "NPU"

        model_manager.PREPROCESSOR_POOL["GPU"] = gpu_pm
        model_manager.PREPROCESSOR_POOL["NPU"] = npu_pm

        with mock.patch("modules.core.config.PREPROCESS_DEVICE", "NPU"):
            model_manager.run_vocal_isolation_direct("gpu-task.wav", "GPU")
            model_manager.run_vocal_isolation_direct("npu-task.wav", "NPU")

        gpu_pm.preprocess_audio.assert_called_once_with("gpu-task.wav", force=False, yield_cb=model_manager.check_preemption)
        npu_pm.preprocess_audio.assert_called_once_with("npu-task.wav", force=False, yield_cb=model_manager.check_preemption)

    def test_check_preemption_waits_if_paused(self):
        """check_preemption must wait for resume when a priority task is live.

        Without a priority registry entry, check_preemption self-heals and clears
        pause_confirmed before the main thread can observe it.
        """
        u_sync, worker_done, errors = _exercise_paused_preemption_wait()
        assert worker_done.is_set(), "check_preemption did not return after resume within timeout"
        assert not errors, f"worker thread raised: {errors}"
        assert u_sync["resume_event"].is_set()


def _register_cpu_task(key: object, *, progress: int, is_priority: bool) -> None:
    with scheduler.STATE.task_registry_lock:
        scheduler.STATE.task_registry[key] = {
            "unit_id": "CPU",
            "progress": progress,
            "stage": "Inference",
            "status": "active",
            "is_priority": is_priority,
        }


def _arm_cpu_pause_sync() -> dict:
    u_sync = scheduler.STATE.unit_sync["CPU"]
    u_sync["pause_requested"].set()
    u_sync["resume_event"].clear()
    scheduler.STATE.preemptible_units.add("CPU")
    return u_sync


def _start_blocked_preemption_worker(
    worker_ready: threading.Event,
    worker_done: threading.Event,
    errors: list[Exception],
) -> threading.Thread:
    def blocked_worker() -> None:
        _register_cpu_task(threading.get_ident(), progress=50, is_priority=False)
        worker_ready.set()
        try:
            model_manager.check_preemption()
        except (RuntimeError, ValueError, TypeError, AttributeError, KeyError) as exc:
            errors.append(exc)
        finally:
            worker_done.set()

    worker_thread = threading.Thread(target=blocked_worker, daemon=True)
    worker_thread.start()
    return worker_thread


def _exercise_paused_preemption_wait() -> tuple[dict, threading.Event, list[Exception]]:
    u_sync = _arm_cpu_pause_sync()
    _register_cpu_task("priority-blocker", progress=10, is_priority=True)
    worker_ready = threading.Event()
    worker_done = threading.Event()
    errors: list[Exception] = []
    worker_thread = _start_blocked_preemption_worker(worker_ready, worker_done, errors)
    assert worker_ready.wait(timeout=5.0), "worker never registered"
    assert u_sync["pause_confirmed"].wait(timeout=5.0), "worker never confirmed pause"
    u_sync["pause_requested"].clear()
    u_sync["resume_event"].set()
    worker_thread.join(timeout=5.0)
    return u_sync, worker_done, errors


def test_run_vocal_isolation_wrapper_and_aggressive_offload():
    """Cover wrapper call path and offload branch in direct isolation."""
    pm = mock.MagicMock()
    model_manager.PREPROCESSOR_POOL["CPU"] = pm

    @contextlib.contextmanager
    def _fake_model_lock_ctx():
        yield None, "CPU"

    # Force CPU preprocessing device selection so _resolve_preprocessor_for_unit
    # deterministically picks the "CPU"-keyed pooled mock above, regardless of
    # what accelerator hardware the machine running the test happens to have.
    with mock.patch("modules.core.config.PREPROCESS_DEVICE", "CPU"):
        with mock.patch("modules.inference.runtime.model_manager.model_lock_ctx", _fake_model_lock_ctx):
            model_manager.run_vocal_isolation("audio.wav")

        with mock.patch("modules.core.config.AGGRESSIVE_OFFLOAD", True):
            pm.separator = "loaded"
            model_manager.run_vocal_isolation_direct("audio.wav", "CPU")
            pm.offload.assert_called()


def _wait_until_empty(pool: dict, timeout: float) -> None:
    """Poll `pool` until it's empty or `timeout` seconds elapse."""
    deadline = time.time() + timeout
    while time.time() < deadline and pool:
        time.sleep(0.05)


def test_model_idle_timeout_reclamation():
    """Verify that the background idle timeout thread successfully offloads models."""
    pm = mock.MagicMock()
    model_manager.PREPROCESSOR_POOL["CPU"] = pm
    model_manager.MODEL_POOL["CPU"] = mock.MagicMock()
    scheduler.STATE.active_sessions = 1

    # Configure timeout of 1 second
    with (
        mock.patch("modules.core.config.MODEL_IDLE_TIMEOUT", 1),
        mock.patch("modules.inference.runtime.model_manager.utils.get_system_telemetry", return_value={}),
    ):
        # Simulates task registration/completion lifecycle
        model_manager.decrement_active_session()
        assert scheduler.STATE.active_sessions == 0

        # Model should still be in pool initially
        assert len(model_manager.MODEL_POOL) == 1

        # Poll for the idle-timeout Timer thread to fire and trigger offload, instead of
        # a fixed sleep -- bounded well above the 1s timeout to absorb CI scheduling jitter.
        _wait_until_empty(model_manager.MODEL_POOL, timeout=10)

        assert len(model_manager.MODEL_POOL) == 0, "idle timeout did not clear MODEL_POOL within 10s"
        pm.unload_model.assert_called_once()


def _run_cleaner_vs_init_race(
    entered_unload: threading.Event, release_unload: threading.Event, unit: dict, init_completed: threading.Event
) -> None:
    """Start unload_models vs init_unit race; return after both threads finish."""
    t_clean = threading.Thread(target=model_manager.unload_models, daemon=True, name="t_clean")
    t_clean.start()
    t_init = threading.Thread(target=_run_init_unit_then_signal, args=(unit, init_completed), daemon=True, name="t_init")
    t_init_started = False
    try:
        assert entered_unload.wait(timeout=5.0), "cleanup thread never reached slow_unload"
        t_init.start()
        t_init_started = True
        # init_unit must still be blocked on the pool lock while cleanup holds it.
        assert not init_completed.wait(timeout=0.2)
    finally:
        # Always release the cleanup thread and wait for both threads to actually
        # finish, even on assertion failure -- otherwise slow_unload stays blocked
        # on release_unload while the patched globals above are already reverted.
        release_unload.set()
        join_and_assert_terminated(t_clean, timeout=5.0, label=t_clean.name)
        if t_init_started:
            join_and_assert_terminated(t_init, timeout=5.0, label=t_init.name)


def _run_init_unit_then_signal(unit: dict, init_completed: threading.Event) -> None:
    model_manager.init_unit(unit)
    init_completed.set()


def test_new_task_waits_if_cleaner_is_running():
    """Verify that if a new task arrives while unload_models is executing, it blocks until cleanup completes.

    Uses threading.Event signals rather than sleeps/elapsed-duration thresholds: slow_unload
    signals when it's actually entered (proving unload_models holds the pool lock) and blocks
    on a release event the test controls, so the test can deterministically observe init_unit
    still blocked before releasing cleanup, instead of inferring blocking from timing.
    """
    model_manager.MODEL_POOL["CPU"] = mock.MagicMock()

    entered_unload = threading.Event()
    release_unload = threading.Event()
    mock_model = mock.MagicMock()

    def slow_unload() -> None:
        entered_unload.set()
        release_unload.wait(timeout=5.0)

    mock_model.unload = slow_unload
    model_manager.MODEL_POOL["CPU"] = mock_model

    init_completed = threading.Event()
    unit = {"id": "CPU", "type": "CPU", "name": "CPU"}
    mock_whisper = mock.MagicMock()

    # Mock get_system_telemetry to prevent psutil issues in clean thread
    with (
        mock.patch("modules.inference.runtime.model_manager.utils.get_system_telemetry", return_value={}),
        mock.patch("modules.inference.engines.engine_factory.create_engine", return_value=mock_whisper),
    ):
        _run_cleaner_vs_init_race(entered_unload, release_unload, unit, init_completed)

    assert init_completed.is_set()
    # And the model should have been re-loaded after cleanup finished
    assert "CPU" in model_manager.MODEL_POOL


def test_update_audio_duration_metadata_failure_path():
    """Cover warning path when duration extraction fails."""
    with (
        mock.patch("modules.inference.runtime.model_manager.utils.get_audio_duration", side_effect=RuntimeError("boom")),
        mock.patch("modules.inference.runtime.model_manager.logger.warning") as mock_warning,
    ):
        model_manager._update_audio_duration_metadata("bad.wav")
    mock_warning.assert_called_once()
    assert "Failed to get audio duration early" in mock_warning.call_args.args[0]
    assert "boom" in str(mock_warning.call_args.args[1])


def test_get_status_returns_expected_payload():
    """Cover status payload helper."""
    model_manager.MODEL_POOL["CPU"] = mock.MagicMock()
    with mock.patch("modules.core.config.HARDWARE_UNITS", [{"id": "CPU", "type": "CPU", "name": "Host CPU"}]):
        status = model_manager.get_status()
    assert "active_units" in status
    assert "total_units" in status
