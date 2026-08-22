"""Shared fixtures, mock harnesses, and dispatch helpers for concurrency e2e tests."""

from __future__ import annotations

import concurrent.futures
import os
import struct
import threading
import uuid
import wave
from collections.abc import Generator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest import mock
from urllib.parse import quote

import pytest

from modules.core import config, utils
from modules.inference import scheduler
from modules.inference.runtime import model_manager
from tests.conftest import FlaskCompatibleClient
from whisper_pro_asr import create_app


@pytest.fixture
def sample_wav() -> Generator[str, None, None]:
    """Generate a temporary sample WAV file for concurrency integration tests."""
    file_path = os.path.abspath(os.path.join(str(config.PREPROCESSING_CACHE_DIR), f"sample_concurrency_{uuid.uuid4().hex}.wav"))
    create_test_wav_file(file_path, duration_sec=1.5)
    try:
        yield file_path
    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass


@contextmanager
def mock_asr_and_ld_pipeline(detected_lang: str = "en", text: str = "hello", confidence: float = 0.95) -> Generator[None, None, None]:
    """Context manager wrapping common pipeline mocks for ASR and Language Detection."""

    def fake_transcription(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        with model_manager.model_lock_ctx(priority=False):
            return {"text": text, "segments": []}

    def fake_voting(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        with model_manager.model_lock_ctx(priority=True):
            return {"detected_language": detected_lang, "confidence": confidence}

    with (
        mock.patch("modules.api.routes.asr._detect_lang_if_needed", return_value=detected_lang),
        mock.patch(
            "modules.api.routes.asr.model_manager.run_transcription",
            side_effect=fake_transcription,
        ),
        mock.patch(
            "modules.inference.pipeline.language_detection.run_voting_detection",
            side_effect=fake_voting,
        ),
    ):
        yield


HW_TOPOLOGY_0_CPU: list[dict[str, str]] = []
HW_TOPOLOGY_1_NPU: list[dict[str, str]] = [
    {"id": "NPU.0", "type": "NPU", "name": "Intel NPU"},
]
HW_TOPOLOGY_2_DUAL: list[dict[str, str]] = [
    {"id": "NPU.0", "type": "NPU", "name": "Intel NPU"},
    {"id": "GPU.0", "type": "GPU", "name": "Intel GPU"},
]
HW_TOPOLOGY_3_TRIPLE: list[dict[str, str]] = [
    {"id": "NPU.0", "type": "NPU", "name": "Intel NPU"},
    {"id": "GPU.0", "type": "GPU", "name": "Intel GPU"},
    {"id": "CUDA.0", "type": "CUDA", "name": "NVIDIA GPU"},
]
HW_TOPOLOGY_4_QUAD: list[dict[str, str]] = [
    {"id": "NPU.0", "type": "NPU", "name": "Intel NPU"},
    {"id": "GPU.0", "type": "GPU", "name": "Intel GPU"},
    {"id": "CUDA.0", "type": "CUDA", "name": "NVIDIA GPU"},
    {"id": "CPU", "type": "CPU", "name": "Host CPU"},
]


def create_test_wav_file(file_path: str, duration_sec: float = 1.0) -> str:
    """Create a minimal valid 16kHz 16-bit mono PCM WAV audio file."""
    sample_rate = 16000
    num_samples = int(sample_rate * duration_sec)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    wav_file: wave.Wave_write | None = None
    with open(file_path, "wb") as raw_file:
        try:
            wav_file = wave.open(raw_file, "wb")
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            pcm_data = bytearray()
            for i in range(num_samples):
                sample_val = int(5000 * (i % 16) / 16.0)
                pcm_data.extend(struct.pack("<h", sample_val))
            wav_file.writeframes(pcm_data)
        finally:
            if wav_file is not None:
                try:
                    wav_file.close()
                except (OSError, ValueError, wave.Error):
                    pass
    return file_path


def reset_scheduler_state_and_pools() -> None:
    """Cancel any pending idle-unload timer and reset scheduler state plus all
    runtime pools (model/preprocessor/diarize/align) to a clean slate. Shared by
    `HarnessContext.reset_pools` and other call sites (e.g.
    tests/integration/concurrency/test_e2e_traffic_volume.py) that need the same
    reset without going through the full HTTP-client harness."""
    model_manager.cancel_idle_cleanup()
    scheduler.STATE = scheduler.SchedulerState()
    model_manager.MODEL_POOL.clear()
    model_manager.PREPROCESSOR_POOL.clear()
    model_manager.DIARIZE_POOL.clear()
    model_manager.ALIGN_POOL.clear()


class HarnessContext:
    """Manages active mock patches and ensures comprehensive teardown."""

    def __init__(self, patchers: list[Any]) -> None:
        self.patchers = patchers

    def stop(self) -> None:
        """Stop all started patches and reset runtime pools."""
        for patcher in reversed(self.patchers):
            patcher.stop()
        self.reset_pools()

    def reset_pools(self) -> None:
        """Reset runtime pools and scheduler state."""
        reset_scheduler_state_and_pools()

    def __enter__(self) -> HarnessContext:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        del exc_type, exc_val, exc_tb
        self.stop()


def _populate_initial_model_pools(hw_list: list[dict[str, str]]) -> None:
    """Populate mock model instances into manager pools for configured units."""
    model_manager.MODEL_POOL.clear()
    model_manager.PREPROCESSOR_POOL.clear()
    model_manager.DIARIZE_POOL.clear()
    model_manager.ALIGN_POOL.clear()

    active_units = hw_list if hw_list else [{"id": "CPU", "type": "CPU", "name": "Host CPU"}]
    for unit in active_units:
        unit_id = unit["id"]
        model_manager.MODEL_POOL[unit_id] = mock.MagicMock()
        model_manager.PREPROCESSOR_POOL[unit_id] = mock.MagicMock()


@contextmanager
def registered_worker_session(is_priority: bool = False) -> Generator[None, None, None]:
    """Reset thread context, register an active/early-registered session, and clean up on exit.

    Shared setup for test worker threads that need to acquire a hardware unit via
    `model_manager.model_lock_ctx`: resets `THREAD_CONTEXT`, increments the active-session
    count, and opens `early_task_registration` for the duration of the `with` block.
    """
    utils.THREAD_CONTEXT.reset()
    model_manager.increment_active_session()
    try:
        with model_manager.early_task_registration(is_priority=is_priority):
            yield
    finally:
        model_manager.decrement_active_session()
        utils.THREAD_CONTEXT.reset()


def setup_concurrency_harness(
    hw_list: list[dict[str, str]],
) -> tuple[HarnessContext, FlaskCompatibleClient]:
    """Patch hardware units and initialize clean scheduler state and test app with rollback on error."""
    started_patchers: list[Any] = []
    try:
        patch_hw = mock.patch("modules.core.config.HARDWARE_UNITS", hw_list)
        patch_hw.start()
        started_patchers.append(patch_hw)

        scheduler.STATE = scheduler.SchedulerState()
        scheduler.STATE.engine_initialized = True
        _populate_initial_model_pools(hw_list)

        patch_telemetry = mock.patch(
            "modules.inference.runtime.model_manager._read_reclamation_memory_snapshot",
            return_value={"app_memory_gb": 0.5, "cuda_vram_mb": None},
        )
        patch_telemetry.start()
        started_patchers.append(patch_telemetry)

        patch_dashboard_psu = mock.patch("modules.monitoring.dashboard.psutil", create=True)
        mock_psu = patch_dashboard_psu.start()
        started_patchers.append(patch_dashboard_psu)
        mock_psu.cpu_percent.return_value = 10.0
        mock_psu.cpu_count.return_value = 8
        mock_psu.virtual_memory.return_value = SimpleNamespace(percent=50.0, used=8 * (1024**3), total=16 * (1024**3))

        patch_dashboard_utils = mock.patch("modules.monitoring.dashboard.utils")
        mock_db_utils = patch_dashboard_utils.start()
        started_patchers.append(patch_dashboard_utils)
        mock_db_utils.get_system_telemetry.return_value = {
            "cpu_percent": 12.5,
            "memory_percent": 42.0,
            "app_cpu_percent": 6.5,
            "memory_used_gb": 7.2,
            "memory_total_gb": 16.0,
            "app_memory_gb": 1.1,
        }

        app = create_app(testing=True)
        client = FlaskCompatibleClient(app)
        harness_ctx = HarnessContext(started_patchers)
        return harness_ctx, client
    except Exception:
        for patcher in reversed(started_patchers):
            patcher.stop()
        reset_scheduler_state_and_pools()
        raise


@contextmanager
def run_concurrency_test_harness(
    hw_topology: list[dict[str, str]],
    confidence: float = 0.95,
    detected_lang: str = "en",
    text: str = "hello",
) -> Generator[FlaskCompatibleClient, None, None]:
    """Context manager for setting up and tearing down concurrency harness with pipeline mocks."""
    harness_ctx, client = setup_concurrency_harness(hw_topology)
    try:
        with mock_asr_and_ld_pipeline(detected_lang=detected_lang, text=text, confidence=confidence):
            yield client
    finally:
        harness_ctx.stop()


def dispatch_single_request(client: FlaskCompatibleClient, spec: dict[str, Any]) -> dict[str, Any]:
    """Execute a single HTTP endpoint request against the test client."""
    endpoint = spec.get("endpoint", "/asr")
    method = spec.get("method", "POST")
    local_path = spec.get("local_path")
    output_format = spec.get("output_format", "json")
    task_type = spec.get("task_type", "transcribe")

    url = f"{endpoint}?output={output_format}&task={task_type}"
    if local_path:
        url += f"&local_path={quote(str(local_path), safe='')}"

    if method.upper() == "GET":
        resp = client.get(url)
    else:
        resp = client.post(url)

    return {
        "status_code": resp.status_code,
        "json": resp.get_json() if resp.status_code == 200 else None,
        "raw": resp.data,
    }


def execute_concurrent_workload(client: FlaskCompatibleClient, specs: list[dict[str, Any]], max_workers: int = 10) -> list[dict[str, Any]]:
    """Execute a list of request specifications concurrently using a thread pool."""
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(dispatch_single_request, client, spec) for spec in specs]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return results


def assert_all_responses_successful(responses: list[dict[str, Any]]) -> None:
    """Assert that every response in a concurrent workload execution returned HTTP 200."""
    for resp in responses:
        assert resp["status_code"] == 200
        assert resp["json"] is not None


@contextmanager
def auto_confirm_priority_waits() -> Generator[None, None, None]:
    """Keep priority pause confirmations flowing during concurrency tests."""
    stop_event = threading.Event()
    errors: list[BaseException] = []

    def run_auto_confirm() -> None:
        try:
            while not stop_event.is_set():
                state = scheduler.STATE
                if state.pause_requested.is_set() and not state.pause_confirmed.is_set():
                    state.pause_confirmed.set()
                for u_sync in list(state.unit_sync.values()):
                    if u_sync["pause_requested"].is_set() and not u_sync["pause_confirmed"].is_set():
                        u_sync["pause_confirmed"].set()
                stop_event.wait(0.01)
        except BaseException as exc:  # background-thread isolation: captured and re-raised
            errors.append(exc)
            raise

    confirm_thread = threading.Thread(target=run_auto_confirm, daemon=True)
    confirm_thread.start()
    try:
        yield
    finally:
        stop_event.set()
        confirm_thread.join(timeout=2.0)
        assert not confirm_thread.is_alive(), "auto_confirm_priority_waits background thread did not stop"
        if errors:
            raise RuntimeError("auto_confirm_priority_waits background thread failed") from errors[0]
