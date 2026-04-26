"""Tests for modules/inference/model_manager.py"""
# pylint: disable=protected-access, too-few-public-methods, redefined-outer-name, unused-import
import threading
import time
import os
import queue
import numpy as np
from unittest import mock
import pytest
from modules.inference import model_manager, scheduler
from modules import utils, config


@pytest.fixture(autouse=True)
def reset_state():
    """Reset model_manager and scheduler global state before each test."""
    model_manager._MODEL_POOL.clear()
    model_manager._PREPROCESSOR_POOL.clear()

    # Mock HARDWARE_UNITS before creating SchedulerState
    with mock.patch("modules.config.HARDWARE_UNITS", [{"id": "CPU", "type": "CPU", "name": "CPU"}]):
        from modules.inference.scheduler import SchedulerState
        scheduler.STATE = SchedulerState()
        scheduler.STATE.engine_initialized = True

    yield

    with mock.patch("modules.config.HARDWARE_UNITS", [{"id": "CPU", "type": "CPU", "name": "CPU"}]):
        from modules.inference.scheduler import SchedulerState
        scheduler.STATE = SchedulerState()


class TestModelLockCtx:
    """Tests for model_lock_ctx."""

    def test_model_lock_ctx_success(self):
        """Test successful hardware unit acquisition."""
        mock_model = mock.MagicMock()
        model_manager._MODEL_POOL["CPU"] = mock_model

        with model_manager.model_lock_ctx() as (model, unit_id):
            assert unit_id == "CPU"
            assert model == mock_model
            # Semaphore value should be 0 (acquired)
            assert scheduler.STATE.model_lock._value == 0

        # Semaphore value should be 1 (released)
        assert scheduler.STATE.model_lock._value == 1

    def test_model_lock_ctx_contention(self):
        """Test contention tracking in model_lock_ctx."""
        scheduler.STATE.model_lock.acquire()  # Busy

        results = []

        def claim():
            try:
                # We need to mock the registry entry for this thread
                thread_id = threading.get_ident()
                with scheduler.STATE.task_registry_lock:
                    scheduler.STATE.task_registry[thread_id] = {"status": "active"}

                with model_manager.model_lock_ctx() as (model, unit_id):
                    results.append(unit_id)
            except Exception as e:
                results.append(str(e))

        t = threading.Thread(target=claim)
        t.start()
        time.sleep(0.1)
        assert scheduler.STATE.queued_sessions == 1

        scheduler.STATE.model_lock.release()
        t.join()
        assert "CPU" in results
        assert scheduler.STATE.queued_sessions == 0


class TestLoadModel:
    """Tests for load_model."""

    def test_load_model_success(self):
        """Test successful engine initialization (lazy initialization of models)."""
        with mock.patch("modules.config.HARDWARE_UNITS", [{"id": "CPU", "type": "CPU", "name": "CPU"}]):
            with mock.patch("modules.inference.preprocessing.PreprocessingManager") as mock_pm_cls:
                success = model_manager.load_model()
                assert success is True
                assert scheduler.STATE.engine_initialized is True
                assert "CPU" in model_manager._PREPROCESSOR_POOL
                mock_pm_cls.assert_called_once()

    def test_init_unit_success(self):
        """Test loading a specific unit's model."""
        unit = {"id": "CPU", "type": "CPU", "name": "CPU"}
        mock_whisper = mock.MagicMock()

        with mock.patch.dict("modules.inference.model_manager._ENGINES", {"WhisperModel": mock_whisper}):
            model_manager._init_unit(unit)
            assert model_manager._MODEL_POOL["CPU"] == mock_whisper.return_value

    def test_init_unit_failure(self):
        """Test error handling during unit initialization."""
        unit = {"id": "CPU", "type": "CPU", "name": "CPU"}
        mock_fail = mock.MagicMock(side_effect=Exception("Load fail"))
        with mock.patch.dict("modules.inference.model_manager._ENGINES", {"WhisperModel": mock_fail}):
            # Should log error but not raise
            model_manager._init_unit(unit)
            assert "CPU" not in model_manager._MODEL_POOL


class TestRunTranscription:
    """Tests for run_transcription."""

    def test_run_transcription_success(self):
        """Test full transcription lifecycle."""
        mock_model = mock.MagicMock()
        mock_info = mock.MagicMock(language="en", language_probability=0.9, duration=10.0)
        # Mock segments as a list of mock objects
        mock_segment = mock.MagicMock(start=0.0, end=1.0, text=" Hello")
        mock_model.transcribe.return_value = ([mock_segment], mock_info)

        model_manager._MODEL_POOL["CPU"] = mock_model

        with mock.patch("modules.config.ENABLE_VOCAL_SEPARATION", False):
            result = model_manager.run_transcription(
                "test.wav", language="en", task="transcribe", batch_size=1)
            assert result["language"] == "en"
            assert "Hello" in result["text"]
            assert result["video_duration_sec"] == 10.0

    def test_run_transcription_vocal_separation(self):
        """Test transcription with vocal separation enabled."""
        mock_model = mock.MagicMock()
        mock_info = mock.MagicMock(language="en", language_probability=0.9, duration=10.0)
        mock_model.transcribe.return_value = ([], mock_info)
        model_manager._MODEL_POOL["CPU"] = mock_model

        pm = mock.MagicMock()
        pm.preprocess_audio.return_value = "isolated.wav"
        model_manager._PREPROCESSOR_POOL["CPU"] = pm

        with mock.patch("modules.config.ENABLE_VOCAL_SEPARATION", True):
            with mock.patch("os.path.exists", return_value=True):
                with mock.patch("os.remove") as mock_remove:
                    model_manager.run_transcription(
                        "original.wav", language="en", task="transcribe", batch_size=1)
                    pm.preprocess_audio.assert_called_with("original.wav", force=False)
                    mock_remove.assert_called_with("isolated.wav")


class TestLanguageDetection:
    """Tests for language detection."""

    def test_run_language_detection_success(self):
        """Test successful language detection."""
        mock_model = mock.MagicMock()
        mock_info = mock.MagicMock(language="fr", language_probability=0.8)
        mock_model.transcribe.return_value = (None, mock_info)
        model_manager._MODEL_POOL["CPU"] = mock_model

        with mock.patch("modules.inference.vad.get_speech_timestamps_from_path", return_value=[{"start": 0, "end": 1}]):
            result = model_manager.run_language_detection("test.wav")
            assert result["detected_language"] == "fr"
            assert result["confidence"] == 0.8

    def test_run_batch_language_detection(self):
        """Test multi-segment language detection."""
        mock_model = mock.MagicMock()
        model_manager._MODEL_POOL["CPU"] = mock_model

        mock_info = mock.MagicMock(language="en", language_probability=0.9)
        # For batch detection, it calls _run_language_detection_core which returns a dict
        with mock.patch("modules.inference.model_manager.run_language_detection_core") as mock_core:
            mock_core.return_value = {"detected_language": "en", "confidence": 0.9}
            with mock.patch("modules.inference.vad.decode_audio", return_value=[0]*(16000*60)):
                results = model_manager.run_batch_language_detection("test.wav", segment_count=2)
                assert len(results) == 2
                assert results[0]["detected_language"] == "en"


class TestResourceManagement:
    """Tests for resource unloading and sessions."""

    def test_decrement_active_session_triggers_unload(self):
        """Test that idle state triggers unload when aggressive offload is on."""
        pm = mock.MagicMock()
        model_manager._PREPROCESSOR_POOL["CPU"] = pm
        model_manager._MODEL_POOL["CPU"] = mock.MagicMock()
        scheduler.STATE.active_sessions = 1

        with mock.patch("modules.config.AGGRESSIVE_OFFLOAD", True), \
                mock.patch("modules.inference.model_manager.utils.get_system_telemetry", return_value={}):
            model_manager.decrement_active_session()
            assert scheduler.STATE.active_sessions == 0
            assert len(model_manager._MODEL_POOL) == 0
            pm.unload_model.assert_called_once()

    def test_unload_models(self):
        """Test explicit model purging."""
        model_manager._MODEL_POOL["CPU"] = mock.MagicMock()
        pm = mock.MagicMock()
        model_manager._PREPROCESSOR_POOL["CPU"] = pm

        with mock.patch("modules.inference.model_manager.utils.get_system_telemetry", return_value={}):
            model_manager.unload_models()
            assert len(model_manager._MODEL_POOL) == 0
            pm.unload_model.assert_called_once()


class TestPreemptionAndPriority:
    """Tests for priority and preemption logic."""

    def test_wait_for_priority(self):
        """Test priority registration."""
        model_manager.wait_for_priority()
        assert utils.THREAD_CONTEXT.is_priority is True

    def test_check_preemption_waits_if_paused(self):
        """Test that _check_preemption waits for resume."""
        scheduler.STATE.pause_requested.set()
        scheduler.STATE.resume_event.clear()

        # We need a task in registry for the current thread
        thread_id = threading.get_ident()
        with scheduler.STATE.task_registry_lock:
            scheduler.STATE.task_registry[thread_id] = {
                "unit_id": "CPU", "progress": 50, "stage": "Inference"}

        # Mock preemptible pool to have our unit back
        scheduler.STATE.preemptible_units.add("CPU")

        # In a separate thread, resume after a bit
        def resume_soon():
            time.sleep(0.1)
            # wait for pause_confirmed
            scheduler.STATE.pause_confirmed.wait()
            scheduler.STATE.pause_requested.clear()
            scheduler.STATE.resume_event.set()

        threading.Thread(target=resume_soon).start()

        # This should block and then return
        model_manager._check_preemption()
        assert scheduler.STATE.resume_event.is_set()


class TestEdgeCases:
    """Tests for various edge cases and error handling."""

    def test_is_engine_initialized(self):
        """Test engine initialization check."""
        scheduler.STATE.engine_initialized = True
        assert model_manager.is_engine_initialized() is True
        scheduler.STATE.engine_initialized = False
        assert model_manager.is_engine_initialized() is False

    def test_is_uvr_actually_loaded(self):
        """Test checking if UVR is actually in RAM."""
        pm = mock.MagicMock()
        pm.separator = "not none"
        model_manager._PREPROCESSOR_POOL["CPU"] = pm
        assert model_manager.is_uvr_actually_loaded() is True

        pm.separator = None
        assert model_manager.is_uvr_actually_loaded() is False


def test_model_manager_booster_edge_cases():
    """Cover miscellaneous uncovered lines in model_manager."""
    # 57: scheduler.STATE.uvr_loaded = True
    with mock.patch("modules.config.ENABLE_VOCAL_SEPARATION", True):
        model_manager.load_model()
        assert scheduler.STATE.uvr_loaded is True

    # 83: logger.info for Intel accelerator fallback
    unit = {"id": "GPU", "type": "GPU", "name": "Intel GPU"}
    with mock.patch("modules.config.ASR_ENGINE", "FASTER-WHISPER"), \
            mock.patch("modules.inference.model_manager._ENGINES", {"WhisperModel": mock.MagicMock()}):
        model_manager._init_unit(unit)

    # 186-187: Cleanup error
    mock_model = mock.MagicMock()
    from argparse import Namespace
    mock_model.transcribe.return_value = ([], Namespace(duration=0, language="en", language_probability=1.0))
    model_manager._MODEL_POOL["CPU"] = mock_model

    with mock.patch("modules.config.ENABLE_VOCAL_SEPARATION", True), \
            mock.patch("modules.inference.model_manager.run_vocal_isolation_direct", return_value="iso.wav"), \
            mock.patch("os.path.exists", return_value=True), \
            mock.patch("os.remove", side_effect=OSError("Locked")):
        model_manager.run_transcription("test.wav", "en", "transcribe")

    # 193: return result if no segments
    assert model_manager._post_process_results({"no_segments": []}) == {"no_segments": []}

    # 243: return audio_path if no preprocessor
    assert model_manager.run_vocal_isolation_direct("test.wav", "NON_EXISTENT") == "test.wav"

    # 285: break in batch LD
    with mock.patch("modules.inference.vad.decode_audio", return_value=np.zeros(100)):
        res = model_manager.run_batch_language_detection_direct(mock_model, "test.wav", 5)
        assert res == []

    # 409: RuntimeError if engine pool is empty
    with mock.patch("modules.inference.model_manager._MODEL_POOL", {}), \
            mock.patch("modules.inference.model_manager._init_unit"):
        with pytest.raises(RuntimeError):
            with model_manager.model_lock_ctx() as (m, u):
                pass
