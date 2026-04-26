"""Tests for modules/model_manager.py"""
# pylint: disable=protected-access, too-few-public-methods, redefined-outer-name, unused-import
import queue
import threading
import time
import os
from unittest import mock
import pytest
from modules import model_manager, utils


@pytest.fixture(autouse=True)
def reset_state():
    """Reset model_manager global state before each test."""
    model_manager._MODEL_POOL = {}
    model_manager._PREPROCESSOR_POOL = {}
    model_manager._ACTIVE_SESSIONS = 0
    model_manager._QUEUED_SESSIONS = 0
    model_manager._PRIORITY_REQUESTS = 0
    model_manager._PAUSE_REQUESTED.clear()
    model_manager._PAUSE_CONFIRMED.clear()
    model_manager._RESUME_EVENT.set()

    # Initialize HW_POOL with a default CPU unit
    model_manager._HW_POOL = queue.Queue()
    unit = {"id": "CPU", "type": "CPU", "name": "CPU"}
    model_manager._HW_POOL.put(unit)
    model_manager._ACCEL_LIMIT = 1

    # Reset locks
    model_manager._MODEL_LOCK = threading.Semaphore(1)
    model_manager._PRIORITY_SEQUENTIAL_LOCK = threading.Semaphore(1)

    yield


class TestModelLockCtx:
    """Tests for model_lock_ctx."""

    def test_model_lock_ctx_success(self):
        """Test successful hardware unit acquisition."""
        with model_manager.model_lock_ctx() as unit:
            assert unit["id"] == "CPU"
            assert utils.THREAD_CONTEXT.assigned_unit == unit
            assert model_manager._MODEL_LOCK._value == 0

        assert utils.THREAD_CONTEXT.assigned_unit is None
        assert model_manager._MODEL_LOCK._value == 1

    def test_model_lock_ctx_contention(self):
        """Test contention logging in model_lock_ctx."""
        model_manager._MODEL_LOCK = threading.Semaphore(0)  # Busy
        model_manager._QUEUED_SESSIONS = 0

        def claim():
            with model_manager.model_lock_ctx():
                pass

        t = threading.Thread(target=claim)
        t.start()
        time.sleep(0.1)
        assert model_manager._QUEUED_SESSIONS == 1
        model_manager._MODEL_LOCK.release()
        t.join()

    def test_model_lock_ctx_wait_logging(self):
        """Test wait duration logging in model_lock_ctx."""
        with mock.patch("time.time", side_effect=[0, 0, 1.0, 1.1, 1.2]):  # Simulate 1s wait
            with model_manager.model_lock_ctx():
                pass


class TestLoadModel:
    """Tests for load_model."""

    def test_load_model_success(self):
        """Test successful engine initialization."""
        mock_whisper = mock.MagicMock()
        mock_pm = mock.MagicMock()

        with mock.patch("modules.model_manager.WhisperModel", return_value=mock_whisper):
            with mock.patch("modules.preprocessing.PreprocessingManager", return_value=mock_pm):
                with mock.patch("modules.config.HARDWARE_UNITS",
                                [{"id": "CPU", "type": "CPU", "name": "CPU"}]):
                    success = model_manager.load_model()
                    assert success is True
                    assert "CPU" in model_manager._MODEL_POOL
                    assert "CPU" in model_manager._PREPROCESSOR_POOL

    def test_load_model_failure(self):
        """Test load_model failure handling."""
        with mock.patch("modules.model_manager.WhisperModel",
                        side_effect=Exception("Load error")):
            with mock.patch("modules.config.HARDWARE_UNITS",
                            [{"id": "CPU", "type": "CPU", "name": "CPU"}]):
                with mock.patch("modules.config.AGGRESSIVE_OFFLOAD", False):
                    success = model_manager.load_model()
                    assert success is False
                    assert "CPU" not in model_manager._MODEL_POOL


class TestRunTranscription:
    """Tests for run_transcription."""

    def test_run_transcription_success(self):
        """Test full transcription lifecycle."""
        mock_model = mock.MagicMock()
        mock_info = mock.MagicMock(language="en", language_probability=0.9, duration=10.0)
        mock_model.transcribe.return_value = (iter([]), mock_info)

        model_manager._MODEL_POOL["CPU"] = mock_model
        model_manager._PREPROCESSOR_POOL["CPU"] = mock.MagicMock()

        with mock.patch("modules.config.ENABLE_VOCAL_SEPARATION", False):
            with mock.patch("modules.model_manager._init_transcription_stats"):
                with mock.patch("modules.model_manager._post_process_vad",
                                side_effect=lambda r, p: r):
                    result = model_manager.run_transcription("test.wav", language="en")
                    assert result["language"] == "en"
                    assert result["text"] == ""  # iter([]) yields no segments
                    mock_model.transcribe.assert_called_once()

    def test_run_transcription_cleanup_failure(self):
        """Test that transcription handles cleanup errors gracefully."""
        mock_model = mock.MagicMock()
        mock_info = mock.MagicMock(language="en", language_probability=0.9, duration=10.0)
        mock_model.transcribe.return_value = (iter([]), mock_info)
        model_manager._MODEL_POOL["CPU"] = mock_model

        with mock.patch("modules.model_manager._preprocess_audio", return_value=(0, "proc.wav")):
            with mock.patch("os.path.exists", return_value=True):
                with mock.patch("os.remove", side_effect=Exception("Remove err")):
                    with mock.patch("modules.model_manager._init_transcription_stats"):
                        model_manager.run_transcription("test.wav")
                        # Should not raise

    def test_run_transcription_load_failure(self):
        """Test error when engine fails to load during lazy initialization."""
        with mock.patch("modules.model_manager.WhisperModel", side_effect=Exception("Load fail")):
            with pytest.raises(Exception, match="Load fail"):
                model_manager.run_transcription("test.wav")


class TestLanguageDetection:
    """Tests for language detection."""

    def test_run_language_detection_success(self):
        """Test successful language detection."""
        mock_model = mock.MagicMock()
        mock_info = mock.MagicMock(language="fr",
                                   language_probability=0.8,
                                   all_language_probs={"fr": 0.8})
        mock_model.transcribe.return_value = (None, mock_info)

        model_manager._MODEL_POOL["CPU"] = mock_model

        with mock.patch("modules.vad.get_speech_timestamps_from_path",
                        return_value=[{"start": 0, "end": 1}]):
            result = model_manager.run_language_detection("test.wav")
            assert result["detected_language"] == "fr"
            assert result["confidence"] == 0.8

    def test_run_language_detection_no_speech(self):
        """Test language detection fallback when no speech is detected."""
        with mock.patch("modules.vad.get_speech_timestamps_from_path", return_value=[]):
            result = model_manager.run_language_detection("test.wav")
            assert result["detected_language"] == "en"
            assert result["confidence"] == 0.0


class TestResourceManagement:
    """Tests for resource reclamation."""

    def test_decrement_active_session_triggers_offload(self):
        """Test that idle state triggers resource reclamation when enabled."""
        pm = mock.MagicMock()
        model_manager._PREPROCESSOR_POOL["CPU"] = pm
        model_manager._ACTIVE_SESSIONS = 1

        with mock.patch("modules.config.AGGRESSIVE_OFFLOAD", True):
            model_manager.decrement_active_session()
            assert model_manager._ACTIVE_SESSIONS == 0
            pm.offload.assert_called_once()


class TestPreprocessingLogic:
    """Tests for _preprocess_audio and _resolve_inference_path."""

    def test_preprocess_audio_disabled(self):
        """Test preprocessing when disabled in config."""
        with mock.patch("modules.config.ENABLE_VOCAL_SEPARATION", False):
            dur, path = model_manager._preprocess_audio("input.wav")
            assert dur == 0.0
            assert path == "input.wav"

    def test_resolve_inference_path_silent_vocals(self):
        """Test fallback to original if vocals are silent."""
        with mock.patch("modules.vad.get_speech_timestamps_from_path", return_value=[]):
            with mock.patch("modules.utils.secure_remove") as mock_remove:
                path = model_manager._resolve_inference_path("original.wav", "processed.wav")
                assert path == "original.wav"
                mock_remove.assert_called_with("processed.wav")

    def test_resolve_inference_path_low_confidence(self):
        """Test that low confidence NO LONGER triggers fallback."""
        with mock.patch("modules.vad.get_speech_timestamps_from_path", return_value=[{"start": 0}]):
            # Even if we were to mock LD, it's not called anymore in Step 2.
            # We assert that the processed path is returned.
            path = model_manager._resolve_inference_path("original.wav", "processed.wav")
            assert path == "processed.wav"

    def test_resolve_inference_path_error(self):
        """Test error handling in _resolve_inference_path."""
        with mock.patch("modules.vad.get_speech_timestamps_from_path",
                        side_effect=Exception("VAD err")):
            path = model_manager._resolve_inference_path("orig.wav", "proc.wav")
            assert path == "proc.wav"  # Fallback to input if logic fails


class TestTaskMetadata:
    """Tests for task registry metadata updates."""

    def test_update_task_metadata(self):
        """Test updating task metadata in registry."""
        thread_id = threading.get_ident()
        model_manager._TASK_REGISTRY[thread_id] = {"task_id": "123"}

        model_manager.update_task_metadata(filename="test.mp3", video_duration=120)

        assert model_manager._TASK_REGISTRY[thread_id]["filename"] == "test.mp3"
        assert model_manager._TASK_REGISTRY[thread_id]["video_duration"] == 120

        del model_manager._TASK_REGISTRY[thread_id]


class TestBatchLogic:
    """Tests for batch isolation and language detection."""

    def test_run_vocal_isolation(self):
        """Test public vocal isolation utility."""
        with mock.patch("modules.model_manager._preprocess_audio") as mock_prep:
            mock_prep.return_value = (1.5, "proc.wav")
            res = model_manager.run_vocal_isolation("orig.wav", force=True)
            assert res == "proc.wav"
            mock_prep.assert_called_with("orig.wav", yield_cb=None, force=True)

    def test_run_batch_language_detection_success(self):
        """Test successful batch language detection scan."""
        mock_model = mock.MagicMock()
        model_manager._MODEL_POOL["CPU"] = mock_model

        mock_res = {"detected_language": "en", "all_probabilities": {"en": 1.0}}

        # Mock VAD dependencies for in-memory processing
        with mock.patch("modules.vad.decode_audio", return_value=[0.0]*(16000*60)):
            with mock.patch("modules.vad.get_speech_timestamps",
                            return_value=[{"start": 0, "end": 60}]):
                with mock.patch("modules.model_manager._run_language_detection_core",
                                return_value=mock_res):
                    results = model_manager.run_batch_language_detection("montage.wav", 2)
                    assert len(results) == 2
                    assert results[0]["detected_language"] == "en"

    def test_run_batch_language_detection_failure(self):
        """Test failure handling in batch scan."""
        mock_model = mock.MagicMock()
        model_manager._MODEL_POOL["CPU"] = mock_model

        with mock.patch("modules.vad.decode_audio", side_effect=Exception("decode err")):
            results = model_manager.run_batch_language_detection("montage.wav", 1)
            assert len(results) == 0  # Returns empty list on fatal decode failure


class TestAdditionalCoverage:
    """Tests to cover remaining gaps."""

    def test_wait_for_priority_yield(self):
        """Test yielding in wait_for_priority."""
        model_manager._PAUSE_REQUESTED.set()
        model_manager._PRIORITY_REQUESTS = 1
        model_manager._RESUME_EVENT.clear()

        mock_lock = mock.MagicMock()

        def resume_after_delay():
            time.sleep(0.1)
            model_manager._RESUME_EVENT.set()

        threading.Thread(target=resume_after_delay).start()
        model_manager.wait_for_priority(model_lock=mock_lock)

        mock_lock.release.assert_called_once()
        mock_lock.acquire.assert_called_once()

    def test_offload_error_handling(self):
        """Test error handling in _check_and_offload_resources."""
        pm = mock.MagicMock()
        pm.offload.side_effect = Exception("Offload fail")
        model_manager._PREPROCESSOR_POOL["CPU"] = pm
        model_manager._ACTIVE_SESSIONS = 0
        model_manager._QUEUED_SESSIONS = 0

        # Should not raise
        model_manager._check_and_offload_resources()

    def test_preprocess_audio_error(self):
        """Test error handling in _preprocess_audio."""
        with mock.patch("modules.config.ENABLE_VOCAL_SEPARATION", True):
            pm = mock.MagicMock()
            pm.process_audio_file.side_effect = Exception("PM err")
            model_manager._PREPROCESSOR_POOL["CPU"] = pm
            model_manager.utils.THREAD_CONTEXT.assigned_unit = {"id": "CPU"}
            dur, path = model_manager._preprocess_audio("test.wav")
            assert path == "test.wav"
            assert dur == 0.0

    def test_run_intel_whisper_path(self):
        """Test the intel-whisper transcription path."""
        mock_model = mock.MagicMock()
        mock_model.transcribe.return_value = {
            "segments": [{"text": "intel", "end": 1.0}],
            "language": "en"
        }

        with mock.patch("modules.vad.decode_audio", return_value=[]):
            with pytest.raises(NotImplementedError,
                               match="Intel Whisper Engine is currently disabled."):
                model_manager._run_intel_whisper_transcription(
                    mock_model, "any.wav", "en", "transcribe"
                )

    def test_log_diagnostics_error(self):
        """Test error handling in _log_audio_diagnostics."""
        with mock.patch("soundfile.info", side_effect=Exception("sf err")):
            model_manager._log_audio_diagnostics("any.wav")

    def test_init_stats_error(self):
        """Test error handling in _init_transcription_stats."""
        with mock.patch("soundfile.info", side_effect=Exception("sf err")):
            model_manager._init_transcription_stats("any.wav")
            assert utils.THREAD_CONTEXT.total_duration == 0

    def test_log_progress_manual(self):
        """Test _log_progress_manual."""
        model_manager._log_progress_manual(5.0, 10.0, time.time() - 1.0, "hello")

    def test_run_language_detection_core_dict(self):
        """Test core language detection with dict return (Intel style)."""
        mock_model = mock.MagicMock()
        mock_model.transcribe.return_value = {"language": "de", "language_probability": 0.7}
        with mock.patch("modules.vad.get_speech_timestamps_from_path", return_value=[{"start": 0}]):
            res = model_manager._run_language_detection_core(mock_model, "any.wav")
            assert res["detected_language"] == "de"

    def test_post_process_vad_edge_cases(self):
        """Test VAD post-processing edge cases."""
        # Empty segments
        assert model_manager._post_process_vad({"segments": []}, "x") == {"segments": []}
        # Repetition filter
        res = {"segments": [{"text": "hi", "probability": 1.0}] * 20}
        with mock.patch("modules.config.HALLUCINATION_REPETITION_THRESHOLD", 5):
            processed = model_manager._post_process_vad(res, "x")
            assert processed["segments"][5]["text"] == ""
        # Error path
        assert model_manager._post_process_vad(None, "x") is None

    def test_wait_for_priority_interrupted(self):
        """Test wait_for_priority interruption resilience."""
        with mock.patch('time.sleep', side_effect=[None, InterruptedError()]):
            # This should eventually return or raise, but we want to cover the sleep line
            try:
                model_manager.wait_for_priority()
            except InterruptedError:
                pass

    def test_is_engine_initialized_error(self):
        """Test is_engine_initialized resilience to errors."""
        with mock.patch('modules.model_manager._MODEL_POOL', mock.MagicMock()):
            # Force an error during iteration
            model_manager._MODEL_POOL.values.side_effect = Exception("Crash")
            assert model_manager.is_engine_initialized() is False

    def test_get_service_stats_with_active_task(self):
        """Test get_service_stats when tasks are present."""
        thread_id = threading.get_ident()
        with model_manager._TASK_REGISTRY_LOCK:
            model_manager._TASK_REGISTRY[thread_id] = {
                "status": "active",
                "progress": 50
            }
        try:
            stats = model_manager.get_service_stats()
            assert stats["active_sessions"] == 1
        finally:
            with model_manager._TASK_REGISTRY_LOCK:
                if thread_id in model_manager._TASK_REGISTRY:
                    del model_manager._TASK_REGISTRY[thread_id]

    def test_cleanup_failed_task_missing(self):
        """Test cleanup_failed_task with non-existent thread_id."""
        # Should not raise
        model_manager.cleanup_failed_task()
