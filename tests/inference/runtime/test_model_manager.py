"""Tests for modules/inference/model_manager.py"""

import threading
import time
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from modules.core import config
from modules.inference import scheduler
from modules.inference.runtime import model_manager, model_segment_processing


def test_model_lock_ctx_success():
    """Test successful hardware unit acquisition."""
    mock_model = mock.MagicMock()
    model_manager.MODEL_POOL["CPU"] = mock_model

    with model_manager.model_lock_ctx() as (model, unit_id):
        assert unit_id == "CPU"
        assert model == mock_model
        # Semaphore value should be 0 (acquired)
        assert scheduler.STATE.model_lock._value == 0

    # Semaphore value should be 1 (released)
    assert scheduler.STATE.model_lock._value == 1


def test_model_lock_ctx_contention():
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
        with mock.patch("modules.core.config.HARDWARE_UNITS", [{"id": "CPU", "type": "CPU", "name": "CPU"}]):
            with mock.patch("modules.inference.pipeline.preprocessing.PreprocessingManager") as mock_pm_cls:
                success = model_manager.load_model()
                assert success is True
                assert scheduler.STATE.engine_initialized is True
                assert "CPU" in model_manager.PREPROCESSOR_POOL
                mock_pm_cls.assert_called_once()

    def test_init_unit_success(self):
        """Test loading a specific unit's model."""
        unit = {"id": "CPU", "type": "CPU", "name": "CPU"}
        mock_engine = mock.MagicMock()

        with mock.patch("modules.inference.engines.engine_factory.create_engine", return_value=mock_engine) as mock_create:
            model_manager.init_unit(unit)
            assert model_manager.MODEL_POOL["CPU"] == mock_engine
            mock_create.assert_called_once_with(config.ASR_ENGINE, config.MODEL_ID, unit)

    def test_init_unit_failure(self):
        """Test error handling during unit initialization."""
        unit = {"id": "CPU", "type": "CPU", "name": "CPU"}
        with mock.patch("modules.inference.engines.engine_factory.create_engine", side_effect=RuntimeError("Load fail")):
            # Should log error but not raise
            model_manager.init_unit(unit)
            assert "CPU" not in model_manager.MODEL_POOL


class TestSharedPreprocessorSelection:
    """Tests for shared preprocessor unit-selection behavior."""

    def test_shared_preprocessor_prefers_matching_hardware_unit(self):
        model_manager.PREPROCESSOR_POOL.clear()
        npu_unit = {"id": "NPU.0", "type": "NPU", "name": "Intel NPU"}

        with (
            mock.patch("modules.core.config.HARDWARE_UNITS", [npu_unit]),
            mock.patch("modules.inference.pipeline.preprocessing.PreprocessingManager") as mock_pm,
        ):
            instance = mock.MagicMock()
            mock_pm.return_value = instance

            result = model_manager._shared_preprocessor_for_type("NPU")

            assert result is instance
            mock_pm.assert_called_once_with(npu_unit)
            assert model_manager.PREPROCESSOR_POOL["PREPROCESS::NPU"] is instance

    def test_shared_preprocessor_falls_back_when_no_matching_unit(self):
        model_manager.PREPROCESSOR_POOL.clear()

        with (
            mock.patch("modules.core.config.HARDWARE_UNITS", [{"id": "CPU", "type": "CPU", "name": "CPU"}]),
            mock.patch("modules.inference.pipeline.preprocessing.PreprocessingManager") as mock_pm,
        ):
            instance = mock.MagicMock()
            mock_pm.return_value = instance

            result = model_manager._shared_preprocessor_for_type("NPU")

            assert result is instance
            mock_pm.assert_called_once_with()
            assert model_manager.PREPROCESSOR_POOL["PREPROCESS::NPU"] is instance


class TestRunTranscription:
    """Tests for run_transcription."""

    def test_run_transcription_success(self):
        """Test full transcription lifecycle."""
        mock_model = mock.MagicMock()
        mock_info = mock.MagicMock(language="en", language_probability=0.9, duration=10.0)
        # Mock segments as a list of mock objects
        mock_segment = mock.MagicMock(start=0.0, end=1.0, text=" Hello")
        mock_model.transcribe.return_value = ([mock_segment], mock_info)

        model_manager.MODEL_POOL["CPU"] = mock_model

        with mock.patch("modules.core.config.ENABLE_VOCAL_SEPARATION", False):
            result = model_manager.run_transcription("test.wav", language="en", task="transcribe", batch_size=1)
            assert result["language"] == "en"
            assert "Hello" in result["text"]
            assert result["video_duration_sec"] == 10.0

    def test_run_transcription_vocal_separation(self):
        """Test transcription with vocal separation enabled."""
        mock_model = mock.MagicMock()
        mock_info = mock.MagicMock(language="en", language_probability=0.9, duration=10.0)
        mock_model.transcribe.return_value = ([], mock_info)
        model_manager.MODEL_POOL["CPU"] = mock_model

        pm = mock.MagicMock()
        pm.preprocess_audio.return_value = "isolated.wav"
        model_manager.PREPROCESSOR_POOL["CPU"] = pm

        # Force CPU preprocessing so the "CPU"-keyed pooled preprocessor is used
        # regardless of host accelerator hardware.
        with mock.patch("modules.core.config.PREPROCESS_DEVICE", "CPU"):
            with mock.patch("modules.core.config.ENABLE_VOCAL_SEPARATION", True):
                with mock.patch("os.path.exists", return_value=True):
                    with mock.patch("os.remove") as mock_remove:
                        model_manager.run_transcription("original.wav", language="en", task="transcribe", batch_size=1)
                        pm.preprocess_audio.assert_called_with("original.wav", force=False, yield_cb=model_manager.check_preemption)
                        mock_remove.assert_called_with("isolated.wav")

    def test_run_transcription_checks_preemption_on_stage_transitions(self):
        """Ensure cooperative preemption checks occur across ASR stage transitions."""
        mock_model = mock.MagicMock()
        mock_info = mock.MagicMock(language="en", language_probability=0.9, duration=10.0)
        mock_segment = mock.MagicMock(start=0.0, end=1.0, text=" Hello")
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        model_manager.MODEL_POOL["CPU"] = mock_model

        pm = mock.MagicMock()
        pm.preprocess_audio.return_value = "isolated.wav"
        model_manager.PREPROCESSOR_POOL["CPU"] = pm

        with (
            mock.patch("modules.core.config.PREPROCESS_DEVICE", "CPU"),
            mock.patch("modules.core.config.ENABLE_VOCAL_SEPARATION", True),
            mock.patch("os.path.exists", return_value=True),
            mock.patch("os.remove"),
            mock.patch("modules.inference.runtime.model_manager.check_preemption") as mock_preempt,
        ):
            model_manager.run_transcription("original.wav", language="en", task="transcribe", batch_size=1)

        assert mock_preempt.call_count >= 5

    def test_run_transcription_starts_with_neutral_inference_stage(self):
        """Initial inference stage should be set to factual phase text."""
        mock_model = mock.MagicMock()
        mock_info = mock.MagicMock(language="en", language_probability=0.9, duration=10.0)
        mock_segment = mock.MagicMock(start=0.0, end=1.0, text=" Hello")
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        model_manager.MODEL_POOL["CPU"] = mock_model

        with (
            mock.patch("modules.core.config.ENABLE_VOCAL_SEPARATION", False),
            mock.patch("modules.inference.runtime.model_manager.scheduler.update_task_progress") as mock_progress,
        ):
            model_manager.run_transcription("test.wav", language="en", task="transcribe", batch_size=1)

        stages = [call.args[1] for call in mock_progress.call_args_list if len(call.args) >= 2 and isinstance(call.args[1], str)]
        assert "Inference" in stages

    def test_run_translation_logs_selected_hardware_unit_before_inference(self):
        """Translation path should log chosen hardware unit before model inference begins."""
        mock_model = mock.MagicMock()
        mock_info = mock.MagicMock(language="en", language_probability=0.9, duration=10.0)
        mock_segment = mock.MagicMock(start=0.0, end=1.0, text=" Hello")
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        model_manager.MODEL_POOL["CPU"] = mock_model

        with (
            mock.patch("modules.core.config.ENABLE_VOCAL_SEPARATION", False),
            mock.patch("modules.inference.runtime.model_manager.logger.info") as mock_log_info,
        ):
            model_manager.run_transcription("test.wav", language="en", task="translate", batch_size=1)

        mock_log_info.assert_any_call("[ASR] Starting %s on hardware unit %s", "translation", "CPU")

    def test_run_transcription_queue_duration_uses_task_id_registry_entry(self):
        """Queue duration should come from the task_id-keyed registry snapshot."""
        mock_model = mock.MagicMock()
        mock_info = mock.MagicMock(language="en", language_probability=0.9, duration=10.0)
        mock_segment = mock.MagicMock(start=0.0, end=1.0, text=" Hello")
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        model_manager.MODEL_POOL["CPU"] = mock_model

        task_id = "task-uuid-1"
        with (
            mock.patch("modules.core.config.ENABLE_VOCAL_SEPARATION", False),
            mock.patch(
                "modules.inference.runtime.model_manager._get_current_task_info",
                return_value=(task_id, None, None, None, None, None),
            ),
            mock.patch("modules.inference.runtime.model_manager.time.time", return_value=200.0),
        ):
            with scheduler.STATE.task_registry_lock:
                scheduler.STATE.task_registry[task_id] = {
                    "start_time": 100.0,
                    "start_active": 130.0,
                }
            result = model_manager.run_transcription("test.wav", language="en", task="transcribe", batch_size=1)

        assert result["performance"]["queue_sec"] == 30.0


class TestLanguageDetection:
    """Tests for language detection."""

    def test_run_language_detection_success(self):
        """Test successful language detection."""
        mock_model = mock.MagicMock()
        mock_info = mock.MagicMock(language="fr", language_probability=0.8)
        mock_model.transcribe.return_value = (None, mock_info)
        model_manager.MODEL_POOL["CPU"] = mock_model

        with mock.patch("modules.inference.pipeline.vad.get_speech_timestamps_from_path", return_value=[{"start": 0, "end": 1}]):
            result = model_manager.run_language_detection("test.wav")
            assert result["detected_language"] == "fr"
            assert result["confidence"] == 0.8

    def test_run_batch_language_detection(self):
        """Test multi-segment language detection."""
        mock_model = mock.MagicMock()
        model_manager.MODEL_POOL["CPU"] = mock_model

        mock.MagicMock(language="en", language_probability=0.9)
        # For batch detection, it calls _run_language_detection_core which returns a dict
        with mock.patch("modules.inference.runtime.model_manager.run_language_detection_core") as mock_core:
            mock_core.return_value = {"detected_language": "en", "confidence": 0.9}
            with mock.patch("modules.inference.pipeline.vad.decode_audio", return_value=[0] * (16000 * 60)):
                results = model_manager.run_batch_language_detection("test.wav", segment_count=2)
                assert len(results) == 2
                assert results[0]["detected_language"] == "en"


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
        model_manager.PREPROCESSOR_POOL["CPU"] = pm
        assert model_manager.is_uvr_actually_loaded() is True

        pm.separator = None
        assert model_manager.is_uvr_actually_loaded() is False


def test_model_manager_booster_edge_cases():
    """Cover miscellaneous uncovered lines in model_manager."""
    # 57: scheduler.STATE.uvr_loaded = True
    with mock.patch("modules.core.config.ENABLE_VOCAL_SEPARATION", True):
        model_manager.load_model()
        assert scheduler.STATE.uvr_loaded is True

    # 83: logger.info for Intel accelerator fallback
    unit = {"id": "GPU", "type": "GPU", "name": "Intel GPU"}
    with (
        mock.patch("modules.core.config.ASR_ENGINE", "FASTER-WHISPER"),
        mock.patch("modules.inference.engines.engine_factory.FasterWhisperEngine"),
    ):
        from modules.inference.engines import engine_factory

        engine_factory.create_engine("FASTER-WHISPER", config.MODEL_ID, unit)

    # 186-187: Cleanup error
    mock_model = mock.MagicMock()
    from argparse import Namespace

    mock_model.transcribe.return_value = ([], Namespace(duration=0, language="en", language_probability=1.0))
    model_manager.MODEL_POOL["CPU"] = mock_model

    with (
        mock.patch("modules.core.config.ENABLE_VOCAL_SEPARATION", True),
        mock.patch("modules.inference.runtime.model_manager.run_vocal_isolation_direct", return_value="iso.wav"),
        mock.patch("os.path.exists", return_value=True),
        mock.patch("os.remove", side_effect=OSError("Locked")),
    ):
        model_manager.run_transcription("test.wav", "en", "transcribe")

    # 193: return result if no segments
    assert model_manager._post_process_results({"no_segments": []}) == {"no_segments": []}

    # 243: return audio_path if no preprocessor
    assert model_manager.run_vocal_isolation_direct("test.wav", "NON_EXISTENT") == "test.wav"

    # 285: break in batch LD
    with mock.patch("modules.inference.pipeline.vad.decode_audio", return_value=np.zeros(0)):
        res = model_manager.run_batch_language_detection_direct(mock_model, "test.wav", 5)
        assert not res

    # 409: RuntimeError if engine pool is empty
    with (
        mock.patch("modules.inference.runtime.model_manager.MODEL_POOL", {}),
        mock.patch("modules.inference.runtime.model_manager.init_unit"),
    ):
        with pytest.raises(RuntimeError):
            with model_manager.model_lock_ctx() as (m, u):
                pass


def test_model_manager_forwards_new_params():
    """Verify that model_manager.run_transcription forwards new parameters to transcribe call."""
    mock_model = mock.MagicMock()
    mock_info = mock.MagicMock(language="en", language_probability=0.95, duration=5.0)
    mock_segment = mock.MagicMock(start=0.0, end=1.0, text="hello")
    mock_model.transcribe.return_value = ([mock_segment], mock_info)

    model_manager.MODEL_POOL["CPU"] = mock_model

    model_manager.run_transcription(
        "test.wav",
        language="en",
        task="transcribe",
        initial_prompt="hello test",
        vad_filter=False,
        word_timestamps=True,
    )

    mock_model.transcribe.assert_called_once_with(
        "test.wav",
        language="en",
        task="transcribe",
        beam_size=config.DEFAULT_BEAM_SIZE,
        initial_prompt="hello test",
        vad_filter=False,
        vad_parameters={"min_silence_duration_ms": config.VAD_MIN_SILENCE_DURATION_MS},
        word_timestamps=True,
    )


def test_preprocessor_resolution_paths_cover_shared_and_cpu_fallbacks():
    """Cover helper branches for preferred and unit-specific preprocessor selection."""
    model_manager.PREPROCESSOR_POOL.clear()

    cpu_pm = mock.MagicMock()
    cpu_pm.device_type = "CPU"
    model_manager.PREPROCESSOR_POOL["CPU"] = cpu_pm

    with mock.patch("modules.core.config.PREPROCESS_DEVICE", "CPU"):
        assert model_manager._resolve_preprocessor_for_unit("CPU") is cpu_pm

    with (
        mock.patch("modules.core.config.PREPROCESS_DEVICE", "NPU"),
        mock.patch("modules.inference.runtime.model_manager.preprocessing.PreprocessingManager") as pm_cls,
    ):
        shared_pm = mock.MagicMock()
        shared_pm.device_type = "NPU"
        pm_cls.return_value = shared_pm
        resolved = model_manager._resolve_preprocessor_for_unit("CPU")
        assert resolved is shared_pm
        assert model_manager.PREPROCESSOR_POOL["PREPROCESS::NPU"] is shared_pm


def test_model_segment_processing_tracks_progress_and_fallbacks():
    """Cover segment-processing progress, metadata, and diarization fallback branches."""
    segment = SimpleNamespace(
        start=0.0,
        end=1.0,
        text=" hello ",
        words=[SimpleNamespace(start=0.0, end=0.5, word="hello")],
    )
    info = SimpleNamespace(duration=10.0, language="en")

    with (
        mock.patch("modules.inference.runtime.model_segment_processing.utils.format_single_srt_block", return_value="block"),
        mock.patch("modules.inference.runtime.model_segment_processing.scheduler.update_task_metadata") as mock_metadata,
        mock.patch("modules.inference.runtime.model_segment_processing.scheduler.update_task_progress") as mock_progress,
    ):
        results = model_segment_processing.consume_transcription_segments(
            [segment],
            info,
            "transcribe",
            diarize=False,
            min_speakers=None,
            max_speakers=None,
            hf_token=None,
            unit_id="CPU",
            processed_path="audio.wav",
            preemption_check=lambda: None,
        )

    assert results == [
        {
            "start": 0.0,
            "end": 1.0,
            "text": "hello",
            "words": [{"start": 0.0, "end": 0.5, "word": "hello", "probability": 1.0}],
        }
    ]
    assert mock_metadata.called
    assert mock_progress.called

    with (
        mock.patch("modules.inference.runtime.model_segment_processing.utils.format_single_srt_block", return_value="block"),
        mock.patch("modules.inference.runtime.model_segment_processing.diarization.run_diarization", side_effect=RuntimeError("boom")),
    ):
        fallback = model_segment_processing.consume_transcription_segments(
            [segment],
            info,
            "transcribe",
            diarize=True,
            min_speakers=None,
            max_speakers=None,
            hf_token=None,
            unit_id="CPU",
            processed_path="audio.wav",
            preemption_check=lambda: None,
        )

    assert fallback == [
        {
            "start": 0.0,
            "end": 1.0,
            "text": "hello",
            "words": [{"start": 0.0, "end": 0.5, "word": "hello", "probability": 1.0}],
        }
    ]
