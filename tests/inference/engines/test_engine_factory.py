"""Tests for modules/inference/engines/engine_factory.py."""

from unittest import mock

import pytest

from modules.core import config
from modules.inference.engines import base, engine_factory


def test_inference_info():
    info = base.InferenceInfo(language="en", language_probability=0.9, duration=10.5)
    assert info.language == "en"
    assert info.language_probability == 0.9
    assert info.duration == 10.5


def test_segment_wrapper():
    seg = base.SegmentWrapper(start=1.0, end=2.0, text="hello")
    assert seg.start == 1.0
    assert seg.end == 2.0
    assert seg.text == "hello"


def test_base_asr_engine():
    base = engine_factory.BaseASREngine()
    with pytest.raises(NotImplementedError):
        base.transcribe("dummy.wav")
    with pytest.raises(NotImplementedError):
        base.detect_language("dummy")
    base.unload()  # should pass as a no-op


def test_faster_whisper_engine():
    mock_faster_whisper = mock.MagicMock()
    mock_model = mock_faster_whisper.WhisperModel.return_value
    mock_model.transcribe.return_value = (iter([]), mock.MagicMock())

    with mock.patch("importlib.import_module", return_value=mock_faster_whisper):
        engine = engine_factory.FasterWhisperEngine(
            model_id="test-model",
            device="cpu",
            device_index=0,
            compute_type="int8",
            cpu_threads=4,
            download_root="/tmp",
        )
        assert engine.model is mock_model

        # Test transcribe
        engine.transcribe("dummy.wav", language="en", word_timestamps=True)
        mock_model.transcribe.assert_called_once_with(
            "dummy.wav",
            language="en",
            task="transcribe",
            beam_size=config.DEFAULT_BEAM_SIZE,
            initial_prompt=None,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": config.VAD_MIN_SILENCE_DURATION_MS},
            word_timestamps=True,
        )

        # Test detect_language
        mock_model.detect_language.return_value = ("en", 0.95, [("en", 0.95), ("fr", 0.05)])
        res = engine.detect_language("dummy_audio")
        mock_model.detect_language.assert_called_once_with("dummy_audio")
        assert res == ("en", 0.95, [("en", 0.95), ("fr", 0.05)])

        # Test unload
        engine.unload()
        assert not hasattr(engine, "model")


def test_openai_whisper_engine():
    mock_whisper = mock.MagicMock()
    mock_model = mock_whisper.load_model.return_value
    mock_model.transcribe.return_value = {
        "language": "fr",
        "segments": [{"start": 0.0, "end": 2.0, "text": "bonjour"}],
    }

    orig_duration = engine_factory.utils.get_audio_duration
    engine_factory.utils.get_audio_duration = mock.MagicMock(return_value=5.0)
    try:
        with mock.patch("importlib.import_module", return_value=mock_whisper):
            engine = engine_factory.OpenaiWhisperEngine(model_id="test-model", device="cpu")
            loaded_model = engine.model
            segs, info = engine.transcribe("dummy.wav", language="fr", beam_size=5, unknown_param="ignored")
            seg_list = list(segs)
            engine.unload()
            assert (
                loaded_model is mock_model,
                info.language,
                info.duration,
                len(seg_list),
                seg_list[0].text,
                hasattr(engine, "model"),
            ) == (True, "fr", 5.0, 1, "bonjour", False)
    finally:
        engine_factory.utils.get_audio_duration = orig_duration


def test_openai_whisper_detect_language_path_and_probs():
    """OpenAI detect_language should load path input and sort probabilities."""
    mock_whisper = mock.MagicMock()
    mock_model = mock_whisper.load_model.return_value
    mock_whisper.load_audio.return_value = "audio-array"
    mock_mel = mock.MagicMock()
    mock_mel.to.return_value = "mel-on-device"
    mock_whisper.log_mel_spectrogram.return_value = mock_mel
    mock_model.device = "cpu"
    mock_model.detect_language.return_value = (
        None,
        {"fr": 0.2, "en": 0.7, "de": 0.1},
    )

    with mock.patch("importlib.import_module", return_value=mock_whisper):
        engine = engine_factory.OpenaiWhisperEngine(model_id="test-model", device="cpu")
        lang, prob, all_probs = engine.detect_language("dummy.wav")

    assert lang == "en"
    assert prob == 0.7
    assert all_probs[0] == ("en", 0.7)


def test_whisperx_engine():
    """WhisperXEngine should delegate model load/transcribe to the isolated worker process,
    and unload() must actually invoke the worker's unload_model cleanup call (not merely
    drop the local model_handle attribute)."""
    with mock.patch("modules.inference.engines.whisperx_engine.worker") as mock_worker:
        mock_worker.call_with_generation.return_value = ("handle-1", 0)
        mock_worker.generation.return_value = 0
        mock_worker.call.side_effect = [
            {"language": "en", "segments": [{"start": 1.0, "end": 3.0, "text": "whisperx"}]},
            None,
        ]

        orig_duration = engine_factory.utils.get_audio_duration
        engine_factory.utils.get_audio_duration = mock.MagicMock(return_value=10.0)
        try:
            engine = engine_factory.WhisperXEngine(model_id="test-model", device="cpu", compute_type="int8")
            model_handle = engine.model_handle
            segs, info = engine.transcribe("dummy.wav", language="en")
            seg_list = list(segs)
            engine.unload()
            assert (
                model_handle,
                info.language,
                info.duration,
                len(seg_list),
                seg_list[0].text,
                hasattr(engine, "model_handle"),
            ) == ("handle-1", "en", 10.0, 1, "whisperx", False)
            mock_worker.call_with_generation.assert_called_once_with(
                "load_model",
                model_id="test-model",
                device="cpu",
                compute_type="int8",
            )
            assert [c.args[0] for c in mock_worker.call.call_args_list] == ["transcribe", "unload_model"]
            mock_worker.call.assert_called_with("unload_model", model_handle="handle-1")
        finally:
            engine_factory.utils.get_audio_duration = orig_duration


def _configure_whisperx_reload_worker(mock_worker) -> None:
    mock_worker.call_with_generation.side_effect = [
        ("handle-1", 1),  # load_model at __init__
        ("handle-2", 2),  # load_model reload after generation changes
    ]
    mock_worker.call.side_effect = [
        {"language": "en", "segments": [{"start": 0.0, "end": 1.0, "text": "first"}]},
        {"language": "en", "segments": [{"start": 0.0, "end": 1.0, "text": "second"}]},
    ]
    mock_worker.generation.side_effect = [1, 2]


def _run_whisperx_reload_transcriptions(engine) -> tuple:
    after_init = (engine.model_handle, engine._generation)
    list(engine.transcribe("dummy.wav", language="en")[0])
    after_stable_call = (engine.model_handle, engine._generation)
    list(engine.transcribe("dummy.wav", language="en")[0])
    after_reload_call = (engine.model_handle, engine._generation)
    return after_init, after_stable_call, after_reload_call


def _whisperx_reload_assertion_payload(mock_worker, snapshots: tuple) -> tuple:
    after_init, after_stable_call, after_reload_call = snapshots
    load_model_calls = [c for c in mock_worker.call_with_generation.call_args_list if c.args[0] == "load_model"]
    transcribe_handles = [c.kwargs["model_handle"] for c in mock_worker.call.call_args_list if c.args[0] == "transcribe"]
    return (
        after_init,
        after_stable_call,
        after_reload_call,
        len(load_model_calls),
        load_model_calls[1].kwargs["model_id"],
        transcribe_handles,
    )


def test_whisperx_engine_reloads_model_handle_after_worker_restart():
    """_ensure_current_model_handle must reload model_handle (a second load_model call)
    when worker.generation() returns a different value than what was recorded at load
    time -- signaling the isolated worker process crashed and respawned, which means
    the old handle's `objects` dict no longer exists. A stable generation across calls
    must NOT trigger a reload (test_whisperx_engine above already covers that path)."""
    with mock.patch("modules.inference.engines.whisperx_engine.worker") as mock_worker:
        _configure_whisperx_reload_worker(mock_worker)
        orig_duration = engine_factory.utils.get_audio_duration
        engine_factory.utils.get_audio_duration = mock.MagicMock(return_value=5.0)
        try:
            engine = engine_factory.WhisperXEngine(model_id="test-model", device="cpu", compute_type="int8")
            snapshots = _run_whisperx_reload_transcriptions(engine)
        finally:
            engine_factory.utils.get_audio_duration = orig_duration

    assert _whisperx_reload_assertion_payload(mock_worker, snapshots) == (
        ("handle-1", 1),
        ("handle-1", 1),
        ("handle-2", 2),
        2,
        "test-model",
        ["handle-1", "handle-2"],
    )


def test_whisperx_detect_language_routes_through_worker():
    """WhisperX detect_language should forward to the isolated worker and return its result verbatim."""
    with mock.patch("modules.inference.engines.whisperx_engine.worker") as mock_worker:
        mock_worker.call_with_generation.return_value = ("handle-1", 0)
        mock_worker.generation.return_value = 0
        mock_worker.call.return_value = ("ro", 0.88, [("ro", 0.88), ("en", 0.12)])
        engine = engine_factory.WhisperXEngine(model_id="test-model", device="cpu", compute_type="int8")
        lang, prob, all_probs = engine.detect_language("audio-data.wav")

    assert (lang, prob, all_probs[0]) == ("ro", 0.88, ("ro", 0.88))
    mock_worker.call.assert_called_with("detect_language", model_handle="handle-1", audio_path="audio-data.wav")


def _engine_import_side_effect(mock_intel_module, mock_openai_module, mock_faster_module):
    """Return the importlib side effect used by the create_engine tests."""
    module_map = {
        "modules.inference.engines.intel_engine": mock_intel_module,
        "whisper": mock_openai_module,
        "faster_whisper": mock_faster_module,
    }
    return lambda name: module_map.get(name, mock.MagicMock())


def test_create_engine_intel_whisper_npu():
    """Intel Whisper on NPU should create the Intel engine."""
    mock_intel_module = mock.MagicMock()
    mock_intel_engine = mock_intel_module.IntelWhisperEngine.return_value
    mock_openai_module = mock.MagicMock()
    mock_faster_module = mock.MagicMock()

    with mock.patch(
        "importlib.import_module",
        side_effect=_engine_import_side_effect(mock_intel_module, mock_openai_module, mock_faster_module),
    ):
        unit_intel = {"id": "npu:0", "type": "NPU", "name": "Intel NPU"}
        engine = engine_factory.create_engine("INTEL-WHISPER", "test-model", unit_intel)

    assert engine is mock_intel_engine
    mock_intel_module.IntelWhisperEngine.assert_called_once_with("test-model", device="npu:0")


def test_create_engine_intel_whisper_cuda_falls_back():
    """Intel Whisper on CUDA should fall back to Faster Whisper."""
    mock_intel_module = mock.MagicMock()
    mock_openai_module = mock.MagicMock()
    mock_faster_module = mock.MagicMock()

    with mock.patch(
        "importlib.import_module",
        side_effect=_engine_import_side_effect(mock_intel_module, mock_openai_module, mock_faster_module),
    ):
        unit_cuda = {"id": "cuda:0", "type": "CUDA", "name": "NVIDIA GPU"}
        engine = engine_factory.create_engine("INTEL-WHISPER", "test-model", unit_cuda)

    assert isinstance(engine, engine_factory.FasterWhisperEngine)
    assert mock_intel_module.IntelWhisperEngine.call_count == 0


def test_create_engine_openai_whisper_cuda():
    """OpenAI Whisper on CUDA should create an OpenAI engine with CUDA device."""
    mock_intel_module = mock.MagicMock()
    mock_openai_module = mock.MagicMock()
    mock_faster_module = mock.MagicMock()

    with mock.patch(
        "importlib.import_module",
        side_effect=_engine_import_side_effect(mock_intel_module, mock_openai_module, mock_faster_module),
    ):
        unit_cuda = {"id": "cuda:0", "type": "CUDA", "name": "NVIDIA GPU"}
        engine = engine_factory.create_engine("OPENAI-WHISPER", "test-model", unit_cuda)

    assert isinstance(engine, engine_factory.OpenaiWhisperEngine)
    assert engine.device == "cuda"


def test_create_engine_whisperx_cpu():
    """WhisperX on CPU should create a WhisperX engine with cpu device."""
    with mock.patch("modules.inference.engines.whisperx_engine.worker") as mock_worker:
        mock_worker.call_with_generation.return_value = ("handle-1", 0)
        unit_cpu = {"id": "cpu", "type": "CPU", "name": "Intel CPU"}
        engine = engine_factory.create_engine("WHISPERX", "test-model", unit_cpu)

    assert isinstance(engine, engine_factory.WhisperXEngine)
    assert engine.device == "cpu"


def test_create_engine_faster_whisper_cuda():
    """Faster Whisper on CUDA should create a Faster Whisper engine."""
    mock_intel_module = mock.MagicMock()
    mock_openai_module = mock.MagicMock()
    mock_faster_module = mock.MagicMock()

    with mock.patch(
        "importlib.import_module",
        side_effect=_engine_import_side_effect(mock_intel_module, mock_openai_module, mock_faster_module),
    ):
        unit_cuda = {"id": "cuda:0", "type": "CUDA", "name": "NVIDIA GPU"}
        engine = engine_factory.create_engine("FASTER-WHISPER", "test-model", unit_cuda)

    assert isinstance(engine, engine_factory.FasterWhisperEngine)


def test_create_engine_faster_whisper_cpu_fallback():
    """Faster Whisper should still create a Faster Whisper engine for GPU fallback units."""
    mock_intel_module = mock.MagicMock()
    mock_openai_module = mock.MagicMock()
    mock_faster_module = mock.MagicMock()

    with mock.patch(
        "importlib.import_module",
        side_effect=_engine_import_side_effect(mock_intel_module, mock_openai_module, mock_faster_module),
    ):
        unit_gpu = {"id": "gpu:0", "type": "GPU", "name": "Intel GPU"}
        engine = engine_factory.create_engine("FASTER-WHISPER", "test-model", unit_gpu)

    assert isinstance(engine, engine_factory.FasterWhisperEngine)


def test_create_engine_rejects_unknown_engine():
    """Unsupported engine names must raise ValueError instead of silently falling back."""
    unit_cpu = {"id": "cpu", "type": "CPU", "name": "Host CPU"}
    with pytest.raises(ValueError, match="Invalid ASR_ENGINE"):
        engine_factory.create_engine("UNKNOWN-ENGINE", "test-model", unit_cpu)


def test_create_engine_raises_for_unsupported_post_validation_value():
    """Factory should still guard unsupported values even if validator is bypassed."""
    unit_cpu = {"id": "cpu", "type": "CPU", "name": "Host CPU"}
    with (
        mock.patch("modules.inference.engines.engine_factory.engine_registry.normalize_and_validate_engine", return_value="OTHER"),
        mock.patch("modules.inference.engines.engine_factory.engine_registry.supported_engines", return_value=["FASTER-WHISPER"]),
    ):
        with pytest.raises(ValueError, match="Unsupported ASR engine"):
            engine_factory.create_engine("IGNORED", "test-model", unit_cpu)


def test_create_faster_whisper_engine_coercion():
    """Verify _create_faster_whisper_engine coerces float16 compute type to int8 on non-CUDA CPU units."""
    unit_cpu = {"id": "cpu", "type": "CPU", "name": "Host CPU"}
    with mock.patch("modules.inference.engines.engine_factory.FasterWhisperEngine") as mock_fw_constructor:
        with mock.patch("modules.inference.engines.engine_factory.config.COMPUTE_TYPE", "float16"):
            engine_factory._create_faster_whisper_engine("test-model-id", unit_cpu)
            mock_fw_constructor.assert_called_once()
            args = mock_fw_constructor.call_args[0]
            kwargs = mock_fw_constructor.call_args[1]
            assert args[0] == "test-model-id"
            assert kwargs["device"] == "cpu"
            assert kwargs["compute_type"] == "int8"


def test_create_whisperx_engine_coercion():
    """Verify _create_whisperx_engine coerces float16 compute type to int8 on non-CUDA CPU units."""
    unit_cpu = {"id": "cpu", "type": "CPU", "name": "Host CPU"}
    with mock.patch("modules.inference.engines.engine_factory.WhisperXEngine") as mock_wx_constructor:
        with mock.patch("modules.inference.engines.engine_factory.config.COMPUTE_TYPE", "float16"):
            engine_factory._create_whisperx_engine("test-model-id", unit_cpu)
            mock_wx_constructor.assert_called_once()
            args = mock_wx_constructor.call_args[0]
            kwargs = mock_wx_constructor.call_args[1]
            assert args[0] == "test-model-id"
            assert kwargs["device"] == "cpu"
            assert kwargs["compute_type"] == "int8"
