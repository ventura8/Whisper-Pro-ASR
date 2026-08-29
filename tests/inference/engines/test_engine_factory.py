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


class _FakeTorch:
    """Minimal stand-in: which torch build is installed is a per-image property."""

    def __init__(self, *, cuda=False, hip=None, xpu=None):
        self.version = type("v", (), {"hip": hip})()
        self.cuda = type("c", (), {"is_available": staticmethod(lambda: cuda)})()
        # Omitted, not False, by default: `xpu=False` models a torch that HAS the attribute
        # and reports no device, while a CUDA or CPU torch has no `torch.xpu` at all. With
        # the old `xpu=False` default the attribute was always present, so the
        # missing-attribute path -- the one every non-Intel image actually takes -- had no
        # test and an AttributeError there would have gone unnoticed.
        if xpu is not None:
            self.xpu = type("x", (), {"is_available": staticmethod(lambda: xpu)})()


def _with_torch(**kwargs):
    return mock.patch("importlib.import_module", return_value=_FakeTorch(**kwargs))


def test_torch_device_uses_cuda_on_an_nvidia_unit():
    with _with_torch(cuda=True):
        assert engine_factory._resolve_torch_device({"type": "CUDA"}) == "cuda"


def test_torch_device_uses_cuda_for_amd_because_rocm_torch_reports_cuda():
    """ROCm torch deliberately keeps the 'cuda' device string and sets torch.version.hip."""
    with _with_torch(cuda=True, hip="7.2"):
        assert engine_factory._resolve_torch_device({"type": "AMD"}) == "cuda"


def test_torch_device_refuses_amd_without_a_rocm_build():
    """A CPU or CUDA torch on an AMD unit must not claim the GPU."""
    with _with_torch(cuda=False, hip=None):
        assert engine_factory._resolve_torch_device({"type": "AMD"}) == "cpu"


def test_torch_device_uses_xpu_on_intel_units():
    for unit_type in ("GPU", "NPU"):
        with _with_torch(xpu=True):
            assert engine_factory._resolve_torch_device({"type": unit_type}) == "xpu"


def test_torch_device_falls_back_when_xpu_is_present_but_reports_no_device():
    """An XPU-capable torch on a host whose Intel GPU it cannot drive."""
    with _with_torch(xpu=False):
        assert engine_factory._resolve_torch_device({"type": "GPU"}) == "cpu"


def test_torch_device_falls_back_when_torch_has_no_xpu_attribute():
    """The CUDA and CPU torch builds: `torch.xpu` does not exist, and asking must not raise."""
    with _with_torch():
        assert engine_factory._resolve_torch_device({"type": "GPU"}) == "cpu"
        assert engine_factory._resolve_torch_device({"type": "NPU"}) == "cpu"


def test_torch_device_is_cpu_without_torch():
    with mock.patch("importlib.import_module", side_effect=ImportError("no torch")):
        assert engine_factory._resolve_torch_device({"type": "CUDA"}) == "cpu"


def test_ctranslate2_device_stays_narrow():
    """CTranslate2 has no ROCm or OpenVINO backend, so it must not follow torch."""
    assert engine_factory._resolve_device_str({"type": "AMD"}) == "cpu"
    assert engine_factory._resolve_device_str({"type": "GPU"}) == "cpu"
    assert engine_factory._resolve_device_str({"type": "CUDA"}) == "cuda"


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
            vad_parameters={
                "min_silence_duration_ms": config.VAD_MIN_SILENCE_DURATION_MS,
                # Without an explicit threshold Silero runs at its 0.5 default, and music
                # and room tone clear that bar often enough to be handed to the decoder.
                "threshold": config.VAD_THRESHOLD,
            },
            word_timestamps=True,
            # Defends against the recorded long-form decoder-loop defect; see
            # FasterWhisperEngine.transcribe for why both are set.
            condition_on_previous_text=False,
            no_repeat_ngram_size=3,
            # language="en" was requested explicitly here, so per-window re-detection is
            # off: it would override the request and then misreport what it returned.
            multilingual=False,
        )

        # Test detect_language
        mock_model.detect_language.return_value = ("en", 0.95, [("en", 0.95), ("fr", 0.05)])
        res = engine.detect_language("dummy_audio")
        mock_model.detect_language.assert_called_once_with("dummy_audio")
        assert res == ("en", 0.95, [("en", 0.95), ("fr", 0.05)])

        # Test unload
        engine.unload()
        assert not hasattr(engine, "model")


def test_faster_whisper_engine_recovery_after_corrupted_cache(tmp_path):
    mock_faster_whisper = mock.MagicMock()
    mock_model = mock.MagicMock()

    call_count = 0

    def fake_whisper_model(model_name_or_path, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("Model file corrupted: unexpected EOF")
        return mock_model

    mock_faster_whisper.WhisperModel.side_effect = fake_whisper_model

    fake_model_dir = tmp_path / "tiny.en"
    fake_model_dir.mkdir()
    (fake_model_dir / "model.bin").write_bytes(b"corrupted")

    with mock.patch("importlib.import_module", return_value=mock_faster_whisper):
        engine = engine_factory.FasterWhisperEngine(
            model_id="tiny.en",
            device="cpu",
            device_index=0,
            compute_type="int8",
            cpu_threads=4,
            download_root=str(tmp_path),
        )
        assert engine.model is mock_model
        assert call_count == 2
        # Corrupt model path was purged
        assert not fake_model_dir.exists()


def test_faster_whisper_engine_recovery_hf_snapshot_cache(tmp_path):
    mock_faster_whisper = mock.MagicMock()
    mock_model = mock.MagicMock()

    call_count = 0

    def fake_whisper_model(model_name_or_path, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("Model file corrupted: snapshot EOF")
        return mock_model

    mock_faster_whisper.WhisperModel.side_effect = fake_whisper_model

    # Create HuggingFace snapshot cache structure: models--org--repo/snapshots/hash
    repo_dir = tmp_path / "models--Systran--faster-whisper-tiny"
    snap_dir = repo_dir / "snapshots" / "abc12345"
    snap_dir.mkdir(parents=True)
    (snap_dir / "model.bin").write_bytes(b"corrupted")

    with mock.patch("importlib.import_module", return_value=mock_faster_whisper):
        engine = engine_factory.FasterWhisperEngine(
            model_id="Systran/faster-whisper-tiny",
            device="cpu",
            device_index=0,
            compute_type="int8",
            cpu_threads=4,
            download_root=str(tmp_path),
        )
        assert engine.model is mock_model
        assert call_count == 2
        assert not snap_dir.exists()


def test_faster_whisper_engine_retry_failure_reraises(tmp_path):
    mock_faster_whisper = mock.MagicMock()
    mock_faster_whisper.WhisperModel.side_effect = RuntimeError("Persistent network error")

    fake_model_dir = tmp_path / "tiny"
    fake_model_dir.mkdir()
    (fake_model_dir / "model.bin").write_bytes(b"corrupted")

    with mock.patch("importlib.import_module", return_value=mock_faster_whisper):
        with pytest.raises(RuntimeError, match="Persistent network error"):
            engine_factory.FasterWhisperEngine(
                model_id="tiny",
                device="cpu",
                device_index=0,
                compute_type="int8",
                cpu_threads=4,
                download_root=str(tmp_path),
            )
    assert not fake_model_dir.exists()


def test_faster_whisper_engine_local_path_purges_without_retry(tmp_path):
    mock_faster_whisper = mock.MagicMock()
    mock_faster_whisper.WhisperModel.side_effect = RuntimeError("Local model file corrupted")

    corrupt_local_dir = tmp_path / "corrupt_local_dir"
    corrupt_local_dir.mkdir()
    (corrupt_local_dir / "model.bin").write_bytes(b"corrupt")

    with mock.patch("importlib.import_module", return_value=mock_faster_whisper):
        with pytest.raises(RuntimeError, match="Local model file corrupted"):
            engine_factory.FasterWhisperEngine(
                model_id=str(corrupt_local_dir),
                device="cpu",
                device_index=0,
                compute_type="int8",
                cpu_threads=4,
                download_root=str(tmp_path),
            )
    # Corrupt local directory is purged and not retried with local path as repo ID
    assert not corrupt_local_dir.exists()


def test_faster_whisper_engine_recovery_failure_raises(tmp_path):
    mock_faster_whisper = mock.MagicMock()
    mock_faster_whisper.WhisperModel.side_effect = RuntimeError("Persistent corruption error")

    valid_dir = tmp_path / "valid_model"
    valid_dir.mkdir()
    (valid_dir / "model.bin").write_bytes(b"x" * (11 * 1024 * 1024))
    (valid_dir / "config.json").write_text("{}")
    (valid_dir / "preprocessor_config.json").write_text("{}")
    (valid_dir / "tokenizer.json").write_text("{}")

    with mock.patch("importlib.import_module", return_value=mock_faster_whisper):
        with pytest.raises(RuntimeError, match="Persistent corruption error"):
            engine_factory.FasterWhisperEngine(
                model_id=str(valid_dir),
                device="cpu",
                device_index=0,
                compute_type="int8",
                cpu_threads=4,
                download_root=str(tmp_path),
            )
    # Valid directory is NOT purged on generic execution error
    assert valid_dir.exists()


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
