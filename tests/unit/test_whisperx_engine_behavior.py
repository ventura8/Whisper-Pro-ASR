"""Additional coverage tests for modules/inference/engines/whisperx_engine.py and whisperx_worker.py."""

from unittest import mock

from modules.inference.engines import whisperx_engine
from modules.inference.engines.whisperx_engine import _unsupported_whisperx_options
from modules.inference.engines.whisperx_worker import _detect_language, _dispatch, _transcribe, _unload_model


def test_unsupported_whisperx_options_collects_all_flags():
    """Unsupported option helper should report all non-supported flags."""
    unsupported = _unsupported_whisperx_options(
        initial_prompt="ctx",
        vad_filter=False,
        word_timestamps=True,
    )
    assert unsupported == ["initial_prompt", "vad_filter", "word_timestamps"]


def test_whisperx_engine_init_loads_model_via_worker():
    """Engine construction should delegate model loading to the isolated worker."""
    with mock.patch("modules.inference.engines.whisperx_engine.worker") as mock_worker:
        mock_worker.call_with_generation.return_value = ("handle-1", 0)
        engine = whisperx_engine.WhisperXEngine(model_id="m", device="cpu", compute_type="int8")

    mock_worker.call_with_generation.assert_called_once_with(
        "load_model",
        model_id="m",
        device="cpu",
        compute_type="int8",
    )
    assert engine.model_handle == "handle-1"


def test_whisperx_engine_transcribe_logs_unsupported_options(caplog):
    """Engine transcribe should log ignored unsupported options and still return segments."""
    caplog.set_level("WARNING")
    with (
        mock.patch("modules.inference.engines.whisperx_engine.worker") as mock_worker,
        mock.patch("modules.inference.engines.base.utils.get_audio_duration", return_value=1.0),
    ):
        mock_worker.call_with_generation.return_value = ("handle-1", 0)
        mock_worker.generation.return_value = 0
        mock_worker.call.return_value = {"language": "en", "segments": [{"start": 0.0, "end": 1.0, "text": "hi"}]}
        engine = whisperx_engine.WhisperXEngine(model_id="m", device="cpu")
        segments, info = engine.transcribe(
            "x.wav",
            initial_prompt="ctx",
            vad_filter=False,
            word_timestamps=True,
        )

    assert len(list(segments)) == 1
    assert info.language == "en"
    assert "Ignoring unsupported options" in caplog.text


def test_whisperx_engine_detect_language_with_path():
    """Engine detect_language should delegate to the worker with an audio path."""
    with mock.patch("modules.inference.engines.whisperx_engine.worker") as mock_worker:
        mock_worker.call_with_generation.return_value = ("handle-1", 0)
        mock_worker.generation.return_value = 0
        mock_worker.call.return_value = ("en", 0.9, [("en", 0.9)])
        engine = whisperx_engine.WhisperXEngine(model_id="m", device="cpu")
        result = engine.detect_language("audio.wav")

    assert result == ("en", 0.9, [("en", 0.9)])
    mock_worker.call.assert_called_with("detect_language", model_handle="handle-1", audio_path="audio.wav")


def test_whisperx_engine_detect_language_with_array():
    """Engine detect_language should delegate to the worker with a preloaded audio array."""
    with mock.patch("modules.inference.engines.whisperx_engine.worker") as mock_worker:
        mock_worker.call_with_generation.return_value = ("handle-1", 0)
        mock_worker.generation.return_value = 0
        mock_worker.call.return_value = ("ro", 0.8, [("ro", 0.8)])
        engine = whisperx_engine.WhisperXEngine(model_id="m", device="cpu")
        audio_array = object()
        result = engine.detect_language(audio_array)

    assert result == ("ro", 0.8, [("ro", 0.8)])
    mock_worker.call.assert_called_with("detect_language", model_handle="handle-1", audio_array=audio_array)


def test_whisperx_engine_unload_releases_worker_handle():
    """Engine unload should release the worker-side model handle and drop model_handle."""
    with mock.patch("modules.inference.engines.whisperx_engine.worker") as mock_worker:
        mock_worker.call_with_generation.return_value = ("handle-1", 0)
        mock_worker.generation.return_value = 0
        engine = whisperx_engine.WhisperXEngine(model_id="m", device="cpu")
        engine.unload()

    mock_worker.call.assert_called_with("unload_model", model_handle="handle-1")
    assert not hasattr(engine, "model_handle")


# --- whisperx_worker.py: handlers run inside the isolated subprocess ---


def test_worker_transcribe_uses_audio_array_when_provided():
    """Worker _transcribe should use a preloaded audio array instead of reloading from path."""
    objects = {"h": mock.MagicMock()}
    objects["h"].transcribe.return_value = {"language": "en"}

    result = _transcribe(objects, model_handle="h", batch_size=8, language="en", task="transcribe", audio_array="preloaded")

    objects["h"].transcribe.assert_called_once_with("preloaded", batch_size=8, language="en", task="transcribe")
    assert result == {"language": "en"}


def test_worker_detect_language_prefers_inner_model():
    """Worker _detect_language should prefer the model's inner .model when available."""
    inner = mock.MagicMock()
    inner.detect_language.return_value = ("ro", 0.88, [("ro", 0.88), ("en", 0.12)])
    model = mock.MagicMock()
    model.model = inner
    objects = {"h": model}

    result = _detect_language(objects, model_handle="h", audio_array="a")

    assert result == ("ro", 0.88, [("ro", 0.88), ("en", 0.12)])


def test_worker_detect_language_uses_direct_method_when_inner_absent():
    """Worker _detect_language should call detect_language directly when no inner model exists."""
    model = mock.MagicMock(spec=["detect_language", "transcribe"])
    model.detect_language.return_value = ("es", 0.77, [("es", 0.77), ("en", 0.23)])
    objects = {"h": model}

    result = _detect_language(objects, model_handle="h", audio_array="a")

    assert result == ("es", 0.77, [("es", 0.77), ("en", 0.23)])


def test_worker_detect_language_falls_back_to_transcribe():
    """Worker _detect_language should fall back to transcribe when detect_language is unsupported."""
    model = mock.MagicMock(spec=["transcribe"])
    model.transcribe.return_value = {"language": "it"}
    objects = {"h": model}

    result = _detect_language(objects, model_handle="h", audio_array="a")

    assert result == ("it", 1.0, [("it", 1.0)])


def test_worker_unload_model_pops_handle():
    """Worker _unload_model should remove the handle from the objects pool."""
    objects = {"h": object()}
    _unload_model(objects, model_handle="h")
    assert "h" not in objects


def test_dispatch_reports_errors_without_raising():
    """Worker _dispatch should convert a handler exception into an error response, not raise."""
    handlers = {"boom": lambda: (_ for _ in ()).throw(RuntimeError("bad"))}
    response = _dispatch(handlers, {"id": 1, "cmd": "boom", "args": {}})
    assert response["ok"] is False
    assert "bad" in response["error"]
