"""WhisperX engine: device selection, worker delegation, and handle invalidation.

Split out of test_engine_factory.py, which had grown past the project's module-length
limit. WhisperX is the one engine that always proxies to its own worker process, so its
tests are about the boundary -- which device the worker reports, and what happens to a
cached model handle when that worker dies and respawns.
"""

from unittest import mock

from modules.inference.engines import engine_factory


def _whisperx_worker_reporting(capabilities):
    worker = mock.MagicMock()
    worker.call.return_value = capabilities
    return mock.patch("importlib.import_module", return_value=worker)


def test_whisperx_uses_cuda_when_its_worker_has_a_cuda_torch():
    """The image supplies WhisperX's torch, so GPU support is the worker's to report."""
    with _whisperx_worker_reporting({"cuda": True}):
        with mock.patch.object(engine_factory, "WhisperXEngine") as mock_whisperx:
            engine_factory._create_whisperx_engine("model", {"id": "cuda:0", "type": "CUDA", "name": "NVIDIA GPU 0"})

    assert mock_whisperx.call_args.kwargs["device"] == "cuda"


def test_whisperx_falls_back_to_cpu_when_the_worker_has_no_cuda():
    """A CPU-only torch raises at model load; degrade instead of failing the request.

    Verified on hardware before the image stopped bundling a CPU torch: WHISPERX on a
    CUDA unit died with "Torch not compiled with CUDA enabled", surfacing only as an
    empty engine pool.
    """
    with _whisperx_worker_reporting({"cuda": False}):
        with mock.patch.object(engine_factory, "WhisperXEngine") as mock_whisperx:
            engine_factory._create_whisperx_engine("model", {"id": "cuda:0", "type": "CUDA", "name": "NVIDIA GPU 0"})

    assert mock_whisperx.call_args.kwargs["device"] == "cpu"


def test_whisperx_does_not_probe_the_worker_for_a_cpu_unit():
    """A CPU unit is CPU regardless; spawning a worker just to ask would be wasteful."""
    with mock.patch("importlib.import_module") as mock_import:
        with mock.patch.object(engine_factory, "WhisperXEngine") as mock_whisperx:
            engine_factory._create_whisperx_engine("model", {"id": "CPU", "type": "CPU", "name": "Host CPU"})

    mock_import.assert_not_called()
    assert mock_whisperx.call_args.kwargs["device"] == "cpu"


def test_whisperx_coerces_float16_to_int8_on_cpu():
    """float16 is not a usable CTranslate2 compute type on CPU."""
    with mock.patch.object(engine_factory.config, "COMPUTE_TYPE", "float16"):
        with mock.patch.object(engine_factory, "WhisperXEngine") as mock_whisperx:
            engine_factory._create_whisperx_engine("model", {"id": "CPU", "type": "CPU", "name": "Host CPU"})

    assert mock_whisperx.call_args.kwargs["compute_type"] == "int8"


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


def test_create_engine_whisperx_cpu():
    """WhisperX on CPU should create a WhisperX engine with cpu device."""
    with mock.patch("modules.inference.engines.whisperx_engine.worker") as mock_worker:
        mock_worker.call_with_generation.return_value = ("handle-1", 0)
        unit_cpu = {"id": "cpu", "type": "CPU", "name": "Intel CPU"}
        engine = engine_factory.create_engine("WHISPERX", "test-model", unit_cpu)

    assert isinstance(engine, engine_factory.WhisperXEngine)
    assert engine.device == "cpu"


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
