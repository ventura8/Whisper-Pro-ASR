"""The PARAKEET / CANARY composite engine (modules/inference/engines/nemo_onnx_engine.py)."""

from unittest import mock

import pytest

from modules.core import config_nemo, engine_registry
from modules.inference.engines import nemo_chunking, nemo_onnx_engine, nemo_runtime
from tests.inference.engines.nemo_fixtures import FakeModel, companion, speech

CUDA_UNIT = {"type": "CUDA", "id": "cuda:0", "name": "RTX"}


def _engine(engine_type=engine_registry.ENGINE_PARAKEET, model=None, device="cuda", whisper=None):
    """An engine whose load returned ``model`` on ``device``, with a mock companion."""
    loaded = model or FakeModel("CUDAExecutionProvider")
    with mock.patch.object(nemo_runtime, "load", return_value=(loaded, device)) as load:
        engine = nemo_onnx_engine.NemoOnnxEngine(engine_type, CUDA_UNIT, companion=whisper or companion(), cache_root="/cache")
    return engine, loaded, load


def test_construction_loads_the_catalog_model_into_the_cache(monkeypatch):
    """The default model loads from the persistent cache, fp32 on CUDA."""
    monkeypatch.delenv("NEMO_MODEL", raising=False)
    engine, _, load = _engine()
    spec, path, quantization, unit = load.call_args.args
    assert spec.name == "nemo-parakeet-tdt-0.6b-v3"
    assert path == "/cache/nemo/nemo-parakeet-tdt-0.6b-v3"
    assert quantization is None, "fp32 on a CUDA unit"
    assert unit is CUDA_UNIT
    assert engine.device == "cuda"


def test_an_uncovered_language_is_passed_to_the_companion_untouched():
    """Japanese reaches Whisper exactly as the pipeline sent it."""
    whisper = companion()
    engine, model, _ = _engine(whisper=whisper)
    result = engine.transcribe("film.wav", language="ja", task="transcribe", initial_prompt="p", clip_timestamps=[1.0, 2.0], beam_size=5)
    assert result is whisper.transcribe.return_value
    whisper.transcribe.assert_called_once_with(
        "film.wav",
        language="ja",
        task="transcribe",
        initial_prompt="p",
        vad_filter=True,
        word_timestamps=False,
        clip_timestamps=[1.0, 2.0],
        beam_size=5,
    )
    assert len(model.calls) == 0


def test_a_covered_language_decodes_the_clips_on_the_nemo_model():
    """German is decoded clip by clip on the NeMo model."""
    engine, model, _ = _engine()
    with mock.patch.object(nemo_chunking, "decode", return_value=speech(10.0)):
        segments, info = engine.transcribe("film.wav", language="de", clip_timestamps=[1.0, 3.0, 5.0, 8.0], vad_filter=False)
        produced = list(segments)
    assert (info.language, info.language_probability, info.duration) == ("de", 1.0, 10.0)
    assert [round(s.start, 2) for s in produced] == [1.0, 5.0]
    batch = model.calls[0]["waveform"]
    assert [len(chunk) for chunk in batch] == [32000, 48000]
    assert "language" not in model.calls[0], "Parakeet takes no language keyword"


def test_segments_are_produced_lazily_and_stop_when_closed(monkeypatch):
    """Preemption closes the stream; nothing past the current batch may be decoded."""
    monkeypatch.setattr(config_nemo, "NEMO_BATCH_SIZE", 1)
    engine, model, _ = _engine()
    with mock.patch.object(nemo_chunking, "decode", return_value=speech(10.0)):
        segments, _ = engine.transcribe("f.wav", language="en", clip_timestamps=[0, 2, 3, 5, 6, 8], vad_filter=False)
    assert model.calls == [], "nothing decoded before the caller starts reading"
    next(segments)
    segments.close()
    assert len(model.calls) == 1


def test_canary_is_told_the_source_and_target_language():
    """Canary translates natively and reports the source language."""
    engine, model, _ = _engine(engine_registry.ENGINE_CANARY, model=FakeModel(result="Guten Tag."))
    with mock.patch.object(nemo_chunking, "decode", return_value=speech(4.0)):
        segments, info = engine.transcribe("f.wav", language="de", task="translate", clip_timestamps=[0.0, 4.0], vad_filter=False)
        [segment] = list(segments)
    assert model.calls[0]["language"] == "de" and model.calls[0]["target_language"] == "en"
    assert info.language == "de", "a translation reports its source language, as Whisper does"
    assert (segment.start, segment.end) == (0.0, 4.0)


def test_parakeet_translation_goes_to_whisper():
    """Parakeet cannot translate, so Whisper does."""
    whisper = companion()
    engine, _, _ = _engine(whisper=whisper)
    engine.transcribe("f.wav", language="de", task="translate")
    assert whisper.transcribe.call_args.kwargs["task"] == "translate"


def test_word_timestamps_are_attached_when_requested():
    """word_timestamps=True attaches Parakeet's rebuilt words."""
    engine, _, _ = _engine()
    with mock.patch.object(nemo_chunking, "decode", return_value=speech(3.0)):
        segments, _ = engine.transcribe("f.wav", language="en", clip_timestamps=[0.0, 2.0], vad_filter=False, word_timestamps=True)
        [segment] = list(segments)
    assert [w.word for w in segment.words][0] == " The"


def test_initial_prompt_is_ignored_on_nemo_and_says_so_at_debug(caplog):
    """A prompt has no NeMo counterpart; it is logged, not applied."""
    engine, _, _ = _engine()
    caplog.set_level("DEBUG")
    with mock.patch.object(nemo_chunking, "decode", return_value=speech(2.0)):
        segments, _ = engine.transcribe("f.wav", language="en", initial_prompt="Names: Ana", clip_timestamps=[0.0, 2.0], vad_filter=False)
        list(segments)
    assert "initial_prompt has no effect" in caplog.text


def test_an_accelerator_failure_mid_decode_moves_the_engine_to_the_cpu(caplog):
    """A provider that rejects a later input shape costs a reload, not the request."""
    engine, _, _ = _engine(model=FakeModel("CUDAExecutionProvider", fail_on=1))
    cpu = FakeModel("CPUExecutionProvider")
    with (
        mock.patch.object(nemo_chunking, "decode", return_value=speech(3.0)),
        mock.patch.object(nemo_runtime, "load_on_cpu", return_value=cpu),
    ):
        segments, _ = engine.transcribe("f.wav", language="en", clip_timestamps=[0.0, 2.0], vad_filter=False)
        assert list(segments)
    assert engine.device == "cpu" and engine.model is cpu
    assert "continuing on the CPU" in caplog.text


def test_a_cpu_failure_mid_decode_propagates():
    """With nowhere left to fall back to, the error surfaces."""
    engine, _, _ = _engine(model=FakeModel("CPUExecutionProvider", fail_on=1), device="cpu")
    with mock.patch.object(nemo_chunking, "decode", return_value=speech(3.0)):
        segments, _ = engine.transcribe("f.wav", language="en", clip_timestamps=[0.0, 2.0], vad_filter=False)
        with pytest.raises(RuntimeError):
            list(segments)


@pytest.mark.parametrize("audio", ["slice.wav", [0.0, 0.1]])
def test_language_detection_is_always_the_companions(audio):
    """Detection is Whisper's, for paths and samples alike."""
    whisper = companion()
    engine, _, _ = _engine(whisper=whisper)
    assert engine.detect_language(audio) == whisper.detect_language.return_value
    whisper.detect_language.assert_called_once_with(audio)


def test_unload_releases_both_models():
    """Unloading frees the NeMo model and the companion."""
    whisper = companion()
    engine, _, _ = _engine(whisper=whisper)
    engine.unload()
    whisper.unload.assert_called_once()
    assert not hasattr(engine, "model")


class _OutOfMemoryOnBatches(FakeModel):
    """A model whose arena cannot hold more than one chunk at a time."""

    def recognize(self, waveform, **kwargs):
        """Fail any multi-chunk batch the way ONNX Runtime's arena does."""
        if isinstance(waveform, list) and len(waveform) > 1:
            self.calls.append({"waveform": waveform, **kwargs})
            raise RuntimeError("BFCArena::AllocateRawInternal Failed to allocate memory for requested buffer")
        return super().recognize(waveform, **kwargs)


def test_an_out_of_memory_batch_is_retried_chunk_by_chunk_on_the_same_device(caplog):
    """An 8 GB card that cannot hold 4 x 20 s still decodes on the GPU, one chunk at a time."""
    engine, model, _ = _engine(model=_OutOfMemoryOnBatches("CUDAExecutionProvider"))
    with mock.patch.object(nemo_chunking, "decode", return_value=speech(10.0)), mock.patch.object(nemo_runtime, "load_on_cpu") as cpu:
        segments, _ = engine.transcribe("f.wav", language="en", clip_timestamps=[0, 2, 3, 5], vad_filter=False)
        assert len(list(segments)) == 2
    cpu.assert_not_called()
    assert engine.device == "cuda" and len(model.calls) == 3
    assert "one at a time" in caplog.text


class _OutOfMemoryAlways(FakeModel):
    """A card so full that even one chunk does not fit."""

    def recognize(self, waveform, **kwargs):
        """Fail every call with the arena allocator's message."""
        self.calls.append({"waveform": waveform, **kwargs})
        raise RuntimeError("BFCArena::AllocateRawInternal Failed to allocate memory for requested buffer")


def test_a_single_chunk_out_of_memory_decodes_that_chunk_on_the_cpu_and_stays_on_the_card(caplog):
    """One transient shortage must not move the engine to the CPU for the life of the worker."""
    engine, _, _ = _engine(model=_OutOfMemoryAlways("CUDAExecutionProvider"))
    spare = FakeModel("CPUExecutionProvider")
    with (
        mock.patch.object(nemo_chunking, "decode", return_value=speech(10.0)),
        mock.patch.object(nemo_runtime, "load_on_cpu", return_value=spare) as cpu,
    ):
        segments, _ = engine.transcribe("f.wav", language="en", clip_timestamps=[0, 2, 3, 5], vad_filter=False)
        assert len(list(segments)) == 2
    cpu.assert_called_once()
    assert engine.device == "cuda" and engine.model is not spare
    assert len(spare.calls) == 2
    assert "decoding it on the CPU" in caplog.text


class _OutOfMemoryThenBroken(FakeModel):
    """One chunk out of memory, then a provider that cannot run the graph at all."""

    def recognize(self, waveform, **kwargs):
        """First call: arena exhausted; every later call: a non-memory provider error."""
        self.calls.append({"waveform": waveform, **kwargs})
        if len(self.calls) == 1:
            raise RuntimeError("BFCArena::AllocateRawInternal Failed to allocate memory for requested buffer")
        raise RuntimeError("provider rejected the input shape")


def test_moving_to_the_cpu_for_good_reuses_the_spare_instead_of_loading_a_second_copy():
    """A spare loaded for one OOM chunk becomes the engine's model; the CPU never holds two."""
    engine, _, _ = _engine(model=_OutOfMemoryThenBroken("CUDAExecutionProvider"))
    spare = FakeModel("CPUExecutionProvider")
    with (
        mock.patch.object(nemo_chunking, "decode", return_value=speech(10.0)),
        mock.patch.object(nemo_runtime, "load_on_cpu", return_value=spare) as cpu,
        mock.patch.object(config_nemo, "NEMO_BATCH_SIZE", 1),
    ):
        segments, _ = engine.transcribe("f.wav", language="en", clip_timestamps=[0, 2, 3, 5], vad_filter=False)
        assert len(list(segments)) == 2
    cpu.assert_called_once()
    assert engine.device == "cpu" and engine.model is spare and engine._cpu_spare is None


def test_cuda_memory_can_be_capped(monkeypatch):
    """NEMO_GPU_MEM_LIMIT_MB becomes ONNX Runtime's gpu_mem_limit, in bytes."""
    monkeypatch.setattr(config_nemo, "NEMO_GPU_MEM_LIMIT_MB", 3072)
    options = nemo_runtime.providers_for(CUDA_UNIT, ["CUDAExecutionProvider", "CPUExecutionProvider"])[0][1]
    assert options["gpu_mem_limit"] == 3072 * 1024 * 1024
    monkeypatch.setattr(config_nemo, "NEMO_GPU_MEM_LIMIT_MB", 0)
    assert "gpu_mem_limit" not in nemo_runtime.providers_for(CUDA_UNIT, ["CUDAExecutionProvider"])[0][1]
