"""End-to-end tests for the out-of-process engine proxy.

These run a real subprocess executing ``inference_worker``'s real streaming handlers.
Only ``engine_factory.create_engine`` is stubbed, so the transport, the segment/info
normalisation and the proxy's iterator contract are all genuinely exercised without
loading multi-gigabyte weights.
"""

import pytest

from modules.inference.engines import isolated_engine, worker_channel, worker_runtime
from modules.inference.engines.base import InferenceInfo, SegmentWrapper
from tests.inference.engines import worker_fixtures

UNIT = {"id": "test:0", "type": "GPU", "name": "Test Accelerator"}


@pytest.fixture(name="engine")
def _engine(monkeypatch):
    channel = worker_channel.WorkerChannel(
        worker_fixtures.engine_worker_main,
        name="test-engine-worker",
        log_tag="TestEngine",
        call_timeout_sec=30.0,
        lock_warn_sec=30.0,
    )
    monkeypatch.setitem(isolated_engine._CHANNELS, "INTEL-WHISPER", channel)
    try:
        yield isolated_engine.IsolatedEngine("INTEL-WHISPER", "3", UNIT)
    finally:
        channel.shutdown()


def test_transcribe_returns_info_before_segments(engine):
    segments, info = engine.transcribe("/tmp/audio.wav")

    assert isinstance(info, InferenceInfo)
    assert info.language == "es"
    assert info.language_probability == pytest.approx(0.87)
    assert info.duration == pytest.approx(12.5)


def test_transcribe_streams_typed_segments(engine):
    segments, _info = engine.transcribe("/tmp/audio.wav")
    collected = list(segments)

    assert len(collected) == 3
    assert all(isinstance(segment, SegmentWrapper) for segment in collected)
    assert [s.text for s in collected] == ["segment 0", "segment 1", "segment 2"]
    assert collected[1].start == pytest.approx(1.0)


def test_segments_arrive_incrementally_rather_than_as_one_batch(engine):
    """The parent must be able to act on a segment before the worker has finished.

    This is what keeps per-segment progress and cooperative preemption meaningful.
    """
    segments, _info = engine.transcribe("/tmp/audio.wav")

    first = next(segments)
    assert first.text == "segment 0"

    segments.close()


def test_abandoning_transcription_preserves_the_worker(engine):
    """Preemption abandons the segment iterator; that must not cost a model reload."""
    segments, _info = engine.transcribe("/tmp/audio.wav")
    next(segments)
    segments.close()

    assert engine._channel.call("loaded_handles") == ["test:0"], "model should still be resident"

    segments, _info = engine.transcribe("/tmp/audio.wav")
    assert len(list(segments)) == 3


def test_isolation_env_is_applied_inside_the_worker(engine):
    """Intel workers must start with CUDA blanked, so no CUDA context can ever form."""
    assert engine._channel.call("isolation_env_seen", handle=engine.handle, name="CUDA_VISIBLE_DEVICES") == ""


def test_isolation_follows_the_unit_vendor_not_the_engine(engine):
    """An Intel unit must not have SYCL forced to CPU, whatever engine runs on it.

    Keying this by engine forced ONEAPI_DEVICE_SELECTOR="*:cpu" for openai-whisper, which
    is torch-based and vendor-agnostic -- silently disabling Intel XPU for precisely the
    engine the intel-xpu image exists to accelerate.
    """
    assert engine._channel.call("isolation_env_seen", handle=engine.handle, name="ONEAPI_DEVICE_SELECTOR") is None


def test_cuda_units_are_blinded_to_the_other_vendors():
    assert worker_runtime.ISOLATION_ENV["CUDA"]["ONEAPI_DEVICE_SELECTOR"] == "*:cpu"
    assert worker_runtime.ISOLATION_ENV["CUDA"]["HIP_VISIBLE_DEVICES"] == ""
    assert "CUDA_VISIBLE_DEVICES" not in worker_runtime.ISOLATION_ENV["CUDA"], "a CUDA unit must keep its own GPU"


def test_no_isolation_value_is_ever_an_empty_sycl_selector():
    """Intel's SYCL runtime aborts the process on an empty ONEAPI_DEVICE_SELECTOR.

    That is not a hypothetical: docker-compose set it to "" and the service crash-looped
    on SIGSEGV as soon as an XPU torch build was present to read it.
    """
    for vendor, env in worker_runtime.ISOLATION_ENV.items():
        assert env.get("ONEAPI_DEVICE_SELECTOR") != "", f"{vendor} would abort a SYCL runtime"


def test_detect_language_rejects_decoded_samples(engine):
    """Sending arrays would copy audio across the pipe; paths only."""
    with pytest.raises(TypeError, match="audio path"):
        engine.detect_language([0.0, 0.1, 0.2])


def test_unload_releases_the_model_but_keeps_the_process(engine):
    engine.unload()
    assert engine._channel.call("loaded_handles") == []
    assert engine._channel.is_running(), "other units' models must survive one unit unloading"


def test_worker_crash_is_survived_by_reloading_the_model(engine):
    """A dead worker becomes a slow request, not a failed one."""
    list(engine.transcribe("/tmp/audio.wav")[0])
    engine._channel.shutdown()

    segments, info = engine.transcribe("/tmp/audio.wav")
    assert info.language == "es"
    assert len(list(segments)) == 3
