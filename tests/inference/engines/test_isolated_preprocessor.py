"""Out-of-process vocal separation must behave exactly like the in-process manager."""

from unittest import mock

import pytest

from modules.inference.engines import worker_channel
from modules.inference.pipeline import preprocessing_worker
from modules.inference.pipeline.preprocessing import isolated
from tests.inference.engines import prep_worker_fixtures

UNIT = {"id": "cuda:0", "type": "CUDA", "name": "NVIDIA GPU 0"}


@pytest.fixture(name="prep")
def _prep(monkeypatch):
    channel = worker_channel.WorkerChannel(
        prep_worker_fixtures.worker_main,
        name="test-uvr-worker",
        log_tag="TestUVR",
        call_timeout_sec=30.0,
        lock_warn_sec=30.0,
    )
    monkeypatch.setitem(isolated._CHANNELS, "CUDA", channel)
    try:
        yield isolated.IsolatedPreprocessor(UNIT)
    finally:
        channel.shutdown()


def test_separation_returns_the_worker_s_output_path(prep):
    assert prep.preprocess_audio("/tmp/clip.wav").endswith(".vocals.wav")


#: Chunk count and per-chunk delay for the preemption test's worker.
#:
#: Deliberately sized against preprocessing_worker._HEARTBEAT_SEC rather than picked to
#: look small. The worker emits a `tick` only when its separation thread is STILL ALIVE
#: after a heartbeat-long join, and `tick` is the only event that makes the parent call
#: yield_cb. With the previous 3 chunks x 0.05s the whole separation finished inside one
#: 0.2s heartbeat, so the parent received exactly ["result"], yield_cb was never called,
#: and this test could not pass under any implementation -- it failed from the commit that
#: introduced it. 12 x 0.15s is ~1.8s, roughly nine heartbeats, so cancellation lands with
#: most chunks still outstanding.
_PREEMPT_CHUNKS = 12
_PREEMPT_CHUNK_DELAY = 0.15


def test_the_preemption_fixture_outlasts_the_worker_heartbeat():
    """Guard the guard: the numbers above only work while they exceed the heartbeat.

    If _HEARTBEAT_SEC is ever raised past the fixture's total runtime, the preemption test
    below silently stops testing preemption instead of failing loudly. This asserts the
    relationship the test depends on.
    """
    total = _PREEMPT_CHUNKS * _PREEMPT_CHUNK_DELAY
    assert total > preprocessing_worker._HEARTBEAT_SEC * 4, (
        f"the fake separation runs {total:.2f}s against a {preprocessing_worker._HEARTBEAT_SEC}s heartbeat; "
        "the parent needs several ticks before it can preempt"
    )


def test_preemption_still_crosses_the_process_boundary(prep):
    """yield_cb raising must abort the worker's separation, not just the parent's wait.

    Losing this would be a silent regression: separation would run to completion on the
    device while the scheduler believed the task had yielded.
    """

    class _Preempted(Exception):
        pass

    prep._channel.call("configure", chunks=_PREEMPT_CHUNKS, delay=_PREEMPT_CHUNK_DELAY)

    calls = {"n": 0}

    def _yield_cb():
        calls["n"] += 1
        if calls["n"] >= 2:
            raise _Preempted()

    with pytest.raises(_Preempted):
        prep.preprocess_audio("/tmp/clip.wav", yield_cb=_yield_cb)

    # Asserted explicitly, because a yield_cb that is never called cannot raise and the
    # pytest.raises above would then fail with "DID NOT RAISE" -- a message that points at
    # preemption rather than at the stream having carried no tick events at all, which is
    # the failure this test actually had.
    assert calls["n"] >= 2, f"yield_cb was called {calls['n']} times; the stream carried no preemption opportunity"

    # The worker must have stopped separating, not merely stopped being waited on. Losing
    # this would be a silent regression: separation would run to completion on the device
    # while the scheduler believed the task had yielded, and the parent could not tell.
    progress = prep._channel.call("progress", handle=prep._handle)
    assert progress["chunks_done"] < progress["chunks_total"], (
        f"worker completed {progress['chunks_done']}/{progress['chunks_total']} chunks; separation was not cancelled"
    )

    # Only then is the channel checked: it must also be reusable rather than stranded.
    assert prep._channel.call("state", handle=prep._handle)["loaded"] in (True, False)


def test_separator_probe_reports_loaded_state(prep):
    assert prep.separator is None, "nothing is loaded before the first separation"
    prep.preprocess_audio("/tmp/clip.wav")
    assert prep.separator is not None


def test_separator_probe_does_not_touch_the_channel(prep):
    """Telemetry polls this every second while a separation may hold the channel lock.

    Querying the worker here blocked every dashboard poll for the whole duration of a
    job -- observed live as "Blocked on lock for >5.0s during call (cmd=state)".
    """
    prep.preprocess_audio("/tmp/clip.wav")

    with mock.patch.object(prep._channel, "call", side_effect=AssertionError("separator must not call the worker")):
        assert prep.separator is not None
        assert prep.separator.onnx_execution_provider is not None


def test_unload_releases_the_model(prep):
    prep.preprocess_audio("/tmp/clip.wav")
    prep.unload_model()
    assert prep.separator is None


def test_offload_is_forwarded(prep):
    """The worker must actually receive the command, not just the parent return cleanly."""
    prep.preprocess_audio("/tmp/clip.wav")
    assert prep._channel.call("progress", handle=prep._handle)["offloaded"] is False

    prep.offload()

    assert prep._channel.call("progress", handle=prep._handle)["offloaded"] is True


def test_cuda_worker_cannot_see_other_vendors(prep):
    """A CUDA UVR worker must not be able to build an OpenVINO or ROCm context."""
    prep._ensure_loaded()
    assert prep._channel.call("isolation_env_seen", name="HIP_VISIBLE_DEVICES") == ""


def test_device_properties_match_the_unit(prep):
    assert prep.unit == UNIT
    assert prep.device_id == "cuda:0"
    assert prep.device_type == "CUDA"
