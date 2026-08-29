"""The parent-side UVR proxy: channel reuse, cached state, and the one-shot retry.

The existing `test_isolated_preprocessor.py` drives a real worker subprocess, which is the
right way to prove the wire protocol but leaves the proxy's own decisions untested -- and
two of them have already caused visible defects:

* `separator` is answered from a cached snapshot rather than an RPC. Querying the worker
  there made every telemetry poll block until separation finished, stalling the dashboard
  for the whole job.
* the retry is deliberately NOT applied to a failure the worker reported. Retrying those
  ran a genuinely failing separation twice.

Nothing here spawns a process; the channel is a fake.
"""

from __future__ import annotations

import threading

import pytest

from modules.inference.engines import worker_channel
from modules.inference.pipeline.preprocessing import isolated


class FakeChannel:
    """Stands in for a WorkerChannel: scripted results, recorded calls."""

    def __init__(self, *, state=None, stream_events=None, call_errors=None):
        """Record the construction arguments."""
        self.lock = threading.Lock()
        self.calls: list[tuple[str, dict]] = []
        self._state = state or {"loaded": False, "providers": []}
        self._stream_events = stream_events or []
        self._call_errors = call_errors or {}
        self.streams_opened = 0
        self.closed_streams = 0

    def call(self, cmd, **kwargs):
        """Record the RPC and return its scripted result."""
        self.calls.append((cmd, kwargs))
        error = self._call_errors.get(cmd)
        if error is not None:
            raise error
        if cmd == "state":
            return self._state
        if cmd == "load":
            return kwargs.get("unit", {}).get("id", "handle")
        return None

    def stream(self, cmd, **kwargs):
        """Record the RPC and return the scripted event generator."""
        self.calls.append((cmd, kwargs))
        self.streams_opened += 1
        channel = self

        def gen():
            """Yield the scripted events, recording the close."""
            try:
                for event in list(channel._stream_events):
                    if isinstance(event, Exception):
                        raise event
                    yield event
            finally:
                channel.closed_streams += 1

        return gen()


@pytest.fixture(autouse=True)
def _clean_channels():
    """Clear the module-global channel registry around each test."""
    isolated._CHANNELS.clear()
    yield
    isolated._CHANNELS.clear()


class TestChannelForDeviceType:
    """One channel per device type -- two means two workers holding two UVR models."""

    def test_the_first_call_creates_a_channel_named_for_the_device(self, monkeypatch):
        """The first call creates a channel named for the device."""
        built = {}

        def fake_channel(worker_main, name, log_tag, call_timeout_sec=0.0):
            """Build a fake channel, recording how it was configured."""
            built.update(name=name, log_tag=log_tag, call_timeout_sec=call_timeout_sec)
            return FakeChannel()

        monkeypatch.setattr(isolated.worker_channel, "WorkerChannel", fake_channel)
        isolated.channel_for("GPU.0")

        assert built["name"] == "uvr-gpu_0-worker", "the dot would make an odd process name"
        assert built["log_tag"] == "UVR GPU.0 worker"
        # WorkerChannel defaults call_timeout_sec to 0.0, which disables the deadline
        # entirely. A separator wedged in the vendor runtime then holds the channel lock for
        # the life of the process and every later request on that device queues behind it.
        assert built["call_timeout_sec"] > 0, "a UVR worker with no deadline can wedge the device"

    def test_a_second_call_for_the_same_device_reuses_the_channel(self, monkeypatch):
        """A second call for the same device reuses the channel."""
        monkeypatch.setattr(isolated.worker_channel, "WorkerChannel", lambda *a, **k: FakeChannel())
        assert isolated.channel_for("NPU") is isolated.channel_for("NPU")

    def test_different_device_types_get_different_channels(self, monkeypatch):
        """Different device types get different channels."""
        monkeypatch.setattr(isolated.worker_channel, "WorkerChannel", lambda *a, **k: FakeChannel())
        assert isolated.channel_for("GPU") is not isolated.channel_for("NPU")

    def test_concurrent_first_calls_build_exactly_one_channel(self, monkeypatch):
        """The loser's worker was orphaned: still running, still holding device memory,
        unreachable and never shut down."""
        built: list[FakeChannel] = []
        ready = threading.Barrier(2, timeout=5.0)

        def fake_channel(*_a, **_k):
            """Build a fake channel, recording its name."""
            channel = FakeChannel()
            built.append(channel)
            return channel

        monkeypatch.setattr(isolated.worker_channel, "WorkerChannel", fake_channel)
        results: list[object] = []

        def arrive():
            """Enter the contended path once both threads are ready."""
            ready.wait()
            results.append(isolated.channel_for("GPU"))

        threads = [threading.Thread(target=arrive, daemon=True) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5.0)
            assert not thread.is_alive()

        assert len(built) == 1, f"{len(built)} workers were spawned for one device"
        assert results[0] is results[1]

    def test_shutdown_all_terminates_every_channel(self, monkeypatch):
        """Shutdown all terminates every channel."""
        shut: list[str] = []

        def fake_channel(worker_main, name, log_tag, call_timeout_sec=0.0):
            """Build a fake channel, recording its name."""
            channel = FakeChannel()
            channel.shutdown = lambda: shut.append(name)
            return channel

        monkeypatch.setattr(isolated.worker_channel, "WorkerChannel", fake_channel)
        isolated.channel_for("GPU")
        isolated.channel_for("NPU")

        isolated.shutdown_all()
        assert sorted(shut) == ["uvr-gpu-worker", "uvr-npu-worker"]


class TestSeparatorProbe:
    """The picklable stand-in telemetry and metrics read."""

    def test_it_is_truthy_so_the_in_process_idiom_keeps_working(self):
        """`if preprocessor.separator` must keep meaning "UVR is using memory"."""
        assert bool(isolated._SeparatorProbe([])) is True

    def test_it_carries_the_providers_the_worker_reported(self):
        """It carries the providers the worker reported."""
        probe = isolated._SeparatorProbe(["OpenVINOExecutionProvider"])
        assert probe.onnx_execution_provider == ["OpenVINOExecutionProvider"]
        assert probe.providers() == ["OpenVINOExecutionProvider"]

    def test_providers_returns_a_copy_so_a_caller_cannot_edit_the_snapshot(self):
        """Providers returns a copy so a caller cannot edit the snapshot."""
        probe = isolated._SeparatorProbe(["CPUExecutionProvider"])
        probe.providers().append("mutated")
        assert probe.onnx_execution_provider == ["CPUExecutionProvider"]


def _preprocessor(channel, unit=None):
    """Build an IsolatedPreprocessor bound to a fake channel."""
    unit = unit or {"id": "GPU.0", "type": "GPU", "name": "Intel GPU"}
    prep = isolated.IsolatedPreprocessor.__new__(isolated.IsolatedPreprocessor)
    prep._channel = channel
    prep._unit = unit
    prep._handle = "GPU.0"
    prep._device_type = unit["type"]
    prep._device_id = unit["id"]
    prep._state_cache = {"loaded": False, "providers": []}
    return prep


class TestSeparatorIsAnsweredFromCache:
    """Telemetry polls this once a second while the channel lock is held all job long."""

    def test_nothing_loaded_reads_as_none(self):
        """Nothing loaded reads as none."""
        assert _preprocessor(FakeChannel()).separator is None

    def test_a_loaded_model_reports_a_probe_carrying_its_providers(self):
        """A loaded model reports a probe carrying its providers."""
        prep = _preprocessor(FakeChannel())
        prep._state_cache = {"loaded": True, "providers": ["CUDAExecutionProvider"]}

        assert prep.separator.onnx_execution_provider == ["CUDAExecutionProvider"]

    def test_reading_it_makes_no_rpc_at_all(self):
        """An RPC here blocked every dashboard poll until separation finished."""
        channel = FakeChannel()
        prep = _preprocessor(channel)
        prep._state_cache = {"loaded": True, "providers": []}

        _ = prep.separator
        assert channel.calls == [], "the snapshot exists precisely so telemetry never waits on the lock"

    def test_the_lock_exposed_is_the_channel_lock(self):
        """metrics calls `.locked()` on this; it is held for the whole separation stream, so
        it answers exactly the question "is UVR working right now". Returning the lock object
        rather than a boolean is what keeps the in-process interface intact -- an earlier
        version exposed something else, the call raised AttributeError inside a guard that
        swallowed it, and the Intel and AMD utilization charts sat at 0% for entire
        separations.
        """
        channel = FakeChannel()
        assert _preprocessor(channel).lock is channel.lock


class TestRefreshState:
    """refresh state."""

    def test_it_caches_what_the_worker_reports(self):
        """It caches what the worker reports."""
        channel = FakeChannel(state={"loaded": True, "providers": ["X"]})
        prep = _preprocessor(channel)

        prep.refresh_state()
        assert prep._state_cache == {"loaded": True, "providers": ["X"]}

    def test_no_handle_means_nothing_is_loaded_and_no_rpc_is_made(self):
        """No handle means nothing is loaded and no rpc is made."""
        channel = FakeChannel()
        prep = _preprocessor(channel)
        prep._handle = None

        prep.refresh_state()
        assert prep._state_cache == {"loaded": False, "providers": []}
        assert channel.calls == []

    def test_a_dead_worker_reads_as_unloaded_rather_than_raising(self):
        """This runs on the telemetry path; an exception here is a dashboard 500."""
        channel = FakeChannel(call_errors={"state": worker_channel.WorkerError("worker died")})
        prep = _preprocessor(channel)
        prep._state_cache = {"loaded": True, "providers": ["X"]}

        prep.refresh_state()
        assert prep._state_cache == {"loaded": False, "providers": []}


class TestOffloadAndUnload:
    """offload and unload."""

    def test_offload_asks_the_worker_to_release_device_memory(self):
        """Offload asks the worker to release device memory."""
        channel = FakeChannel()
        _preprocessor(channel).offload()
        assert channel.calls == [("offload", {"handle": "GPU.0"})]

    def test_offload_without_a_handle_is_a_no_op(self):
        """Offload without a handle is a no op."""
        channel = FakeChannel()
        prep = _preprocessor(channel)
        prep._handle = None
        prep.offload()
        assert channel.calls == []

    def test_offload_tolerates_a_dead_worker(self):
        """Offload tolerates a dead worker."""
        channel = FakeChannel(call_errors={"offload": worker_channel.WorkerError("gone")})
        _preprocessor(channel).offload()  # must not raise

    def test_unload_drops_the_handle_and_the_cached_state(self):
        """Unload drops the handle and the cached state."""
        channel = FakeChannel()
        prep = _preprocessor(channel)
        prep._state_cache = {"loaded": True, "providers": ["X"]}

        prep.unload_model()

        assert channel.calls == [("unload", {"handle": "GPU.0"})]
        assert prep._handle is None
        assert prep._state_cache == {"loaded": False, "providers": []}

    def test_unload_clears_state_even_when_the_worker_is_already_dead(self):
        """A dead worker has released everything this call would have freed, but the parent
        must still stop believing a model is resident."""
        channel = FakeChannel(call_errors={"unload": worker_channel.WorkerError("gone")})
        prep = _preprocessor(channel)
        prep._state_cache = {"loaded": True, "providers": ["X"]}

        prep.unload_model()

        assert prep._handle is None
        assert prep._state_cache == {"loaded": False, "providers": []}

    def test_unload_without_a_handle_is_a_no_op(self):
        """Unload without a handle is a no op."""
        channel = FakeChannel()
        prep = _preprocessor(channel)
        prep._handle = None
        prep.unload_model()
        assert channel.calls == []
