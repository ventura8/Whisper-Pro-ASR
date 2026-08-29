"""The parent-side engine proxy: stream framing, channel lifecycle, and detection.

`test_isolated_engine.py` drives a real worker subprocess; this covers the proxy's own
logic against a fake channel, including the failure shapes a subprocess test cannot
conveniently produce -- a stream that never sends `info`, and a worker that dies during
unload.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from modules.inference.engines import isolated_engine, worker_channel


class FakeChannel:
    """A test double for channel."""

    def __init__(self, *, events=None, call_result=None, call_error=None, generation=0):
        """Record the construction arguments."""
        self._events = events or []
        self._call_result = call_result
        self._call_error = call_error
        self._generation = generation
        self.calls: list[tuple[str, dict]] = []
        self.cancelled = False
        self.shutdowns = 0
        self.closed = 0

    def generation(self):
        """The channel's current worker generation."""
        return self._generation

    def call(self, cmd, **kwargs):
        """Record the RPC and return its scripted result."""
        self.calls.append((cmd, kwargs))
        if self._call_error is not None:
            raise self._call_error
        return self._call_result

    def call_with_generation(self, cmd, **kwargs):
        """Record the RPC and return (handle, generation)."""
        self.calls.append((cmd, kwargs))
        return kwargs.get("unit", {}).get("id", "handle"), self._generation

    def stream(self, cmd, **kwargs):
        """Record the RPC and return the scripted event generator."""
        self.calls.append((cmd, kwargs))
        channel = self

        def gen():
            """Yield the scripted events, recording the close."""
            try:
                yield from list(channel._events)
            finally:
                channel.closed += 1

        return gen()

    def cancel_in_stream(self):
        """Record that the parent asked the stream to stop."""
        self.cancelled = True

    def shutdown(self):
        """Record that the worker was terminated."""
        self.shutdowns += 1


@pytest.fixture(autouse=True)
def _clean_channels():
    """Isolate the module-global channel registry, restoring whatever was already there.

    Saved and restored rather than cleared: clearing discards live channels owned by another
    test in the same worker, and each one holds a worker process that is then unreachable and
    never shut down.
    """
    saved = dict(isolated_engine._CHANNELS)
    isolated_engine._CHANNELS.clear()
    yield
    isolated_engine._CHANNELS.clear()
    isolated_engine._CHANNELS.update(saved)


def _engine(channel, engine_type="FASTER-WHISPER"):
    """Build the object under test against a fake channel."""
    engine = isolated_engine.IsolatedEngine.__new__(isolated_engine.IsolatedEngine)
    engine.engine_type = engine_type
    engine.model_id = "/models/ct2"
    engine.unit = {"id": "cuda:0", "type": "CUDA", "name": "GPU 0"}
    engine._channel = channel
    engine.handle = "cuda:0"
    engine._generation = channel.generation()
    return engine


class TestChannelRegistry:
    """One channel per engine type -- isolation relocates models, it must not copy them."""

    def test_a_channel_is_created_once_per_engine_type(self, monkeypatch):
        """A channel is created once per engine type."""
        built = []
        monkeypatch.setattr(isolated_engine.worker_channel, "WorkerChannel", lambda *a, **k: built.append(k.get("name")) or FakeChannel())
        first = isolated_engine.channel_for("FASTER-WHISPER")
        second = isolated_engine.channel_for("FASTER-WHISPER")

        assert first is second
        assert built == ["faster_whisper-worker"], "the engine name becomes a process name"

    def test_different_engine_types_get_their_own_workers(self, monkeypatch):
        """Mutually exclusive runtimes -- CUDA vs OpenVINO, CUDA torch vs ROCm torch -- can
        only coexist in one deployment by living in separate processes."""
        monkeypatch.setattr(isolated_engine.worker_channel, "WorkerChannel", lambda *a, **k: FakeChannel())
        assert isolated_engine.channel_for("FASTER-WHISPER") is not isolated_engine.channel_for("INTEL-WHISPER")

    def test_concurrent_first_calls_build_exactly_one_worker(self, monkeypatch):
        """Concurrent first calls build exactly one worker."""
        built: list[Any] = []
        ready = threading.Barrier(2, timeout=5.0)
        monkeypatch.setattr(isolated_engine.worker_channel, "WorkerChannel", lambda *a, **k: built.append(1) or FakeChannel())
        results = []

        def arrive():
            """Enter the contended path once both threads are ready."""
            ready.wait()
            results.append(isolated_engine.channel_for("WHISPERX"))

        threads = [threading.Thread(target=arrive, daemon=True) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5.0)
            assert not thread.is_alive()

        assert len(built) == 1, "the loser's worker would be orphaned, holding device memory"
        assert results[0] is results[1]

    def test_active_channels_returns_a_copy(self, monkeypatch):
        """Active channels returns a copy."""
        monkeypatch.setattr(isolated_engine.worker_channel, "WorkerChannel", lambda *a, **k: FakeChannel())
        isolated_engine.channel_for("FASTER-WHISPER")

        snapshot = isolated_engine.active_channels()
        snapshot.clear()
        assert isolated_engine.active_channels(), "callers must not be able to empty the registry"

    def test_shutdown_all_terminates_every_worker(self, monkeypatch):
        """Shutdown all terminates every worker."""
        channels = []
        monkeypatch.setattr(isolated_engine.worker_channel, "WorkerChannel", lambda *a, **k: channels.append(FakeChannel()) or channels[-1])
        isolated_engine.channel_for("FASTER-WHISPER")
        isolated_engine.channel_for("INTEL-WHISPER")

        isolated_engine.shutdown_all()
        assert [c.shutdowns for c in channels] == [1, 1]


class TestReloadingAfterAWorkerRestart:
    """A crash and respawn leaves the handle pointing at a pool that no longer exists."""

    def test_a_matching_generation_does_not_reload(self):
        """A matching generation does not reload."""
        channel = FakeChannel(generation=3)
        engine = _engine(channel)
        engine._generation = 3

        engine._ensure_loaded()
        assert channel.calls == []

    def test_a_bumped_generation_reloads_the_model_transparently(self):
        """This is what turns a worker death into a slow request rather than a failed one."""
        channel = FakeChannel(generation=5)
        engine = _engine(channel)
        engine._generation = 4

        engine._ensure_loaded()

        # engine_device follows every load: execution_status prefers a loaded engine's own
        # device over a prediction, and the proxy has to ask the worker for it.
        assert [c[0] for c in channel.calls] == ["load_model", "engine_device"]
        assert engine._generation == 5


class TestTranscribeStreamFraming:
    """`info` arrives first, before any segment is consumed."""

    def test_info_is_read_ahead_and_segments_follow(self):
        """Info is read ahead and segments follow."""
        channel = FakeChannel(
            events=[
                {"event": "info", "info": {"language": "es", "language_probability": 0.8, "duration": 4.2}},
                {"event": "segment", "segment": {"start": 0.0, "end": 1.0, "text": "hola"}},
            ]
        )
        engine = _engine(channel)

        segments, info = engine.transcribe("/clip.wav", language="es")

        assert (info.language, info.duration) == ("es", 4.2)
        assert [s.text for s in segments] == ["hola"]

    def test_events_that_are_not_segments_are_ignored_by_the_segment_iterator(self):
        """Events that are not segments are ignored by the segment iterator."""
        channel = FakeChannel(
            events=[
                {"event": "info", "info": {}},
                {"event": "progress", "percent": 40},
                {"event": "segment", "segment": {"start": 0.0, "end": 1.0, "text": "one"}},
            ]
        )
        segments, _ = _engine(channel).transcribe("/clip.wav")
        assert [s.text for s in segments] == ["one"]

    def test_a_stream_with_no_info_event_is_a_named_worker_error(self):
        """Without this the caller unpacks None and fails somewhere far from the cause."""
        channel = FakeChannel(events=[{"event": "segment", "segment": {}}])

        with pytest.raises(worker_channel.WorkerError, match="sent no info event"):
            _engine(channel).transcribe("/clip.wav")
        assert channel.closed == 1, "the stream must be closed so the worker stops decoding"

    def test_an_empty_stream_is_also_a_named_worker_error(self):
        """An empty stream is also a named worker error."""
        channel = FakeChannel(events=[])
        with pytest.raises(worker_channel.WorkerError, match="sent no info event"):
            _engine(channel).transcribe("/clip.wav")

    def test_decode_params_reach_the_worker_as_one_params_payload(self):
        """Decode params reach the worker as one params payload."""
        channel = FakeChannel(events=[{"event": "info", "info": {}}])
        engine = _engine(channel)

        engine.transcribe("/clip.wav", language="fr", word_timestamps=True, beam_size=5)

        cmd, kwargs = channel.calls[0]
        assert cmd == "transcribe"
        assert kwargs["params"]["language"] == "fr"
        assert kwargs["params"]["word_timestamps"] is True
        assert kwargs["params"]["beam_size"] == 5, "engine-specific kwargs must pass through"


class TestPayloadReconstruction:
    """payload reconstruction."""

    def test_a_segment_payload_becomes_a_wrapper(self):
        """A segment payload becomes a wrapper."""
        wrapper = isolated_engine._segment_from_dict({"start": 1.0, "end": 2.0, "text": "hi", "words": [{"start": 1.0}]})
        assert (wrapper.start, wrapper.end, wrapper.text) == (1.0, 2.0, "hi")
        assert wrapper.words == [{"start": 1.0}]

    def test_a_sparse_segment_payload_gets_safe_defaults(self):
        """A sparse segment payload gets safe defaults."""
        wrapper = isolated_engine._segment_from_dict({})
        assert (wrapper.start, wrapper.end, wrapper.text, wrapper.words) == (0.0, 0.0, "", None)

    def test_info_probabilities_are_rebuilt_as_tuples(self):
        """Info probabilities are rebuilt as tuples."""
        info = isolated_engine._info_from_dict({"language": "de", "all_language_probs": [["de", 0.9], ["en", 0.1]]})
        assert info.all_language_probs == [("de", 0.9), ("en", 0.1)]

    def test_info_without_probabilities_reports_none(self):
        """Info without probabilities reports none."""
        assert isolated_engine._info_from_dict({}).all_language_probs is None

    def test_info_without_a_language_defaults_to_english(self):
        """Info without a language defaults to english."""
        assert isolated_engine._info_from_dict({}).language == "en"


class TestDetectLanguage:
    """detect language."""

    def test_a_path_is_forwarded_and_the_result_unpacked(self):
        """A path is forwarded and the result unpacked."""
        channel = FakeChannel(call_result={"language": "it", "probability": 0.7, "all_probs": [("it", 0.7)]})
        assert _engine(channel).detect_language("/clip.wav") == ("it", 0.7, [("it", 0.7)])

    def test_decoded_samples_are_refused_rather_than_copied_across_the_pipe(self):
        """Sending an array would copy the whole clip through the pipe, and the pipeline's
        batch detection already runs entirely worker-side."""
        with pytest.raises(TypeError, match="requires an audio path"):
            _engine(FakeChannel()).detect_language([0.0, 1.0])

    def test_batch_detection_yields_only_detection_events(self):
        """Batch detection yields only detection events."""
        channel = FakeChannel(events=[{"event": "detection", "index": 0}, {"event": "progress"}, {"event": "detection", "index": 1}])
        results = list(_engine(channel).detect_language_batch("/clip.wav", segment_count=2))
        assert [r["index"] for r in results] == [0, 1]


class TestLifecycle:
    """lifecycle."""

    def test_cancel_asks_the_channel_to_stop_the_in_flight_stream(self):
        """Cancel asks the channel to stop the in flight stream."""
        channel = FakeChannel()
        _engine(channel).cancel()
        assert channel.cancelled is True

    def test_unload_drops_this_units_model_in_the_worker(self):
        """Unload drops this units model in the worker."""
        channel = FakeChannel()
        _engine(channel).unload()
        assert channel.calls == [("unload_model", {"handle": "cuda:0"})]

    def test_unload_against_a_dead_worker_is_not_an_error(self):
        """A dead worker has already released everything this call would have freed."""
        channel = FakeChannel(call_error=worker_channel.WorkerError("worker gone"))
        _engine(channel).unload()  # must not raise

    def test_the_proxy_marks_itself_as_out_of_process_on_the_type(self):
        """language_detection_core needs that answer across an import cycle, and reads it off
        the type -- a MagicMock instance would invent the attribute and claim to be one."""
        assert isolated_engine.IsolatedEngine.IS_ISOLATED_ENGINE is True
