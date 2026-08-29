"""The child-side request loop that every isolated worker runs.

This is the protocol both worker families speak, and almost none of it was reachable from
the existing suites: they exercise the *parent* half (`worker_channel`) against a real
subprocess, so the child's own dispatch, cancellation and serialization paths were only
ever executed inside a process the tests could not inspect.

Nothing here spawns anything. `serve` and its helpers take a `Connection`, and a fake one
is enough to drive every branch -- which is the point: the behaviours below are what make
cooperative preemption survive the process boundary, and they should not need a subprocess
to pin down.
"""

from __future__ import annotations

import logging
import pickle
from typing import Any

import pytest

from modules.inference.engines import worker_runtime


class FakeConn:
    """A `Connection` stand-in: scripted inbound messages, recorded outbound ones."""

    def __init__(self, inbound: list[Any] | None = None, *, control: list[Any] | None = None):
        """Record the construction arguments."""
        self._inbound = list(inbound or [])
        # Control messages are what `poll`/`recv` see during a stream, separately from the
        # request queue -- the real pipe is one channel, but keeping them apart here lets a
        # test say "cancel arrives after the second event" without ordering games.
        self._control = list(control or [])
        self.sent: list[Any] = []
        self.send_failures: list[Exception] = []
        self.recv_error: type[Exception] | None = None

    def recv(self):
        """Return the next scripted message, requests before control."""
        # Requests first, control second. The real pipe is one channel and the parent only
        # sends a cancel once a stream is already running, so serving control messages ahead
        # of the request that starts the stream would be an ordering the parent cannot
        # produce -- and it made the cancel arrive before the handler existed.
        if self.recv_error is not None:
            raise self.recv_error("pipe gone")
        if self._inbound:
            return self._inbound.pop(0)
        if self._control:
            return self._control.pop(0)
        raise EOFError("no more requests")

    def poll(self, _timeout=None):
        """Whether a control message is pending."""
        return bool(self._control)

    def send(self, message):
        """Record an outbound message, or raise the next scripted failure."""
        if self.send_failures:
            raise self.send_failures.pop(0)
        self.sent.append(message)


def _events(conn: FakeConn) -> list[str]:
    """The event kinds the fake channel recorded."""
    return [m.get("event") for m in conn.sent]


class TestIterRequests:
    """How a worker's life ends: the two ways a parent stops it, neither an error."""

    def test_requests_are_yielded_in_order(self):
        """Requests are yielded in order."""
        conn = FakeConn([{"cmd": "a"}, {"cmd": "b"}])
        assert list(worker_runtime.iter_requests(conn)) == [{"cmd": "a"}, {"cmd": "b"}]

    def test_the_none_stop_token_ends_the_loop_without_yielding_it(self):
        """The none stop token ends the loop without yielding it."""
        conn = FakeConn([{"cmd": "a"}, None, {"cmd": "never"}])
        assert list(worker_runtime.iter_requests(conn)) == [{"cmd": "a"}]

    @pytest.mark.parametrize("error", [EOFError, OSError])
    def test_a_closed_pipe_ends_the_loop_quietly(self, error: type[Exception]):
        """A parent that exits closes the pipe; that is a normal shutdown, not a failure."""
        conn = FakeConn([{"cmd": "a"}])
        conn.recv_error = error
        assert list(worker_runtime.iter_requests(conn)) == []


class TestDispatch:
    """One request, one response -- including when the handler misbehaves."""

    def test_a_successful_handler_result_is_returned_under_ok(self):
        """A successful handler result is returned under ok."""
        conn = FakeConn([{"id": 7, "cmd": "ping", "args": {"value": 3}}])
        worker_runtime.serve(conn, {"ping": lambda value: value * 2}, {})
        assert conn.sent == [{"id": 7, "ok": True, "result": 6}]

    def test_an_unknown_command_is_an_error_response_not_a_crash(self):
        """An unknown command is an error response not a crash."""
        conn = FakeConn([{"id": 1, "cmd": "nope"}])
        worker_runtime.serve(conn, {"ping": lambda: None}, {})
        assert conn.sent[0]["ok"] is False
        assert "Unknown command 'nope'" in conn.sent[0]["error"]

    def test_a_handler_exception_becomes_an_error_response_and_the_loop_survives(self):
        """The whole reason the catch is broad: a dead loop reaches the parent as a pipe
        death with no text, which is indistinguishable from a segfault in a vendor runtime.
        """

        def boom():
            """Raise, so the failure path is exercised."""
            raise ValueError("engine exploded")

        conn = FakeConn([{"id": 1, "cmd": "boom"}, {"id": 2, "cmd": "ok"}])
        worker_runtime.serve(conn, {"boom": boom, "ok": lambda: "still here"}, {})

        assert conn.sent[0] == {"id": 1, "ok": False, "error": "ValueError: engine exploded"}
        assert conn.sent[1] == {"id": 2, "ok": True, "result": "still here"}

    def test_a_non_exception_failure_is_still_contained(self):
        """`tuple([Exception])` must keep catching everything a bare `except Exception` did."""

        def boom():
            """Raise, so the failure path is exercised."""
            raise RecursionError("too deep")

        conn = FakeConn([{"id": 1, "cmd": "boom"}])
        worker_runtime.serve(conn, {"boom": boom}, {})
        assert conn.sent[0]["error"].startswith("RecursionError")

    @pytest.mark.parametrize("junk", [None, "not-a-dict", 42, {"no_cmd": True}])
    def test_junk_and_stale_cancels_are_ignored_rather_than_answered(self, junk):
        """A cancel for an already-finished stream arrives after it; replying to its id
        would resolve a request nobody is awaiting."""
        conn = FakeConn([{"control": "cancel"}, {"id": 5, "cmd": "ping"}] if junk is None else [junk, {"id": 5, "cmd": "ping"}])
        worker_runtime.serve(conn, {"ping": lambda: "ok"}, {})
        assert conn.sent == [{"id": 5, "ok": True, "result": "ok"}]


class TestStreaming:
    """Many events per request, and the cancel check that sits between them."""

    def test_events_are_forwarded_then_terminated_with_done(self):
        """Events are forwarded then terminated with done."""

        def counter(limit):
            """Yield the requested number of segment events."""
            for index in range(limit):
                yield {"event": "segment", "index": index}

        conn = FakeConn([{"id": 3, "cmd": "count", "stream": True, "args": {"limit": 2}}])
        worker_runtime.serve(conn, {}, {"count": counter})

        assert _events(conn) == ["segment", "segment", "done"]
        assert [m.get("index") for m in conn.sent[:2]] == [0, 1]

    def test_an_event_without_its_own_kind_is_labelled_data(self):
        """An event without its own kind is labelled data."""
        conn = FakeConn([{"id": 1, "cmd": "s", "stream": True}])
        worker_runtime.serve(conn, {}, {"s": lambda: iter([{"payload": 1}])})
        assert _events(conn) == ["data", "done"]

    def test_an_unknown_streaming_command_reports_an_error_event(self):
        """An unknown streaming command reports an error event."""
        conn = FakeConn([{"id": 1, "cmd": "nope", "stream": True}])
        worker_runtime.serve(conn, {}, {"count": lambda: iter(())})
        assert _events(conn) == ["error"]
        assert "Unknown streaming command 'nope'" in conn.sent[0]["error"]

    def test_a_cancel_between_events_stops_early_and_terminates_with_cancelled(self):
        """`cancelled`, not `done`. Reporting the same terminator as a completed stream made
        the parent's drain treat a preempted, partial transcription as a finished one.
        """
        started = []

        def endless():
            """Yield events until the stream is cancelled."""
            for index in range(100):
                started.append(index)
                yield {"event": "segment", "index": index}

        conn = FakeConn([{"id": 4, "cmd": "s", "stream": True}], control=[{"control": "cancel"}])
        worker_runtime.serve(conn, {}, {"s": endless})

        # The event the handler had already produced is forwarded, and only then is the
        # cancel honoured. This previously asserted ["cancelled"] alone, which pinned a
        # defect: an event pulled off the handler was dropped, and for a transcription the
        # first one is the `info` event carrying detected language and duration -- the only
        # metadata the parent would ever receive.
        assert _events(conn) == ["segment", "cancelled"], "the already-pulled event must not be discarded"
        assert len(started) == 1, "the handler must not be pumped after a cancel"

    def test_a_generator_is_closed_so_its_own_cleanup_runs(self):
        """Chunk buffers and temp slices are released in the handler's finally, at the point
        we stop pulling -- not whenever the object happens to be collected."""
        closed = []

        def handler():
            """A scripted stream handler."""
            try:
                yield {"event": "segment"}
                yield {"event": "segment"}
            finally:
                closed.append(True)

        conn = FakeConn([{"id": 1, "cmd": "s", "stream": True}], control=[{"control": "cancel"}])
        worker_runtime.serve(conn, {}, {"s": handler})
        assert closed == [True]

    def test_a_handler_that_raises_reports_an_error_event(self):
        """A handler that raises reports an error event."""

        def handler():
            """A scripted stream handler."""
            yield {"event": "segment"}
            raise RuntimeError("decode failed")

        conn = FakeConn([{"id": 9, "cmd": "s", "stream": True}])
        worker_runtime.serve(conn, {}, {"s": handler})

        assert _events(conn) == ["segment", "error"]
        assert conn.sent[-1]["error"] == "RuntimeError: decode failed"

    def test_a_handler_that_raises_before_yielding_still_reports_an_error(self):
        """`generator` is still None in the finally; closing it must not mask the error."""

        def handler():
            """A scripted stream handler."""
            raise ValueError("bad args")
            yield  # pragma reachable only after the raise, kept so this is a generator

        conn = FakeConn([{"id": 9, "cmd": "s", "stream": True}])
        worker_runtime.serve(conn, {}, {"s": handler})
        assert conn.sent[-1]["error"] == "ValueError: bad args"

    def test_a_handler_returning_something_without_close_is_tolerated(self):
        """A handler returning something without close is tolerated."""
        conn = FakeConn([{"id": 1, "cmd": "s", "stream": True}])
        worker_runtime.serve(conn, {}, {"s": lambda: [{"event": "segment"}]})
        assert _events(conn) == ["segment", "done"]


class TestCancelDetection:
    """`_cancel_requested` drains the control channel without blocking."""

    def test_no_pending_message_is_not_a_cancel(self):
        """No pending message is not a cancel."""
        assert worker_runtime._cancel_requested(FakeConn()) is False

    def test_a_non_cancel_control_message_is_drained_but_ignored(self):
        """A non cancel control message is drained but ignored."""
        conn = FakeConn(control=[{"control": "ping"}])
        assert worker_runtime._cancel_requested(conn) is False

    def test_a_cancel_anywhere_in_the_drained_batch_counts(self):
        """A cancel anywhere in the drained batch counts."""
        conn = FakeConn(control=[{"control": "ping"}, {"control": "cancel"}])
        assert worker_runtime._cancel_requested(conn) is True

    @pytest.mark.parametrize("error", [EOFError, OSError])
    def test_a_broken_pipe_reads_as_a_cancel(self, error: type[Exception]):
        """The parent is gone, so there is nobody to send the remaining events to."""
        conn = FakeConn(control=[{"control": "ping"}])
        conn.recv_error = error
        assert worker_runtime._cancel_requested(conn) is True


class TestSend:
    """An unserializable payload must not take the worker down."""

    def test_an_ordinary_message_is_sent_once(self):
        """An ordinary message is sent once."""
        conn = FakeConn()
        worker_runtime._send(conn, {"id": 1, "ok": True})
        assert conn.sent == [{"id": 1, "ok": True}]

    @pytest.mark.parametrize("error", [pickle.PicklingError("nope"), AttributeError("nope"), TypeError("nope")])
    def test_an_unpicklable_payload_degrades_to_a_serialized_error(self, error: Exception):
        """A handler returning an engine object instead of a plain payload raises here, out
        of reach of _dispatch's guard -- and the parent would see only a pipe death."""
        conn = FakeConn()
        conn.send_failures = [error]
        worker_runtime._send(conn, {"id": 4, "ok": True, "result": object()})

        assert len(conn.sent) == 1
        assert conn.sent[0]["ok"] is False
        assert conn.sent[0]["error"].startswith(f"Unserializable worker response: {type(error).__name__}")
        assert conn.sent[0]["id"] == 4

    def test_a_pipe_that_also_refuses_the_error_reply_is_given_up_on_quietly(self):
        """Both sends failing means the parent is gone; raising here would kill the loop."""
        conn = FakeConn()
        conn.send_failures = [TypeError("unpicklable"), OSError("pipe closed")]
        worker_runtime._send(conn, {"id": 4, "result": object()})
        assert conn.sent == []


class TestWorkerLogging:
    """A spawned child never runs the app's logging setup."""

    def test_a_handler_is_installed_and_tagged_when_the_root_logger_is_bare(self, monkeypatch):
        """A handler is installed and tagged when the root logger is bare."""
        root = logging.getLogger()
        monkeypatch.setattr(root, "handlers", [])
        try:
            worker_runtime.configure_worker_logging("prep-worker")
            assert root.handlers, "a worker with no handler logs nowhere, so a crash is invisible"
            assert "prep-worker" in root.handlers[0].formatter._fmt
        finally:
            root.handlers = []

    def test_an_already_configured_root_logger_is_left_alone(self, monkeypatch):
        """In-process tests and embedded use share this module; it must not hijack them."""
        existing = logging.NullHandler()
        monkeypatch.setattr(logging.getLogger(), "handlers", [existing])
        worker_runtime.configure_worker_logging("worker")
        assert logging.getLogger().handlers == [existing]


class TestIsolationEnv:
    """Which accelerators a worker is made blind to, keyed by the unit it serves."""

    @pytest.mark.parametrize("unit_type", ["GPU", "NPU"])
    def test_an_intel_unit_hides_both_gpu_vendors(self, unit_type: str):
        """The pairing config.py documents as a driver-level crash: an OpenVINO context and
        a CUDA context in one process."""
        assert worker_runtime.ISOLATION_ENV[unit_type] == {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": ""}

    def test_a_cuda_unit_pins_sycl_to_the_cpu_rather_than_hiding_intel_outright(self):
        """A cuda unit pins sycl to the cpu rather than hiding intel outright."""
        assert worker_runtime.ISOLATION_ENV["CUDA"]["ONEAPI_DEVICE_SELECTOR"] == "*:cpu"

    def test_an_amd_unit_hides_cuda_and_pins_sycl(self):
        """An amd unit hides cuda and pins sycl."""
        assert worker_runtime.ISOLATION_ENV["AMD"] == {"ONEAPI_DEVICE_SELECTOR": "*:cpu", "CUDA_VISIBLE_DEVICES": ""}

    def test_an_unknown_unit_type_gets_no_overrides(self):
        """Keyed by the unit's vendor, so a CPU unit is simply absent -- and openai-whisper
        on an Intel XPU must not be blinded by a lookup keyed on the engine instead."""
        assert worker_runtime.ISOLATION_ENV.get("CPU") is None
