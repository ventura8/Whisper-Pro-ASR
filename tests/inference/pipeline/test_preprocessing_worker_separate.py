"""The worker-side separation stream: heartbeats, completion, failure and cancellation.

`_separate` is what makes cooperative preemption survive the process boundary. It runs the
real separation on a thread and emits `tick` while that thread is alive, because a plain
request/response call would have given the parent no point at which to cancel. The
behaviours below are the contract the parent's stream loop is written against, and none of
them are exercised by the in-process preprocessing tests.

Driven by explicit events rather than by sleeping: a timing-based test of a heartbeat loop
either flakes on a loaded machine or passes without proving anything.
"""

from __future__ import annotations

import threading

import pytest

from modules.inference.pipeline import preprocessing_worker


class _FakeManager:
    """A manager whose separation the test controls, step by step."""

    def __init__(self, *, release: threading.Event | None = None, error: Exception | None = None):
        self._release = release
        self._error = error
        self.started = threading.Event()
        self.yield_calls = 0

    def preprocess_audio(self, audio_path, force=False, yield_cb=None, stage="Vocal Separation"):
        self.started.set()
        if self._error is not None:
            raise self._error
        if self._release is not None:
            # Wait in slices, calling back between them, so a cancel can land at a
            # boundary exactly as a real chunked separation would let it.
            for _ in range(200):
                if self._release.wait(timeout=0.01):
                    break
                if yield_cb:
                    self.yield_calls += 1
                    yield_cb()
        return audio_path + ".vocals.wav"


@pytest.fixture(name="install_manager")
def _install_manager():
    """Register a fake manager under a handle, and remove it afterwards."""
    handles: list[str] = []

    def install(handle: str, manager) -> str:
        preprocessing_worker._MANAGERS[handle] = manager
        handles.append(handle)
        return handle

    yield install
    for handle in handles:
        preprocessing_worker._MANAGERS.pop(handle, None)


def test_a_completed_separation_ends_with_the_result_path(install_manager):
    """The terminal event carries the output path the parent returns to its caller."""
    install_manager("u0", _FakeManager())

    events = list(preprocessing_worker._separate("u0", "/tmp/clip.wav"))

    assert events[-1] == {"event": "result", "path": "/tmp/clip.wav.vocals.wav"}


def test_ticks_are_emitted_while_the_separation_thread_is_alive(install_manager):
    """`tick` is the only thing that gives the parent a chance to cancel.

    Without at least one, a separation shorter than a heartbeat produces a stream of just
    ["result"] and preemption can never fire -- which is precisely how the isolated
    preemption test came to assert against a callback that was never invoked.
    """
    release = threading.Event()
    manager = _FakeManager(release=release)
    install_manager("u1", manager)

    stream = preprocessing_worker._separate("u1", "/tmp/clip.wav")
    try:
        # The first event IS the proof the worker started -- it can only be emitted from the
        # heartbeat loop, which runs after the separation thread is up. The previous
        # `assert manager.started.wait(...) or True` was a no-op (`or True` makes it
        # unconditionally true) that also paid a five-second wait for nothing.
        first = next(stream)
        assert first == {"event": "tick"}
        assert manager.started.is_set(), "a tick was emitted before the separation thread started"
    finally:
        release.set()
        stream.close()


def test_a_manager_failure_surfaces_as_a_runtime_error(install_manager):
    """A worker-side exception must reach the parent as an error, not a silent stop."""
    install_manager("u2", _FakeManager(error=ValueError("separator exploded")))

    with pytest.raises(RuntimeError, match="separator exploded"):
        list(preprocessing_worker._separate("u2", "/tmp/clip.wav"))


def test_closing_the_generator_cancels_and_joins_the_thread(install_manager):
    """Closing is how the runtime cancels; it must stop the work, not orphan it."""
    release = threading.Event()
    manager = _FakeManager(release=release)
    install_manager("u3", manager)

    stream = preprocessing_worker._separate("u3", "/tmp/clip.wav")
    next(stream)
    assert manager.started.is_set()

    # close() runs the generator's finally: abort is set and the thread joined, so the
    # separation stops rather than continuing to hold the device after the parent left.
    stream.close()

    assert manager.yield_calls > 0, "the manager never reached a cancellable boundary"
    assert not any(t.name == "uvr-separate" and t.is_alive() for t in threading.enumerate())


def test_an_unknown_handle_is_reported_rather_than_crashing_the_worker():
    """The request loop turns this into an error response; it must not be a hard crash."""
    with pytest.raises(KeyError):
        list(preprocessing_worker._separate("no-such-handle", "/tmp/clip.wav"))
