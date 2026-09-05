"""Deterministic unit coverage for the isolated preprocessing stream worker."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from modules.inference.pipeline import preprocessing_worker


class _FakeManager:
    """Controllable preprocessor substitute used without ONNX or UVR."""

    def __init__(self, mode: str = "complete") -> None:
        self.mode = mode
        self.started = threading.Event()
        self.release = threading.Event()
        self.cancelled = threading.Event()
        self.calls: list[tuple[str, bool, str]] = []

    def preprocess_audio(self, audio_path, force=False, yield_cb=None, stage="Vocal Separation"):
        self.calls.append((audio_path, force, stage))
        if self.mode == "error":
            raise ValueError("manager failure")
        if self.mode == "blocked":
            self.started.set()
            assert self.release.wait(timeout=1), "test must release the fake manager"
            try:
                if yield_cb:
                    yield_cb()
            except preprocessing_worker._Cancelled:
                self.cancelled.set()
                raise
        return f"{audio_path}.vocals.wav"


@pytest.fixture(name="fake_manager")
def _fake_manager(monkeypatch):
    """Install one fake manager and restore the global worker registry afterwards."""
    original_managers = dict(preprocessing_worker._MANAGERS)
    manager = _FakeManager()
    preprocessing_worker._MANAGERS.clear()
    monkeypatch.setitem(preprocessing_worker._MANAGERS, "test:0", manager)
    try:
        yield manager
    finally:
        preprocessing_worker._MANAGERS.clear()
        preprocessing_worker._MANAGERS.update(original_managers)


def test_separate_yields_result_for_completed_manager(fake_manager):
    """A completed manager emits exactly one terminal result with its output path."""
    events = list(preprocessing_worker._separate("test:0", "/tmp/clip.wav", force=True, stage="Isolation"))

    assert events == [{"event": "result", "path": "/tmp/clip.wav.vocals.wav"}]
    assert fake_manager.calls == [("/tmp/clip.wav", True, "Isolation")]


def test_separate_formats_manager_error_without_result_event(fake_manager):
    """Manager failures propagate with context and never masquerade as a result."""
    fake_manager.mode = "error"
    stream = preprocessing_worker._separate("test:0", "/tmp/broken.wav")

    with pytest.raises(RuntimeError, match="ValueError: manager failure"):
        next(stream)

    with pytest.raises(StopIteration):
        next(stream)


def test_separate_emits_heartbeat_while_manager_is_running(fake_manager, monkeypatch):
    """A blocked manager emits a tick before its controlled completion."""
    fake_manager.mode = "blocked"
    monkeypatch.setattr(preprocessing_worker, "_HEARTBEAT_SEC", 0.001)
    stream = preprocessing_worker._separate("test:0", "/tmp/long.wav")

    assert next(stream) == {"event": "tick"}
    assert fake_manager.started.is_set()
    fake_manager.release.set()
    assert list(stream) == [{"event": "result", "path": "/tmp/long.wav.vocals.wav"}]


def test_closing_separation_stream_aborts_the_manager_yield_callback(fake_manager, monkeypatch):
    """Closing the stream sets abort before the manager reaches its next yield boundary."""
    fake_manager.mode = "blocked"
    abort_set = threading.Event()
    real_threading = threading

    class _AbortEvent:
        def __init__(self) -> None:
            self._event = real_threading.Event()

        def is_set(self) -> bool:
            return self._event.is_set()

        def set(self) -> None:
            self._event.set()
            abort_set.set()

    monkeypatch.setattr(
        preprocessing_worker,
        "threading",
        SimpleNamespace(Event=_AbortEvent, Thread=real_threading.Thread),
    )
    monkeypatch.setattr(preprocessing_worker, "_HEARTBEAT_SEC", 0.001)
    stream = preprocessing_worker._separate("test:0", "/tmp/cancel.wav")

    assert next(stream) == {"event": "tick"}
    assert fake_manager.started.is_set()
    closer = real_threading.Thread(target=stream.close, daemon=True)
    closer.start()
    assert abort_set.wait(timeout=1)
    fake_manager.release.set()
    closer.join(timeout=1)

    assert not closer.is_alive(), "stream closure must not strand the worker thread"
    assert fake_manager.cancelled.is_set()
