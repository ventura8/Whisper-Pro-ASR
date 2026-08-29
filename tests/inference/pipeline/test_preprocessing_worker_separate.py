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

import os
import threading
from unittest import mock

import pytest

from modules.inference.pipeline import preprocessing_worker


class _FakeManager:
    """A manager whose separation the test controls, step by step."""

    def __init__(self, *, release: threading.Event | None = None, error: Exception | None = None):
        """Record the construction arguments."""
        self._release = release
        self._error = error
        self.started = threading.Event()
        self.yield_calls = 0
        # The lifecycle surface the pool tests below drive. separator stays None until a
        # model is actually loaded, which is exactly what _state reports on.
        self.separator = None
        self.unloaded = False
        self.offloaded = False

    def unload_model(self):
        """Record that the model was released."""
        self.unloaded = True

    def offload(self):
        """Record that device memory was released."""
        self.offloaded = True

    def preprocess_audio(self, audio_path, force=False, yield_cb=None, stage="Vocal Separation"):
        """Run the scripted separation, honouring the cancel callback."""
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
        """Register a fake manager under a handle."""
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

    # Snapshot taken before the stream exists, so the difference is this stream's own
    # thread and nothing else. Filtering the live enumerate() by name alone still sees any
    # "uvr-separate" thread another test in the same worker happens to be running, which
    # makes the assertion about the whole session rather than about this stream.
    before = set(threading.enumerate())

    stream = preprocessing_worker._separate("u3", "/tmp/clip.wav")
    next(stream)
    assert manager.started.is_set()

    ours = [t for t in threading.enumerate() if t not in before and t.name == "uvr-separate"]
    assert ours, "the separation thread was never started"

    # close() runs the generator's finally: abort is set and the thread joined, so the
    # separation stops rather than continuing to hold the device after the parent left.
    stream.close()

    assert manager.yield_calls > 0, "the manager never reached a cancellable boundary"
    for thread in ours:
        assert not thread.is_alive(), "close() must join the separation thread, not orphan it"


def test_an_unknown_handle_is_reported_rather_than_crashing_the_worker():
    """The request loop turns this into an error response; it must not be a hard crash."""
    with pytest.raises(KeyError):
        list(preprocessing_worker._separate("no-such-handle", "/tmp/clip.wav"))


class TestTheManagerPool:
    """The worker-side pool, mirroring the one the parent used to hold in-process.

    Untested until now because these run in the spawned child: the existing cases above
    drive `_separate` through an already-installed fake, so nothing exercised how a manager
    gets created, found, or released -- including the env-before-import ordering that is the
    entire reason UVR is isolated per vendor.
    """

    @pytest.fixture(autouse=True)
    def _clean_pool(self):
        """Clear the module-global pool around each test."""
        preprocessing_worker._MANAGERS.clear()
        yield
        preprocessing_worker._MANAGERS.clear()

    def test_loading_creates_the_manager_and_keys_it_by_unit_id(self, monkeypatch):
        """Loading creates the manager and keys it by unit id."""
        built = {}

        def manager_factory(unit):
            """Build a fake manager, recording the unit."""
            built["unit"] = unit
            return _FakeManager()

        monkeypatch.setattr(
            preprocessing_worker.importlib,
            "import_module",
            lambda _n: type("M", (), {"PreprocessingManager": staticmethod(manager_factory)}),
        )
        handle = preprocessing_worker._load({"id": "GPU.0", "name": "Intel GPU"})

        assert handle == "GPU.0"
        assert built["unit"]["id"] == "GPU.0"

    def test_device_visibility_is_applied_before_the_runtime_is_imported(self, monkeypatch):
        """The ordering the isolation depends on: an Intel worker must have
        CUDA_VISIBLE_DEVICES="" set before ONNX Runtime can enumerate anything."""
        seen = {}
        # delenv on the real mapping, not setattr(os, "environ", {}). Replacing the whole
        # object leaves every other reader -- subprocess, tempfile, any library imported
        # during the test -- looking at an empty environment for the duration, which is a far
        # larger blast radius than the one variable this test is about.
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

        def import_module(_name):
            """Return the fake module the test scripted."""
            seen["env"] = preprocessing_worker.os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
            return type("M", (), {"PreprocessingManager": staticmethod(lambda unit: _FakeManager())})

        monkeypatch.setattr(preprocessing_worker.importlib, "import_module", import_module)
        # _load applies the isolation env to the real os.environ -- that is the behaviour
        # under test. delenv above records nothing when the variable is already unset, so
        # without this the "" leaks into every later test in the session: the provider
        # tests then read it as "this process is CUDA-blinded" and stop blocking OpenVINO.
        # They failed only in a serial full-suite run, and passed under -n auto by luck of
        # which worker got them.
        with mock.patch.dict(os.environ, {}, clear=False):
            preprocessing_worker._load({"id": "NPU"}, env={"CUDA_VISIBLE_DEVICES": ""})

        assert seen["env"] == "", "the runtime was imported before the isolation env landed"

    def test_a_second_load_reuses_the_resident_manager(self, monkeypatch):
        """A second load reuses the resident manager."""
        calls = []
        monkeypatch.setattr(
            preprocessing_worker.importlib,
            "import_module",
            lambda _n: type("M", (), {"PreprocessingManager": staticmethod(lambda unit: calls.append(1) or _FakeManager())}),
        )
        preprocessing_worker._load({"id": "GPU.0"})
        preprocessing_worker._load({"id": "GPU.0"})
        assert len(calls) == 1, "reloading UVR onto a device that already has it is the leak this guards"

    def test_an_unknown_handle_is_a_named_keyerror(self):
        """An unknown handle is a named keyerror."""
        with pytest.raises(KeyError, match="No preprocessor loaded for handle 'ghost'"):
            preprocessing_worker._get("ghost")

    def test_unloading_releases_the_model_and_drops_the_entry(self):
        """Unloading releases the model and drops the entry."""
        manager = _FakeManager()
        preprocessing_worker._MANAGERS["u"] = manager

        assert preprocessing_worker._unload("u") is True
        assert manager.unloaded is True
        assert "u" not in preprocessing_worker._MANAGERS

    def test_unloading_an_absent_handle_reports_false(self):
        """Unloading an absent handle reports false."""
        assert preprocessing_worker._unload("ghost") is False

    def test_offloading_keeps_the_manager_but_releases_the_device(self):
        """Offload is the idle path: the process stays up so the next task does not respawn."""
        manager = _FakeManager()
        preprocessing_worker._MANAGERS["u"] = manager

        assert preprocessing_worker._offload("u") is True
        assert manager.offloaded is True
        assert "u" in preprocessing_worker._MANAGERS

    def test_offloading_an_absent_handle_reports_false(self):
        """Offloading an absent handle reports false."""
        assert preprocessing_worker._offload("ghost") is False


class TestStateReporting:
    """What the parent's proxy answers from, without another round trip."""

    @pytest.fixture(autouse=True)
    def _clean_pool(self):
        """Clear the module-global pool around each test."""
        preprocessing_worker._MANAGERS.clear()
        yield
        preprocessing_worker._MANAGERS.clear()

    def test_an_absent_handle_reports_unloaded_rather_than_raising(self):
        """telemetry polls this; a KeyError here would surface as a dashboard 500."""
        assert preprocessing_worker._state("ghost") == {"loaded": False, "providers": []}

    def test_a_manager_with_no_separator_yet_reports_unloaded(self):
        """A manager with no separator yet reports unloaded."""
        preprocessing_worker._MANAGERS["u"] = _FakeManager()
        assert preprocessing_worker._state("u") == {"loaded": False, "providers": []}

    def test_a_loaded_separator_reports_its_execution_providers(self):
        """This is the evidence the dashboard shows for which device UVR actually ran on."""
        manager = _FakeManager()
        manager.separator = type("S", (), {"onnx_execution_provider": ["OpenVINOExecutionProvider"]})()
        preprocessing_worker._MANAGERS["u"] = manager

        assert preprocessing_worker._state("u") == {"loaded": True, "providers": ["OpenVINOExecutionProvider"]}

    def test_a_separator_reporting_no_providers_still_reports_loaded(self):
        """A separator reporting no providers still reports loaded."""
        manager = _FakeManager()
        manager.separator = type("S", (), {"onnx_execution_provider": None})()
        preprocessing_worker._MANAGERS["u"] = manager
        assert preprocessing_worker._state("u") == {"loaded": True, "providers": []}
