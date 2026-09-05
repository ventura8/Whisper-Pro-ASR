"""Fake preprocessing worker: exercises the real separation stream without ONNX/UVR."""

import os
import time

from modules.inference.engines import worker_runtime
from modules.inference.pipeline import preprocessing_worker as prep_worker

#: How the next _FakeManager is built. Configurable per test through the `configure`
#: command, because separation speed is not a detail here -- it decides whether the parent
#: sees any preemption opportunity at all. See _configure.
_CONFIG = {"chunks": 3, "delay": 0.05}


def _configure(chunks: int, delay: float) -> dict:
    """Set the shape of the next manager, so a test can make separation slow enough to preempt.

    The worker emits a `tick` only when its separation thread is still alive after a
    _HEARTBEAT_SEC join, and `tick` is the only event that drives the parent's yield_cb.
    Work shorter than one heartbeat therefore produces a stream of exactly ["result"] and
    no preemption opportunity whatsoever -- which is precisely how the preemption test came
    to assert against a callback that was never invoked.
    """
    _CONFIG["chunks"] = int(chunks)
    _CONFIG["delay"] = float(delay)
    return dict(_CONFIG)


class _FakeManager:
    """Stands in for PreprocessingManager, honouring yield_cb like the real one."""

    def __init__(self, chunks=3, delay=0.05):
        self.chunks = chunks
        self.delay = delay
        self.separator = object()
        self.offloaded = False
        self.unloaded = False
        #: How far separation actually got. A cancelled run must stop short of self.chunks;
        #: without this the parent can only see that its own wait ended.
        self.chunks_done = 0

    def preprocess_audio(self, audio_path, force=False, yield_cb=None, stage="Vocal Separation"):
        for _ in range(self.chunks):
            time.sleep(self.delay)
            self.chunks_done += 1
            if yield_cb:
                yield_cb()  # chunk boundary -- this is where preemption lands
        return audio_path + ".vocals.wav"

    def offload(self):
        self.offloaded = True

    def unload_model(self):
        self.unloaded = True
        self.separator = None


def _fake_load(unit, env=None):
    for key, value in (env or {}).items():
        os.environ[key] = value
    handle = unit["id"]
    # Shape taken from _CONFIG, not from `env`. The previous
    # `(env or {}).get("chunks_total", 3)` could never fire: `env` is the parent's
    # _worker_env(), which carries device-visibility variables and has never held a
    # "chunks_total" key, so every manager was silently built with the 3-chunk default.
    prep_worker._MANAGERS[handle] = _FakeManager(chunks=_CONFIG["chunks"], delay=_CONFIG["delay"])
    return handle


def _isolation_env_seen(name):
    return os.environ.get(name)


def _progress(handle):
    """Report what the worker-side manager actually did, for assertions in the parent."""
    manager = prep_worker._MANAGERS[handle]
    return {
        "chunks_done": manager.chunks_done,
        "chunks_total": manager.chunks,
        "offloaded": manager.offloaded,
        "unloaded": manager.unloaded,
    }


def worker_main(conn):
    worker_runtime.serve(
        conn,
        handlers={
            "load": _fake_load,
            "unload": prep_worker._unload,
            "offload": prep_worker._offload,
            "state": prep_worker._state,
            "isolation_env_seen": _isolation_env_seen,
            "progress": _progress,
            "configure": _configure,
        },
        stream_handlers={"separate": prep_worker._separate},
    )
