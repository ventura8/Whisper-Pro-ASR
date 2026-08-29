"""A trivial worker used to exercise the generic channel without loading an engine.

Lives in an importable module because ``spawn`` re-imports the target by reference in
the child; a locally-defined function could not be pickled.
"""

import os
import time
from types import SimpleNamespace

from modules.inference.engines import inference_worker, worker_runtime


def _echo(value):
    return {"echo": value, "pid": os.getpid()}


def _boom():
    raise ValueError("handler exploded")


def _die():
    os._exit(1)  # simulate a native crash, bypassing normal teardown


def _count(total, delay=0.0):
    for i in range(total):
        if delay:
            time.sleep(delay)
        yield {"event": "segment", "index": i}


def worker_main(conn):
    worker_runtime.serve(
        conn,
        handlers={"echo": _echo, "boom": _boom, "die": _die},
        stream_handlers={"count": _count},
    )


class _FakeEngine:
    """Stands in for a real engine so the worker's streaming path can be tested
    without loading multi-gigabyte weights."""

    def __init__(self, segment_count=3):
        self.segment_count = segment_count
        self.unloaded = False

    def transcribe(self, audio_path, **params):
        info = SimpleNamespace(language="es", language_probability=0.87, duration=12.5, all_language_probs=[("es", 0.87), ("en", 0.1)])

        def _segments():
            for i in range(self.segment_count):
                time.sleep(0.01)
                yield SimpleNamespace(start=float(i), end=float(i + 1), text=f"segment {i}", words=None)

        return _segments(), info

    def detect_language(self, audio):
        return "es", 0.91, [("es", 0.91)]

    def unload(self):
        self.unloaded = True


def _fake_load_model(engine_type, model_id, unit, env=None):
    """Register a fake engine, bypassing engine_factory but keeping the real pool."""
    for key, value in (env or {}).items():
        os.environ[key] = value
    handle = unit["id"]
    inference_worker._ENGINES[handle] = _FakeEngine(segment_count=int(model_id) if model_id.isdigit() else 3)
    return handle


def _isolation_env_seen(handle, name):
    """Report an env var as the worker process sees it, to prove isolation took effect."""
    return os.environ.get(name)


def engine_worker_main(conn):
    """Worker exercising inference_worker's real streaming handlers with a fake engine."""
    worker_runtime.serve(
        conn,
        handlers={
            "load_model": _fake_load_model,
            "unload_model": inference_worker._unload_model,
            "unload_all": inference_worker._unload_all,
            "loaded_handles": inference_worker._loaded_handles,
            "isolation_env_seen": _isolation_env_seen,
        },
        stream_handlers={"transcribe": inference_worker._transcribe},
    )
