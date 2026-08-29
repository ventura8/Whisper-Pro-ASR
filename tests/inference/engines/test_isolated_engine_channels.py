"""One worker channel per engine type, even when units initialise concurrently.

A channel owns a spawned process holding a loaded model. Building two for the same key and
keeping one leaves the other orphaned for the life of the service, still holding whatever
device memory it loaded -- on an NPU, enough to make the surviving load fail outright.
"""

# pylint: disable=protected-access
# The unit under test is the module's internals; reaching them by name is the point.

import threading
from unittest import mock

import pytest

from modules.inference.engines import isolated_engine


@pytest.fixture(name="clean_channels")
def _clean_channels():
    saved = dict(isolated_engine._CHANNELS)
    isolated_engine._CHANNELS.clear()
    yield
    isolated_engine._CHANNELS.clear()
    isolated_engine._CHANNELS.update(saved)


def test_concurrent_callers_share_one_channel(clean_channels):
    """Every caller gets the same object, and exactly one channel is ever constructed.

    The construction is deliberately slowed so the threads overlap inside the cache-miss
    window. Without the guard this asserts on, that window is where both callers see an
    empty cache, both build, and the second overwrites the first.
    """
    built = []
    start = threading.Barrier(8)

    def slow_channel(*_args, **_kwargs):
        # Long enough that every thread is inside the miss path before the first finishes.
        threading.Event().wait(0.05)
        channel = mock.MagicMock(name=f"channel-{len(built)}")
        built.append(channel)
        return channel

    results = []
    lock = threading.Lock()

    def claim():
        start.wait(timeout=5)
        channel = isolated_engine.channel_for("FASTER-WHISPER")
        with lock:
            results.append(channel)

    with mock.patch.object(isolated_engine.worker_channel, "WorkerChannel", side_effect=slow_channel):
        threads = [threading.Thread(target=claim) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

    assert len(built) == 1, f"{len(built)} channels were constructed; the losers own orphaned worker processes"
    assert len(results) == 8
    assert all(channel is results[0] for channel in results)
    assert isolated_engine._CHANNELS["FASTER-WHISPER"] is results[0]


def test_distinct_engine_types_get_distinct_channels(clean_channels):
    """The key is the engine type: mutually exclusive runtimes need separate processes."""
    with mock.patch.object(isolated_engine.worker_channel, "WorkerChannel", side_effect=lambda *a, **k: mock.MagicMock()):
        faster = isolated_engine.channel_for("FASTER-WHISPER")
        intel = isolated_engine.channel_for("INTEL-WHISPER")

    assert faster is not intel
    assert isolated_engine.channel_for("FASTER-WHISPER") is faster
