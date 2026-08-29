"""One worker channel per engine type, even when units initialise concurrently.

A channel owns a spawned process holding a loaded model. Building two for the same key and
keeping one leaves the other orphaned for the life of the service, still holding whatever
device memory it loaded -- on an NPU, enough to make the surviving load fail outright.
"""

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
        # daemon=True with an asserted join, matching the isolated engine proxy test: a
        # non-daemon thread that deadlocks hangs interpreter shutdown after the suite has
        # already finished, and a bare join(timeout=...) returns quietly whether the thread
        # ended or not -- so a deadlock surfaced as a confusing count assertion below
        # instead of at the join that detected it.
        threads = [threading.Thread(target=claim, daemon=True) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
            assert not thread.is_alive(), "a channel_for caller deadlocked"

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


def test_the_channel_is_given_a_finite_call_deadline(clean_channels):
    """A call with no deadline holds the channel lock forever, and the lock is shared.

    Channels are keyed by engine type, so every unit of that type queues behind one lock.
    On the Intel NUC an INTEL-WHISPER generate() stopped returning and took the GPU unit,
    the NPU unit and nine queued tasks with it for 2h40m -- because WorkerChannel defaults
    call_timeout_sec to 0.0, which disables the deadline entirely, and this call site
    passed nothing. whisperx_worker_client had always passed one.
    """
    with mock.patch.object(isolated_engine.worker_channel, "WorkerChannel", side_effect=lambda *a, **k: mock.MagicMock()) as built:
        isolated_engine.channel_for("INTEL-WHISPER")

    timeout = built.call_args.kwargs.get("call_timeout_sec")
    assert timeout is not None, "no deadline was passed; a hung call would never be broken"
    assert timeout > 0, f"call_timeout_sec={timeout} disables the deadline, which is the defect"
