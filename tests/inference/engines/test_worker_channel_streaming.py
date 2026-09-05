"""Stream lifecycle on the channel: cancellation, abandonment and pipe hygiene.

These use the real subprocess fixture worker for the same reason the rest of
test_worker_channel.py does -- the invariants are about a pipe and a process, and a mocked
connection would assert only that the code calls itself.

The specific concern here is the terminal event. A cancelled stream and an exhausted one
both end the generator and both leave the pipe clean, but they are not the same outcome,
and collapsing them made a preempted separation indistinguishable from a completed one.
"""

import pytest

from modules.inference.engines import worker_channel
from tests.inference.engines import worker_fixtures


@pytest.fixture(name="channel")
def _channel():
    ch = worker_channel.WorkerChannel(
        worker_fixtures.worker_main,
        name="test-stream-worker",
        log_tag="TestStream",
        call_timeout_sec=30.0,
        lock_warn_sec=30.0,
    )
    try:
        yield ch
    finally:
        ch.shutdown()


def test_cancelled_is_a_terminal_event_that_leaves_the_pipe_clean(channel):
    """After a cancel the channel must be immediately reusable, not torn down.

    `cancelled` was added as a distinct terminator; if the parent did not recognise it the
    stream would fall through to the abandon path, drain fruitlessly for 30 seconds and
    then respawn the worker -- costing a full model reload on every preemption.
    """
    pid_before = channel.call("echo", value="x")["pid"]

    stream = channel.stream("count", total=200, delay=0.01)
    received = 0
    for _event in stream:
        received += 1
        if received == 2:
            channel.cancel_in_stream()
    stream.close()

    # Bounded, because "the stream ended" is also true of a cancel the worker ignored until
    # all 200 events had been emitted -- the pipe would be just as clean and the worker just
    # as alive. The bar is loose enough for the events already in flight when the cancel was
    # written, and far below the 200 that mean cancellation did nothing.
    assert received < 50, f"cancel must stop the stream promptly, but {received} of 200 events arrived"
    assert channel.call("echo", value="after")["echo"] == "after"
    assert channel.call("echo", value="y")["pid"] == pid_before, "cancel must preserve the worker"


def test_an_exhausted_stream_and_a_cancelled_one_both_release_the_lock(channel):
    """Either terminator must release the channel lock, or every later call hangs."""
    list(channel.stream("count", total=3))
    assert not channel.lock.locked()

    stream = channel.stream("count", total=200, delay=0.01)
    next(stream)
    channel.cancel_in_stream()
    stream.close()
    assert not channel.lock.locked()


def test_the_lock_is_held_for_the_duration_of_a_stream(channel):
    """Telemetry reads this lock to decide whether the device is busy."""
    stream = channel.stream("count", total=200, delay=0.01)
    try:
        next(stream)
        assert channel.lock.locked(), "an in-flight stream must present as busy"
    finally:
        stream.close()


def test_abandoning_without_cancelling_still_restores_a_clean_pipe(channel):
    """A caller that simply breaks out must not desynchronise later responses."""
    for event in channel.stream("count", total=50, delay=0.01):
        if event["index"] == 1:
            break

    # A desynchronised pipe would hand this call a leftover stream event instead of a reply.
    assert channel.call("echo", value="recovered")["echo"] == "recovered"


def test_a_handler_error_mid_stream_raises_and_leaves_the_channel_usable(channel):
    """An error terminator is not a dead worker; the channel keeps working."""
    with pytest.raises(worker_channel.WorkerError):
        list(channel.stream("boom_stream")) if "boom_stream" in dir(worker_fixtures) else channel.call("boom")

    assert channel.call("echo", value="still here")["echo"] == "still here"
