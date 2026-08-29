"""Transport tests for the generic isolated-worker channel.

These spawn real subprocesses on purpose. The whole point of the channel is behaviour
under process death and cancellation, none of which a mocked pipe would exercise.
"""

import os

import pytest

from modules.inference.engines import worker_channel
from tests.inference.engines import worker_fixtures


@pytest.fixture(name="channel")
def _channel():
    ch = worker_channel.WorkerChannel(
        worker_fixtures.worker_main,
        name="test-worker",
        log_tag="TestWorker",
        call_timeout_sec=30.0,
        lock_warn_sec=30.0,
    )
    try:
        yield ch
    finally:
        ch.shutdown()


def test_call_round_trips_through_a_real_subprocess(channel):
    result = channel.call("echo", value="hello")
    assert result["echo"] == "hello"
    assert result["pid"] != os.getpid(), "worker must run in its own process"


def test_handler_exception_surfaces_without_killing_the_worker(channel):
    with pytest.raises(worker_channel.WorkerError, match="handler exploded"):
        channel.call("boom")

    # The request loop must have survived, so the channel is still usable.
    assert channel.call("echo", value="still alive")["echo"] == "still alive"


def test_unknown_command_is_reported_as_an_error(channel):
    with pytest.raises(worker_channel.WorkerError, match="Unknown command"):
        channel.call("nope")


def test_stream_yields_events_then_completes(channel):
    events = list(channel.stream("count", total=4))
    assert [e["index"] for e in events] == [0, 1, 2, 3]
    assert all(e["event"] == "segment" for e in events)


def test_stream_releases_the_lock_so_later_calls_work(channel):
    list(channel.stream("count", total=2))
    assert channel.call("echo", value="after")["echo"] == "after"


def test_abandoning_a_stream_early_does_not_strand_the_channel(channel):
    """Breaking out mid-stream must not leave the lock held or the pipe desynchronised."""
    for event in channel.stream("count", total=50, delay=0.01):
        if event["index"] == 1:
            break

    # A stranded lock would hang here; a desynchronised pipe would return a stale event.
    assert channel.call("echo", value="recovered")["echo"] == "recovered"


def test_abandoning_a_stream_keeps_the_worker_and_its_models(channel):
    """Preemption abandons streams routinely, so it must not cost a model reload."""
    pid_before = channel.call("echo", value="x")["pid"]

    for event in channel.stream("count", total=50, delay=0.01):
        if event["index"] == 1:
            break

    assert channel.call("echo", value="y")["pid"] == pid_before, "worker should have been preserved, not respawned"


def test_cancel_stops_the_stream_early(channel):
    received = []
    stream = channel.stream("count", total=200, delay=0.01)
    for event in stream:
        received.append(event["index"])
        if len(received) == 2:
            channel.cancel_in_stream()
    stream.close()

    # Bounded near the two events consumed before cancelling, not merely under the total.
    # "< 200" passed even if cancel did nothing and the stream simply ended on its own, so
    # it asserted the loop terminates rather than that cancellation works. The slack absorbs
    # events already in flight in the pipe when the cancel is written.
    assert len(received) <= 20, f"cancel should stop the stream promptly, got {len(received)} events"


def test_worker_death_raises_and_then_respawns(channel):
    first_pid = channel.call("echo", value="x")["pid"]
    generation_before = channel.generation()

    with pytest.raises(worker_channel.WorkerError):
        channel.call("die")

    assert channel.generation() > generation_before, "generation must invalidate cached handles"

    second_pid = channel.call("echo", value="y")["pid"]
    assert second_pid != first_pid, "a fresh worker should have been spawned"


def test_shutdown_terminates_the_process(channel):
    channel.call("echo", value="x")
    assert channel.is_running()

    channel.shutdown()
    assert not channel.is_running()

    # Shutdown is not terminal: the next call spawns a new worker.
    assert channel.call("echo", value="again")["echo"] == "again"


def test_shutdown_is_idempotent(channel):
    channel.call("echo", value="x")
    channel.shutdown()
    channel.shutdown()
    assert not channel.is_running()
