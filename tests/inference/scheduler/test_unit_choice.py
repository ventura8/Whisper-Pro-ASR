"""Which idle unit a task takes, on a host whose pool holds units the engine cannot drive.

The pool rotates: a released unit goes to the tail. On a CUDA+Intel host every auto-detect
request runs language detection before transcription, so detection took the CUDA unit and
the transcription took the Intel one -- where FASTER-WHISPER loads on the CPU. Measured on
an RTX 3080 laptop: twenty times slower than the card beside it, banner still saying CUDA.
"""

import queue
import threading
from unittest import mock

import pytest

from modules.inference.scheduler import unit_choice

CUDA = {"id": "cuda:0", "type": "CUDA", "name": "NVIDIA GPU 0"}
INTEL = {"id": "GPU", "type": "GPU", "name": "Intel GPU"}
NPU = {"id": "NPU", "type": "NPU", "name": "Intel NPU"}


def _pool(*units):
    pool = queue.Queue()
    for unit in units:
        pool.put(unit)
    return pool


@pytest.fixture(name="faster_whisper_everywhere")
def _faster_whisper_everywhere():
    """The shipped default: one engine for every unit, and it only drives CUDA/CPU."""
    with mock.patch.object(unit_choice.config, "engine_for_unit", return_value="FASTER-WHISPER"):
        yield


class TestPreferringADrivableUnit:
    """The rotation case, and the case the v1.3.0 design was written for."""

    def test_an_idle_cuda_unit_beats_the_intel_unit_at_the_head(self, faster_whisper_everywhere):
        """The pool after language detection returned the CUDA unit to the tail."""
        pool = _pool(INTEL, CUDA)
        assert unit_choice.take_idle_unit(pool) is CUDA
        assert list(pool.queue) == [INTEL], "the unit not taken stays, in place"

    def test_the_head_is_taken_when_it_is_already_drivable(self, faster_whisper_everywhere):
        """Nothing is reordered for its own sake."""
        pool = _pool(CUDA, INTEL)
        assert unit_choice.take_idle_unit(pool) is CUDA
        assert list(pool.queue) == [INTEL]

    def test_a_non_drivable_unit_is_still_taken_when_nothing_better_is_idle(self, faster_whisper_everywhere):
        """The throughput trade stands: with the CUDA unit busy, the task runs on the
        Intel unit and decodes on the CPU rather than waiting."""
        pool = _pool(INTEL)
        assert unit_choice.take_idle_unit(pool) is INTEL
        assert not list(pool.queue)

    def test_the_rest_of_the_pool_keeps_its_order(self, faster_whisper_everywhere):
        """Only the chosen unit leaves; the others keep their rotation order."""
        pool = _pool(INTEL, NPU, CUDA)
        assert unit_choice.take_idle_unit(pool) is CUDA
        assert list(pool.queue) == [INTEL, NPU]

    def test_an_empty_pool_raises_empty_like_get_did(self, faster_whisper_everywhere):
        """Callers already handle queue.Empty; the contract must not change under them."""
        with pytest.raises(queue.Empty):
            unit_choice.take_idle_unit(_pool())


class TestHybridEngines:
    """With an engine per unit, every unit is drivable by its own engine and the head wins."""

    def test_the_head_is_taken_when_each_unit_has_its_own_engine(self):
        """HYBRID_ENGINES: the Intel unit runs INTEL-WHISPER, so it is drivable and first."""

        def per_unit(unit):
            return "INTEL-WHISPER" if unit["type"] in ("GPU", "NPU") else "FASTER-WHISPER"

        with mock.patch.object(unit_choice.config, "engine_for_unit", side_effect=per_unit):
            pool = _pool(INTEL, CUDA)
            assert unit_choice.take_idle_unit(pool) is INTEL


class TestQueueBookkeeping:
    """A taker that bypasses Queue.get must leave the queue as get would have."""

    def test_a_blocked_putter_is_woken(self, faster_whisper_everywhere):
        """`get` notifies `not_full`; a bounded pool with a waiting producer depends on it.

        The producer is known to be blocked in ``put`` -- its entry into ``not_full.wait``
        is observed -- before the unit is taken, and it must come back promptly: without the
        notification it would sit in ``wait`` until its own timeout, which is what the short
        join catches. Taking the unit before the producer blocked would have let it succeed
        on a free slot with no notification at all.
        """
        pool = queue.Queue(maxsize=1)
        pool.put(INTEL)
        waiting = threading.Event()
        original_wait = pool.not_full.wait

        def observed_wait(timeout=None):
            waiting.set()
            return original_wait(timeout)

        pool.not_full.wait = observed_wait
        woken = []

        def producer():
            pool.put(CUDA, timeout=10)
            woken.append(True)

        thread = threading.Thread(target=producer)
        thread.start()
        assert waiting.wait(timeout=2), "the producer never blocked on the full pool"
        assert unit_choice.take_idle_unit(pool) is INTEL
        thread.join(timeout=1.5)
        assert not thread.is_alive(), "the producer was not released by the notification"
        assert woken == [True]
        assert list(pool.queue) == [CUDA]
