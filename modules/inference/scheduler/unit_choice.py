"""Which idle unit a task takes, when more than one is free.

The pool is a FIFO: a unit comes off the head and a released unit goes back on the tail.
That rotation is blind to what a unit can run. On a host with a CUDA card and an Intel
iGPU the pool holds both -- deliberately, so vocal isolation can use the iGPU while
CTranslate2 decodes on the card -- and every auto-detect request runs language detection
before transcription. Detection takes the CUDA unit and returns it to the tail; the
transcription then takes the head, which is now the Intel unit, on which FASTER-WHISPER
has no backend and loads a second model on the CPU.

Measured on an RTX 3080 laptop (2026-09-11): every transcription ran at CPU speed, twenty
times slower than the same clip on the card it was sitting next to, while the startup
banner reported CUDA throughout -- the banner describes the first unit, not the one a task
lands on. Per-clip language detection went from 220 ms to 6 s for the same reason.

The v1.3.0 decision to keep the non-drivable unit in the pool stands: when the drivable
unit is busy, a task landing on the other one still finishes, and two tasks run where one
did. What changes is only the case where a drivable unit is *idle* -- it is preferred, so
the CPU fallback happens under load and never by rotation.
"""

import queue

from modules.core import config, engine_registry


def take_idle_unit(pool: queue.Queue) -> dict:
    """Take an idle unit from ``pool``, preferring one the ASR engine can drive.

    Raises :class:`queue.Empty` when nothing is idle, exactly as ``pool.get(block=False)``
    did, so callers keep their existing handling. Done under the queue's own mutex with the
    same bookkeeping ``Queue.get`` performs, so a concurrent taker sees a consistent pool.
    """
    with pool.mutex:
        units = list(pool.queue)
        if not units:
            raise queue.Empty
        chosen = next((unit for unit in units if _drivable(unit)), units[0])
        pool.queue.remove(chosen)
        pool.not_full.notify()
    return chosen


def _drivable(unit: dict) -> bool:
    """Whether the engine that would serve on ``unit`` can actually execute there."""
    engine = config.engine_for_unit(unit)
    return engine_registry.engine_supports_unit(engine, unit.get("type", ""))
