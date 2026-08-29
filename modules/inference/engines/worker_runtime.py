"""Child-side request loop for isolated engine workers.

The parent half lives in ``worker_channel``. This module is imported *inside* the
worker process and deliberately pulls in nothing heavier than the standard library at
import time, so a spawned child can set up its runtime (library paths, device
visibility) before any engine import happens.

Two request shapes are supported:

``call``
    one request, one response -- ``{"id", "ok", "result"|"error"}``.

``stream``
    one request, many events. The handler is a generator; each value it yields is
    forwarded as an event, and the runtime checks for a parent ``cancel`` between
    yields. That check is what preserves cooperative preemption across the process
    boundary: the parent's per-segment ``preemption_check()`` sends ``cancel``, and the
    handler is closed at its next yield rather than decoding to the end.
"""

import logging
import os
import pickle
import sys
from collections.abc import Callable, Iterator
from multiprocessing.connection import Connection
from typing import Any

logger = logging.getLogger(__name__)


class CancelledByParent(Exception):
    """Raised inside a stream handler when the parent asks it to stop."""


#: Device-visibility overrides applied inside a worker *before* it imports any runtime.
#: Blanking the other vendor's variable is what guarantees that, for example, an Intel
#: worker never creates a CUDA context -- the pairing modules/core/config.py documents as
#: a driver-level crash when it happens in one process.
#:
#: Keyed by the *unit's* vendor rather than by the engine, because which accelerators a
#: worker must be blind to follows the hardware it serves. Keying by engine was wrong for
#: openai-whisper: it is torch-based and runs on any vendor, so forcing SYCL to CPU would
#: have silently disabled Intel XPU for exactly the engine the intel-xpu image exists for.
ISOLATION_ENV: dict[str, dict[str, str]] = {
    "GPU": {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": ""},
    "NPU": {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": ""},
    "CUDA": {"ONEAPI_DEVICE_SELECTOR": "*:cpu", "HIP_VISIBLE_DEVICES": ""},
    "AMD": {"ONEAPI_DEVICE_SELECTOR": "*:cpu", "CUDA_VISIBLE_DEVICES": ""},
}


def configure_worker_logging(tag: str) -> None:
    """Send this worker's records to stderr, prefixed so its output is attributable.

    A spawned child never runs the application's logging setup, so without this its
    records go nowhere and a crash inside the vendor runtime is invisible from the parent.
    ``tag`` distinguishes the worker families in a shared log ("worker", "prep-worker").
    """
    if logging.getLogger().handlers:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(f"%(asctime)s  [{tag}:{os.getpid()}] %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def iter_requests(conn: Connection) -> Iterator[Any]:
    """Yield each request from the parent until the pipe closes or it sends the stop token.

    A closed pipe and an explicit ``None`` both mean "stop"; neither is an error worth
    reporting, because both are how a parent ends a worker's life normally.
    """
    while True:
        try:
            request = conn.recv()
        except (EOFError, OSError):
            return
        if request is None:
            return
        yield request


def serve(
    conn: Connection,
    handlers: dict[str, Callable[..., Any]],
    stream_handlers: dict[str, Callable[..., Iterator[Any]]],
) -> None:
    """Blocking request loop. Runs entirely inside the child process."""
    for request in iter_requests(conn):
        if not isinstance(request, dict) or "cmd" not in request:
            # A stale cancel for an already-finished stream, or junk. Ignore rather
            # than treating it as a request and replying to an id nobody awaits.
            continue

        if request.get("stream"):
            _run_stream(conn, stream_handlers, request)
        else:
            _send(conn, _dispatch(handlers, request))


def _dispatch(handlers: dict[str, Callable[..., Any]], request: dict[str, Any]) -> dict[str, Any]:
    """Return every handler failure to the parent as a protocol error.

    Catches Exception broadly on purpose: an engine raising something unanticipated
    must not kill the worker's request loop, because the parent would then see an
    opaque pipe death instead of the actual error text.
    """
    request_id = request.get("id")
    cmd = request.get("cmd")
    handler = handlers.get(cmd)
    if handler is None:
        return {"id": request_id, "ok": False, "error": f"Unknown command '{cmd}'"}
    try:
        return {"id": request_id, "ok": True, "result": handler(**request.get("args", {}))}
    except Exception as exc:  # pylint: disable=broad-exception-caught  # noqa: BLE001 - see docstring
        logger.exception("[Worker] Command '%s' failed", cmd)
        return {"id": request_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _run_stream(
    conn: Connection,
    stream_handlers: dict[str, Callable[..., Iterator[Any]]],
    request: dict[str, Any],
) -> None:
    request_id = request.get("id")
    cmd = request.get("cmd")
    handler = stream_handlers.get(cmd)
    if handler is None:
        _send(conn, {"id": request_id, "event": "error", "error": f"Unknown streaming command '{cmd}'"})
        return

    generator = None
    try:
        generator = handler(**request.get("args", {}))
        cancelled = _pump_events(conn, generator, request_id=request_id, cmd=cmd)
        # "cancelled", not "done". A cancelled stream stopped part-way through the work, and
        # reporting the same terminal event as a completed one made the parent's drain treat
        # a partial result as a successful run -- so preemption looked indistinguishable
        # from completion on the wire. "done" now means exhausted, and only that.
        _send(conn, {"id": request_id, "event": _TERMINATOR[cancelled]})
    except Exception as exc:  # pylint: disable=broad-exception-caught  # noqa: BLE001 - same rationale as _dispatch
        logger.exception("[Worker] Streaming command '%s' failed", cmd)
        _send(conn, {"id": request_id, "event": "error", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        _close_generator(generator)


#: Which terminal event ends a stream, keyed by whether the parent cancelled it.
_TERMINATOR = {False: "done", True: "cancelled"}


def _close_generator(generator: Any) -> None:
    """Close a handler's generator so it runs its own cleanup.

    Releasing chunk buffers and temp files has to happen at the point we stop pulling from
    it, not whenever the object is eventually collected. Tolerates a handler that returned
    something without ``close`` -- and a handler that never got as far as returning at all.
    """
    if generator is not None and hasattr(generator, "close"):
        generator.close()


def _pump_events(conn: Connection, generator: Iterator[Any], *, request_id: Any, cmd: Any) -> bool:
    """Forward each event to the parent, stopping early if it asks. Returns whether it did.

    The cancel check sits between events rather than around the whole stream: that is the
    point at which a handler is safely interruptible, and it is what preserves cooperative
    preemption across the process boundary.
    """
    for event in generator:
        if _cancel_requested(conn):
            logger.info("[Worker] '%s' cancelled by parent", cmd)
            return True
        _send(conn, {"id": request_id, "event": event.get("event", "data"), **event})
    return False


def _cancel_requested(conn: Connection) -> bool:
    """Drain any pending parent control messages, reporting whether cancel arrived.

    Non-blocking: polls with a zero timeout so a stream that is never cancelled pays
    only the cost of one syscall per event.
    """
    cancelled = False
    while conn.poll(0):
        try:
            message = conn.recv()
        except (EOFError, OSError):
            return True
        if isinstance(message, dict) and message.get("control") == "cancel":
            cancelled = True
    return cancelled


def _send(conn: Connection, message: dict[str, Any]) -> None:
    """Send a message, degrading to a serialized error rather than killing the loop.

    A handler that returns something unpicklable (an engine object instead of a plain
    payload) would otherwise raise here, out of reach of _dispatch's own guard, and
    take the worker down with an opaque pipe death.
    """
    try:
        conn.send(message)
    except (pickle.PicklingError, AttributeError, TypeError) as exc:
        try:
            conn.send(
                {
                    "id": message.get("id"),
                    "ok": False,
                    "event": "error",
                    "error": f"Unserializable worker response: {type(exc).__name__}: {exc}",
                }
            )
        except OSError:
            pass
