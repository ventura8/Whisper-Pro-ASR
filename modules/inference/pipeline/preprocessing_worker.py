"""Child-side host for out-of-process vocal separation.

**Why this module sits beside the preprocessing package rather than inside it.**
``spawn`` imports the module holding ``worker_main`` in the child, and importing a
submodule imports its package ``__init__`` first. ``preprocessing/__init__.py`` imports
``modules.core.config``, whose hardware detection probes CUDA -- so living inside the
package would load the NVIDIA driver into an Intel-only worker *before* :func:`_load`
could apply any isolation, which is exactly what was observed as a stray 194 MiB CUDA
context in the Intel UVR worker. ``modules/inference/engines/__init__.py`` is kept empty
for the same reason; this package cannot be, so the worker moved out instead.



UVR is the app's other accelerator consumer, and isolating it buys three things the
in-process path cannot:

* **Reclamation.** ONNX Runtime's CUDA provider held ~2.4 GB of VRAM in the API process
  on this codebase, and no in-process purge returns it. Killing the worker does.
* **Cross-vendor safety.** An OpenVINO GPU context and a CUDA context cannot share an
  interpreter (see ``modules/core/config.py``), which is why preprocessing was forced
  onto the ASR vendor. Separate processes lift that restriction, so an Intel iGPU can do
  UVR while CUDA does ASR -- the case AMD deployments care about most, since ROCm-torch
  and CUDA-torch equally cannot coexist.
* **Crash containment.** A native fault in a vendor execution provider kills a worker
  rather than the API.

Separation is cooperatively preemptible at chunk boundaries. That has to survive the
process boundary, so the work runs on a thread here while the request generator emits
heartbeats; the parent cancels between heartbeats and the thread's yield callback raises
at its next chunk. A plain request/response call would have silently dropped preemption.
"""

import logging
import os
import threading
from collections.abc import Iterator
from multiprocessing.connection import Connection
from typing import Any, Optional

from modules.inference.engines import worker_runtime

logger = logging.getLogger(__name__)

#: unit_id -> PreprocessingManager. Mirrors the pool the parent used to hold.
_MANAGERS: dict[str, Any] = {}

#: How often the separation generator yields while work is in flight. Small enough that a
#: cancel is acted on promptly, large enough not to spam the pipe during a long job.
_HEARTBEAT_SEC = 0.2


class _Cancelled(Exception):
    """Raised inside the separation thread when the parent cancels."""


def _load(unit: dict, env: Optional[dict] = None) -> str:
    """Create the manager for ``unit``, applying device visibility first."""
    handle = unit["id"]
    if handle in _MANAGERS:
        return handle
    for key, value in (env or {}).items():
        os.environ[key] = value

    # Deferred until after env is applied, so the vendor runtime this worker is meant to
    # use is the only one its ONNX Runtime can see.
    from modules.inference.pipeline import preprocessing  # pylint: disable=import-outside-toplevel  # noqa: PLC0415

    logger.info("[PrepWorker] Creating preprocessor for %s", unit.get("name", handle))
    _MANAGERS[handle] = preprocessing.PreprocessingManager(unit)
    return handle


def _get(handle: str):
    manager = _MANAGERS.get(handle)
    if manager is None:
        raise KeyError(f"No preprocessor loaded for handle '{handle}'")
    return manager


def _unload(handle: str) -> bool:
    manager = _MANAGERS.pop(handle, None)
    if manager is None:
        return False
    manager.unload_model()
    return True


def _offload(handle: str) -> bool:
    manager = _MANAGERS.get(handle)
    if manager is None:
        return False
    manager.offload()
    return True


def _state(handle: str) -> dict[str, Any]:
    """Report what the parent's proxy needs to answer without another round trip."""
    manager = _MANAGERS.get(handle)
    separator = getattr(manager, "separator", None) if manager else None
    return {
        "loaded": separator is not None,
        "providers": list(getattr(separator, "onnx_execution_provider", []) or []),
    }


def _separate(handle: str, audio_path: str, force: bool = False, stage: str = "Vocal Separation") -> Iterator[dict[str, Any]]:
    """Run separation on a thread, heartbeating so cancellation still works.

    Yields ``tick`` while the thread runs and finally ``result`` with the output path.
    Closing the generator (which the runtime does on cancel) sets the abort flag and
    joins the thread, so a cancelled request does not leave UVR running on the device.
    """
    manager = _get(handle)
    abort = threading.Event()
    outcome: dict[str, Any] = {}

    def _yield_cb():
        if abort.is_set():
            raise _Cancelled()

    def _run():
        try:
            outcome["path"] = manager.preprocess_audio(audio_path, force=force, yield_cb=_yield_cb, stage=stage)
        except _Cancelled:
            outcome["cancelled"] = True
        except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            outcome["error"] = f"{type(exc).__name__}: {exc}"

    thread = threading.Thread(target=_run, name="uvr-separate", daemon=True)
    thread.start()
    try:
        while thread.is_alive():
            thread.join(timeout=_HEARTBEAT_SEC)
            if thread.is_alive():
                yield {"event": "tick"}
        if "error" in outcome:
            raise RuntimeError(outcome["error"])
        yield {"event": "result", "path": outcome.get("path", audio_path)}
    finally:
        abort.set()
        thread.join(timeout=30)


def worker_main(conn: Connection) -> None:
    """Entry point for the spawned preprocessing worker."""
    worker_runtime.configure_worker_logging("prep-worker")
    worker_runtime.serve(
        conn,
        handlers={"load": _load, "unload": _unload, "offload": _offload, "state": _state},
        stream_handlers={"separate": _separate},
    )
