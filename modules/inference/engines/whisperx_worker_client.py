"""Parent-side client for the isolated WhisperX subprocess.

See ``whisperx_worker`` for why WhisperX runs out-of-process: it depends on an older,
mutually incompatible torch/transformers/huggingface-hub stack than the rest of the app.

The transport -- lazily spawned long-lived worker, serialized calls, generation stamping
so cached handles are invalidated when the process dies, and a shutdown that can
interrupt a hung unbounded call -- lives in :mod:`worker_channel`, which every isolated
engine shares. This module is the WhisperX-specific binding: it names the worker, keeps
the historic ``WHISPERX_WORKER_*`` environment overrides, and preserves the module-level
function API that ``whisperx_engine`` and ``diarization`` already call.
"""

import logging
import os
from typing import Any

from modules.inference.engines import whisperx_worker
from modules.inference.engines.worker_channel import WorkerChannel, WorkerError

logger = logging.getLogger(__name__)


class WhisperXWorkerError(WorkerError):
    """Raised when the isolated WhisperX worker reports a failure."""


#: A finite deadline prevents a dead worker from retaining the channel lock forever.
#: The channel shuts it down on expiry and the next call lazily respawns it.
_CALL_TIMEOUT_SEC = float(os.environ.get("WHISPERX_WORKER_CALL_TIMEOUT_SEC", "1800"))
#: Operational warn threshold when a caller blocks waiting for serialized worker access.
_LOCK_WARN_SEC = float(os.environ.get("WHISPERX_WORKER_LOCK_WARN_SEC", "5"))

_CHANNEL = WorkerChannel(
    whisperx_worker.worker_main,
    name="whisperx-worker",
    log_tag="WhisperXWorker",
    error_cls=WhisperXWorkerError,
    call_timeout_sec=_CALL_TIMEOUT_SEC,
    lock_warn_sec=_LOCK_WARN_SEC,
)


def channel() -> WorkerChannel:
    """Return the underlying channel (for lifecycle callers and tests)."""
    return _CHANNEL


def generation() -> int:
    """Return the current worker generation, reaping a since-died process first.

    Callers that cache handles into the worker's ``objects`` pool (diarization's
    ALIGN_POOL/DIARIZE_POOL, WhisperXEngine's model_handle) compare against this before
    trusting a cached handle: a handle from a prior generation refers to an ``objects``
    dict that no longer exists.
    """
    return _CHANNEL.generation()


def call_with_generation(cmd: str, **args: Any) -> tuple[Any, int]:
    """Send a command and return ``(result, generation)`` under one lock hold."""
    return _CHANNEL.call_with_generation(cmd, **args)


def call(cmd: str, **args: Any) -> Any:
    """Send a command to the isolated WhisperX worker and block for the result."""
    return _CHANNEL.call(cmd, **args)


def shutdown() -> None:
    """Terminate the worker, reclaiming all the RAM it holds.

    This is how the app's periodic "purge everything" idle cleanup guarantees WhisperX's
    memory is actually released; a fresh worker is spawned on next use.
    """
    _CHANNEL.shutdown()
