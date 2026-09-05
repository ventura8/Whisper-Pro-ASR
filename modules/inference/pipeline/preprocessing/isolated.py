"""Parent-side proxy presenting an out-of-process preprocessor as a normal one.

``IsolatedPreprocessor`` implements the surface ``PreprocessingManager`` exposes to the
rest of the app -- ``preprocess_audio``, ``offload``, ``unload_model``, the ``separator``
probe used by telemetry and metrics, and the unit/device properties -- so the runtime
cannot tell the difference while UVR actually runs in its own process.

Channels are keyed by **device type**, not unit: all CUDA units share one worker, all
Intel units another. That keeps the resident UVR model count identical to the in-process
design while letting different vendors' execution providers live in different processes,
which is the whole point -- an OpenVINO GPU context and a CUDA context cannot coexist in
one interpreter, and neither can ROCm-torch and CUDA-torch.
"""

import logging
import threading
from typing import Any, Optional

from modules.inference.engines import worker_channel, worker_runtime
from modules.inference.pipeline import preprocessing_worker

logger = logging.getLogger(__name__)


_CHANNELS: dict[str, worker_channel.WorkerChannel] = {}

#: Guards the cache-miss path below. "One channel per device type" is the invariant the
#: whole module rests on -- two channels means two worker processes holding two copies of
#: the UVR model on one device, and shutdown_all() only ever sees the one left in the dict.
_CHANNELS_LOCK = threading.Lock()


def channel_for(device_type: str) -> worker_channel.WorkerChannel:
    """Return (creating on first use) the single channel for ``device_type``."""
    channel = _CHANNELS.get(device_type)
    if channel is not None:
        return channel
    with _CHANNELS_LOCK:
        # Re-checked under the lock: two tasks starting together on the same device both
        # saw None and both constructed a channel, and the loser's worker was orphaned --
        # still running, still holding device memory, unreachable and never shut down.
        channel = _CHANNELS.get(device_type)
        if channel is None:
            slug = str(device_type).lower().replace(".", "_")
            channel = worker_channel.WorkerChannel(
                preprocessing_worker.worker_main,
                name=f"uvr-{slug}-worker",
                log_tag=f"UVR {device_type} worker",
            )
            _CHANNELS[device_type] = channel
    return channel


def shutdown_all() -> None:
    """Terminate every preprocessing worker, returning the device memory UVR held."""
    for device_type, channel in list(_CHANNELS.items()):
        logger.info("[Isolated] Shutting down UVR %s worker", device_type)
        channel.shutdown()


class _SeparatorProbe:  # pylint: disable=too-few-public-methods
    """Stands in for the real ``Separator`` for the two things callers ask of it.

    telemetry checks ``is not None`` and metrics reads ``onnx_execution_provider``; the
    real object lives in the worker and is not picklable, so this carries the answers.
    """

    def __init__(self, providers: list[str]) -> None:
        self.onnx_execution_provider = providers

    def __bool__(self) -> bool:
        return True


class IsolatedPreprocessor:
    """Runs UVR in a worker process, presenting the in-process manager interface."""

    def __init__(self, assigned_unit: Optional[dict] = None) -> None:
        self._unit = assigned_unit
        self._device_id = assigned_unit["id"] if assigned_unit else "CPU"
        self._device_type = assigned_unit["type"] if assigned_unit else "CPU"
        self._channel = channel_for(self._device_type)
        self._handle: Optional[str] = None
        #: Snapshot of the worker's model state; see the `separator` property for why
        #: this is cached rather than fetched on demand.
        self._state_cache: dict[str, Any] = {"loaded": False, "providers": []}

    # --- properties the runtime reads ------------------------------------------

    @property
    def unit(self):
        """The hardware unit this preprocessor was created for, or None."""
        return self._unit

    @property
    def device_id(self):
        """The unit's device id, as the scheduler and the logs name it."""
        return self._device_id

    @property
    def device_type(self):
        """The unit's device family (CUDA, AMD, GPU, NPU, CPU)."""
        return self._device_type

    @property
    def lock(self):
        """The worker channel's lock, presented as the in-process manager's ``lock``.

        metrics_discovery calls ``pm.lock.locked()`` to decide whether an accelerator is
        busy with vocal separation. IsolatedPreprocessor had no ``lock`` at all, so that
        call raised AttributeError, was swallowed by the guard around it, and every
        isolated UVR run reported the device as idle -- the Intel and AMD utilization
        charts sat at 0% for the whole of a separation, which reads as "the accelerator is
        not being used", the single most misleading thing this dashboard can say.

        The channel holds this lock for the entire duration of a separation stream, so its
        state answers exactly the question metrics is asking. Returned as the lock object
        rather than a boolean to keep the in-process interface -- callers use `.locked()`.
        """
        return self._channel.lock

    @property
    def separator(self):
        """Whether a UVR model is resident in the worker, plus its providers.

        Answered from a cached snapshot rather than an RPC. Telemetry polls this once a
        second for the dashboard, while the channel lock is held for the whole duration
        of a separation stream -- querying the worker here made every poll block until
        separation finished, stalling the dashboard for the entire job.

        Returns None when nothing is loaded, so ``if preprocessor.separator`` keeps
        meaning "UVR is using memory", exactly as it did in-process.
        """
        if not self._state_cache.get("loaded"):
            return None
        return _SeparatorProbe(self._state_cache.get("providers", []))

    def refresh_state(self) -> None:
        """Pull the worker's model state into the cache. Never call while streaming."""
        if self._handle is None:
            self._state_cache = {"loaded": False, "providers": []}
            return
        try:
            self._state_cache = self._channel.call("state", handle=self._handle)
        except worker_channel.WorkerError:
            self._state_cache = {"loaded": False, "providers": []}

    # --- work -------------------------------------------------------------------

    def _worker_env(self) -> dict[str, str]:
        """Vendor isolation plus the preprocessing settings the parent already resolved.

        WHISPER_WORKER_CONTEXT stops the worker re-probing hardware; without it the
        ctranslate2 CUDA probe run during config import maps the NVIDIA driver into an
        Intel-only UVR worker, which is exactly the isolation this class exists to give.
        """
        env = dict(worker_runtime.ISOLATION_ENV.get(self._device_type, {}))
        env["ASR_PREPROCESS_DEVICE"] = str(self._device_type)
        # NOTE: WHISPER_WORKER_CONTEXT is deliberately NOT set for OpenVINO targets.
        # Skipping hardware detection also skips the OpenVINO initialisation the ONNX
        # Runtime OpenVINO provider depends on, and creating a GPU session without it
        # segfaults the worker. Non-OpenVINO vendors keep the cheaper skip.
        if self._device_type not in ("GPU", "NPU"):
            env["WHISPER_WORKER_CONTEXT"] = "1"
        return env

    def _ensure_loaded(self) -> str:
        """Create the worker-side manager, reloading transparently after a crash."""
        unit = self._unit or {"id": self._device_id, "type": self._device_type, "name": self._device_id}
        self._handle = self._channel.call("load", unit=unit, env=self._worker_env())
        return self._handle

    def preprocess_audio(self, audio_path, force=False, yield_cb=None, stage="Vocal Separation"):
        """Separate vocals in the worker, retrying once if the worker dies.

        Intel's OpenVINO GPU execution provider can take the worker down with a SIGSEGV
        during separation -- a native crash in the vendor runtime, not something Python
        can prevent. A fresh worker then separates the same audio successfully, so one
        retry turns a hard 500 into a slower success. This is the crash containment
        isolation exists for: in-process, the same fault killed the whole service, which
        is why cross-vendor preprocessing used to be disabled outright.

        Only worker death is retried. A failure the worker reports as an error is a real
        problem and propagates unchanged -- which is what WorkerReportedError, caught and
        re-raised first below, exists to express: previously both arrived as WorkerError
        and a reported failure was silently run a second time.
        """
        try:
            return self._separate_once(audio_path, force, yield_cb, stage)
        except worker_channel.WorkerReportedError:
            raise
        except worker_channel.WorkerError as exc:
            logger.warning("[Isolated] UVR %s worker died (%s); retrying once on a fresh worker.", self._device_type, exc)
            self._handle = None
            self._state_cache = {"loaded": False, "providers": []}
            return self._separate_once(audio_path, force, yield_cb, stage)

    def _separate_once(self, audio_path, force, yield_cb, stage):
        """One separation attempt against the current worker."""
        handle = self._ensure_loaded()
        result_path = audio_path
        stream = self._channel.stream("separate", handle=handle, audio_path=audio_path, force=force, stage=stage)
        try:
            for event in stream:
                if event.get("event") == "result":
                    result_path = event.get("path") or audio_path
                    break
                if yield_cb:
                    yield_cb()
        finally:
            stream.close()
        # Safe now: the stream is closed, so the channel lock is free.
        self.refresh_state()
        return result_path

    def offload(self) -> None:
        """Ask the worker to release the UVR model's device memory, keeping the process."""
        if self._handle is None:
            return
        try:
            self._channel.call("offload", handle=self._handle)
        except worker_channel.WorkerError as exc:
            logger.info("[Isolated] UVR offload skipped for %s: %s", self._device_id, exc)

    def unload_model(self) -> None:
        """Drop the worker's UVR model, leaving the process up for other units."""
        if self._handle is None:
            return
        try:
            self._channel.call("unload", handle=self._handle)
        except worker_channel.WorkerError as exc:
            # A dead worker already released everything this call would have freed.
            logger.info("[Isolated] UVR unload skipped for %s: %s", self._device_id, exc)
        finally:
            self._handle = None
            self._state_cache = {"loaded": False, "providers": []}
