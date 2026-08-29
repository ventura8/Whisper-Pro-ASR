"""Parent-side proxy presenting an out-of-process engine as a normal engine.

``IsolatedEngine`` satisfies :class:`BaseASREngine`, so the scheduler, model pool and
transcription pipeline treat it exactly like an in-process engine while the model
actually lives in a worker owned by :mod:`worker_channel`.

Channels are keyed by *engine type*, not hardware unit: one CUDA worker holds every CUDA
unit's model, one Intel worker holds every Intel unit's. That keeps the resident-model
count identical to the in-process design -- isolation moves the models, it does not
copy them -- while still letting mutually exclusive runtimes (CUDA vs OpenVINO, CUDA
torch vs ROCm torch) coexist in one deployment.
"""

import logging
import threading
from collections.abc import Iterator
from typing import Any, Optional

from modules.inference.engines import inference_worker, worker_channel, worker_runtime
from modules.inference.engines.base import BaseASREngine, InferenceInfo, SegmentWrapper

logger = logging.getLogger(__name__)


_CHANNELS: dict[str, worker_channel.WorkerChannel] = {}

# Serialises the cache-miss path. Two units of the same engine type initialising together
# each built a channel, and the loser's was overwritten in the dict while still owning a
# spawned worker process -- orphaned for the life of the service, holding whatever device
# memory its model had loaded. The same race the shared preprocessor pool had.
_CHANNELS_LOCK = threading.Lock()


def channel_for(engine_type: str) -> worker_channel.WorkerChannel:
    """Return (creating on first use) the single channel for ``engine_type``."""
    channel = _CHANNELS.get(engine_type)
    if channel is not None:
        return channel
    with _CHANNELS_LOCK:
        # Re-checked under the lock: another caller may have finished building this key
        # while this one waited, and replacing its entry would strand a live worker.
        channel = _CHANNELS.get(engine_type)
        if channel is None:
            slug = engine_type.lower().replace("-", "_")
            channel = worker_channel.WorkerChannel(
                inference_worker.worker_main,
                name=f"{slug}-worker",
                log_tag=f"{engine_type} worker",
            )
            _CHANNELS[engine_type] = channel
    return channel


def active_channels() -> dict[str, worker_channel.WorkerChannel]:
    """Every channel created so far, for lifecycle callers (idle purge, shutdown)."""
    return dict(_CHANNELS)


def shutdown_all() -> None:
    """Terminate every worker, reclaiming all device memory they hold.

    This is the reclamation the in-process path cannot achieve: killing the process
    returns the CUDA/ROCm/OpenVINO context's device memory to the OS outright.
    """
    for engine_type, channel in list(_CHANNELS.items()):
        logger.info("[Isolated] Shutting down %s worker", engine_type)
        channel.shutdown()


class IsolatedEngine(BaseASREngine):
    """Runs a real engine in a worker process, presenting the in-process interface."""

    #: Marks this class as an out-of-process engine proxy, so a caller can recognise one
    #: without importing it. language_detection_core needs that answer and sits on the
    #: other side of an import cycle; a class attribute crosses it where an import cannot.
    #: Read off the *type*, never the instance -- a MagicMock invents any attribute asked
    #: of an instance and would claim to be a proxy.
    IS_ISOLATED_ENGINE = True

    def __init__(self, engine_type: str, model_id: str, unit: dict) -> None:
        self.engine_type = engine_type
        self.model_id = model_id
        self.unit = unit
        self._channel = channel_for(engine_type)
        self.handle, self._generation = self._load()

    def _worker_env(self) -> dict[str, str]:
        """Environment the worker starts from: vendor isolation plus resolved settings.

        The worker skips hardware detection (WHISPER_WORKER_CONTEXT), so anything the
        parent derived from probing has to be handed over explicitly. COMPUTE_TYPE
        especially: a worker that re-detected as CPU would quietly load a CUDA model as
        int8 instead of the float16 the parent resolved.
        """
        from modules.core import config  # pylint: disable=import-outside-toplevel  # noqa: PLC0415 - avoids an import cycle at module load

        env = dict(worker_runtime.ISOLATION_ENV.get(self.unit.get("type", ""), {}))
        env["WHISPER_WORKER_CONTEXT"] = "1"
        env["ASR_ENGINE"] = self.engine_type
        # The engine is dictated to the worker but the device was not, so the worker
        # re-detected hardware and could disagree with the unit it serves -- emitting
        # "INTEL-WHISPER requested but no Intel GPU/NPU available. Falling back to
        # FASTER-WHISPER" while faithfully running the Intel engine the parent handed it.
        # The device it serves is not a thing for it to work out; it is a given.
        unit_type = self.unit.get("type", "")
        if unit_type:
            env["ASR_DEVICE"] = unit_type
        env["ASR_COMPUTE_TYPE"] = str(getattr(config, "COMPUTE_TYPE", "AUTO"))
        return env

    def _load(self) -> tuple[str, int]:
        return self._channel.call_with_generation(
            "load_model",
            engine_type=self.engine_type,
            model_id=self.model_id,
            unit=self.unit,
            env=self._worker_env(),
        )

    def _ensure_loaded(self) -> None:
        """Reload if the worker restarted since this engine's model was loaded.

        A crash and respawn leaves the handle pointing at a pool that no longer exists;
        reloading transparently is what turns a worker death into a slow request rather
        than a failed one.
        """
        if self._generation == self._channel.generation():
            return
        logger.info("[Isolated] %s worker restarted; reloading model for %s", self.engine_type, self.unit["id"])
        self.handle, self._generation = self._load()

    def transcribe(
        self,
        audio_path: str,
        *,
        language: Optional[str] = None,
        task: str = "transcribe",
        initial_prompt: Optional[str] = None,
        vad_filter: bool = True,
        word_timestamps: bool = False,
        **kwargs: Any,
    ) -> tuple[Iterator[SegmentWrapper], InferenceInfo]:
        """Start a streaming transcription and return ``(segments, info)``.

        ``info`` arrives as the stream's first event, so the caller gets it before
        consuming segments -- matching the in-process contract. The returned generator
        owns the worker for its lifetime; abandoning it early (which cooperative
        preemption does routinely) cancels the worker-side decode.
        """
        self._ensure_loaded()
        params = {
            "language": language,
            "task": task,
            "initial_prompt": initial_prompt,
            "vad_filter": vad_filter,
            "word_timestamps": word_timestamps,
            **kwargs,
        }
        stream = self._channel.stream("transcribe", handle=self.handle, audio_path=audio_path, params=params)

        info = self._read_leading_info(stream)
        return self._iter_segments(stream), info

    def _read_leading_info(self, stream: Iterator[dict]) -> InferenceInfo:
        for event in stream:
            if event.get("event") == "info":
                return _info_from_dict(event["info"])
            break
        stream.close()
        raise worker_channel.WorkerError(f"{self.engine_type} worker sent no info event")

    def _iter_segments(self, stream: Iterator[dict]) -> Iterator[SegmentWrapper]:
        for event in stream:
            if event.get("event") == "segment":
                yield _segment_from_dict(event["segment"])

    def detect_language(self, audio: Any) -> tuple[str, float, list[tuple[str, float]]]:
        """Detect language for an audio *path*.

        Only paths are accepted: sending decoded samples would copy the audio across the
        pipe, and the pipeline's batch detection already runs entirely worker-side.
        """
        if not isinstance(audio, str):
            raise TypeError("IsolatedEngine.detect_language requires an audio path, not decoded samples")
        self._ensure_loaded()
        result = self._channel.call("detect_language", handle=self.handle, audio_path=audio)
        return result["language"], result["probability"], result["all_probs"]

    def detect_language_batch(self, audio_path: str, segment_count: int) -> Iterator[dict[str, Any]]:
        """Stream per-window detection results, decoded entirely inside the worker."""
        self._ensure_loaded()
        for event in self._channel.stream(
            "detect_language_batch",
            handle=self.handle,
            audio_path=audio_path,
            segment_count=segment_count,
        ):
            if event.get("event") == "detection":
                yield event

    def cancel(self) -> None:
        """Ask the worker to stop the in-flight stream (cooperative preemption)."""
        self._channel.cancel_in_stream()

    def unload(self) -> None:
        """Drop this unit's model in the worker, leaving the process up for other units."""
        try:
            self._channel.call("unload_model", handle=self.handle)
        except worker_channel.WorkerError as exc:
            # A dead worker has already released everything this call would have freed.
            logger.info("[Isolated] Unload skipped for %s: %s", self.unit["id"], exc)


def _info_from_dict(payload: dict[str, Any]) -> InferenceInfo:
    probs = payload.get("all_language_probs")
    return InferenceInfo(
        language=payload.get("language", "en"),
        language_probability=payload.get("language_probability", 0.0),
        duration=payload.get("duration", 0.0),
        all_language_probs=[tuple(item) for item in probs] if probs else None,
    )


def _segment_from_dict(payload: dict[str, Any]) -> SegmentWrapper:
    return SegmentWrapper(
        start=payload.get("start", 0.0),
        end=payload.get("end", 0.0),
        text=payload.get("text", ""),
        words=payload.get("words"),
    )
