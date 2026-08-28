"""WhisperX engine wrapper.

WhisperX runs in an isolated subprocess (``whisperx_worker_client``) because
it depends on an older, mutually incompatible ``torch``/``transformers``/
``huggingface-hub`` stack than the rest of the app. See
``whisperx_worker`` for the full rationale.
"""

import logging
from collections.abc import Iterator
from typing import Any, Optional

from modules.core import config
from modules.inference.engines import whisperx_worker_client as worker
from modules.inference.engines.base import BaseASREngine, InferenceInfo, SegmentWrapper, build_inference_info, iter_segment_wrappers

logger = logging.getLogger(__name__)


class WhisperXEngine(BaseASREngine):
    """WhisperX engine supporting batch inference, backed by an isolated worker process."""

    def __init__(self, model_id: str, device: str, compute_type: str = "int8") -> None:
        self.model_id = model_id
        self.device = device
        self.compute_type = compute_type
        # Stamp generation atomically with load so a mid-load worker death cannot
        # leave a handle paired with a stale generation from a separate generation().
        self.model_handle, self._generation = worker.call_with_generation(
            "load_model",
            model_id=model_id,
            device=device,
            compute_type=compute_type,
        )

    def _ensure_current_model_handle(self) -> None:
        """Reload model_handle if the worker has restarted since it was cached.

        A worker crash+respawn (see whisperx_worker_client.generation()) leaves
        the old handle pointing at an `objects` dict that no longer exists;
        sending it to the new worker would fail every call until this engine
        is evicted and rebuilt. Reload transparently instead."""
        if not hasattr(self, "model_handle") or self._generation == worker.generation():
            return
        logger.info("[WhisperX] Worker restarted since model load; reloading model handle.")
        self.model_handle, self._generation = worker.call_with_generation(
            "load_model",
            model_id=self.model_id,
            device=self.device,
            compute_type=self.compute_type,
        )

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
        unsupported_opts = _unsupported_whisperx_options(initial_prompt, vad_filter, word_timestamps)
        if unsupported_opts:
            logger.warning("[WhisperX] Ignoring unsupported options: %s", ", ".join(unsupported_opts))

        self._ensure_current_model_handle()
        batch_size = kwargs.get("batch_size", config.DEFAULT_BATCH_SIZE)
        result = worker.call(
            "transcribe",
            model_handle=self.model_handle,
            audio_path=audio_path,
            batch_size=batch_size,
            language=language,
            task=task,
        )
        return iter_segment_wrappers(result), build_inference_info(result, audio_path, language)

    def detect_language(self, audio: Any) -> tuple[str, float, list[tuple[str, float]]]:
        """Identify language with WhisperX/faster-whisper backend when available.

        Accepts either an audio file path or an already-decoded numpy array
        (used by the batched multi-segment detection path).
        """
        self._ensure_current_model_handle()
        audio_kwargs = {"audio_path": audio} if isinstance(audio, str) else {"audio_array": audio}
        language, probability, all_probs = worker.call("detect_language", model_handle=self.model_handle, **audio_kwargs)
        return language, probability, all_probs

    def unload(self) -> None:
        if not hasattr(self, "model_handle"):
            return
        try:
            if self._generation == worker.generation():
                worker.call("unload_model", model_handle=self.model_handle)
        finally:
            del self.model_handle


def _unsupported_whisperx_options(initial_prompt: Optional[str], vad_filter: bool, word_timestamps: bool) -> list[str]:
    unsupported = []
    if initial_prompt:
        unsupported.append("initial_prompt")
    if not vad_filter:
        unsupported.append("vad_filter")
    if word_timestamps:
        unsupported.append("word_timestamps")
    return unsupported
