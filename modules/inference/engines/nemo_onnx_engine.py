"""PARAKEET and CANARY: an NVIDIA NeMo model on onnx-asr, with a Whisper companion.

One object per hardware unit, like every other engine, so the scheduler, the model pool and
the pipeline need nothing new. Inside it are two models:

* the NeMo model (Parakeet-TDT-0.6B-v3 or Canary-1B-v2 by default), which transcribes the
  25 European languages it was trained on -- several times faster than Whisper large-v3 and,
  for Parakeet, without the autoregressive decoder that invents text into silence;
* a Whisper companion (a FasterWhisperEngine on ``ASR_MODEL``), because the NeMo models have
  no language-identification head and cover a quarter of what Bazarr sends. It answers every
  ``detect_language`` call -- so ``/detect-language``, gap filling and per-region detection
  behave exactly as on FASTER-WHISPER -- and transcribes whatever nemo_routing sends its way.

The NeMo half honours ``clip_timestamps`` and yields its segments lazily, a batch of chunks
at a time, so decoding by speech region and pause/resume (resumable_decode) work unchanged:
closing the stream stops it at the next batch boundary.
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Iterator
from typing import Any, Optional

import numpy as np

from modules.core import config_nemo
from modules.inference.engines import nemo_chunking, nemo_routing, nemo_runtime, nemo_segments
from modules.inference.engines.base import BaseASREngine, InferenceInfo, SegmentWrapper

logger = logging.getLogger(__name__)

# Deliberately broad handlers catch through this name: pylint flags a bare `except Exception` and inline disables are banned.
_ANY_EXCEPTION: tuple[type[Exception], ...] = (Exception,)


class NemoOnnxEngine(BaseASREngine):
    """A NeMo model for the languages it knows, a Whisper companion for everything else."""

    def __init__(self, engine_type: str, unit: dict, *, companion: Any, cache_root: str) -> None:
        self.engine_type = engine_type
        self.unit = unit
        self.spec = config_nemo.model_spec(engine_type)
        self._companion = companion
        self._quantization = config_nemo.quantization_for(unit.get("type", ""))
        self._path = config_nemo.model_dir(cache_root, self.spec)
        self._cpu_spare: Any = None
        self.model, self.device = nemo_runtime.load(self.spec, self._path, self._quantization, unit)
        logger.info(
            "[NeMo] %s ready on %s (device=%s, quantization=%s); Whisper companion on %s",
            self.spec.name,
            unit.get("name", unit.get("id")),
            self.device,
            self._quantization or "fp32",
            getattr(companion, "device", "?"),
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
        """Route the call (nemo_routing), then decode it on whichever model it belongs to.

        The companion receives the call exactly as it arrived -- clips, beam size, prompt and
        all -- so a routed request is indistinguishable from one served by FASTER-WHISPER.
        """
        route = nemo_routing.choose(self.spec, language, task)
        if route.target == nemo_routing.TARGET_WHISPER:
            logger.info("[NeMo] route=whisper reason=%s", route.reason)
            return self._companion.transcribe(
                audio_path,
                language=language,
                task=task,
                initial_prompt=initial_prompt,
                vad_filter=vad_filter,
                word_timestamps=word_timestamps,
                **kwargs,
            )
        # The prompt is the one Whisper option with no NeMo counterpart that changes output;
        # it has a non-empty default, so this is debug, not a warning on every request.
        if initial_prompt:
            logger.debug("[NeMo] initial_prompt has no effect on %s", self.spec.name)
        return self._transcribe_on_nemo(audio_path, route, vad_filter=vad_filter, word_timestamps=word_timestamps, kwargs=kwargs)

    def _transcribe_on_nemo(self, audio_path: Any, route: nemo_routing.Route, *, vad_filter: bool, word_timestamps: bool, kwargs: dict):
        audio = nemo_chunking.decode(audio_path)
        spans = nemo_chunking.plan(
            audio,
            clip_timestamps=kwargs.get("clip_timestamps"),
            vad_filter=vad_filter,
            vad_parameters=kwargs.get("vad_parameters"),
            max_sec=config_nemo.NEMO_MAX_CHUNK_SEC,
        )
        logger.info("[NeMo] route=nemo reason=%s clips=%d", route.reason, len(spans))
        info = InferenceInfo(
            language=route.language or "en",
            language_probability=1.0,
            duration=len(audio) / nemo_chunking.SAMPLE_RATE,
            all_language_probs=None,
        )
        return self._segments(audio, spans, route, keep_words=word_timestamps), info

    def _segments(self, audio: np.ndarray, spans: list, route: nemo_routing.Route, *, keep_words: bool) -> Iterator[SegmentWrapper]:
        """Decode ``spans`` a batch at a time, yielding each batch's segments in time order."""
        recognize_kwargs = route.recognize_kwargs(self.spec)
        for batch in nemo_chunking.batches(spans, config_nemo.NEMO_BATCH_SIZE):
            results = self._recognize([nemo_chunking.samples(audio, span) for span in batch], recognize_kwargs)
            for span, result in zip(batch, results):
                yield from nemo_segments.to_segments(result, span, timed=self.spec.word_timestamps, keep_words=keep_words)

    def _recognize(self, chunks: list, recognize_kwargs: dict) -> list:
        """One batch, degrading step by step rather than failing the request.

        1. The batch as given.
        2. On an accelerator, a batch that ran out of device memory is retried one chunk at a
           time on the same device. A batch's activations scale with its size, and on a card
           shared with the companion and vocal isolation that is the usual failure: an RTX
           3080 laptop (8 GB) failed a 4 x 20 s Parakeet batch in the arena allocator while
           one chunk at a time fits (2026-10-09). Leaving the GPU for that would trade a
           transient shortage for a 30x slowdown for the life of the worker. A single chunk
           that still does not fit is decoded on a CPU copy of the model, and the engine
           stays on the accelerator.
        3. Anything else on an accelerator -- a plugin that rejects a later input shape the
           warm-up never exercised -- moves the engine to the CPU for good, and it reports so.
        """
        try:
            return list(self.model.recognize(chunks, **recognize_kwargs))
        except _ANY_EXCEPTION as exc:
            if self.device == "cpu":
                raise
            return self._recover(exc, chunks, recognize_kwargs)

    def _recover(self, exc: Exception, chunks: list, recognize_kwargs: dict) -> list:
        """Steps 2 and 3 of :meth:`_recognize`, for a batch that failed on an accelerator."""
        if _is_out_of_memory(exc):
            return self._recover_out_of_memory(chunks, recognize_kwargs)
        logger.error("[NeMo] %s failed on %s mid-decode (%s); continuing on the CPU.", self.spec.name, self.device, exc)
        self.model = nemo_runtime.load_on_cpu(self.spec, self._path, self._quantization)
        self.device = "cpu"
        return list(self.model.recognize(chunks, **recognize_kwargs))

    def _recover_out_of_memory(self, chunks: list, recognize_kwargs: dict) -> list:
        """Step 2 of :meth:`_recognize`: a device-memory shortage never leaves the accelerator."""
        if len(chunks) > 1:
            logger.warning("[NeMo] %s ran out of memory on %s; retrying %d chunks one at a time.", self.spec.name, self.device, len(chunks))
            return [result for chunk in chunks for result in self._recognize([chunk], recognize_kwargs)]
        # A single chunk that still does not fit is a momentary shortage, not a graph the
        # device cannot run: decode just this chunk on the CPU and keep the accelerator.
        logger.warning("[NeMo] %s ran out of memory on %s for one chunk; decoding it on the CPU.", self.spec.name, self.device)
        return list(self._cpu_model().recognize(chunks, **recognize_kwargs))

    def _cpu_model(self) -> Any:
        """A CPU copy of the model for one-off chunks, loaded on first use and then reused."""
        if self._cpu_spare is None:
            self._cpu_spare = nemo_runtime.load_on_cpu(self.spec, self._path, self._quantization)
        return self._cpu_spare

    def detect_language(self, audio: Any):
        """Whisper's language identification, via the companion. The NeMo models have none."""
        return self._companion.detect_language(audio)

    def unload(self) -> None:
        """Release both models."""
        companion = getattr(self, "_companion", None)
        if companion is not None:
            companion.unload()
        if hasattr(self, "model"):
            del self.model
        self._cpu_spare = None
        gc.collect()


#: How ONNX Runtime reports an exhausted device: its arena allocator, or CUDA itself.
_OUT_OF_MEMORY_MARKERS = ("failed to allocate memory", "out of memory", "bfcarena")


def _is_out_of_memory(exc: Exception) -> bool:
    """Whether ``exc`` is a device-memory shortage rather than a graph the provider cannot run."""
    text = str(exc).lower()
    return any(marker in text for marker in _OUT_OF_MEMORY_MARKERS)
