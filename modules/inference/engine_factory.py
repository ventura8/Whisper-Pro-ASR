"""
Engine Factory for Whisper Pro ASR.
Supports FASTER-WHISPER, INTEL-WHISPER, OPENAI-WHISPER, and WHISPERX.
"""

import importlib
import logging
from dataclasses import dataclass
from typing import Any, Iterator, List, Optional, Tuple

from modules.core import config, engine_registry, utils

logger = logging.getLogger(__name__)


@dataclass
class InferenceInfo:
    """Standardized info structure returned by engines."""

    language: str
    language_probability: float
    duration: float
    all_language_probs: Optional[List[Tuple[str, float]]] = None


@dataclass
class SegmentWrapper:
    """Standardized segment structure yielded by engines."""

    start: float
    end: float
    text: str
    words: Optional[List[Any]] = None


class BaseASREngine:
    """Base interface for all ASR engines."""

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
    ) -> Tuple[Iterator[Any], Any]:
        """Runs transcription on the audio path and returns (segments_iterator, info_object)."""
        raise NotImplementedError()

    def detect_language(self, audio: Any) -> Tuple[str, float, List[Tuple[str, float]]]:
        """Identify the language of the audio data without full transcription."""
        raise NotImplementedError()

    def unload(self) -> None:
        """Release underlying model resources from memory."""


class FasterWhisperEngine(BaseASREngine):
    """CTranslate2 faster-whisper engine."""

    def __init__(
        self,
        model_id: str,
        *,
        device: str,
        device_index: int = 0,
        compute_type: str = "int8",
        cpu_threads: int = 4,
        download_root: Optional[str] = None,
    ):
        faster_whisper = importlib.import_module("faster_whisper")
        self.model = faster_whisper.WhisperModel(
            model_id,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            download_root=download_root,
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
    ) -> Tuple[Iterator[Any], Any]:
        params = {
            "beam_size": config.DEFAULT_BEAM_SIZE,
            "initial_prompt": initial_prompt,
            "vad_filter": vad_filter,
            "vad_parameters": {"min_silence_duration_ms": config.VAD_MIN_SILENCE_DURATION_MS},
            "word_timestamps": word_timestamps,
        }
        params.update(kwargs)
        return self.model.transcribe(audio_path, language=language, task=task, **params)

    def detect_language(self, audio: Any) -> Tuple[str, float, List[Tuple[str, float]]]:
        """Identify the language of the audio data without full transcription.
        Returns (lang_code, probability, all_probabilities).
        """
        return self.model.detect_language(audio)

    def unload(self) -> None:
        if hasattr(self, "model"):
            del self.model


class OpenaiWhisperEngine(BaseASREngine):
    """Standard PyTorch openai-whisper engine."""

    def __init__(self, model_id: str, device: str):
        self.whisper = importlib.import_module("whisper")
        self.device = device
        self.model = self.whisper.load_model(model_id, device=device)

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
    ) -> Tuple[Iterator[Any], Any]:
        # Filter and convert keyword arguments for OpenAI Whisper
        params = {
            "initial_prompt": initial_prompt,
            "word_timestamps": word_timestamps,
        }
        # Add permissible kwargs that standard transcribe accepts
        for k in [
            "beam_size",
            "best_of",
            "patience",
            "length_penalty",
            "temperature",
            "compression_ratio_threshold",
            "logprob_threshold",
            "no_speech_threshold",
            "fp16",
        ]:
            if k in kwargs:
                params[k] = kwargs[k]

        result = self.model.transcribe(audio_path, language=language, task=task, **params)

        detected_lang = result.get("language", language or "en")
        duration = utils.get_audio_duration(audio_path)

        info = InferenceInfo(
            language=detected_lang,
            language_probability=1.0,
            duration=duration,
            all_language_probs=[(detected_lang, 1.0)],
        )

        def segments_generator():
            for seg in result.get("segments", []):
                yield SegmentWrapper(
                    start=seg.get("start", 0.0),
                    end=seg.get("end", 0.0),
                    text=seg.get("text", ""),
                    words=seg.get("words"),
                )

        return segments_generator(), info

    def detect_language(self, audio: Any) -> Tuple[str, float, List[Tuple[str, float]]]:
        """Identify language using OpenAI Whisper language head."""
        if isinstance(audio, str):
            audio = self.whisper.load_audio(audio)

        mel = self.whisper.log_mel_spectrogram(audio).to(self.model.device)
        _, probs = self.model.detect_language(mel)
        ordered = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        lang_code, lang_prob = ordered[0]
        return lang_code, float(lang_prob), [(k, float(v)) for k, v in ordered]

    def unload(self) -> None:
        if hasattr(self, "model"):
            del self.model


class WhisperXEngine(BaseASREngine):
    """WhisperX engine supporting batch inference."""

    def __init__(self, model_id: str, device: str, compute_type: str = "int8"):
        self.whisperx = importlib.import_module("whisperx")
        self.device = device
        self.compute_type = compute_type
        self.model = self.whisperx.load_model(model_id, device=device, compute_type=compute_type)

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
    ) -> Tuple[Iterator[Any], Any]:
        unsupported_opts = []
        if initial_prompt:
            unsupported_opts.append("initial_prompt")
        if not vad_filter:
            unsupported_opts.append("vad_filter")
        if word_timestamps:
            unsupported_opts.append("word_timestamps")
        if unsupported_opts:
            logger.warning(
                "[WhisperX] Ignoring unsupported options: %s",
                ", ".join(unsupported_opts),
            )

        audio = self.whisperx.load_audio(audio_path)
        batch_size = kwargs.get("batch_size", config.DEFAULT_BATCH_SIZE)

        # WhisperX transcribe options
        result = self.model.transcribe(audio, batch_size=batch_size, language=language, task=task)

        detected_lang = result.get("language", language or "en")
        duration = utils.get_audio_duration(audio_path)

        info = InferenceInfo(
            language=detected_lang,
            language_probability=1.0,
            duration=duration,
            all_language_probs=[(detected_lang, 1.0)],
        )

        def segments_generator():
            for seg in result.get("segments", []):
                yield SegmentWrapper(
                    start=seg.get("start", 0.0),
                    end=seg.get("end", 0.0),
                    text=seg.get("text", ""),
                    words=seg.get("words"),
                )

        return segments_generator(), info

    def detect_language(self, audio: Any) -> Tuple[str, float, List[Tuple[str, float]]]:
        """Identify language with WhisperX/faster-whisper backend when available."""
        if isinstance(audio, str):
            audio = self.whisperx.load_audio(audio)

        if hasattr(self.model, "model") and hasattr(self.model.model, "detect_language"):
            lang_code, lang_prob, all_probs_list = self.model.model.detect_language(audio)
            return lang_code, float(lang_prob), [(k, float(v)) for k, v in all_probs_list]

        if hasattr(self.model, "detect_language"):
            lang_code, lang_prob, all_probs_list = self.model.detect_language(audio)
            return lang_code, float(lang_prob), [(k, float(v)) for k, v in all_probs_list]

        # Fallback: infer language from lightweight transcribe result.
        result = self.model.transcribe(audio, batch_size=1, task="transcribe")
        detected_lang = result.get("language", "en")
        return detected_lang, 1.0, [(detected_lang, 1.0)]

    def unload(self) -> None:
        if hasattr(self, "model"):
            del self.model


def create_engine(engine_type: str, model_id: str, unit: dict) -> BaseASREngine:
    """Factory method to instantiate the correct ASR engine wrapper."""
    engine_type = engine_registry.normalize_and_validate_engine(engine_type)

    if engine_type == engine_registry.ENGINE_INTEL_WHISPER:
        if unit["type"] in ["GPU", "NPU"]:
            logger.info("[EngineFactory] Loading IntelWhisperEngine on %s", unit["name"])
            intel_engine = importlib.import_module("modules.inference.intel_engine")
            return intel_engine.IntelWhisperEngine(model_id, device=unit["id"])

        logger.info(
            "[EngineFactory] INTEL-WHISPER requested on %s. Falling back to FasterWhisperEngine.",
            unit["name"],
        )
        engine_type = engine_registry.ENGINE_FASTER_WHISPER

    if engine_type == engine_registry.ENGINE_OPENAI_WHISPER:
        logger.info("[EngineFactory] Loading OpenaiWhisperEngine on %s", unit["name"])
        device_str = "cuda" if unit["type"] == "CUDA" else "cpu"
        return OpenaiWhisperEngine(model_id, device=device_str)

    if engine_type == engine_registry.ENGINE_WHISPERX:
        logger.info("[EngineFactory] Loading WhisperXEngine on %s", unit["name"])
        device_str = "cuda" if unit["type"] == "CUDA" else "cpu"
        return WhisperXEngine(model_id, device=device_str, compute_type=config.COMPUTE_TYPE)

    if engine_type != engine_registry.ENGINE_FASTER_WHISPER:
        supported = ", ".join(engine_registry.supported_engines())
        raise ValueError(f"Unsupported ASR engine '{engine_type}'. Supported values: {supported}")

    # FASTER-WHISPER (CTranslate2)
    logger.info("[EngineFactory] Loading FasterWhisperEngine (CTranslate2) on %s", unit["name"])

    # Resolve target device for Faster-Whisper
    if unit["type"] == "CUDA":
        target_device = "cuda"
    else:
        target_device = "cpu"
        if unit["type"] in ["NPU", "GPU"]:
            logger.info(
                "[EngineFactory] Intel accelerator detected. Faster-Whisper will fall back to CPU for Whisper slot."
            )

    return FasterWhisperEngine(
        model_id,
        device=target_device,
        device_index=unit.get("index", 0),
        compute_type=config.COMPUTE_TYPE,
        cpu_threads=config.ASR_THREADS,
        download_root=config.OV_CACHE_DIR,
    )
