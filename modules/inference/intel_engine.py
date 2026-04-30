"""
Intel Whisper Engine using OpenVINO GenAI

This module provides a Whisper engine implementation optimized for Intel hardware
(NPU/GPU/CPU) using the OpenVINO GenAI pipeline.
"""
import os
import logging
import gc
import importlib
from argparse import Namespace
from typing import Optional, Any, Tuple, List
import numpy as np
from modules import config
from modules.inference import vad

logger = logging.getLogger(__name__)


class IntelWhisperEngine:
    """
    ASR Engine using OpenVINO GenAI for Intel hardware acceleration.
    """

    def __init__(self, model_path: str, device: str = "NPU"):
        self.device = device
        self.model_path = model_path
        self.pipeline = None

        logger.info(
            "[Intel] Initializing OpenVINO GenAI pipeline on %s...", device)
        try:
            # Lazy load the module
            ov_genai = importlib.import_module("openvino_genai")
            # The pipeline handles model loading and hardware compilation.
            self.pipeline = ov_genai.WhisperPipeline(model_path, device)
            logger.info("[Intel] OpenVINO GenAI pipeline loaded successfully.")
        except (RuntimeError, ValueError, ImportError) as e:
            if os.path.exists(model_path):
                logger.error("[Intel] Initialization failed. Path: %s. Content: %s",
                             model_path, os.listdir(model_path))
            logger.error("[Intel] Initialization error details: %s", e)
            raise

    def transcribe(self, audio_data: Any, language: Optional[str] = None,
                   task: str = 'transcribe', **kwargs: Any) -> Tuple[Any, Namespace]:
        """
        Transcribe audio data using the OpenVINO pipeline.

        Parameters:
            audio_data: Numpy array (16kHz, float32) OR string (file path)
            language: ISO 639-1 language code (e.g., 'en')
            task: 'transcribe' or 'translate'
        """
        if self.pipeline is None:
            raise RuntimeError("Intel Whisper pipeline not initialized.")

        # Support direct path input
        if isinstance(audio_data, str):
            logger.debug("[Intel] Input is path, decoding: %s", audio_data)
            audio_data = vad.decode_audio(audio_data)

        # Silence suppression (Matched VAD Filter behavior)
        if kwargs.get('vad_filter', False):
            audio_data = self._apply_vad(audio_data, **kwargs)
            # Fix: If VAD suppressed everything, return empty early to avoid hallucinations
            if np.all(audio_data == 0):
                return (s for s in []), Namespace(
                    language=language or "en",
                    language_probability=0.0,
                    duration=0.0
                )

        # Robust generation configuration retrieval
        gen_config = self._prepare_gen_config(language, task, **kwargs)

        try:
            # --- [TENSOR SANITIZATION] ---
            audio_data = self._sanitize_audio(audio_data)

            # --- [SINGLE PASS INFERENCE] ---
            # Using CPU backend (per config.py) to support Beam Search + Timestamps in one pass.
            logger.info("[Intel] Detecting: Beam: %d, Timestamps: True",
                        gen_config.num_beams)
            result = self.pipeline.generate(audio_data, gen_config)

            segments, info = self._parse_response(result, language)
            return (s for s in segments), info
        except (RuntimeError, ValueError) as e:
            logger.error("[Intel] Single-pass transcription failed: %s", e)
            raise

    def _apply_vad(self, audio_data: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply Voice Activity Detection to suppress silence."""
        try:
            v_threshold = kwargs.get('vad_threshold', 0.35)
            v_min_silence = kwargs.get(
                'min_silence_duration_ms', config.VAD_MIN_SILENCE_DURATION_MS)
            v_pad = kwargs.get('speech_pad_ms', config.VAD_SPEECH_PAD_MS)

            speech_ts = vad.get_speech_timestamps(
                audio_data,
                threshold=v_threshold,
                min_silence_duration_ms=v_min_silence,
                speech_pad_ms=v_pad
            )

            if not speech_ts:
                logger.info("[Intel] VAD found no speech.")
                return np.zeros_like(audio_data)

            # Mask non-speech with zero to preserve temporal context while removing noise
            mask = np.zeros_like(audio_data, dtype=bool)
            for ts in speech_ts:
                start_idx = int(ts['start'] * 16000)
                end_idx = int(ts['end'] * 16000)
                mask[start_idx:end_idx] = True
            audio_data[~mask] = 0.0
            logger.debug(
                "[Intel] VAD suppression applied to %d speech segments.", len(speech_ts))
        except (RuntimeError, ValueError) as e:
            logger.warning("[Intel] VAD suppression failed: %s", e)
        return audio_data

    def _prepare_gen_config(self, language: Optional[str], task: str,
                            **kwargs: Any) -> Any:
        """Prepare WhisperGenerationConfig for inference."""
        ov_genai = importlib.import_module("openvino_genai")
        try:
            gen_config = self.pipeline.get_generation_config()
        except (RuntimeError, AttributeError):
            gen_config = ov_genai.WhisperGenerationConfig()

        # Task resolution
        gen_config.task = task
        supported_tasks = getattr(gen_config, 'task_to_id', {})
        if task not in supported_tasks and supported_tasks:
            logger.debug(
                "[Intel] Task '%s' not in model mapping, defaulting to transcribe.", task)
            gen_config.task = 'transcribe'

        # Language resolution
        if language:
            token = self._resolve_language(language, gen_config)
            if token:
                gen_config.language = token
            else:
                logger.warning(
                    "[Intel] Language '%s' not found in model map. Using auto-detection.", language)

        # Quality & Performance Parameters (Aligned with CPU/CUDA defaults)
        gen_config.num_beams = kwargs.get('beam_size', config.DEFAULT_BEAM_SIZE)
        gen_config.max_new_tokens = kwargs.get('max_new_tokens', 448)
        gen_config.temperature = kwargs.get('temperature', 0.0)
        gen_config.length_penalty = 1.0
        gen_config.return_timestamps = True

        # Support Initial Prompt
        initial_prompt = kwargs.get('initial_prompt', config.INITIAL_PROMPT)
        if initial_prompt:
            try:
                gen_config.initial_prompt = initial_prompt
            except (RuntimeError, AttributeError) as e:
                logger.debug("[Intel] Could not set initial_prompt: %s", e)

        return gen_config

    def _resolve_language(self, language: str,
                          gen_config: Any) -> Optional[str]:
        """Map language code to model tokens."""
        supported_langs = getattr(gen_config, 'lang_to_id', {})
        if not supported_langs:
            return None

        # Try a prioritized matching strategy
        candidates = [language, f"<|{language}|>", f"<|{language.lower()}|>"]
        for cand in candidates:
            if cand in supported_langs:
                return cand

        lang_bare = language.lower().strip("<|>")
        for k in supported_langs.keys():
            if lang_bare in k.lower():
                return k
        return None

    def _sanitize_audio(self, audio_data: Any) -> np.ndarray:
        """Ensure audio tensor meets requirements."""
        if not isinstance(audio_data, np.ndarray):
            audio_data = np.array(audio_data, dtype=np.float32)
        if audio_data.ndim > 1:
            audio_data = audio_data.flatten()
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        if not audio_data.flags.c_contiguous:
            audio_data = np.ascontiguousarray(audio_data)
        return audio_data

    def unload(self) -> None:
        """Release the OpenVINO GenAI pipeline and clear memory."""
        if self.pipeline is not None:
            logger.info("[Intel] Unloading pipeline from %s", self.device)
            self.pipeline = None
            # Force GC to release the C++ resources associated with the pipeline
            gc.collect()

    def detect_language(self, audio_data: Any) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Identify the language of the audio data without full transcription.
        Returns (lang_code, probability, all_probabilities).
        """
        if self.pipeline is None:
            raise RuntimeError("Intel Whisper pipeline not initialized.")

        # Ensure sanitized numpy array
        audio_data = self._sanitize_audio(audio_data)

        # Detect language using the pipeline
        # OpenVINO GenAI WhisperPipeline.generate with a specific config can be used
        # to just get the language token.
        gen_config = self.pipeline.get_generation_config()
        gen_config.max_new_tokens = 1  # Only get the language token
        gen_config.return_timestamps = False

        try:
            result = self.pipeline.generate(audio_data, gen_config)
            lang_code = getattr(result, 'language', "en")
            # We don't get full probs from OV GenAI easily without full inference,
            # so we return 1.0 for the detected one.
            return lang_code, 1.0, [(lang_code, 1.0)]
        except (RuntimeError, ValueError) as e:
            logger.error("[Intel] Language detection failed: %s", e)
            # Fallback
            return "en", 0.0, [("en", 0.0)]

    def _parse_response(self, result: Any,
                        requested_language: Optional[str]) -> Tuple[List[Namespace], Namespace]:
        """Convert OpenVINO GenAI result to standard (segments, info) tuple."""
        segments = []
        res_segments = getattr(result, 'chunks', None)
        if res_segments is not None:
            for segment in res_segments:
                start = getattr(segment, 'start_ts', getattr(segment, 'start_time', 0.0))
                end = getattr(segment, 'end_ts', getattr(segment, 'end_time', 0.0))
                text = getattr(segment, 'text', "")
                # Create a mock segment object
                segments.append(Namespace(
                    text=text,
                    start=float(start),
                    end=float(end)
                ))

        # Extract final text and language info
        info = Namespace(
            language=getattr(result, 'language', requested_language or "en"),
            language_probability=1.0,
            duration=0.0  # Placeholder
        )
        return segments, info
