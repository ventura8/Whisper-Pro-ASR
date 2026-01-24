"""
Intel Whisper Engine using OpenVINO GenAI

This module provides a Whisper engine implementation optimized for Intel hardware
(NPU/GPU/CPU) using the OpenVINO GenAI pipeline.
"""
# pylint: disable=broad-exception-caught, invalid-name, import-error
# pylint: disable=import-outside-toplevel, too-few-public-methods
import os
import logging
import numpy as np
import openvino_genai as ov_genai
from . import config

logger = logging.getLogger(__name__)


class IntelWhisperEngine:
    """
    ASR Engine using OpenVINO GenAI for Intel hardware acceleration.
    """

    def __init__(self, model_path, device="NPU"):
        self.device = device
        self.model_path = model_path
        self.pipeline = None

        logger.info(
            "[Intel] Initializing OpenVINO GenAI pipeline on %s...", device)
        try:
            # The pipeline handles model loading and hardware compilation.
            self.pipeline = ov_genai.WhisperPipeline(model_path, device)
            logger.info("[Intel] OpenVINO GenAI pipeline loaded successfully.")
        except Exception as e:
            if os.path.exists(model_path):
                logger.error("[Intel] Initialization failed. Path: %s. Content: %s",
                             model_path, os.listdir(model_path))
            logger.error("[Intel] Initialization error details: %s", e)
            raise

    def transcribe(self, audio_data, language=None, task='transcribe', **kwargs):
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
            from . import vad
            logger.debug("[Intel] Input is path, decoding: %s", audio_data)
            audio_data = vad.decode_audio(audio_data)

        # Silence suppression (Matched VAD Filter behavior)
        if kwargs.get('vad_filter', False):
            audio_data = self._apply_vad(audio_data, **kwargs)
            # Fix: If VAD suppressed everything, return empty early to avoid hallucinations
            if np.all(audio_data == 0):
                return {
                    "text": "", "chunks": [],
                    "language": language or "en",
                    "language_probability": 0.0
                }

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

            return self._parse_response(result, language)
        except Exception as e:
            logger.error("[Intel] Single-pass transcription failed: %s", e)
            raise

    def _apply_vad(self, audio_data, **kwargs):
        """Apply Voice Activity Detection to suppress silence."""
        try:
            from . import vad
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
        except Exception as e:
            logger.warning("[Intel] VAD suppression failed: %s", e)
        return audio_data

    def _prepare_gen_config(self, language, task, **kwargs):
        """Prepare WhisperGenerationConfig for inference."""
        try:
            gen_config = self.pipeline.get_generation_config()
        except Exception:
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
            except Exception as e:
                logger.debug("[Intel] Could not set initial_prompt: %s", e)

        return gen_config

    def _resolve_language(self, language, gen_config):
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

    def _sanitize_audio(self, audio_data):
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

    def _parse_response(self, result, requested_language):
        """Convert OpenVINO GenAI result to standard dict."""
        chunks = []
        res_chunks = getattr(result, 'chunks', None)
        if res_chunks is not None:
            for chunk in res_chunks:
                start = getattr(chunk, 'start_ts', getattr(chunk, 'start_time', 0.0))
                end = getattr(chunk, 'end_ts', getattr(chunk, 'end_time', 0.0))
                text = getattr(chunk, 'text', "")
                chunks.append({
                    "text": text,
                    "start": float(start),
                    "end": float(end),
                    "probability": 1.0
                })

        # Extract final text
        text_out = ""
        if hasattr(result, 'texts') and result.texts:
            text_out = result.texts[0].strip()
        elif hasattr(result, 'text'):
            text_out = str(result.text).strip()

        if not text_out and chunks:
            text_out = " ".join([c["text"] for c in chunks]).strip()

        return {
            "text": text_out,
            "chunks": chunks,
            "language": getattr(result, 'language', requested_language or "en"),
            "language_probability": 1.0
        }
