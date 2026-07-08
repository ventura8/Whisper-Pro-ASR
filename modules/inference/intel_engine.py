"""
Intel Whisper Engine using OpenVINO GenAI

This module provides a Whisper engine implementation optimized for Intel hardware
(NPU/GPU/CPU) using the OpenVINO GenAI pipeline.
"""

import gc
import importlib
import logging
import os
from argparse import Namespace
from typing import Any, List, Optional, Tuple

import numpy as np

from modules.core import config, utils
from modules.inference import vad

logger = logging.getLogger(__name__)


def find_split_points(audio_len_sec: float, speech_ts: List[dict], target_chunk_len: float = 300.0) -> List[float]:
    """
    Find optimal split points at silent intervals (gaps between speech segments)
    closest to target_chunk_len to avoid splitting words.
    """
    split_points = [0.0]
    gaps = []
    if not speech_ts:
        gaps.append((0.0, audio_len_sec))
    else:
        if speech_ts[0]["start"] > 0:
            gaps.append((0.0, speech_ts[0]["start"]))
        for i in range(len(speech_ts) - 1):
            if speech_ts[i + 1]["start"] > speech_ts[i]["end"]:
                gaps.append((speech_ts[i]["end"], speech_ts[i + 1]["start"]))
        if speech_ts[-1]["end"] < audio_len_sec:
            gaps.append((speech_ts[-1]["end"], audio_len_sec))

    while split_points[-1] < audio_len_sec:
        current_start = split_points[-1]
        target_end = current_start + target_chunk_len
        if target_end >= audio_len_sec - 10.0:
            split_points.append(audio_len_sec)
            break

        window_start = target_end - 30.0
        window_end = target_end + 30.0

        candidates = []
        for g_start, g_end in gaps:
            mid = (g_start + g_end) / 2.0
            if window_start <= mid <= window_end:
                candidates.append(mid)

        if candidates:
            best_mid = min(candidates, key=lambda x: abs(x - target_end))
            split_points.append(best_mid)
        else:
            # Fallback to wider window [target_end - 60, target_end + 60]
            window_start = target_end - 60.0
            window_end = target_end + 60.0
            for g_start, g_end in gaps:
                mid = (g_start + g_end) / 2.0
                if window_start <= mid <= window_end:
                    candidates.append(mid)
            if candidates:
                best_mid = min(candidates, key=lambda x: abs(x - target_end))
                split_points.append(best_mid)
            else:
                split_points.append(target_end)

    return split_points


class IntelWhisperEngine:
    """
    ASR Engine using OpenVINO GenAI for Intel hardware acceleration.
    """

    def __init__(self, model_path: str, device: str = "NPU"):
        self.device = device
        self.model_path = model_path
        self.pipeline = None

        logger.info("[Intel] Initializing OpenVINO GenAI pipeline on %s...", device)
        try:
            # Lazy load the module
            ov_genai = importlib.import_module("openvino_genai")
            # The pipeline handles model loading and hardware compilation.
            self.pipeline = ov_genai.WhisperPipeline(model_path, device)
            logger.info("[Intel] OpenVINO GenAI pipeline loaded successfully.")
        except (RuntimeError, ValueError, ImportError) as e:
            if os.path.isdir(model_path):
                logger.error("[Intel] Initialization failed. Path: %s. Content: %s", model_path, os.listdir(model_path))
            logger.error("[Intel] Initialization error details: %s", e)
            raise

    def transcribe(
        self, audio_data: Any, language: Optional[str] = None, task: str = "transcribe", **kwargs: Any
    ) -> Tuple[Any, Namespace]:
        """
        Transcribe audio data using the OpenVINO pipeline in a chunked manner
        to support incremental progress logging, live updates, and status checks.
        """
        if self.pipeline is None:
            raise RuntimeError("Intel Whisper pipeline not initialized.")

        # Support direct path input
        if isinstance(audio_data, str):
            logger.debug("[Intel] Input is path, decoding: %s", audio_data)
            audio_data = vad.decode_audio(audio_data)

        audio_data = self.sanitize_audio(audio_data)
        total_duration = len(audio_data) / 16000.0

        # Get speech timestamps for VAD filtering and split points
        speech_ts = []
        try:
            speech_ts = vad.get_speech_timestamps(
                audio_data,
                threshold=kwargs.get("vad_threshold", 0.35),
                min_silence_duration_ms=kwargs.get("min_silence_duration_ms", config.VAD_MIN_SILENCE_DURATION_MS),
                speech_pad_ms=kwargs.get("speech_pad_ms", config.VAD_SPEECH_PAD_MS),
            )
        except (RuntimeError, ValueError) as e:
            logger.warning("[Intel] VAD detection failed: %s", e)

        # Silence suppression (Matched VAD Filter behavior)
        if kwargs.get("vad_filter", False):
            if speech_ts:
                # Mask non-speech with zero to preserve temporal context while removing noise
                mask = np.zeros_like(audio_data, dtype=bool)
                for ts_item in speech_ts:
                    mask[int(ts_item["start"] * 16000) : int(ts_item["end"] * 16000)] = True
                audio_data[~mask] = 0.0
                logger.debug("[Intel] VAD suppression applied to %d speech segments.", len(speech_ts))
            else:
                logger.info("[Intel] VAD found no speech.")
                return (s for s in []), Namespace(language=language or "en", language_probability=0.0, duration=0.0)

        # Robust generation configuration retrieval
        gen_config = self.prepare_gen_config(language, task, **kwargs)

        # Construct the immediate info Namespace. If language is auto-detected (None),
        # we will populate it dynamically once the first chunk completes.
        info = Namespace(language=language or "en", language_probability=1.0, duration=total_duration)

        # Find split points for chunking
        split_points = find_split_points(total_duration, speech_ts, float(config.INTEL_ASR_CHUNK_DURATION))
        total_chunks = len(split_points) - 1

        def segment_generator():
            for i in range(total_chunks):
                chunk_start = split_points[i]
                chunk_end = split_points[i + 1]
                start_idx = int(chunk_start * 16000)
                end_idx = int(chunk_end * 16000)
                chunk_audio = audio_data[start_idx:end_idx]

                # Skip completely silent/zeroed-out chunks if VAD is active
                if kwargs.get("vad_filter", False) and np.all(chunk_audio == 0.0):
                    continue

                logger.info(
                    "[Intel] Transcribing chunk %d/%d (Audio range: %s - %s)...",
                    i + 1,
                    total_chunks,
                    utils.format_duration(chunk_start),
                    utils.format_duration(chunk_end),
                )

                try:
                    logger.info("[Intel] Detecting: Beam: %d, Timestamps: True", gen_config.num_beams)
                    result = self.pipeline.generate(chunk_audio, gen_config)
                    res_segments, res_info = self._parse_response(result, info.language)

                    # Update and lock language dynamically if auto-detected on the first chunk
                    if i == 0 and not language:
                        detected_lang = res_info.language
                        info.language = detected_lang
                        token = self.resolve_language(detected_lang, gen_config)
                        if token:
                            gen_config.language = token

                    for seg in res_segments:
                        yield Namespace(text=seg.text, start=seg.start + chunk_start, end=seg.end + chunk_start)
                except (RuntimeError, ValueError) as e:
                    logger.error("[Intel] Chunk transcription failed: %s", e)
                    raise

        return segment_generator(), info

    def apply_vad(self, audio_data: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply Voice Activity Detection to suppress silence."""
        try:
            v_threshold = kwargs.get("vad_threshold", 0.35)
            v_min_silence = kwargs.get("min_silence_duration_ms", config.VAD_MIN_SILENCE_DURATION_MS)
            v_pad = kwargs.get("speech_pad_ms", config.VAD_SPEECH_PAD_MS)

            speech_ts = vad.get_speech_timestamps(
                audio_data, threshold=v_threshold, min_silence_duration_ms=v_min_silence, speech_pad_ms=v_pad
            )

            if not speech_ts:
                logger.info("[Intel] VAD found no speech.")
                return np.zeros_like(audio_data)

            # Mask non-speech with zero to preserve temporal context while removing noise
            mask = np.zeros_like(audio_data, dtype=bool)
            for ts in speech_ts:
                start_idx = int(ts["start"] * 16000)
                end_idx = int(ts["end"] * 16000)
                mask[start_idx:end_idx] = True
            audio_data[~mask] = 0.0
            logger.debug("[Intel] VAD suppression applied to %d speech segments.", len(speech_ts))
        except (RuntimeError, ValueError) as e:
            logger.warning("[Intel] VAD suppression failed: %s", e)
        return audio_data

    def prepare_gen_config(self, language: Optional[str], task: str, **kwargs: Any) -> Any:
        """Prepare WhisperGenerationConfig for inference."""
        ov_genai = importlib.import_module("openvino_genai")
        try:
            gen_config = self.pipeline.get_generation_config()
        except (RuntimeError, AttributeError):
            gen_config = ov_genai.WhisperGenerationConfig()

        # Task resolution
        gen_config.task = task
        supported_tasks = getattr(gen_config, "task_to_id", {})
        if task not in supported_tasks and supported_tasks:
            logger.debug("[Intel] Task '%s' not in model mapping, defaulting to transcribe.", task)
            gen_config.task = "transcribe"

        # Language resolution
        if language:
            token = self.resolve_language(language, gen_config)
            if token:
                gen_config.language = token
            else:
                logger.warning("[Intel] Language '%s' not found in model map. Using auto-detection.", language)

        # Quality & Performance Parameters (Aligned with CPU/CUDA defaults)
        gen_config.num_beams = kwargs.get("beam_size", config.DEFAULT_BEAM_SIZE)
        gen_config.max_new_tokens = kwargs.get("max_new_tokens", 448)
        gen_config.temperature = kwargs.get("temperature", 0.0)
        gen_config.length_penalty = 1.0
        gen_config.return_timestamps = True

        # Support Initial Prompt
        initial_prompt = kwargs.get("initial_prompt", config.INITIAL_PROMPT)
        if initial_prompt:
            try:
                gen_config.initial_prompt = initial_prompt
            except (RuntimeError, AttributeError) as e:
                logger.debug("[Intel] Could not set initial_prompt: %s", e)

        return gen_config

    def resolve_language(self, language: str, gen_config: Any) -> Optional[str]:
        """Map language code to model tokens."""
        supported_langs = getattr(gen_config, "lang_to_id", {})
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

    def sanitize_audio(self, audio_data: Any) -> np.ndarray:
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
        audio_data = self.sanitize_audio(audio_data)

        # Detect language using the pipeline
        # OpenVINO GenAI WhisperPipeline.generate with a specific config can be used
        # to just get the language token.
        gen_config = self.pipeline.get_generation_config()
        gen_config.max_new_tokens = 1  # Only get the language token
        gen_config.return_timestamps = False

        try:
            result = self.pipeline.generate(audio_data, gen_config)
            lang_code = getattr(result, "language", "en")
            # We don't get full probs from OV GenAI easily without full inference,
            # so we return 1.0 for the detected one.
            return lang_code, 1.0, [(lang_code, 1.0)]
        except (RuntimeError, ValueError) as e:
            logger.error("[Intel] Language detection failed: %s", e)
            # Fallback
            return "en", 0.0, [("en", 0.0)]

    def _parse_response(self, result: Any, requested_language: Optional[str]) -> Tuple[List[Namespace], Namespace]:
        """Convert OpenVINO GenAI result to standard (segments, info) tuple."""
        segments = []
        res_segments = getattr(result, "chunks", None)
        if res_segments is not None:
            for segment in res_segments:
                start = getattr(segment, "start_ts", getattr(segment, "start_time", 0.0))
                end = getattr(segment, "end_ts", getattr(segment, "end_time", 0.0))
                text = getattr(segment, "text", "")
                # Create a mock segment object
                segments.append(Namespace(text=text, start=float(start), end=float(end)))

        # Extract final text and language info
        info = Namespace(
            language=getattr(result, "language", requested_language or "en"),
            language_probability=1.0,
            duration=0.0,  # Placeholder
        )
        return segments, info
