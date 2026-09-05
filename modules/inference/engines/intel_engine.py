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

from modules.core import config, model_integrity, model_provisioning, utils
from modules.inference.pipeline import vad

logger = logging.getLogger(__name__)
_VAD_FAILURE_SENTINEL = object()


def find_split_points(audio_len_sec: float, speech_ts: List[dict], target_chunk_len: float = 300.0) -> List[float]:
    """
    Find optimal split points at silent intervals (gaps between speech segments)
    closest to target_chunk_len to avoid splitting words.
    """
    split_points = [0.0]
    gaps = _build_speech_gaps(audio_len_sec, speech_ts)

    while split_points[-1] < audio_len_sec:
        target_end = split_points[-1] + target_chunk_len
        if target_end >= audio_len_sec - 10.0:
            split_points.append(audio_len_sec)
            break
        split_points.append(_resolve_next_split_point(gaps, target_end))

    return split_points


def _build_speech_gaps(audio_len_sec: float, speech_ts: List[dict]) -> List[tuple]:
    if not speech_ts:
        return [(0.0, audio_len_sec)]
    gaps = _leading_speech_gap(speech_ts)
    gaps.extend(_intermediate_speech_gaps(speech_ts))
    gaps.extend(_trailing_speech_gap(speech_ts, audio_len_sec))
    return gaps


def _leading_speech_gap(speech_ts: List[dict]) -> List[tuple]:
    if speech_ts[0]["start"] > 0:
        return [(0.0, speech_ts[0]["start"])]
    return []


def _intermediate_speech_gaps(speech_ts: List[dict]) -> List[tuple]:
    gaps = []
    for i in range(len(speech_ts) - 1):
        if speech_ts[i + 1]["start"] > speech_ts[i]["end"]:
            gaps.append((speech_ts[i]["end"], speech_ts[i + 1]["start"]))
    return gaps


def _trailing_speech_gap(speech_ts: List[dict], audio_len_sec: float) -> List[tuple]:
    if speech_ts[-1]["end"] < audio_len_sec:
        return [(speech_ts[-1]["end"], audio_len_sec)]
    return []


def _resolve_next_split_point(gaps: List[tuple], target_end: float) -> float:
    candidates = _gap_midpoints_in_window(gaps, target_end, 30.0)
    if candidates:
        return _closest_midpoint(candidates, target_end)
    candidates = _gap_midpoints_in_window(gaps, target_end, 60.0)
    if candidates:
        return _closest_midpoint(candidates, target_end)
    return target_end


def _should_return_empty_vad_result(kwargs, speech_ts) -> bool:
    if not kwargs.get("vad_filter", False):
        return False
    if speech_ts is _VAD_FAILURE_SENTINEL:
        return False
    return not speech_ts


def _empty_vad_result(language):
    return (s for s in []), Namespace(language=language or "en", language_probability=0.0, duration=0.0)


def _gap_midpoints_in_window(gaps: List[tuple], target_end: float, window_half: float) -> List[float]:
    window_start = target_end - window_half
    window_end = target_end + window_half
    return [((g_start + g_end) / 2.0) for g_start, g_end in gaps if window_start <= ((g_start + g_end) / 2.0) <= window_end]


def _closest_midpoint(candidates: List[float], target_end: float) -> float:
    return min(candidates, key=lambda x: abs(x - target_end))


def _init_intel_pipeline(model_path: str, device: str):
    ov_genai = importlib.import_module("openvino_genai")
    return ov_genai.WhisperPipeline(model_path, device)


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
            self._load_pipeline(model_path, device)
        except (RuntimeError, ValueError, ImportError, OSError) as e:
            self._log_init_error(model_path, e)
            if not self._maybe_retry_pipeline(model_path, device):
                raise
        self._verify_device_executes(model_path)

    def _verify_device_executes(self, model_path: str) -> None:
        """Prove the device can actually run inference, not merely build a pipeline.

        The NPU builds a WhisperPipeline happily and then fails on every generate() with
        `L0 pfnAppendGraphExecute ... ZE_RESULT_ERROR_UNKNOWN`, because the exported IR is
        dynamic-shaped ('[?,?,1280]') and the NPU plugin needs static upper bounds. Without
        this check the service starts healthy, reports 'ASR Runtime: OpenVINO (NPU)', and
        returns HTTP 500 for every request -- a broken service that looks configured.

        Scoped to the NPU deliberately: the GPU path is exercised constantly and does not
        need a warmup tax on every start.
        """
        if not getattr(config, "VERIFY_RUNTIME", True):
            return
        if not self.device.upper().startswith("NPU"):
            return

        try:
            self.pipeline.generate(np.zeros(16000, dtype=np.float32))
            logger.info("[Intel] NPU verified: a warmup inference completed.")
            return
        except (RuntimeError, ValueError) as e:
            detail = str(e).replace("\n", " ")[:200]

        logger.error(
            "[Intel] The NPU built the pipeline but cannot execute it: %s. This model's "
            "OpenVINO IR is dynamic-shaped and the NPU plugin requires static upper bounds.",
            detail,
        )
        if not self._fallback_to_cpu(model_path):
            raise RuntimeError(
                "Intel NPU cannot execute this model (dynamic input shapes) and the CPU "
                "fallback could not be loaded either. Re-run with ASR_DEVICE=GPU."
            )

    def _fallback_to_cpu(self, model_path: str) -> bool:
        """Move ASR to the CPU rather than serve 500s, and say so loudly.

        CPU, not the iGPU, and deliberately so. This engine is serving an NPU unit on a
        machine whose GPU unit is already running ASR; sending this one to the GPU too
        would put both units on one device and serialise them. The NPU keeps doing what it
        is good for -- UVR preprocessing -- while ASR for this unit runs on the CPU, so the
        two units genuinely proceed in parallel.

        A silent fallback is the failure mode this project keeps getting bitten by, so this
        logs at ERROR and rewrites self.device, and the status then reports what really ran.
        """
        try:
            self._load_pipeline(model_path, "CPU")
        except (RuntimeError, ValueError, ImportError, OSError) as e:
            logger.error("[Intel] Fallback to CPU also failed: %s", e)
            return False
        self.device = "CPU"
        logger.error(
            "[Intel] ASR FALLING BACK to the CPU for this NPU unit; the NPU cannot execute "
            "this model. The NPU remains available for UVR preprocessing."
        )
        return True

    def _maybe_retry_pipeline(self, model_path: str, device: str) -> bool:
        if not self._restore_corrupted_model(model_path):
            return False
        try:
            self._load_pipeline(model_path, device)
            return True
        except (RuntimeError, ValueError, ImportError, OSError) as retry_error:
            logger.error("[Intel] Retry after corrupted-model purge failed: %s", retry_error)
        return False

    @staticmethod
    def _restore_corrupted_model(model_path: str) -> bool:
        """Purge a corrupted OpenVINO directory and fetch the IR again. False if it cannot.

        Purging deletes the directory, so a retry that skipped the re-provision could only
        fail with "path does not exist" -- a bounded retry guaranteed to spend an attempt and
        bury the real initialization error under a less informative one.
        """
        if not os.path.isdir(model_path) or model_integrity.verify_openvino_model_dir(model_path):
            return False
        if not model_integrity.purge_corrupted_path(model_path, description=f"OpenVINO model ({model_path})"):
            return False
        if model_provisioning.ensure_openvino_whisper(model_path):
            return True
        logger.error("[Intel] Could not re-provision the OpenVINO IR after purging %s.", model_path)
        return False

    def _load_pipeline(self, model_path: str, device: str):
        self.pipeline = _init_intel_pipeline(model_path, device)
        logger.info("[Intel] OpenVINO GenAI pipeline loaded successfully.")

    def _log_init_error(self, model_path: str, e: Exception):
        if os.path.isdir(model_path):
            logger.error("[Intel] Initialization failed. Path: %s. Content: %s", model_path, os.listdir(model_path))
        logger.error("[Intel] Initialization error details: %s", e)

    def transcribe(self, audio_data: Any, language: Optional[str] = None, task: str = "transcribe", **kwargs: Any) -> Tuple[Any, Namespace]:
        """
        Transcribe audio data using the OpenVINO pipeline in a chunked manner
        to support incremental progress logging, live updates, and status checks.
        """
        if self.pipeline is None:
            raise RuntimeError("Intel Whisper pipeline not initialized.")

        audio_data = self._prepare_transcription_audio(audio_data)
        total_duration = len(audio_data) / 16000.0
        speech_ts = self._get_speech_ts_safe(audio_data, **kwargs)
        speech_ts_for_split = [] if speech_ts is _VAD_FAILURE_SENTINEL else speech_ts
        audio_data = self._apply_vad_filter_if_requested(audio_data, speech_ts, **kwargs)
        empty_result = self._empty_result_if_no_vad_speech(kwargs, speech_ts, language)
        if empty_result is not None:
            return empty_result

        gen_config = self.prepare_gen_config(language, task, **kwargs)
        info = Namespace(language=language or "en", language_probability=1.0, duration=total_duration)
        split_points = find_split_points(total_duration, speech_ts_for_split, float(config.INTEL_ASR_CHUNK_DURATION))
        total_chunks = len(split_points) - 1
        generator = self._build_segment_generator(
            audio_data,
            split_points,
            total_chunks,
            gen_config=gen_config,
            info=info,
            language=language,
            **kwargs,
        )
        return generator, info

    def _empty_result_if_no_vad_speech(self, kwargs, speech_ts, language):
        if not _should_return_empty_vad_result(kwargs, speech_ts):
            return None
        return _empty_vad_result(language)

    def _prepare_transcription_audio(self, audio_data: Any):
        if isinstance(audio_data, str):
            logger.debug("[Intel] Input is path, decoding: %s", audio_data)
            audio_data = vad.decode_audio(audio_data)
        return IntelWhisperEngine.sanitize_audio(self, audio_data)

    def _get_speech_ts_safe(self, audio_data, **kwargs) -> list:
        try:
            return vad.get_speech_timestamps(
                audio_data,
                threshold=kwargs.get("vad_threshold", 0.35),
                min_silence_duration_ms=kwargs.get("min_silence_duration_ms", config.VAD_MIN_SILENCE_DURATION_MS),
                speech_pad_ms=kwargs.get("speech_pad_ms", config.VAD_SPEECH_PAD_MS),
            )
        except (RuntimeError, ValueError) as e:
            logger.warning("[Intel] VAD detection failed: %s", e)
            return _VAD_FAILURE_SENTINEL

    def _apply_vad_filter_if_requested(self, audio_data, speech_ts, **kwargs):
        if kwargs.get("vad_filter", False) and speech_ts:
            audio_data = self._apply_vad_mask(audio_data, speech_ts)
            logger.debug("[Intel] VAD suppression applied to %d speech segments.", len(speech_ts))
        return audio_data

    def _lock_first_chunk_language(self, detected_lang, gen_config, info):
        info.language = detected_lang
        token = self.resolve_language(detected_lang, gen_config)
        if token:
            gen_config.language = token

    def _build_segment_generator(self, audio_data, split_points, total_chunks, *, gen_config, info, language, **kwargs):
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
                        self._lock_first_chunk_language(res_info.language, gen_config, info)

                    for seg in res_segments:
                        yield Namespace(text=seg.text, start=seg.start + chunk_start, end=seg.end + chunk_start)
                except (RuntimeError, ValueError) as e:
                    logger.error("[Intel] Chunk transcription failed: %s", e)
                    raise

        return segment_generator()

    def _apply_vad_mask(self, audio_data: np.ndarray, speech_ts: List[dict]) -> np.ndarray:
        """Create a boolean mask from speech timestamps and zero out non-speech.

        Returns a new masked audio array, leaving the original audio_data unchanged.
        """
        # Work on a copy to avoid mutating the caller's buffer
        masked_audio = np.copy(audio_data)
        mask = np.zeros_like(audio_data, dtype=bool)
        for ts in speech_ts:
            start_idx = int(ts["start"] * 16000)
            end_idx = int(ts["end"] * 16000)
            mask[start_idx:end_idx] = True
        masked_audio[~mask] = 0.0
        return masked_audio

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

            # Use shared helper for mask creation
            audio_data = self._apply_vad_mask(audio_data, speech_ts)
            logger.debug("[Intel] VAD suppression applied to %d speech segments.", len(speech_ts))
        except (RuntimeError, ValueError) as e:
            logger.warning("[Intel] VAD suppression failed: %s", e)
        return audio_data

    def prepare_gen_config(self, language: Optional[str], task: str, **kwargs: Any) -> Any:
        """Prepare WhisperGenerationConfig for inference."""
        gen_config = self._load_generation_config()
        self._resolve_generation_task(gen_config, task)
        self._resolve_generation_language(gen_config, language)
        self._apply_generation_quality_params(gen_config, **kwargs)
        self._apply_initial_prompt(gen_config, kwargs.get("initial_prompt", config.INITIAL_PROMPT))
        return gen_config

    def _load_generation_config(self) -> Any:
        ov_genai = importlib.import_module("openvino_genai")
        try:
            return self.pipeline.get_generation_config()
        except (RuntimeError, AttributeError):
            return ov_genai.WhisperGenerationConfig()

    def _resolve_generation_task(self, gen_config: Any, task: str):
        gen_config.task = task
        supported_tasks = getattr(gen_config, "task_to_id", {})
        if task not in supported_tasks and supported_tasks:
            logger.debug("[Intel] Task '%s' not in model mapping, defaulting to transcribe.", task)
            gen_config.task = "transcribe"

    def _resolve_generation_language(self, gen_config: Any, language: Optional[str]):
        if not language:
            return
        token = self.resolve_language(language, gen_config)
        if token:
            gen_config.language = token
            return
        logger.warning("[Intel] Language '%s' not found in model map. Using auto-detection.", language)

    def _resolve_num_beams(self, requested: int) -> int:
        """Clamp beam search to greedy on Intel GPU/NPU devices.

        OpenVINO GenAI's WhisperPipeline cannot run beam search on a non-CPU device:
        the beam scorer reads logits through ov::Tensor::data(), which is unimplemented
        for the remote tensors a GPU/NPU pipeline produces. It surfaces as
        "iremote_tensor.hpp:32: Not Implemented" (or, with timestamps enabled, as a beam
        vs. logits batch-size mismatch). Greedy decoding is the only working mode, and
        it supports timestamps. Verified on openvino-genai 2026.3.1 / Intel UHD Graphics.
        """
        if requested <= 1 or self.device.upper().startswith("CPU"):
            return requested
        logger.warning(
            "[Intel] Beam search (beam_size=%d) is unsupported by OpenVINO GenAI on %s; falling back to greedy decoding.",
            requested,
            self.device,
        )
        return 1

    def _apply_generation_quality_params(self, gen_config: Any, **kwargs):
        gen_config.num_beams = self._resolve_num_beams(kwargs.get("beam_size", config.DEFAULT_BEAM_SIZE))
        gen_config.max_new_tokens = kwargs.get("max_new_tokens", 448)
        gen_config.temperature = kwargs.get("temperature", 0.0)
        gen_config.length_penalty = 1.0
        gen_config.return_timestamps = True
        # Break runaway decoding. Without this the long-form clip produced 15 consecutive
        # repeats of 'el lujo de la luna' -- a phrase absent from the ground truth -- all
        # inside one chunk, 357s to 373s, the same five words back to back.
        #
        # repetition_penalty, NOT no_repeat_ngram_size. WhisperGenerationConfig exposes
        # no_repeat_ngram_size and accepts the assignment (it reads back as 3, from a
        # default of 2**64-1), but OpenVINO GenAI's WhisperPipeline does not act on it:
        # setting it left the loop at exactly 15 repeats, unchanged. Its presence on the
        # config object makes it look supported, so this comment is the only warning.
        # repetition_penalty is honoured -- the same run dropped to 10 repeats of a real
        # fixture sentence, within the 12 the timeline itself contains.
        #
        # Greedy decoding only. OpenVINO GenAI validates the two against each other and
        # rejects the combination outright:
        #
        #     'repetition_penalty' is not currently supported by beam search and should be
        #     1.0f, but got 1.15
        #
        # which is a hard RuntimeError on every request, not a warning. It never showed on
        # an Intel GPU or NPU because _resolve_num_beams already clamps those to greedy;
        # the CPU path keeps beam search, so it surfaced only there -- including the
        # NPU-cannot-execute fallback, which rewrites self.device to CPU and thereby
        # re-enables beam search on a config built for greedy. Verified on the Intel NUC.
        #
        # Beam search is the better accuracy trade where it is available, so it wins here
        # and the loop guard applies to the greedy paths, which is where the runaway
        # decoding was actually recorded.
        if gen_config.num_beams <= 1:
            gen_config.repetition_penalty = kwargs.get("repetition_penalty", 1.15)

    def _apply_initial_prompt(self, gen_config: Any, initial_prompt: Optional[str]):
        if not initial_prompt:
            return
        try:
            gen_config.initial_prompt = initial_prompt
        except (RuntimeError, AttributeError) as e:
            logger.debug("[Intel] Could not set initial_prompt: %s", e)

    def resolve_language(self, language: str, gen_config: Any) -> Optional[str]:
        """Map language code to model tokens."""
        supported_langs = getattr(gen_config, "lang_to_id", {})
        if not supported_langs:
            return None
        token = _resolve_language_token_exact(language, supported_langs)
        if token:
            return token
        return _resolve_language_token_partial(language, supported_langs)

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


def _resolve_language_token_exact(language: str, supported_langs: dict) -> Optional[str]:
    candidates = [language, f"<|{language}|>", f"<|{language.lower()}|>"]
    for cand in candidates:
        if cand in supported_langs:
            return cand
    return None


def _resolve_language_token_partial(language: str, supported_langs: dict) -> Optional[str]:
    lang_bare = language.lower().strip("<|>")
    for key in supported_langs.keys():
        if lang_bare in key.lower():
            return key
    return None
