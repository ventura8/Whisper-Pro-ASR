"""OpenAI Whisper engine wrapper."""

import importlib
import logging
from typing import Any, Optional

from modules.inference.engines.base import BaseASREngine, build_inference_info, iter_segment_wrappers

logger = logging.getLogger(__name__)


def _is_multi_candidate(beam_size: Any, best_of: Any) -> bool:
    """Whether these decoding params score more than one candidate per step.

    Either knob alone is enough: whisper's temperature-fallback loop raises ``best_of``
    on its own, so a request that only set ``beam_size=1`` can still reach the
    multi-candidate path.
    """
    return (beam_size or 0) > 1 or (best_of or 0) > 1


class OpenaiWhisperEngine(BaseASREngine):
    """Standard PyTorch openai-whisper engine."""

    def __init__(self, model_id: str, device: str):
        self.whisper = importlib.import_module("whisper")
        self.device = device
        # Download into the persistent model cache; the library's default lives under
        # HOME, which is not a mounted volume, so every container restart would re-fetch
        # multiple gigabytes.
        config = importlib.import_module("modules.core.config")
        download_root = getattr(config, "OPENAI_WHISPER_CACHE_DIR", None)
        self.model = self.whisper.load_model(model_id, device=device, download_root=download_root)

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
    ):
        params = {
            "initial_prompt": initial_prompt,
            "word_timestamps": word_timestamps,
        }
        for key in [
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
            if key in kwargs:
                params[key] = kwargs[key]
        self._clamp_multi_candidate_decoding(params)

        result = self.model.transcribe(audio_path, language=language, task=task, **params)
        return iter_segment_wrappers(result), build_inference_info(result, audio_path, language)

    def _clamp_multi_candidate_decoding(self, params: dict) -> None:
        """Force single-candidate (greedy) decoding on Intel XPU.

        Any decoding path that scores more than one candidate per step -- beam search
        (beam_size > 1) or best-of sampling (best_of > 1, which whisper's automatic
        temperature-fallback loop can trigger even when the request did not ask for it)
        -- reorders the KV cache across the candidate dimension with an index_select.
        On this project's torch 2.13.0+xpu / Intel(R) Graphics [0x7d51], that op does not
        return: beam_size=5 on 3 seconds of silence ran >9 minutes at 99% CPU with no
        result, and the same construct produced 6864 repeated
        "Indexing.cpp: Assertion `srcIndex < srcSelectDimSize` failed" lines in an earlier
        run. A minimal reproduction outside this codebase confirmed encoder, language
        detection, and single-candidate (greedy) decoder steps all complete correctly and
        quickly -- the multi-candidate KV-cache reorder is the specific op that fails.

        Mirrors IntelWhisperEngine._resolve_num_beams, which clamps the same way on
        OpenVINO GPU/NPU for an analogous reason (a batched/remote-tensor op that is
        unimplemented there).
        """
        if self.device != "xpu":
            return
        beam_size = params.get("beam_size")
        best_of = params.get("best_of")
        if not _is_multi_candidate(beam_size, best_of):
            return
        logger.warning(
            "[OpenAI-Whisper] beam_size=%s/best_of=%s is unusable on Intel XPU (KV-cache "
            "index_select over the candidate dimension hangs); forcing greedy decoding.",
            beam_size,
            best_of,
        )
        params["beam_size"] = None
        params["best_of"] = None

    def detect_language(self, audio: Any):
        """Identify language using OpenAI Whisper language head."""
        if isinstance(audio, str):
            audio = self.whisper.load_audio(audio)

        # The encoder takes exactly one 30-second window, so anything shorter has to be
        # padded before the mel is computed -- otherwise detect_language asserts
        # "incorrect audio shape" and fails. Found on an Intel Arc iGPU: gap-fill detects
        # the language of sub-second slices (medians around 0.6s), and every one of those
        # calls failed, so gap-fill silently recovered nothing on this engine while the
        # request still returned 200.
        audio = self.whisper.pad_or_trim(audio)

        # n_mels must match the model's own encoder, not the library's default. The
        # default is 80, but large-v3 (this project's default model) was trained on 128
        # mel bins; the mismatch fails every call with a conv-input-channel shape error,
        # so every request silently paid for a second, full transcribe() just to recover a
        # language guess. Verified on Intel(R) Graphics [0x7d51] with large-v3.
        mel = self.whisper.log_mel_spectrogram(audio, n_mels=self.model.dims.n_mels).to(self.model.device)
        _, probs = self.model.detect_language(mel)
        ordered = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        lang_code, lang_prob = ordered[0]
        return lang_code, float(lang_prob), [(k, float(v)) for k, v in ordered]

    def unload(self) -> None:
        if hasattr(self, "model"):
            del self.model
