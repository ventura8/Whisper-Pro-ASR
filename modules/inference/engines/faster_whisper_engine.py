"""CTranslate2 faster-whisper engine wrapper."""

import importlib
import logging
import os
from typing import Any, Optional

from modules.core import config, model_integrity
from modules.inference.engines.base import BaseASREngine

logger = logging.getLogger(__name__)


def _init_whisper_model(
    model_id: str,
    *,
    device: str,
    device_index: int,
    compute_type: str,
    cpu_threads: int,
    download_root: Optional[str],
):
    faster_whisper = importlib.import_module("faster_whisper")
    return faster_whisper.WhisperModel(
        model_id,
        device=device,
        device_index=device_index,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
        download_root=download_root,
    )


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
        try:
            self.model = _init_whisper_model(
                model_id,
                device=device,
                device_index=device_index,
                compute_type=compute_type,
                cpu_threads=cpu_threads,
                download_root=download_root,
            )
        except (RuntimeError, ValueError, OSError, EOFError) as first_err:
            if not self._maybe_retry_after_purge(
                model_id,
                first_err,
                device=device,
                device_index=device_index,
                compute_type=compute_type,
                cpu_threads=cpu_threads,
                download_root=download_root,
            ):
                raise

    def _maybe_retry_after_purge(
        self,
        model_id: str,
        first_err: Exception,
        *,
        device: str,
        device_index: int,
        compute_type: str,
        cpu_threads: int,
        download_root: Optional[str],
    ) -> bool:
        if os.path.isdir(model_id):
            self._purge_corrupted_local_dir(model_id, first_err)
            return False

        return self._try_retry_cached_model(
            model_id,
            first_err,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            download_root=download_root,
        )

    @staticmethod
    def _is_foreign_model_dir(path: str) -> bool:
        """True when the directory holds another engine's weights.

        A failed CTranslate2 load is not evidence that the directory is corrupt: it may
        simply belong to a different runtime. Purging an OpenVINO IR here would delete a
        valid model the Intel engine still needs.
        """
        if model_integrity.verify_openvino_model_dir(path):
            logger.warning(
                "[FasterWhisper] Refusing to purge %s: it is a valid OpenVINO model directory, not a corrupt CTranslate2 one.",
                path,
            )
            return True
        return False

    def _purge_corrupted_local_dir(self, model_id: str, first_err: Exception):
        if self._is_foreign_model_dir(model_id):
            return
        if not model_integrity.verify_ct2_model_dir(model_id):
            logger.warning(
                "[FasterWhisper] Corrupted local model dir detected (%s, error: %s). Purging...",
                model_id,
                first_err,
            )
            model_integrity.purge_corrupted_path(model_id, description=f"Faster-Whisper model path ({model_id})")

    def _find_latest_snapshot(self, snapshots_dir: str) -> Optional[str]:
        if not os.path.isdir(snapshots_dir):
            return None
        snapshots = [f.path for f in os.scandir(snapshots_dir) if f.is_dir()]
        return max(snapshots, key=os.path.getmtime, default=None)

    def _resolve_hf_snapshot_dir(self, model_id: str, download_root: Optional[str]) -> Optional[str]:
        if not download_root:
            return None
        direct_path = os.path.join(download_root, model_id)
        if os.path.isdir(direct_path):
            return direct_path
        hf_repo_name = f"models--{model_id.replace('/', '--')}"
        snapshots_dir = os.path.join(download_root, hf_repo_name, "snapshots")
        return self._find_latest_snapshot(snapshots_dir) or direct_path

    def _try_retry_cached_model(
        self,
        model_id: str,
        first_err: Exception,
        *,
        device: str,
        device_index: int,
        compute_type: str,
        cpu_threads: int,
        download_root: Optional[str],
    ) -> bool:
        target_dir = self._resolve_hf_snapshot_dir(model_id, download_root)
        if not (target_dir and os.path.isdir(target_dir) and not model_integrity.verify_ct2_model_dir(target_dir)):
            return False

        if self._is_foreign_model_dir(target_dir):
            return False

        logger.warning(
            "[FasterWhisper] Corrupted cached model dir detected (%s, error: %s). Purging...",
            target_dir,
            first_err,
        )
        model_integrity.purge_corrupted_path(target_dir, description=f"Faster-Whisper model path ({target_dir})")
        return self._reload_whisper_model_safe(
            model_id,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            download_root=download_root,
        )

    def _reload_whisper_model_safe(
        self,
        model_id: str,
        *,
        device: str,
        device_index: int,
        compute_type: str,
        cpu_threads: int,
        download_root: Optional[str],
    ) -> bool:
        try:
            self.model = _init_whisper_model(
                model_id,
                device=device,
                device_index=device_index,
                compute_type=compute_type,
                cpu_threads=cpu_threads,
                download_root=download_root,
            )
            logger.info("[FasterWhisper] Model successfully recovered and reloaded on retry.")

            return True
        except (RuntimeError, ValueError, OSError, EOFError) as retry_err:
            logger.error("[FasterWhisper] Retry after purge failed: %s", retry_err)
            return False

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
            "beam_size": config.DEFAULT_BEAM_SIZE,
            "initial_prompt": initial_prompt,
            "vad_filter": vad_filter,
            "vad_parameters": {
                "min_silence_duration_ms": config.VAD_MIN_SILENCE_DURATION_MS,
                "threshold": config.VAD_THRESHOLD,
            },
            "word_timestamps": word_timestamps,
            # Defends against the recorded long-form decoder-loop defect ("el veloz zorro
            # marron salta sobre el perro perezoso" x18; verified 2026-09-01 on RTX 3080).
            # faster-whisper defaults to condition_on_previous_text=True, which is exactly
            # the mechanism that sustains a loop: once a chunk degenerates, its own
            # repeated output becomes the prompt for the next chunk, so the model keeps
            # regenerating it. False breaks that feedback path -- WhisperX's own internal
            # defaults do the same, which is consistent with it not showing this defect.
            # no_repeat_ngram_size is CTranslate2-native n-gram blocking: it stops the
            # exact same run of tokens (a full sentence, here) from being reproduced
            # verbatim inside one chunk, which condition_on_previous_text=False alone does
            # not prevent. Both are overridable via kwargs for a caller that wants the
            # library defaults back.
            "condition_on_previous_text": False,
            "no_repeat_ngram_size": 3,
            # Re-detect the language on every 30-second window instead of committing to
            # one for the whole file. Whisper decodes a single language per window, so on
            # audio that changes language the decoder emits the dominant language's words
            # over everything else -- the recorded `windows` and `quiet` defects.
            #
            # It is free: faster-whisper runs the detection on the encoder output it has
            # already computed for that window, not a second pass. Measured on 24
            # single-language clips, mean word overlap was 0.8684 either way and no clip
            # changed its reported language, so there is nothing to trade off on ordinary
            # audio.
            #
            # `language is None` matters. multilingual=True overrides an explicitly
            # requested language: forcing "fr" on Spanish audio returns Spanish text while
            # still reporting language="fr". Better transcription, but it silently ignores
            # what the caller asked for and then misreports it, so an explicit request
            # keeps the old single-language behaviour.
            "multilingual": language is None,
        }
        params.update(kwargs)
        return self.model.transcribe(audio_path, language=language, task=task, **params)

    def detect_language(self, audio: Any):
        """Identify the language of audio data without full transcription."""
        return self.model.detect_language(audio)

    def unload(self) -> None:
        if hasattr(self, "model"):
            del self.model
