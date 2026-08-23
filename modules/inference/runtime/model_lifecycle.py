"""Model reclamation helpers for runtime lifecycle management."""

from __future__ import annotations

import ctypes
import gc
import logging
from typing import Any

from modules.core import utils
from modules.inference import scheduler
from modules.inference.engines import whisperx_worker_client

logger = logging.getLogger(__name__)


def _read_reclamation_memory_snapshot() -> dict[str, float | int | None]:
    """Capture process RSS and optional CUDA VRAM usage for reclaim logs."""
    telemetry = utils.get_system_telemetry() or {}
    return {
        "app_memory_gb": float(telemetry.get("app_memory_gb", 0.0) or 0.0),
        "cuda_vram_mb": utils.get_nvidia_vram_usage_mb(),
    }


def _format_reclamation_memory(snapshot: dict[str, float | int | None]) -> str:
    ram_gb = float(snapshot.get("app_memory_gb", 0.0) or 0.0)
    cuda_vram_mb = snapshot.get("cuda_vram_mb")
    if isinstance(cuda_vram_mb, int):
        return f"RAM(RSS)={ram_gb:.2f} GB, CUDA VRAM={cuda_vram_mb} MB"
    return f"RAM(RSS)={ram_gb:.2f} GB"


def _format_reclamation_delta(
    before: dict[str, float | int | None],
    after: dict[str, float | int | None],
) -> str:
    ram_before = float(before.get("app_memory_gb", 0.0) or 0.0)
    ram_after = float(after.get("app_memory_gb", 0.0) or 0.0)
    parts = [f"RAM(RSS)={ram_before - ram_after:+.2f} GB"]

    vram_before = before.get("cuda_vram_mb")
    vram_after = after.get("cuda_vram_mb")
    if isinstance(vram_before, int) and isinstance(vram_after, int):
        parts.append(f"CUDA VRAM={vram_before - vram_after:+d} MB")

    return ", ".join(parts)


def _clear_whisper_models(model_pool: dict[str, Any]) -> int:
    with scheduler.STATE.model_lock:
        whisper_count = len(model_pool)
        for unit_id in list(model_pool.keys()):
            model = model_pool.pop(unit_id)
            try:
                if hasattr(model, "unload"):
                    model.unload()
                elif hasattr(model, "pipeline"):
                    model.pipeline = None
            except tuple([Exception]) as exc:
                logger.debug("[Engine] Error unloading model %s: %s", unit_id, exc)
            del model
        model_pool.clear()
    return whisper_count


def _clear_uvr_models(preprocessor_pool: dict[str, Any]) -> int:
    uvr_count = len(preprocessor_pool)
    for unit_id in list(preprocessor_pool.keys()):
        preprocessor = preprocessor_pool.pop(unit_id)
        try:
            preprocessor.unload_model()
        except tuple([Exception]) as exc:
            logger.debug("[Engine] Error unloading UVR %s: %s", unit_id, exc)
        del preprocessor
    preprocessor_pool.clear()
    return uvr_count


def _clear_whisperx_models(
    diarize_pool: dict[str, Any],
    align_pool: dict[str, Any],
) -> None:
    # Handles into the isolated WhisperX worker process become invalid the
    # moment the worker is torn down, so the pools just need clearing —
    # shutting down the worker outright is what actually reclaims its
    # RAM/VRAM (see whisperx_worker_client for why it can't be done
    # in-process).
    diarize_pool.clear()
    align_pool.clear()
    whisperx_worker_client.shutdown()


def _run_garbage_collection_and_reclamation(engines: dict[str, Any]) -> None:
    gc.collect()
    gc.collect()  # Second pass for circular references

    if engines["ctranslate2"]:
        try:
            engines["ctranslate2"].clear_caches()
        except (AttributeError, RuntimeError):
            pass

    # Release GPU memory if applicable
    utils.clear_gpu_cache()

    # Force OS memory reclamation
    try:
        # Linux/Docker optimization
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except (OSError, AttributeError):
        pass
