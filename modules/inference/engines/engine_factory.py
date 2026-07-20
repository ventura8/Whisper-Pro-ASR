"""Factory and compatibility exports for ASR engine wrappers."""

import importlib
import logging

from modules.core import config, engine_registry
from modules.core import utils as core_utils
from modules.inference.engines.base import BaseASREngine
from modules.inference.engines.faster_whisper_engine import FasterWhisperEngine
from modules.inference.engines.openai_whisper_engine import OpenaiWhisperEngine
from modules.inference.engines.whisperx_engine import WhisperXEngine

logger = logging.getLogger(__name__)

# Backward-compatible module alias used by existing call sites and tests.
utils = core_utils


def create_engine(engine_type: str, model_id: str, unit: dict) -> BaseASREngine:
    """Factory method to instantiate the correct ASR engine wrapper."""
    engine_type = engine_registry.normalize_and_validate_engine(engine_type)

    if engine_type == engine_registry.ENGINE_INTEL_WHISPER:
        engine_type = _resolve_intel_whisper_engine(engine_type, unit)

    if engine_type == engine_registry.ENGINE_INTEL_WHISPER:
        logger.info("[EngineFactory] Loading IntelWhisperEngine on %s", unit["name"])
        intel_engine = importlib.import_module("modules.inference.engines.intel_engine")
        return intel_engine.IntelWhisperEngine(model_id, device=unit["id"])

    return _create_non_intel_engine(engine_type, model_id, unit)


def _create_non_intel_engine(engine_type: str, model_id: str, unit: dict) -> BaseASREngine:
    if engine_type == engine_registry.ENGINE_OPENAI_WHISPER:
        logger.info("[EngineFactory] Loading OpenaiWhisperEngine on %s", unit["name"])
        return OpenaiWhisperEngine(model_id, device=_resolve_device_str(unit))

    if engine_type == engine_registry.ENGINE_WHISPERX:
        return _create_whisperx_engine(model_id, unit)

    if engine_type != engine_registry.ENGINE_FASTER_WHISPER:
        supported = ", ".join(engine_registry.supported_engines())
        raise ValueError(f"Unsupported ASR engine '{engine_type}'. Supported values: {supported}")

    return _create_faster_whisper_engine(model_id, unit)


def _resolve_intel_whisper_engine(engine_type: str, unit: dict) -> str:
    if unit["type"] in ["GPU", "NPU"]:
        return engine_type
    logger.info(
        "[EngineFactory] INTEL-WHISPER requested on %s. Falling back to FasterWhisperEngine.",
        unit["name"],
    )
    return engine_registry.ENGINE_FASTER_WHISPER


def _resolve_device_str(unit: dict) -> str:
    if unit["type"] == "CUDA":
        return "cuda"
    return "cpu"


def _coerce_cpu_compute_type(target_device: str, compute_type: str) -> str:
    """Coerce float16 compute_type to int8 when running CTranslate2/WhisperX on CPU."""
    if target_device == "cpu" and compute_type == "float16":
        return "int8"
    return compute_type


def _create_faster_whisper_engine(model_id: str, unit: dict) -> FasterWhisperEngine:
    logger.info("[EngineFactory] Loading FasterWhisperEngine (CTranslate2) on %s", unit["name"])
    target_device = _resolve_device_str(unit)
    if target_device == "cpu" and unit["type"] in ["NPU", "GPU"]:
        logger.info("[EngineFactory] Intel accelerator detected. Faster-Whisper will fall back to CPU for Whisper slot.")

    compute_type = _coerce_cpu_compute_type(target_device, config.COMPUTE_TYPE)

    return FasterWhisperEngine(
        model_id,
        device=target_device,
        device_index=unit.get("index", 0),
        compute_type=compute_type,
        cpu_threads=config.ASR_THREADS,
        download_root=config.OV_CACHE_DIR,
    )


def _create_whisperx_engine(model_id: str, unit: dict) -> WhisperXEngine:
    logger.info("[EngineFactory] Loading WhisperXEngine on %s", unit["name"])
    target_device = _resolve_device_str(unit)
    compute_type = _coerce_cpu_compute_type(target_device, config.COMPUTE_TYPE)
    return WhisperXEngine(model_id, device=target_device, compute_type=compute_type)
