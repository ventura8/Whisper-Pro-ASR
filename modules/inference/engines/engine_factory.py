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
        device = _resolve_torch_device(unit)
        logger.info("[EngineFactory] Loading OpenaiWhisperEngine on %s (device=%s)", unit["name"], device)
        return OpenaiWhisperEngine(model_id, device=device)

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
    """Device string for CTranslate2 engines, which support only CPU and CUDA."""
    if unit["type"] == "CUDA":
        return "cuda"
    return "cpu"


def _resolve_torch_device(unit: dict) -> str:
    """Device string for torch-based engines, probed from the torch build present.

    Wider than :func:`_resolve_device_str` because torch reaches hardware CTranslate2
    cannot. Which build is installed is per-image, so this asks torch rather than
    assuming: the intel target ships an XPU build, the opt-in amd-rocm-torch target a
    ROCm one, and everything else CUDA or CPU.

    ROCm torch deliberately reports ``cuda`` as its device string (and sets
    ``torch.version.hip``), so an AMD unit maps to "cuda" rather than anything ROCm-named.
    """
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return "cpu"

    probe = _TORCH_DEVICE_PROBES.get(unit.get("type"))
    if probe is None:
        return "cpu"
    return probe(torch)


def _torch_cuda_device(torch) -> str:
    """CUDA when this torch was built with it, CPU otherwise."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def _torch_rocm_device(torch) -> str:
    """ROCm torch reports itself as CUDA; a CUDA or CPU torch must not claim an AMD GPU."""
    if getattr(torch.version, "hip", None) and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _torch_xpu_device(torch) -> str:
    """XPU only when this build has torch.xpu at all and it reports a device.

    `getattr(torch, "xpu", None)` rather than hasattr-and-call: the CUDA and CPU builds have
    no ``torch.xpu`` attribute whatsoever, which is the common case in every non-Intel image.
    """
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available():
        return "xpu"
    return "cpu"


#: Unit type -> how to ask torch for its device string. A table rather than an if-chain so
#: adding a vendor does not add a branch to the dispatcher (Radon rank A is enforced).
_TORCH_DEVICE_PROBES = {
    "CUDA": _torch_cuda_device,
    "AMD": _torch_rocm_device,
    "GPU": _torch_xpu_device,
    "NPU": _torch_xpu_device,
}


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


def _resolve_whisperx_device(unit: dict) -> str:
    """Pick WhisperX's device from what its worker can actually do.

    WhisperX runs in a segregated stack whose torch comes from the image, so the parent
    cannot infer GPU support: handing a CPU-only torch device="cuda" fails at model load
    with "Torch not compiled with CUDA enabled", which surfaces only as an empty engine
    pool. Ask the worker instead, and degrade to CPU rather than failing.
    """
    # AMD is deliberately exempt from the CPU short-circuit. _resolve_device_str answers for
    # CTranslate2, which has no ROCm backend and so reports "cpu" for an AMD unit -- but
    # WhisperX is torch-based, and a ROCm torch reports itself as CUDA-capable. Gating on
    # that answer meant an AMD host with the ROCm stack never even asked the worker and
    # always ran WhisperX on the CPU. Intel GPU/NPU keep the short-circuit: WhisperX has no
    # XPU path, so probing them is pure latency.
    if unit.get("type") != "AMD" and _resolve_device_str(unit) == "cpu":
        return "cpu"
    try:
        worker = importlib.import_module("modules.inference.engines.whisperx_worker_client")
        if worker.call("capabilities").get("cuda"):
            return "cuda"
        logger.info("[EngineFactory] WhisperX worker reports no CUDA torch; running on CPU for %s.", unit["name"])
    except (ImportError, RuntimeError, OSError) as exc:
        logger.info("[EngineFactory] Could not query WhisperX worker capabilities (%s); running on CPU.", exc)
    return "cpu"


def _create_whisperx_engine(model_id: str, unit: dict) -> WhisperXEngine:
    """Build the WhisperX engine on whichever device its worker supports."""
    target_device = _resolve_whisperx_device(unit)
    logger.info("[EngineFactory] Loading WhisperXEngine on %s (device=%s)", unit["name"], target_device)
    compute_type = _coerce_cpu_compute_type(target_device, config.COMPUTE_TYPE)
    return WhisperXEngine(model_id, device=target_device, compute_type=compute_type)
