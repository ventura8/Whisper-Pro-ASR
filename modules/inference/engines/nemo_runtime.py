"""Loading a NeMo model through onnx-asr on the device its unit names, honestly.

ONNX Runtime is not installed as a dependency: every image carries the build for its vendor
under /app/libs/<vendor>, and bootstrap puts the right one on sys.path. So this module asks
the build that actually got imported what it can do, rather than assuming the unit's type
implies a provider -- a CUDA unit in a worker that imported the CPU build would otherwise
report CUDA while every chunk ran on the CPU.

Providers are always passed explicitly. onnx-asr's default is *every* provider the build
offers, which on the OpenVINO build silently means OpenVINO on its default device for a
unit that was supposed to run on the CPU.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

import numpy as np

from modules.core import config, config_nemo, model_integrity, nemo_catalog

logger = logging.getLogger(__name__)

# Deliberately broad handlers catch through this name: pylint flags a bare `except Exception` and inline disables are banned.
_ANY_EXCEPTION: tuple[type[Exception], ...] = (Exception,)

_CPU = "CPUExecutionProvider"
_CUDA = "CUDAExecutionProvider"
_OPENVINO = "OpenVINOExecutionProvider"

#: One second of silence: enough to make every session allocate and compile once, so a
#: provider that cannot execute the graph fails here, at load, and not on the first request.
_WARMUP_SAMPLES = 16000


def ensure_vendor_runtime(unit_type: str) -> None:
    """Make the OpenVINO build of ONNX Runtime the imported one for an Intel GPU unit.

    The worker's bootstrap normally already did, from its ASR_DEVICE; this covers the image
    whose default build is another vendor's, the same hot-reload vocal separation relies on.
    """
    if str(unit_type or "").upper() != "GPU":
        return
    resolver = importlib.import_module("modules.inference.pipeline.openvino_resolver")
    resolver.ensure_openvino_onnxruntime("GPU")


def providers_for(unit: dict, available: list[str]) -> list:
    """The provider list for ``unit``, restricted to what the imported build offers."""
    accelerated = _ACCELERATED_PROVIDERS.get(str(unit.get("type", "")).upper())
    if accelerated and accelerated[0] in available:
        provider, options = accelerated
        return [(provider, options(unit)), _CPU]
    return [_CPU]


def _cuda_options(unit: dict) -> dict:
    """CUDA provider options for this unit's card.

    ``kSameAsRequested`` rather than the default power-of-two arena growth: the NeMo model
    shares the card with its Whisper companion, and doubling reservations measured 4.97 GB
    peak for Parakeet against 4.34 GB requested-size (RTX 3080 laptop, batch 4 x 20 s).
    """
    dispatch = importlib.import_module("modules.inference.pipeline.openvino_provider_dispatch")
    _, options = dispatch.cuda_provider_config(str(unit.get("id") or "cuda:0"))
    cuda = {**options[0], "arena_extend_strategy": "kSameAsRequested"}
    if config_nemo.NEMO_GPU_MEM_LIMIT_MB:
        cuda["gpu_mem_limit"] = config_nemo.NEMO_GPU_MEM_LIMIT_MB * 1024 * 1024
    return cuda


def _openvino_options(unit: dict) -> dict:
    """OpenVINO provider options for this Intel GPU.

    No ``cache_dir``: a second same-process session reloading the kernel cache segfaults on
    this driver stack (see preprocessing/provider.py), and the companion is not the only
    OpenVINO user a worker may hold. One stream, because a NeMo decode is one batch at a time.
    """
    resolver = importlib.import_module("modules.inference.pipeline.openvino_resolver")
    device = resolver.find_matching_openvino_device(str(unit.get("id") or "GPU"), resolver.get_available_openvino_devices())
    return {"device_type": device, "num_streams": "1"}


#: Unit type -> (provider, options builder) for the units a NeMo model accelerates on. Every
#: other type -- NPU, AMD, CPU -- runs on the CPU provider; see engine_registry.
_ACCELERATED_PROVIDERS = {
    "CUDA": (_CUDA, _cuda_options),
    "GPU": (_OPENVINO, _openvino_options),
}


def session_options(ort: Any) -> Any:
    """Thread limits from config (AGENTS.md: every runtime respects ASR_THREADS), quiet logs.

    Severity 3 (errors only): ONNX Runtime 1.29 warns once per session that no plugin device
    is registered for the CUDA provider while using it correctly, and the startup banner,
    not a vendor runtime, is this project's authority on where work runs.
    """
    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, int(config.ASR_THREADS))
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    return options


def _quiet(ort: Any) -> None:
    """Errors only, process-wide.

    The per-session severity in session_options is not enough on its own: ONNX Runtime 1.29
    logs "No registered plugin EP device found for 'CUDAExecutionProvider'" through its
    default logger while resolving providers, before any session option applies -- ten
    warning lines per model load, measured on an RTX 3080, while the provider was in fact
    used correctly.
    """
    setter = getattr(ort, "set_default_logger_severity", None)
    if callable(setter):
        setter(3)


def load(spec: Any, path: str, quantization: str | None, unit: dict) -> tuple[Any, str]:
    """Load ``spec`` from ``path`` for ``unit`` and return ``(model, device)``.

    ``device`` is where the encoder actually runs. A provider that loads but cannot execute
    the graph -- the OpenVINO GPU plugin and the int8 export, for one -- fails the warm-up;
    the model is then reloaded on the CPU, logged as an error, and reported as "cpu" so the
    dashboard shows the fallback instead of the accelerator that was asked for.
    """
    ensure_vendor_runtime(unit.get("type", ""))
    ort = importlib.import_module("onnxruntime")
    _quiet(ort)
    providers = providers_for(unit, list(ort.get_available_providers()))
    _log_runtime(ort, providers, unit)
    try:
        model = _load_and_warm(spec, path, quantization, providers, ort)
    except _ANY_EXCEPTION as exc:
        if providers == [_CPU]:
            raise
        logger.error("[NeMo] %s could not run on %s (%s); reloading on the CPU.", spec.name, unit.get("name", unit.get("id")), exc)
        model = load_on_cpu(spec, path, quantization)
    return model, actual_device(model, unit)


def load_on_cpu(spec: Any, path: str, quantization: str | None) -> Any:
    """``spec`` on the CPU execution provider: the fallback for a failing accelerator."""
    return _load_and_warm(spec, path, quantization, [_CPU], importlib.import_module("onnxruntime"))


def _load_and_warm(spec: Any, path: str, quantization: str | None, providers: list, ort: Any) -> Any:
    _require_provisioned(spec, path, quantization)
    onnx_asr = importlib.import_module("onnx_asr")
    model = onnx_asr.load_model(spec.name, path, quantization=quantization, providers=providers, sess_options=session_options(ort))
    if spec.word_timestamps:
        model = model.with_timestamps()
    warmup = {"language": "en", "target_language": "en"} if spec.needs_language else {}
    model.recognize(np.zeros(_WARMUP_SAMPLES, dtype=np.float32), **warmup)
    return model


def _require_provisioned(spec: Any, path: str, quantization: str | None) -> None:
    """Refuse to load unless provisioning left the pinned files at ``path``.

    onnx-asr is handed a full model name, and for a name it downloads whatever is missing
    from the Hugging Face repository head. A failed or purged provisioning would then load
    an unpinned graph, so the catalog revision -- the supply-chain guard -- has to be
    enforced here, at load, rather than trusted from an earlier step.
    """
    required = nemo_catalog.required_files(spec, quantization)
    if not model_integrity.verify_nemo_onnx_dir(path, required):
        raise RuntimeError(f"{spec.name} is not provisioned at {path}; refusing to let onnx-asr download an unpinned copy.")


def actual_device(model: Any, unit: dict) -> str:
    """The device string the encoder session settled on, in the spelling execution_status reads."""
    provider = _encoder_provider(model)
    if provider == _CUDA:
        return "cuda"
    if provider == _OPENVINO:
        return str(unit.get("id") or "GPU")
    return "cpu"


def _encoder_provider(model: Any) -> str:
    """The first provider of the encoder session, found through onnx-asr's adapters.

    ``with_timestamps()`` wraps the model in another adapter, and both keep the recogniser as
    ``asr``; the encoder session is ``_encoder`` on it. Private names, so read defensively: an
    onnx-asr that moved them reports "unknown" rather than failing a load that worked.
    """
    asr = getattr(model, "asr", None)
    session = getattr(asr, "_encoder", None)
    try:
        return str(session.get_providers()[0])
    except _ANY_EXCEPTION:
        return "unknown"


def _log_runtime(ort: Any, providers: list, unit: dict) -> None:
    """The evidence validation reads: which build, from where, offering what, asked for what."""
    requested = [provider[0] if isinstance(provider, tuple) else provider for provider in providers]
    logger.info(
        "[NeMo] ONNX Runtime %s from %s on %s: available %s, requested %s",
        getattr(ort, "__version__", "?"),
        getattr(ort, "__file__", "?"),
        unit.get("id"),
        list(ort.get_available_providers()),
        requested,
    )
