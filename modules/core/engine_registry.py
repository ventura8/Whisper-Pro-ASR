"""Central registry and resolution helpers for ASR engine selection."""

import importlib.util
import os

ENGINE_FASTER_WHISPER = "FASTER-WHISPER"
ENGINE_INTEL_WHISPER = "INTEL-WHISPER"
ENGINE_OPENAI_WHISPER = "OPENAI-WHISPER"
ENGINE_WHISPERX = "WHISPERX"

SUPPORTED_ASR_ENGINES = (
    ENGINE_FASTER_WHISPER,
    ENGINE_INTEL_WHISPER,
    ENGINE_OPENAI_WHISPER,
    ENGINE_WHISPERX,
)

# AUTO fallback order requested by product policy.
#: The engine ASR_ENGINE=AUTO resolves to, on every host.
#:
#: Deliberately hardware-independent. AUTO used to pick a different engine per accelerator
#: -- OPENAI-WHISPER on AMD, INTEL-WHISPER on Intel GPU/NPU -- which made the decoding
#: behaviour, and therefore the transcript, a property of whichever machine a request
#: happened to land on. One engine everywhere means a deployment is reproducible across
#: the fleet, and it is the engine this project measures against.
#:
#: An accelerator-specific engine is still available; it just has to be asked for, with
#: ASR_ENGINE=INTEL-WHISPER or ASR_ENGINE=OPENAI-WHISPER.
AUTO_DEFAULT_ENGINE = ENGINE_FASTER_WHISPER

#: Hardware preference order for choosing which *unit* AUTO runs on. Unlike the engine,
#: this is still hardware-dependent: given a choice of slots, the faster one wins.
AUTO_DEVICE_PRIORITY = ("CUDA", "AMD", "GPU", "NPU", "CPU")


def supported_engines() -> list[str]:
    """Return the supported ASR engine names."""
    return list(SUPPORTED_ASR_ENGINES)


def normalize_and_validate_engine(engine_name: str) -> str:
    """Normalize an engine name and raise ValueError if unsupported."""
    normalized = (engine_name or "").strip().upper()
    if normalized in SUPPORTED_ASR_ENGINES:
        return normalized
    supported = ", ".join(SUPPORTED_ASR_ENGINES)
    raise ValueError(f"Invalid ASR_ENGINE '{engine_name}'. Supported values: {supported}")


def _available_hardware_types(hardware_units: list[dict]) -> set[str]:
    return {str(unit.get("type", "")).upper() for unit in hardware_units}


def _validate_hardware_units(hardware_units: list[dict]):
    """Ensure hardware_units list conforms to structure expectations."""
    _validate_hardware_units_container(hardware_units)
    for idx, unit in enumerate(hardware_units):
        _validate_hardware_unit_item(idx, unit)


def _validate_hardware_units_container(hardware_units):
    if not isinstance(hardware_units, list) or not hardware_units:
        raise ValueError("hardware_units must be a non-empty list of unit dictionaries")


def _validate_hardware_unit_item(idx: int, unit):
    if not isinstance(unit, dict) or "type" not in unit:
        raise ValueError(f"hardware_units[{idx}] must be a dict containing a 'type' key")


def resolve_auto_engine(hardware_units: list[dict], requested_device: str = "AUTO") -> tuple[str, str]:
    """
    Resolve ASR engine for ASR_ENGINE=AUTO.

    The engine is always AUTO_DEFAULT_ENGINE; only the reported hardware tier depends on
    the machine. ``requested_device`` still selects which tier is named, so an explicit
    ASR_DEVICE is reflected in the resolution string and the logs.

    Returns:
        (resolved_engine, matched_hardware_tier)
    """
    _validate_hardware_units(hardware_units)

    selected_device = requested_device.strip().upper()
    if selected_device != "AUTO" and selected_device in AUTO_DEVICE_PRIORITY:
        return AUTO_DEFAULT_ENGINE, selected_device

    available_types = _available_hardware_types(hardware_units)
    for hardware_tier in AUTO_DEVICE_PRIORITY:
        if hardware_tier in available_types:
            return AUTO_DEFAULT_ENGINE, hardware_tier
    # Defensive fallback. In practice CPU should always be present.
    return AUTO_DEFAULT_ENGINE, "CPU"


def resolve_auto_device(hardware_units: list[dict]) -> str:
    """Resolve preferred device tier for AUTO in priority order."""
    _validate_hardware_units(hardware_units)
    available_types = _available_hardware_types(hardware_units)
    for hardware_tier in AUTO_DEVICE_PRIORITY:
        if hardware_tier in available_types:
            return hardware_tier
    return "CPU"


#: Which unit types each engine can actually drive.
#:
#: This is a property of the runtime, not a preference. CTranslate2 (Faster-Whisper) has
#: CUDA and CPU backends and nothing else -- no OpenVINO, no ROCm -- so an Intel or AMD
#: unit handed to it degrades to the CPU while the logs name the accelerator. OpenVINO
#: (Intel-Whisper) reads only Intel devices. The torch engines follow torch's own device
#: support.
#:
#: Measured consequences of not enforcing this: with ASR_ENGINE=FASTER-WHISPER on a pool of
#: cuda:0 + Intel GPU.0, tasks landing on GPU.0 ran on the CPU and blew the throughput
#: budget; the same shape on an AMD unit took 27% of a CUDA validation onto the CPU.
ENGINE_UNIT_SUPPORT: dict[str, tuple[str, ...]] = {
    ENGINE_FASTER_WHISPER: ("CUDA", "CPU"),
    ENGINE_INTEL_WHISPER: ("GPU", "NPU", "CPU"),
    # "GPU"/"NPU", not "XPU". Detection emits CUDA/AMD/GPU/NPU/CPU as unit types -- "XPU"
    # is torch's device string, not a unit type, so it matched nothing and Intel units were
    # filtered out of the pool for this engine entirely. engine_factory._resolve_torch_device
    # maps GPU/NPU to the xpu device where the intel-xpu image's torch offers one, and to
    # the CPU otherwise, which is the documented behaviour for the plain intel image.
    ENGINE_OPENAI_WHISPER: ("CUDA", "AMD", "GPU", "NPU", "CPU"),
    ENGINE_WHISPERX: ("CUDA", "CPU"),
}


def engine_supports_unit(engine: str, unit_type: str) -> bool:
    """Return whether ``engine`` can execute on a unit of ``unit_type``."""
    supported = ENGINE_UNIT_SUPPORT.get(engine)
    # An unknown engine is not something to silently exclude hardware over.
    if supported is None:
        return True
    # Upper-cased before the membership test: the table's keys are upper-case, and callers
    # pass a unit's raw "type" through unchanged. A "cuda" or "Gpu" would silently answer
    # False and drop a working accelerator out of the pool.
    return str(unit_type or "").strip().upper() in supported


#: The third-party module each engine needs at runtime. An engine whose module is absent
#: from the image cannot serve a single request, so this is what distinguishes "configured"
#: from "installed".
ENGINE_REQUIRED_MODULE: dict[str, str] = {
    ENGINE_FASTER_WHISPER: "faster_whisper",
    ENGINE_INTEL_WHISPER: "openvino_genai",
    ENGINE_OPENAI_WHISPER: "whisper",
    ENGINE_WHISPERX: "whisperx",
}


#: Engines whose runtime is shipped outside the interpreter's default path, with the
#: location in an environment variable. WhisperX is kept off sys.path on purpose: it pulls
#: its own torch stack and runs in an isolated worker that injects the path itself. Probing
#: only with find_spec therefore reports a correctly installed engine as missing.
ENGINE_LIB_PATH_ENV: dict[str, str] = {
    ENGINE_WHISPERX: "WHISPERX_LIB_PATH",
}


def engine_is_installed(engine: str) -> bool:
    """Return whether this image actually ships the runtime ``engine`` needs."""
    module = ENGINE_REQUIRED_MODULE.get(engine)
    if not module:
        return True
    try:
        if importlib.util.find_spec(module) is not None:
            return True
    except (ImportError, ValueError, ModuleNotFoundError):
        pass
    lib_path = os.environ.get(ENGINE_LIB_PATH_ENV.get(engine, ""), "")
    return bool(lib_path) and os.path.isdir(os.path.join(lib_path, module))
