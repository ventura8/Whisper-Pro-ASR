"""Central registry and resolution helpers for ASR engine selection."""

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
AUTO_ENGINE_PRIORITY = (
    ("CUDA", ENGINE_FASTER_WHISPER),
    ("GPU", ENGINE_INTEL_WHISPER),
    ("NPU", ENGINE_INTEL_WHISPER),
    ("CPU", ENGINE_FASTER_WHISPER),
)


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
    if not isinstance(hardware_units, list) or not hardware_units:
        raise ValueError("hardware_units must be a non-empty list of unit dictionaries")
    for idx, unit in enumerate(hardware_units):
        if not isinstance(unit, dict) or "type" not in unit:
            raise ValueError(f"hardware_units[{idx}] must be a dict containing a 'type' key")


def resolve_auto_engine(hardware_units: list[dict]) -> tuple[str, str]:
    """
    Resolve ASR engine for ASR_ENGINE=AUTO.

    Returns:
        (resolved_engine, matched_hardware_tier)
    """
    _validate_hardware_units(hardware_units)

    available_types = _available_hardware_types(hardware_units)
    for hardware_tier, resolved_engine in AUTO_ENGINE_PRIORITY:
        if hardware_tier in available_types:
            return resolved_engine, hardware_tier
    # Defensive fallback. In practice CPU should always be present.
    return ENGINE_FASTER_WHISPER, "CPU"


def resolve_auto_device(hardware_units: list[dict]) -> str:
    """Resolve preferred device tier for AUTO in priority order."""
    _validate_hardware_units(hardware_units)
    available_types = _available_hardware_types(hardware_units)
    for hardware_tier, _ in AUTO_ENGINE_PRIORITY:
        if hardware_tier in available_types:
            return hardware_tier
    return "CPU"
