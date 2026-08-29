"""Where a unit's transcription and vocal isolation actually execute.

A unit sitting in the scheduler pool says where a task is *dispatched*, not where the work
lands. The two differ constantly, and always in the same direction -- the accelerator is
named and the CPU does the work:

- CTranslate2 has no OpenVINO or ROCm backend, so FASTER-WHISPER on an Intel or AMD unit
  decodes on the CPU.
- torch has no NPU backend, so an NPU unit handed a torch engine runs on the CPU.
- The Intel NPU cannot execute Whisper's dynamic-shaped IR at all; IntelWhisperEngine warms
  up, fails, and rewrites its own device to CPU.
- UVR falls back to CPUExecutionProvider whenever the image's ONNX Runtime has no provider
  for the device it was pointed at.

Every one of those is invisible on a dashboard that shows only the unit's name, which is
why this module exists: it answers "and where did it really run?" per unit, preferring what
a loaded object reports over what configuration predicts.
"""

from __future__ import annotations

from typing import Any

from modules.core import config, engine_registry

#: Unit/device types that are not the CPU. Used to decide whether landing on the CPU counts
#: as a fallback (it only does when the unit was an accelerator in the first place).
_ACCELERATORS = ("CUDA", "GPU", "NPU", "AMD", "XPU")

#: Short, unambiguous labels. "GPU" alone is the one that misleads -- OpenVINO calls an
#: Intel iGPU "GPU", and a dashboard showing that next to an NVIDIA card is unreadable.
_DEVICE_LABELS = {
    "CUDA": "CUDA",
    "GPU": "Intel GPU",
    "NPU": "Intel NPU",
    "XPU": "Intel GPU",
    "AMD": "AMD GPU",
    "CPU": "CPU",
}

#: Device-string prefixes an engine may report, longest-first so "cuda" cannot shadow a
#: longer match. Engines disagree on spelling: OpenVINO uses "GPU.0"/"NPU.0", torch uses
#: "cuda"/"xpu"/"cpu", and ROCm torch reports "cuda" with torch.version.hip set.
_DEVICE_PREFIXES = (
    ("cuda", "CUDA"),
    ("xpu", "XPU"),
    ("gpu", "GPU"),
    ("npu", "NPU"),
    ("amd", "AMD"),
    ("hip", "AMD"),
    ("rocm", "AMD"),
    ("cpu", "CPU"),
)

#: ONNX Runtime provider -> the silicon it runs on. OpenVINO is deliberately absent: it
#: serves both the Intel GPU and the NPU and its name does not say which, so the unit's own
#: type answers that (see _device_from_providers).
_PROVIDER_DEVICES = {
    "CUDAExecutionProvider": "CUDA",
    "TensorrtExecutionProvider": "CUDA",
    "ROCMExecutionProvider": "AMD",
    "MIGraphXExecutionProvider": "AMD",
    "DmlExecutionProvider": "AMD",
    "CPUExecutionProvider": "CPU",
}

_OPENVINO_PROVIDER = "OpenVINOExecutionProvider"


def normalize_device(value: Any) -> str:
    """Canonicalise whatever an engine or config calls a device, or "" if unrecognised."""
    text = str(value or "").strip().lower()
    for prefix, canonical in _DEVICE_PREFIXES:
        if text.startswith(prefix):
            return canonical
    return ""


def _describe(device: str, unit: dict, *, measured: bool) -> dict[str, Any]:
    """One execution answer: what ran it, and whether that is a step down from the unit."""
    unit_type = str(unit.get("type") or "").upper()
    return {
        "device": _DEVICE_LABELS.get(device, device or "unknown"),
        "accelerated": device in _ACCELERATORS,
        # The signal worth surfacing: an accelerator unit whose work landed on the CPU.
        "fallback": device == "CPU" and unit_type in _ACCELERATORS,
        # False means "derived from configuration", not "wrong" -- it is what will happen
        # once something loads. True means a live object was asked.
        "measured": measured,
    }


def asr_execution(unit: dict, engine: Any = None) -> dict[str, Any]:
    """Where transcription for ``unit`` runs, from the loaded engine when there is one.

    A loaded engine is authoritative because it may have moved itself: IntelWhisperEngine
    rewrites ``self.device`` to CPU when its NPU warmup fails, which is the single case a
    configuration-only answer gets wrong.
    """
    measured = _attribute_to_unit(normalize_device(getattr(engine, "device", None)), unit)
    if measured:
        return _describe(measured, unit, measured=True)
    return _describe(_resolved_asr_device(unit), unit, measured=False)


def _attribute_to_unit(device: str, unit: dict) -> str:
    """Correct a device string the engine cannot disambiguate on its own.

    ROCm torch deliberately reports itself as "cuda" and sets ``torch.version.hip``, so an
    AMD unit running the ROCm build reported CUDA -- naming the wrong vendor on the one
    dashboard whose job is to say where the work landed. The unit knows which silicon it is;
    the engine's device string does not.
    """
    if device == "CUDA" and str(unit.get("type") or "").upper() == "AMD":
        return "AMD"
    return device


def _resolved_asr_device(unit: dict) -> str:
    """What the factory would pick for this unit, before anything is loaded."""
    unit_type = str(unit.get("type") or "").upper()
    try:
        engine = config.engine_for_unit(unit)
    except (AttributeError, KeyError, TypeError):
        engine = getattr(config, "ASR_ENGINE", "")
    if engine_registry.engine_supports_unit(engine, unit_type):
        return unit_type or "CPU"
    return "CPU"


def uvr_execution(unit: dict, preprocessor: Any = None) -> dict[str, Any]:
    """Where vocal isolation for ``unit`` runs, from the loaded separator when there is one.

    The ONNX execution provider is the ground truth here and is reported even for isolated
    preprocessing, because the worker hands its provider list back through _SeparatorProbe.
    """
    providers = getattr(getattr(preprocessor, "separator", None), "onnx_execution_provider", None)
    measured = _device_from_providers(providers, unit)
    if measured:
        return _describe(measured, unit, measured=True)
    configured = normalize_device(getattr(preprocessor, "device_type", None)) or normalize_device(config.PREPROCESS_DEVICE)
    return _describe(configured or "CPU", unit, measured=False)


def _device_from_providers(providers: Any, unit: dict) -> str:
    """The device the first recognised ONNX provider names, or "" when there is none.

    Order matters: ONNX Runtime lists providers by priority and always appends
    CPUExecutionProvider as the last resort, so the FIRST recognised entry is the one that
    actually gets the nodes.
    """
    if not isinstance(providers, (list, tuple)):
        return ""
    for provider in providers:
        device = _provider_device(str(provider or ""), unit)
        if device:
            return device
    return ""


def _provider_device(name: str, unit: dict) -> str:
    """The silicon one provider name refers to, or "" when it is not one we know."""
    if name == _OPENVINO_PROVIDER:
        # OpenVINO serves both Intel devices and its name does not say which, so the unit
        # the separator was built for is the only thing that can answer.
        # Every recognised unit type is returned as itself, CPU included. Collapsing
        # anything that was not GPU/NPU to "GPU" meant an OpenVINO separator on the CPU unit
        # reported the GPU -- and was therefore counted as accelerated, which is the one
        # answer this module exists to get right.
        unit_type = normalize_device(unit.get("type"))
        return unit_type or "GPU"
    return _PROVIDER_DEVICES.get(name, "")
