"""Settings for the PARAKEET and CANARY engines (NVIDIA NeMo models on onnx-asr).

Kept out of config.py, which is at its module-length limit. The engine, provisioning and
the startup banner read these from here directly, the way config_segmentation is read.

What ``ASR_MODEL`` means does not change. It still names the Whisper weights, which for these
engines are the *companion's* -- the model that answers language detection and transcribes
every language the NeMo model does not cover.
"""

from __future__ import annotations

import logging
import os

from modules.core import engine_registry, nemo_catalog
from modules.core.config_paths import float_env

logger = logging.getLogger(__name__)

#: Longest audio, in seconds, handed to the model as one item. onnx-asr documents 20-30 s
#: as the ceiling for these exports; longer speech is split at its quietest point first.
NEMO_MAX_CHUNK_SEC = float_env("NEMO_MAX_CHUNK_SEC", 20.0, minimum=5.0, maximum=30.0)

#: Chunks per ``recognize`` call. Throughput rises with it (Parakeet on an RTX 3080 laptop:
#: 82x real time at 1, 123x at 4) and so does device memory: Canary-1B-v2 fp32 measured
#: 6.6 GB at 1 and 7.8 GB at 2 on the same card. It is also the most work a preemption can
#: wait for, since the stream is only checked between batches.
NEMO_BATCH_SIZE = int(float_env("NEMO_BATCH_SIZE", 4, minimum=1, maximum=32))

#: Ceiling on ONNX Runtime's CUDA arena, in MiB; 0 leaves it unbounded. The arena keeps
#: whatever it has grown to, and on a card shared with the Whisper companion and the CUDA
#: vocal-isolation worker that is what runs the others out of memory: an RTX 3080 laptop
#: (8 GB) filled to 7.8 GB with Parakeet, large-v3 int8_float16 and UVR resident, and the
#: next allocation failed in whichever process asked for it (2026-10-09). A cap makes the
#: NeMo session fail -- and retry chunk by chunk -- instead of starving its neighbours.
NEMO_GPU_MEM_LIMIT_MB = int(float_env("NEMO_GPU_MEM_LIMIT_MB", 0, minimum=0, maximum=1048576))

_QUANTIZATION_CHOICES = ("AUTO", "INT8", "FP32")
_COMPANION_DEVICE_CHOICES = ("AUTO", "CPU")


def _choice(name: str, default: str, choices: tuple[str, ...]) -> str:
    """An upper-cased enumerated setting; anything unrecognised is the default, with a warning."""
    raw = (os.environ.get(name) or default).strip().upper()
    if raw in choices:
        return raw
    logger.warning("[Config] %s=%r is not one of %s; using %s.", name, raw, ", ".join(choices), default)
    return default


#: AUTO picks per unit (see quantization_for); INT8 or FP32 forces one for every unit.
NEMO_QUANTIZATION = _choice("NEMO_QUANTIZATION", "AUTO", _QUANTIZATION_CHOICES)

#: Where the Whisper companion runs. AUTO is the unit itself -- the GPU on a CUDA unit. CPU
#: keeps it off the card, for a card that cannot hold both models: Canary-1B-v2 fp32 alone
#: takes 6.6 GB, which leaves an 8 GB card no room for large-v3 beside it.
NEMO_COMPANION_DEVICE = _choice("NEMO_COMPANION_DEVICE", "AUTO", _COMPANION_DEVICE_CHOICES)

#: Unit types whose NeMo engine runs on ONNX Runtime's CPU execution provider by design.
#: int8 halves memory there at the same speed (Parakeet on an i9-11900H, 4 threads: 4.0x
#: real time int8, 4.3x fp32). Everywhere else it is fp32, because the accelerators are
#: worse at int8 than at fp32, not better: CUDA ran Parakeet int8 at 7.6x against 123x fp32,
#: and the OpenVINO GPU plugin failed to execute the int8 graph at all (Intel UHD, 2026-10-09).
_CPU_EP_UNIT_TYPES = frozenset({"CPU", "NPU", "AMD"})


#: (engine, NEMO_MODEL) pairs already refused, so the error is logged once, not per call.
_REFUSED: set = set()


def model_spec(engine: str) -> nemo_catalog.NemoModelSpec:
    """The catalog entry ``engine`` loads: ``NEMO_MODEL`` when it names one of its own, else the default.

    A name outside the catalog, or one belonging to the other family, is refused with an
    error rather than downloaded or loaded: the catalog is the allowlist, and a Canary name
    under PARAKEET would put an encoder-decoder in the transducer's place.
    """
    default = nemo_catalog.FAMILY_DEFAULT[engine]
    requested = (os.environ.get("NEMO_MODEL") or "").strip()
    if not requested:
        return nemo_catalog.CATALOG[default]
    if requested in nemo_catalog.FAMILY_MODELS[engine]:
        return nemo_catalog.CATALOG[requested]
    # Logged once per value: /status asks for the spec on every dashboard poll, and the
    # same error every few seconds would bury every other line of the log.
    if (engine, requested) not in _REFUSED:
        _REFUSED.add((engine, requested))
        allowed = ", ".join(nemo_catalog.FAMILY_MODELS[engine])
        logger.error("NEMO_MODEL=%s is not a %s model (allowed: %s); loading %s.", requested, engine, allowed, default)
    return nemo_catalog.CATALOG[default]


def quantization_for(unit_type: str) -> str | None:
    """The onnx-asr quantization for a unit: ``"int8"``, or ``None`` for the fp32 export."""
    if NEMO_QUANTIZATION == "INT8":
        return "int8"
    if NEMO_QUANTIZATION == "FP32":
        return None
    return "int8" if str(unit_type or "").strip().upper() in _CPU_EP_UNIT_TYPES else None


def quantizations_for(units: list) -> list:
    """Every distinct quantization a pool of units will load, in a stable order."""
    found: list = []
    for unit in units:
        quantization = quantization_for(unit.get("type", ""))
        if quantization not in found:
            found.append(quantization)
    return found or [None]


def model_dir(cache_root: str, spec: nemo_catalog.NemoModelSpec) -> str:
    """Where ``spec``'s files live: real files under the persistent cache, never hub symlinks."""
    return os.path.join(cache_root, "nemo", spec.name)


def banner_lines(engine: str, companion_model: str) -> list[tuple[str, str]]:
    """Startup-banner ``(label, value)`` rows describing the NeMo engine; none for other engines."""
    if not engine_registry.is_nemo_engine(engine):
        return []
    spec = model_spec(engine)
    return [
        ("NeMo Model", f"{spec.name} ({len(spec.languages)} languages, {spec.revision[:8]})"),
        ("NeMo Quantization", f"{NEMO_QUANTIZATION} | Chunk {NEMO_MAX_CHUNK_SEC:g}s x {NEMO_BATCH_SIZE}"),
        ("Whisper Companion", f"{companion_model} on {NEMO_COMPANION_DEVICE} (language ID + other languages)"),
    ]
