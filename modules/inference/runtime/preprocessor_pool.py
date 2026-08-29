"""Choosing which preprocessor a task's vocal isolation runs on.

Separate from the ASR unit assignment on purpose: UVR runs on ONNX Runtime and reaches
devices the ASR engine cannot, so the scheduler pool is the wrong list to pick from. Split
out of model_manager to keep that module inside the project's module-length limit.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import MutableMapping
from typing import Any

from modules.core import config
from modules.inference.pipeline import preprocessing

logger = logging.getLogger(__name__)

PreprocessorPool = MutableMapping[str, Any]

# Serialises the shared-preprocessor cache-miss path. Building a PreprocessingManager loads
# the UVR model onto a device, so two tasks arriving together used to build two of them for
# the same key, and the loser's manager was overwritten in the pool while still holding its
# ONNX session and its share of device memory -- leaked for the life of the process, and on
# an NPU enough to make the second load fail outright.
_SHARED_PREPROCESSOR_LOCK = threading.Lock()


def _is_accelerated_preprocess_device() -> bool:
    return config.PREPROCESS_DEVICE in config.ACCELERATED_PREPROCESS_DEVICES


def _normalized(value: Any) -> str:
    """Upper-cased device type, so a lookup cannot miss on casing alone.

    The two lookups below compared raw strings while _should_colocate_with_unit upper-cases
    first. A unit or an env value spelled "gpu"
    therefore matched nothing here, fell through to _shared_preprocessor_for_type, and built
    a *second* manager -- loading UVR onto a device that already had it, under a different
    key, for the life of the process.
    """
    return str(value or "").upper()


def _pool_preprocessor_by_type(pool: PreprocessorPool, preferred_type: str) -> Any | None:
    # Snapshot: the pool is inserted into by other threads building shared preprocessors,
    # and iterating it live raises "dictionary changed size during iteration" mid-request.
    wanted = _normalized(preferred_type)
    for preprocessor in list(pool.values()):
        if _normalized(getattr(preprocessor, "device_type", None)) == wanted:
            return preprocessor
    return None


def _unit_preprocessor_by_type(pool: PreprocessorPool, preferred_type: str) -> Any | None:
    wanted = _normalized(preferred_type)
    for unit in config.HARDWARE_UNITS:
        if _normalized(unit.get("type")) == wanted:
            preprocessor = pool.get(unit.get("id"))
            if preprocessor is not None:
                return preprocessor
    return None


def _shared_preprocessor_for_type(pool: PreprocessorPool, preferred_type: str) -> Any:
    # _normalized, like every other lookup here. A raw key meant "gpu" and "GPU" produced
    # two different pool entries and therefore two managers, each loading UVR onto the same
    # device -- the exact duplicate-load defect _normalized was introduced to stop, reached
    # by the one path that had not adopted it.
    shared_key = f"PREPROCESS::{_normalized(preferred_type)}"
    preprocessor = pool.get(shared_key)
    if preprocessor is None:
        with _SHARED_PREPROCESSOR_LOCK:
            # Re-checked under the lock: another caller may have finished building this key
            # while this one was waiting, and overwriting its entry would strand a loaded
            # model with no owner.
            preprocessor = pool.get(shared_key)
            if preprocessor is None:
                preprocessor = _create_shared_preprocessor(preferred_type)
                pool[shared_key] = preprocessor
    return preprocessor


def _create_shared_preprocessor(preferred_type: str) -> Any:
    """Build the preprocessor for one device type. Callers hold _SHARED_PREPROCESSOR_LOCK."""
    # Search every detected accelerator, not just the scheduler pool. HARDWARE_UNITS
    # is filtered to what the ASR engine can drive, and preprocessing has different
    # reach: UVR is ONNX Runtime, so it runs on an Intel iGPU that CTranslate2 cannot
    # touch. Looking only at the filtered pool made a hybrid NVIDIA+Intel host fall
    # through to create_manager() with no unit, which resolves to CUDA -- so
    # ASR_PREPROCESS_DEVICE=GPU silently ran UVR on the NVIDIA card while the startup
    # banner named the iGPU.
    # Scheduler pool first, then anything detection found but the ASR engine filter
    # removed. A unit still in the pool is the better assignment; a pruned one is
    # still a real device that UVR can use.
    # Both snapshots, widest last. DETECTED_UNITS is taken after the Intel restriction and
    # the requested-device narrowing, so it is still an ASR-filtered view;
    # DETECTED_HARDWARE_UNITS is everything detection found, before either. Reading only the
    # narrower one lost a unit those filters had dropped -- an ASR_DEVICE=NPU request prunes
    # the iGPU, which UVR can still use -- and reading only the wider one skipped the
    # scheduler-adjacent list this function was built around. The order is the preference
    # order: a unit still in the pool is the best assignment, then one the engine filter
    # removed, then anything else the machine has.
    candidates = [
        *config.HARDWARE_UNITS,
        *getattr(config, "DETECTED_UNITS", []),
        *getattr(config, "DETECTED_HARDWARE_UNITS", []),
    ]
    matched_unit = next((u for u in candidates if _normalized(u.get("type")) == _normalized(preferred_type)), None)
    if matched_unit is not None:
        preprocessor = preprocessing.create_manager(matched_unit)
    else:
        logger.warning(
            "[Preprocess] ASR_PREPROCESS_DEVICE=%s was requested but no such unit was detected; falling back to the default device.",
            preferred_type,
        )
        preprocessor = preprocessing.create_manager()
    return preprocessor


def preferred_preprocessor(pool: PreprocessorPool) -> Any:
    """Return a preprocessor pinned to the configured preprocess device when available."""
    preferred_type = config.PREPROCESS_DEVICE
    preprocessor = _pool_preprocessor_by_type(pool, preferred_type)
    if preprocessor is not None:
        return preprocessor

    preprocessor = _unit_preprocessor_by_type(pool, preferred_type)
    if preprocessor is not None:
        return preprocessor

    return _shared_preprocessor_for_type(pool, preferred_type)


def _accelerator_unit_count() -> int:
    """How many detected accelerator units exist, independent of what has been built yet.

    Colocation used to be decided from the number of *bound preprocessors*, which is a
    property of construction order rather than of the machine: at startup nothing is built,
    so a two-accelerator host looked single-accelerator and routed every unit's isolation to
    the shared preferred device. The dashboard then showed the Intel GPU on the NVIDIA card,
    and the first task genuinely ran there before the second unit bound anything. The
    hardware does not change while the process runs; the pool does.
    """
    return sum(1 for unit in config.HARDWARE_UNITS if _normalized(unit.get("type")) in config.ACCELERATED_PREPROCESS_DEVICES)


def observed_preprocessor_for_unit(pool: PreprocessorPool, unit_id: str) -> Any | None:
    """The preprocessor that *would* run this unit's isolation, without ever building one.

    Mirrors :func:`resolve_preprocessor_for_unit`'s routing, and exists because telemetry
    cannot use that function directly: its shared-type branch constructs a manager on a
    miss, which loads UVR onto a device -- from a dashboard poll.

    Reporting ``pool.get(unit_id)`` instead was the defect this replaces. On a hybrid
    NVIDIA+Intel host with ASR_PREPROCESS_DEVICE resolving to the Intel iGPU, separation
    runs on the shared ``PREPROCESS::GPU`` manager while the unit-keyed lookup returns the
    CUDA unit's own -- so the dashboard reported UVR on CUDA, unmeasured, while the log
    read "[UVR] Isolation complete on Intel(R) UHD Graphics (iGPU)". Measured on the
    RTX 3080 + UHD laptop, which is exactly the host this module exists to be honest about.
    That log line carries the ONNX provider as well now, so the two sources can be compared
    directly rather than one of them having to be trusted.
    """
    if not _is_accelerated_preprocess_device():
        return pool.get(unit_id)
    if _unit_would_colocate(unit_id):
        return pool.get(unit_id)
    preferred_type = config.PREPROCESS_DEVICE
    return (
        _pool_preprocessor_by_type(pool, preferred_type)
        or _unit_preprocessor_by_type(pool, preferred_type)
        or pool.get(f"PREPROCESS::{_normalized(preferred_type)}")
    )


def _unit_would_colocate(unit_id: str) -> bool:
    """Whether this unit will run its own isolation once its preprocessor exists.

    The observed lookup runs against a pool that is often empty -- /status is polled long
    before any task binds a preprocessor -- so asking only about what is built reports a
    routing that will not happen. This answers from the hardware instead, which is what
    decides it.
    """
    unit = next((u for u in config.HARDWARE_UNITS if u.get("id") == unit_id), None)
    return _type_would_colocate(_normalized((unit or {}).get("type")))


def _type_would_colocate(unit_type: str) -> bool:
    """Whether a unit of this type runs its own isolation rather than routing elsewhere."""
    if unit_type not in config.ACCELERATED_PREPROCESS_DEVICES:
        return False
    return unit_type == str(config.PREPROCESS_DEVICE).upper() or _accelerator_unit_count() > 1


def resolve_preprocessor_for_unit(pool: PreprocessorPool, unit_id: str) -> Any | None:
    """Return the preprocessor a task holding ``unit_id`` should run vocal isolation on."""
    if not _is_accelerated_preprocess_device():
        return pool.get(unit_id)
    unit_preprocessor = pool.get(unit_id)
    if _should_colocate_with_unit(unit_preprocessor):
        return unit_preprocessor
    return preferred_preprocessor(pool)


def _should_colocate_with_unit(unit_preprocessor: Any | None) -> bool:
    """Whether to run UVR on the task's own unit rather than the configured one.

    Co-locating spreads concurrent tasks across distinct hardware -- a GPU task and an NPU
    task preprocessing at the same time -- but only when there is something to spread
    across. With a single accelerator there is no parallelism to protect, and co-locating
    just overrides what the operator asked for: on a hybrid NVIDIA+Intel host,
    ASR_PREPROCESS_DEVICE=GPU ran UVR on CUDA because the one CUDA unit passed this check,
    while the startup banner named the Intel iGPU.

    Takes no pool: the decision now counts accelerator *units*, which is a property of the
    machine, not of how many preprocessors happen to have been built (see
    _accelerator_unit_count).
    """
    if unit_preprocessor is None:
        return False
    unit_type = str(getattr(unit_preprocessor, "device_type", "")).upper()
    if unit_type not in config.ACCELERATED_PREPROCESS_DEVICES:
        return False
    return unit_type == str(config.PREPROCESS_DEVICE).upper() or _accelerator_unit_count() > 1
