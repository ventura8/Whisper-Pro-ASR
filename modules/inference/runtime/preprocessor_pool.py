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


def _pool_preprocessor_by_type(pool: PreprocessorPool, preferred_type: str) -> Any | None:
    # Snapshot: the pool is inserted into by other threads building shared preprocessors,
    # and iterating it live raises "dictionary changed size during iteration" mid-request.
    for preprocessor in list(pool.values()):
        if getattr(preprocessor, "device_type", None) == preferred_type:
            return preprocessor
    return None


def _unit_preprocessor_by_type(pool: PreprocessorPool, preferred_type: str) -> Any | None:
    for unit in config.HARDWARE_UNITS:
        if unit.get("type") == preferred_type:
            preprocessor = pool.get(unit.get("id"))
            if preprocessor is not None:
                return preprocessor
    return None


def _shared_preprocessor_for_type(pool: PreprocessorPool, preferred_type: str) -> Any:
    shared_key = f"PREPROCESS::{preferred_type}"
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
    candidates = [*config.HARDWARE_UNITS, *getattr(config, "DETECTED_UNITS", [])]
    matched_unit = next((u for u in candidates if u.get("type") == preferred_type), None)
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


def _accelerator_preprocessor_count(pool: PreprocessorPool) -> int:
    """How many distinct accelerators currently have a preprocessor bound to a unit.

    Counts only unit-keyed entries: the shared ``PREPROCESS::<type>`` fallbacks are not
    hardware assignments and would otherwise make a single-accelerator host look like a
    parallel one.
    """
    return sum(
        1
        for key, preprocessor in list(pool.items())
        if not str(key).startswith("PREPROCESS::")
        and str(getattr(preprocessor, "device_type", "")).upper() in config.ACCELERATED_PREPROCESS_DEVICES
    )


def resolve_preprocessor_for_unit(pool: PreprocessorPool, unit_id: str) -> Any | None:
    """Return the preprocessor a task holding ``unit_id`` should run vocal isolation on."""
    if not _is_accelerated_preprocess_device():
        return pool.get(unit_id)
    unit_preprocessor = pool.get(unit_id)
    if _should_colocate_with_unit(pool, unit_preprocessor):
        return unit_preprocessor
    return preferred_preprocessor(pool)


def _should_colocate_with_unit(pool: PreprocessorPool, unit_preprocessor: Any | None) -> bool:
    """Whether to run UVR on the task's own unit rather than the configured one.

    Co-locating spreads concurrent tasks across distinct hardware -- a GPU task and an NPU
    task preprocessing at the same time -- but only when there is something to spread
    across. With a single accelerator there is no parallelism to protect, and co-locating
    just overrides what the operator asked for: on a hybrid NVIDIA+Intel host,
    ASR_PREPROCESS_DEVICE=GPU ran UVR on CUDA because the one CUDA unit passed this check,
    while the startup banner named the Intel iGPU.
    """
    if unit_preprocessor is None:
        return False
    unit_type = str(getattr(unit_preprocessor, "device_type", "")).upper()
    if unit_type not in config.ACCELERATED_PREPROCESS_DEVICES:
        return False
    return unit_type == str(config.PREPROCESS_DEVICE).upper() or _accelerator_preprocessor_count(pool) > 1
