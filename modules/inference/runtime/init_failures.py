"""Why a unit's engine failed to load, in the words a caller can act on.

Split out of ``model_manager`` to keep that module inside the project's 600-line limit.
The record itself lives here too, so the description and the store it feeds stay together:
``model_manager`` re-exports ``LAST_INIT_ERROR`` and the scheduler reads it from there, and
because it is one dict object rather than a copy, a write through either name is visible to
both.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

#: unit_id -> why its last load attempt failed, so the scheduler can report the actual
#: cause instead of the downstream "engine pool is empty" symptom.
LAST_INIT_ERROR: dict[str, str] = {}


def record_init_failure(engine_type: str, unit: dict, error: Exception) -> None:
    """Log a load failure and record it against the unit, when the unit can be identified."""
    unit_id = (unit.get("id") if isinstance(unit, dict) else None) or ""
    logger.error("[Engine] Failed to load %s: %s", unit_id or "an unidentified unit", error)
    if unit_id:
        LAST_INIT_ERROR[unit_id] = describe_init_failure(engine_type, unit, error)


def describe_init_failure(engine_type: str, unit: dict, error: Exception) -> str:
    """Turn a load failure into something a caller can act on.

    A missing engine dependency is the common case and the least obvious one: not every
    image ships every engine (WhisperX is only in the `full` and `nvidia-whisperx`
    targets), so the honest message names the engine and the image rather than the empty
    pool it causes.
    """
    engine = f"ASR engine {engine_type}" if engine_type else "The ASR engine"
    if isinstance(error, ImportError):
        return (
            f"{engine} is not available in this image "
            f"(missing dependency: {error.name or error}). Use an image that ships it, or select a different ASR_ENGINE."
        )
    return f"{engine} failed to load{_unit_suffix(unit)}: {type(error).__name__}: {error}"


def _unit_suffix(unit: dict) -> str:
    """ " on <unit>", or nothing when the unit cannot be named.

    Omitted rather than filled with a sentinel: "failed to load on ?" reads like a unit
    literally named "?" on the dashboard, sending a reader looking for something that does
    not exist.
    """
    if not isinstance(unit, dict):
        return ""
    label = unit.get("name") or unit.get("id") or ""
    return f" on {label}" if label else ""
