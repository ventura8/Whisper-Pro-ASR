"""Serialisation of OpenVINO session creation, scoped by accelerator family.

Two threads building an OpenVINO session on the same Intel device family at once is not
safe, so initialisation is serialised per family (all GPU slots share one lock, all NPU
slots another) rather than globally -- a global lock would make a GPU unit wait behind an
NPU unit that it does not contend with.
"""

from __future__ import annotations

import threading

from modules.inference.pipeline import openvino_resolver

_LOCKS: dict[str, threading.Lock] = {}
_GUARD = threading.Lock()


def lock_key(device_id: str, device_type: str) -> str:
    """Return the lock key for a device: its OpenVINO family, or the device itself."""
    target = (device_id or device_type or "OPENVINO").upper()
    family = openvino_resolver.openvino_device_family(target) or target
    if family in {"GPU", "NPU"}:
        return family
    return target


def lock_for(device_id: str, device_type: str) -> threading.Lock:
    """Return a stable lock scoped by accelerator family/slot for OpenVINO init."""
    key = lock_key(device_id, device_type)
    with _GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _LOCKS[key] = lock
        return lock
