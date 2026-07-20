"""AMD-specific hardware utilization resolution helpers.

This module exists to keep `metrics_discovery.py` focused (and under size limits)
while centralizing the AMD resolution decision tree.
"""

from __future__ import annotations

from typing import Any, Callable

from modules.monitoring.metrics_amd import _amd_sysfs_paths, _probe_amd_smi_metrics


def resolve_amd_utilization(
    unit_id: Any,
    *,
    amd_smi_loads: list[int] | None = None,
    resolve_index: Callable[[Any], int],
    inactive_zero_result: Callable[..., int | None],
    store_real_accelerator_sample: Callable[[str, int, int], None],
    fetch_single_accelerator_load: Callable[..., int],
) -> int:
    """Resolve AMD utilization using (1) app activity gating and then (2) SMI loads."""
    idx = resolve_index(unit_id)

    inactive_result = inactive_zero_result("AMD", unit_id, idx, exclude_nvidia=False)
    if inactive_result is not None:
        return inactive_result

    smi_loads = amd_smi_loads if amd_smi_loads is not None else _probe_amd_smi_metrics()
    if idx < len(smi_loads):
        store_real_accelerator_sample("AMD", idx, smi_loads[idx])
        return smi_loads[idx]

    return fetch_single_accelerator_load(
        unit_id,
        idx,
        "AMD",
        _amd_sysfs_paths(idx),
        windows_cmd=None,
        busy_value=100,
        exclude_nvidia=False,
    )
