"""Render the measured execution devices from a /status payload read on stdin.

Kept apart from the validation scripts because both the local and the remote runner need
it, and because the distinction it prints is the one the startup banner cannot make: the
banner reports the device that was *requested*, while these rows report the device an
actual inference touched. A row that names an accelerator with ``measured=False`` has
proved nothing about hardware.
"""

import json
import sys


def _format(unit: dict) -> str:
    asr, uvr = unit["asr_execution"], unit["uvr_execution"]
    return (
        f"  {unit['name'][:26]:28} "
        f"ASR={asr['device']:10} measured={str(asr['measured']):5} | "
        f"UVR={uvr['device']:10} measured={uvr['measured']}"
    )


def _report_lines() -> list[str]:
    """One row per hardware unit on stdin, or the reason no rows could be produced."""
    try:
        units = json.load(sys.stdin).get("hardware_units", [])
    except (ValueError, TypeError):
        return ["  /status was not readable -- no measured device evidence for this row"]
    if not units:
        return ["  /status reported no hardware units"]
    return [_format(unit) for unit in units]


def main() -> int:
    """Print one row per hardware unit, or a reason no rows could be printed.

    Always exits 0: this only reports evidence, and a missing row is itself the finding.
    """
    for line in _report_lines():
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
