"""Tests for the small standalone scripts: status_units, check_metrics, generate_audio_matrix.

These three are entry points rather than libraries, so each is exercised the way it is
actually invoked -- stdin for one, a sysfs glob for another, and `__main__` for the third --
rather than by calling internals that production never reaches directly.
"""

import io
import json
import runpy
import sys
from pathlib import Path
from unittest import mock

import pytest

from scripts import check_metrics, status_units


def _unit(name="Intel(R) AI Boost", asr_device="NPU", asr_measured=True, uvr_device="NPU", uvr_measured=True):
    """Build one /status hardware_units entry."""
    return {
        "name": name,
        "asr_execution": {"device": asr_device, "measured": asr_measured},
        "uvr_execution": {"device": uvr_device, "measured": uvr_measured},
    }


def test_status_units_prints_a_row_per_unit(capsys, monkeypatch):
    """Each hardware unit becomes one row naming both devices and their measured flags."""
    payload = {"hardware_units": [_unit(), _unit(name="Intel(R) Graphics", asr_device="GPU", uvr_measured=False)]}
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))

    assert status_units.main() == 0

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) == 2
    assert "Intel(R) AI Boost" in lines[0]
    assert "ASR=NPU" in lines[0]
    assert "measured=True" in lines[0]
    assert "UVR=GPU" not in lines[0]
    assert "measured=False" in lines[1]


def test_status_units_truncates_a_long_device_name(capsys, monkeypatch):
    """The name column is bounded, so a long device name cannot skew every row."""
    payload = {"hardware_units": [_unit(name="A" * 60)]}
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))

    assert status_units.main() == 0

    assert "A" * 26 in capsys.readouterr().out
    assert "A" * 27 not in capsys.readouterr().out


def test_status_units_reports_unreadable_status_rather_than_raising(capsys, monkeypatch):
    """A body that is not JSON is a missing-evidence row, not a crash.

    This runs against a live service over SSH, so a truncated or error body has to degrade
    into "no evidence" -- a traceback here would abort the validation run that was only
    trying to record what the service said.
    """
    monkeypatch.setattr(sys, "stdin", io.StringIO("<html>502 Bad Gateway</html>"))

    assert status_units.main() == 0

    assert "not readable" in capsys.readouterr().out


def test_status_units_reports_an_empty_unit_list(capsys, monkeypatch):
    """A well-formed payload with no units still says so rather than printing nothing."""
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"hardware_units": []})))

    assert status_units.main() == 0

    assert "no hardware units" in capsys.readouterr().out


def test_check_metrics_reports_every_discovered_path(capsys):
    """Discovered sysfs paths are printed, and readable GPU attributes have their value read."""

    def _glob(pattern):
        if "gpu_busy_percent" in pattern:
            return ["/sys/class/drm/card0/device/gpu_busy_percent"]
        if "class/accel" in pattern:
            return ["/sys/class/accel/accel0/device/utilization"]
        return []

    with mock.patch.object(check_metrics.glob, "glob", side_effect=_glob):
        with mock.patch("builtins.open", mock.mock_open(read_data="42\n")):
            check_metrics.find_metrics()

    out = capsys.readouterr().out
    assert "gpu_busy_percent" in out
    assert "42%" in out
    assert "/sys/class/accel/accel0/device/utilization" in out


def test_check_metrics_survives_an_unreadable_attribute(capsys):
    """A sysfs attribute that exists but refuses to be read is skipped, not fatal.

    Intel's driver exposes attributes that raise on read depending on power state, and this
    script's whole job is to report what it can see.
    """
    with mock.patch.object(check_metrics.glob, "glob", return_value=["/sys/class/drm/card0/device/gpu_busy_percent"]):
        with mock.patch("builtins.open", side_effect=OSError("Device or resource busy")):
            check_metrics.find_metrics()

    assert "gpu_busy_percent" in capsys.readouterr().out


def test_generate_audio_matrix_entrypoint_delegates_to_the_cli(monkeypatch):
    """The wrapper exists only to fix sys.path and hand off to scripts.audio_matrix.cli.run."""
    called = {}

    def _run():
        called["ran"] = True
        return 3

    monkeypatch.setattr("scripts.audio_matrix.cli.run", _run)
    script = Path(__file__).resolve().parents[2] / "scripts" / "generate_audio_matrix.py"

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(script), run_name="__main__")

    assert called.get("ran") is True
    assert exit_info.value.code == 3
