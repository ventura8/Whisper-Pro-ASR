"""Tests for Windows accelerator counter helpers."""

from unittest import mock

from modules.core import process_exec
from modules.monitoring import windows_accelerator_counters as counters


def test_normalize_windows_counter_value_rejects_negative():
    assert counters._normalize_windows_counter_value("-1") is None


def test_normalize_windows_counter_value_caps_at_100():
    assert counters._normalize_windows_counter_value("150.5") == 100


def test_read_first_int_value_skips_invalid_paths(tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_text("not-a-number", encoding="utf-8")
    good = tmp_path / "good.txt"
    good.write_text("42\n", encoding="utf-8")
    assert counters._read_first_int_value([str(bad), str(good)]) == 42


def test_run_windows_accelerator_counter_returns_none_on_linux():
    with mock.patch("platform.system", return_value="Linux"):
        assert counters._run_windows_accelerator_counter(counters._gpu_counter_command()) is None


def test_run_windows_accelerator_counter_handles_command_error():
    with (
        mock.patch("platform.system", return_value="Windows"),
        mock.patch("modules.monitoring.windows_accelerator_counters.which", return_value="powershell"),
        mock.patch(
            "modules.core.process_exec.check_output_text",
            side_effect=process_exec.CommandExecutionError(["powershell"], 1),
        ),
    ):
        assert counters._run_windows_accelerator_counter(counters._npu_counter_command()) is None
