"""Unit tests for NVIDIA VRAM helper functions."""

from unittest import mock

from modules.core import nvidia_vram_helpers, process_exec, utils


def test_get_nvidia_vram_usage_mb_sums_visible_devices():
    """VRAM probe should sum memory.used across all visible NVIDIA devices."""
    with (
        mock.patch("modules.core.nvidia_vram_helpers.which", return_value="/usr/bin/nvidia-smi"),
        mock.patch("modules.core.process_exec.check_output_text", return_value="1024\n512\n"),
    ):
        assert utils.get_nvidia_vram_usage_mb() == 1536


def test_get_nvidia_vram_usage_mb_returns_none_without_nvidia_smi():
    """VRAM probe should degrade gracefully when nvidia-smi is unavailable."""
    with mock.patch("modules.core.nvidia_vram_helpers.which", return_value=None):
        assert utils.get_nvidia_vram_usage_mb() is None


def test_get_nvidia_vram_usage_mb_query_failure_returns_none():
    """VRAM query errors should degrade to None instead of raising."""
    with (
        mock.patch("modules.core.nvidia_vram_helpers.which", return_value="/usr/bin/nvidia-smi"),
        mock.patch(
            "modules.core.process_exec.check_output_text",
            side_effect=process_exec.CommandExecutionError(["nvidia-smi"], 1),
        ),
    ):
        assert nvidia_vram_helpers.get_nvidia_vram_usage_mb() is None


def test_get_nvidia_vram_usage_mb_skips_invalid_lines():
    """Invalid memory lines should be skipped while summing valid values."""
    with (
        mock.patch("modules.core.nvidia_vram_helpers.which", return_value="/usr/bin/nvidia-smi"),
        mock.patch("modules.core.process_exec.check_output_text", return_value="1024\nbad\n\n512\n"),
    ):
        assert nvidia_vram_helpers.get_nvidia_vram_usage_mb() == 1536


def test_get_nvidia_vram_usage_mb_returns_none_for_unparseable_output():
    """All-invalid nvidia-smi output should return None."""
    with (
        mock.patch("modules.core.nvidia_vram_helpers.which", return_value="/usr/bin/nvidia-smi"),
        mock.patch("modules.core.process_exec.check_output_text", return_value="bad\n"),
    ):
        assert nvidia_vram_helpers.get_nvidia_vram_usage_mb() is None
