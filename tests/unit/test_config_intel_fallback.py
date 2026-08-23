"""Regression tests for Intel fallback detection branches in config loading."""

import importlib
import os
from pathlib import Path
from unittest import mock

import pytest

import modules.core.config as config_module
import modules.core.config_helpers as config_helpers_module

pytestmark = pytest.mark.usefixtures("restore_config_after_reload")


def _exists_with_intel_nodes(path: str, real_exists):
    if path in {"/dev/accel/accel0", "/dev/dri"}:
        return True
    return real_exists(path)


def test_intel_node_fallback_used_when_openvino_device_list_is_empty():
    """Empty OpenVINO enumeration should still register Intel node-fallback resources."""
    real_exists = os.path.exists

    with mock.patch.dict(os.environ, {"ASR_DEVICE": "AUTO", "MAX_GPU_UNITS": "1", "MAX_NPU_UNITS": "1"}):
        with (
            mock.patch("ctranslate2.get_cuda_device_count", return_value=0),
            mock.patch("openvino.Core") as mock_core_ctor,
            mock.patch("modules.core.config_helpers._has_amd_hardware", return_value=False),
            mock.patch("modules.core.config_helpers._is_intel_drm_present", return_value=True),
            mock.patch("os.path.exists", side_effect=lambda p: _exists_with_intel_nodes(p, real_exists)),
        ):
            mock_core = mock.MagicMock()
            mock_core.available_devices = []
            mock_core_ctor.return_value = mock_core
            importlib.reload(config_module)

            unit_types = {u["type"] for u in config_module.HARDWARE_UNITS}
            assert "GPU" in unit_types
            assert "NPU" in unit_types


def test_intel_node_fallback_used_when_openvino_has_only_cpu():
    """OpenVINO CPU-only enumeration should still trigger Intel node fallback."""
    real_exists = os.path.exists

    with mock.patch.dict(os.environ, {"ASR_DEVICE": "AUTO", "MAX_GPU_UNITS": "1", "MAX_NPU_UNITS": "1"}):
        with (
            mock.patch("ctranslate2.get_cuda_device_count", return_value=0),
            mock.patch("openvino.Core") as mock_core_ctor,
            mock.patch("modules.core.config_helpers._has_amd_hardware", return_value=False),
            mock.patch("modules.core.config_helpers._is_intel_drm_present", return_value=True),
            mock.patch("os.path.exists", side_effect=lambda p: _exists_with_intel_nodes(p, real_exists)),
        ):
            mock_core = mock.MagicMock()
            mock_core.available_devices = ["CPU"]
            mock_core_ctor.return_value = mock_core
            importlib.reload(config_module)

            unit_types = {u["type"] for u in config_module.HARDWARE_UNITS}
            assert "GPU" in unit_types
            assert "NPU" in unit_types


def test_is_intel_drm_present_false_when_sysfs_missing():
    """Missing /sys/class/drm must not claim an Intel GPU is present."""
    is_intel_drm_present_fn = getattr(config_helpers_module, "_is_intel_drm_present")
    with mock.patch("os.path.isdir", return_value=False):
        assert is_intel_drm_present_fn() is False


def test_is_intel_drm_present_false_when_vendor_files_missing():
    """Unpopulated DRM sysfs must not claim an Intel GPU is present."""
    is_intel_drm_present_fn = getattr(config_helpers_module, "_is_intel_drm_present")
    with (
        mock.patch("os.path.isdir", return_value=True),
        mock.patch("modules.core.config_helpers._iter_drm_vendor_files", return_value=[]),
    ):
        assert is_intel_drm_present_fn() is False


def test_is_intel_drm_present_true_for_intel_vendor(tmp_path: Path):
    """Intel vendor id 0x8086 on a render node must be treated as Intel DRM."""
    is_intel_drm_present_fn = getattr(config_helpers_module, "_is_intel_drm_present")
    vendor = tmp_path / "vendor"
    vendor.write_text("0x8086\n", encoding="utf-8")
    with (
        mock.patch("os.path.isdir", return_value=True),
        mock.patch("modules.core.config_helpers._iter_drm_vendor_files", return_value=[str(vendor)]),
    ):
        assert is_intel_drm_present_fn() is True
