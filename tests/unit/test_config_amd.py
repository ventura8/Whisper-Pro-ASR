"""Tests for AMD configuration support in modules/config.py"""

import importlib
import os
from unittest import mock

import modules.core.config as config_module
import modules.core.config_helpers as config_helpers_module


class TestConfigAmd:
    """Test suite for AMD hardware config and resolution."""

    def _reset_config_module(self):
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch("ctranslate2.get_cuda_device_count", return_value=0),
            mock.patch("openvino.Core", side_effect=RuntimeError("no intel")),
            mock.patch("modules.core.config_helpers._has_amd_hardware", return_value=False),
        ):
            importlib.reload(config_module)

    def _build_baseline_env(self, env: dict[str, str]) -> dict[str, str]:
        baseline = {k: os.environ[k] for k in ("PATH", "SYSTEMROOT", "WINDIR", "TMP", "TEMP") if k in os.environ}
        baseline.update(env)
        return baseline

    def test_amd_gpu_detection(self):
        """Test that AMD GPU is detected and registered."""
        env = self._build_baseline_env({"ASR_DEVICE": "AUTO", "ASR_PREPROCESS_DEVICE": "AUTO", "MAX_AMD_UNITS": "1"})
        try:
            with (
                mock.patch.dict(os.environ, env, clear=True),
                mock.patch("ctranslate2.get_cuda_device_count", return_value=0),
                mock.patch("openvino.Core", side_effect=RuntimeError("no intel")),
                mock.patch("modules.core.config_helpers._has_amd_hardware", return_value=True),
                mock.patch(
                    "modules.core.config_helpers.os.path.exists",
                    side_effect=lambda p: p in ["/dev/dxg", "/app/libs/cpu"],
                ),
            ):
                importlib.reload(config_module)
                assert config_module.DEVICE == "AMD"
                assert config_module.PREPROCESS_DEVICE == "AMD"
                assert any(u["type"] == "AMD" for u in config_module.HARDWARE_UNITS)
        finally:
            self._reset_config_module()

    def test_amd_gpu_skipped_when_max_amd_zero(self):
        """Test that AMD GPU registration is skipped when MAX_AMD_UNITS is 0."""
        env = self._build_baseline_env({"ASR_DEVICE": "AUTO", "ASR_PREPROCESS_DEVICE": "AUTO", "MAX_AMD_UNITS": "0"})
        try:
            with (
                mock.patch.dict(os.environ, env, clear=True),
                mock.patch("ctranslate2.get_cuda_device_count", return_value=0),
                mock.patch("openvino.Core", side_effect=RuntimeError("no intel")),
                mock.patch("modules.core.config_helpers._has_amd_hardware", return_value=True),
            ):
                importlib.reload(config_module)
                assert not any(u["type"] == "AMD" for u in config_module.HARDWARE_UNITS)
        finally:
            self._reset_config_module()

    def test_amd_gpu_ctranslate2_cpu_fallback(self):
        """Test that FASTER-WHISPER engine falls back to CPU on AMD device."""
        env = self._build_baseline_env({"ASR_DEVICE": "AMD", "ASR_ENGINE": "FASTER-WHISPER", "MAX_AMD_UNITS": "1"})
        try:
            with (
                mock.patch.dict(os.environ, env, clear=True),
                mock.patch("ctranslate2.get_cuda_device_count", return_value=0),
                mock.patch("modules.core.config_helpers._has_amd_hardware", return_value=True),
                mock.patch(
                    "modules.core.config_helpers.os.path.exists",
                    side_effect=lambda p: p in ["/dev/dxg", "/app/libs/cpu"],
                ),
            ):
                importlib.reload(config_module)
                assert config_module.ASR_ENGINE_DEVICE == "cpu"
                assert config_module.ASR_ENGINE_COMPUTE_TYPE == "int8"
        finally:
            self._reset_config_module()

    def test_is_amd_drm_present(self):
        """Test _is_amd_drm_present with and without DRM folder and with non-AMD vendor ID."""
        is_amd_drm_present_fn = getattr(config_helpers_module, "_is_amd_drm_present")
        # 1. When DRM folder does not exist
        with mock.patch("modules.core.config_helpers.os.path.isdir", return_value=False):
            assert not is_amd_drm_present_fn()

        # 2. When DRM folder exists and returns AMD vendor ID
        with mock.patch("modules.core.config_helpers.os.path.isdir", return_value=True):
            with mock.patch(
                "modules.core.config_helpers._iter_drm_vendor_files",
                return_value=["/sys/class/drm/card0/device/vendor"],
            ):
                with mock.patch("modules.core.config_helpers._read_vendor_id", return_value="0x1002"):
                    assert is_amd_drm_present_fn()

        # 3. When DRM folder exists and returns non-AMD vendor ID
        with mock.patch("modules.core.config_helpers.os.path.isdir", return_value=True):
            with mock.patch(
                "modules.core.config_helpers._iter_drm_vendor_files",
                return_value=["/sys/class/drm/card0/device/vendor"],
            ):
                with mock.patch("modules.core.config_helpers._read_vendor_id", return_value="0x8086"):
                    assert not is_amd_drm_present_fn()

    def test_has_amd_hardware_kfd(self):
        """Test _has_amd_hardware returns True when /dev/kfd exists."""
        has_amd_hardware_fn = getattr(config_helpers_module, "_has_amd_hardware")
        with mock.patch("modules.core.config_helpers.os.path.exists", side_effect=lambda p: p == "/dev/kfd"):
            assert has_amd_hardware_fn()

    def test_has_amd_hardware_dri(self):
        """Test _has_amd_hardware returns True when /dev/dri and AMD DRM present."""
        has_amd_hardware_fn = getattr(config_helpers_module, "_has_amd_hardware")
        with mock.patch("modules.core.config_helpers.os.path.exists", side_effect=lambda p: p == "/dev/dri"):
            with mock.patch("modules.core.config_helpers._is_amd_drm_present", return_value=True):
                assert has_amd_hardware_fn()

    def test_has_amd_hardware_none(self):
        """Test _has_amd_hardware returns False when no device exists."""
        has_amd_hardware_fn = getattr(config_helpers_module, "_has_amd_hardware")
        with (
            mock.patch("modules.core.config_helpers.os.path.exists", return_value=False),
            mock.patch("modules.core.config_helpers._has_amd_wsl_hardware", return_value=False),
        ):
            assert not has_amd_hardware_fn()

    def test_has_amd_hardware_rocm(self):
        """Test ROCm hardware detection positive and negative branches."""
        has_amd_hardware_fn = getattr(config_helpers_module, "_has_amd_hardware")
        check_ort_providers_fn = getattr(config_helpers_module, "_check_amd_ort_providers")
        fake_ort = mock.MagicMock()
        fake_ort.get_available_providers.return_value = ["ROCMExecutionProvider"]
        with mock.patch("modules.core.config_helpers.os.path.exists", side_effect=lambda p: p == "/dev/kfd"):
            assert has_amd_hardware_fn() is True
        with mock.patch("importlib.import_module", return_value=fake_ort):
            assert check_ort_providers_fn() is True
        with (
            mock.patch("modules.core.config_helpers.os.path.exists", return_value=False),
            mock.patch("modules.core.config_helpers._has_amd_wsl_hardware", return_value=False),
            mock.patch("modules.core.config_helpers._is_amd_drm_present", return_value=False),
        ):
            assert has_amd_hardware_fn() is False

    def test_has_amd_hardware_dml(self):
        """Test DirectML / WSL hardware detection positive and negative branches."""
        has_amd_hardware_fn = getattr(config_helpers_module, "_has_amd_hardware")
        check_ort_providers_fn = getattr(config_helpers_module, "_check_amd_ort_providers")
        fake_ort = mock.MagicMock()
        fake_ort.get_available_providers.return_value = ["DmlExecutionProvider"]
        with mock.patch("modules.core.config_helpers._has_amd_wsl_hardware", return_value=True):
            assert has_amd_hardware_fn() is True
        with (
            mock.patch("modules.core.config_helpers.sys.platform", "win32"),
            mock.patch.dict("os.environ", {"ASR_DEVICE": "AMD"}),
            mock.patch("importlib.import_module", return_value=fake_ort),
        ):
            assert check_ort_providers_fn() is True
        with (
            mock.patch("modules.core.config_helpers._has_amd_wsl_hardware", return_value=False),
            mock.patch("modules.core.config_helpers.os.path.exists", return_value=False),
        ):
            assert has_amd_hardware_fn() is False

    def test_has_amd_wsl_driver_folder_missing(self):
        """Test _has_amd_wsl_driver_folder returns False for nonexistent folder."""
        folder_fn = getattr(config_helpers_module, "_has_amd_wsl_driver_folder")
        assert folder_fn("/nonexistent/folder") is False

    def test_is_amd_wsl_driver_present_no_dir(self):
        """Test _is_amd_wsl_driver_present returns False when drivers dir is missing."""
        driver_fn = getattr(config_helpers_module, "_is_amd_wsl_driver_present")
        with mock.patch("os.path.isdir", return_value=False):
            assert driver_fn() is False

    def test_is_amd_wsl_driver_present_os_error(self):
        """Test _is_amd_wsl_driver_present handles OSError gracefully."""
        driver_fn = getattr(config_helpers_module, "_is_amd_wsl_driver_present")
        with mock.patch("os.path.isdir", return_value=True):
            with mock.patch("os.listdir", side_effect=OSError("denied")):
                assert driver_fn() is False

    def test_is_amd_wsl_driver_present_found(self):
        """Test _is_amd_wsl_driver_present returns True when driver is found."""
        driver_fn = getattr(config_helpers_module, "_is_amd_wsl_driver_present")
        with mock.patch("os.path.isdir", return_value=True):
            with mock.patch("os.listdir", return_value=["driver_folder"]):
                with mock.patch("modules.core.config_helpers._has_amd_wsl_driver_folder", return_value=True):
                    assert driver_fn() is True

    def test_has_amd_wsl_hardware_present(self):
        """Test _has_amd_wsl_hardware returns True when nodes exist."""
        wsl_hw_fn = getattr(config_helpers_module, "_has_amd_wsl_hardware")
        with mock.patch("os.path.exists", side_effect=lambda p: p in ["/dev/dxg", "/opt/rocm/lib/librocdxg.so"]):
            with mock.patch("modules.core.config_helpers._is_amd_wsl_driver_present", return_value=True):
                assert wsl_hw_fn() is True

    def test_check_amd_dml_non_windows(self):
        """Test _check_amd_dml returns False on non-win32 platform."""
        check_dml = getattr(config_helpers_module, "_check_amd_dml")
        with mock.patch("modules.core.config_helpers.sys.platform", "linux"):
            assert check_dml(["DmlExecutionProvider"]) is False

    def test_check_amd_dml_windows(self):
        """Test _check_amd_dml on win32 platform for positive and negative cases."""
        check_dml = getattr(config_helpers_module, "_check_amd_dml")
        with mock.patch("modules.core.config_helpers.sys.platform", "win32"):
            with mock.patch.dict("os.environ", {"ASR_DEVICE": "AMD"}):
                assert check_dml(["DmlExecutionProvider"]) is True
            with mock.patch.dict("os.environ", {"ASR_DEVICE": "CPU", "ASR_PREPROCESS_DEVICE": "CPU"}):
                assert check_dml(["DmlExecutionProvider"]) is False
            with mock.patch.dict("os.environ", {"ASR_DEVICE": "AMD"}):
                assert check_dml(["CPUExecutionProvider"]) is False

    def test_has_rocm_hardware(self):
        """Test _has_rocm_hardware detects ROCm provider and rejects non-ROCm providers."""
        has_rocm = getattr(config_helpers_module, "_has_rocm_hardware")
        assert has_rocm(["ROCMExecutionProvider"]) is True
        assert has_rocm(["CPUExecutionProvider", "CUDAExecutionProvider"]) is False

    def test_check_amd_ort_providers_error(self):
        """Test _check_amd_ort_providers handles ImportError."""
        check_ort = getattr(config_helpers_module, "_check_amd_ort_providers")
        with mock.patch("importlib.import_module", side_effect=ImportError("no ort")):
            assert check_ort() is False

    def test_detect_amd_hardware_exception(self):
        """Test _detect_amd_hardware handles exceptions without raising."""
        detect_amd = getattr(config_helpers_module, "_detect_amd_hardware")
        with mock.patch("modules.core.config_helpers._has_amd_hardware", side_effect=RuntimeError("boom")):
            detect_amd(1, [], {})
