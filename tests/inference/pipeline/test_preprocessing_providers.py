"""Preprocessing pipeline tests (split from test_preprocessing.py)."""

import logging
from unittest import mock

import pytest

from modules.inference.pipeline import openvino_resolver, preprocessing
from modules.inference.pipeline.preprocessing import PreprocessingManager

logger = logging.getLogger(__name__)


@pytest.fixture
def prep_manager():
    """Fixture to provide a clean PreprocessingManager instance."""
    unit = {"id": "CPU", "type": "CPU", "name": "CPU"}
    return PreprocessingManager(assigned_unit=unit)


@pytest.fixture(autouse=True)
def reset_openvino_family_circuit_breaker():
    openvino_resolver.clear_openvino_disabled_families()
    yield
    openvino_resolver.clear_openvino_disabled_families()


class TestResolveProviders:
    """Tests for provider resolution logic."""

    def test_resolve_cuda(self, prep_manager):
        prep_manager._device_type = "CUDA"
        prep_manager._device_id = "cuda:1"
        available = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        providers, options = prep_manager._resolve_providers(available)
        assert "CUDAExecutionProvider" in providers
        assert options[0]["device_id"] == 1

    def test_resolve_openvino_gpu(self, prep_manager):
        prep_manager._device_type = "GPU"
        prep_manager._device_id = "GPU.0"
        available = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
        with mock.patch("modules.inference.pipeline.preprocessing.config") as mock_cfg:
            mock_cfg.OV_CACHE_DIR = "/tmp/ov"
            providers, options = prep_manager._resolve_providers(available)
            assert "OpenVINOExecutionProvider" in providers
            assert options[0]["device_type"] == "GPU.0"

    def test_resolve_auto_cuda(self, prep_manager):
        prep_manager._device_type = "AUTO"
        available = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        providers, _ = prep_manager._resolve_providers(available)
        assert "CUDAExecutionProvider" in providers

    def test_resolve_auto_ov(self, prep_manager):
        prep_manager._device_type = "AUTO"
        available = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
        with mock.patch(
            "modules.inference.pipeline.preprocessing._get_available_openvino_devices",
            return_value=["CPU", "GPU.0"],
        ):
            providers, _ = prep_manager._resolve_providers(available)
        assert "OpenVINOExecutionProvider" in providers

    def test_resolve_auto_cpu(self, prep_manager):
        prep_manager._device_type = "AUTO"
        available = ["CPUExecutionProvider"]
        providers, _ = prep_manager._resolve_providers(available)
        assert providers == ["CPUExecutionProvider"]

    def test_resolve_cuda_no_index(self, prep_manager):
        """Test CUDA resolution without an explicit index."""
        prep_manager._device_type = "CUDA"
        prep_manager._device_id = "CUDA"
        available = ["CUDAExecutionProvider"]
        _, options = prep_manager._resolve_providers(available)
        assert options[0]["device_id"] == "0"

    def test_resolve_cuda_preserves_non_default_gpu_index(self, prep_manager):
        """CUDA resolution should preserve explicit non-default GPU ids in multi-GPU hosts."""
        prep_manager._device_type = "CUDA"
        prep_manager._device_id = "cuda:3"
        available = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        providers, options = prep_manager._resolve_providers(available)

        assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
        assert options[0]["device_id"] == 3

    def test_resolve_openvino_custom_device(self, prep_manager):
        """Test OpenVINO resolution with a custom device ID."""
        prep_manager._device_type = "OpenVINO"
        prep_manager._device_id = "GPU.1"
        available = ["OpenVINOExecutionProvider"]
        with mock.patch("modules.inference.pipeline.preprocessing.config") as mock_cfg:
            mock_cfg.OV_CACHE_DIR = "/tmp/ov"
            _, options = prep_manager._resolve_providers(available)
            assert options[0]["device_type"] == "GPU.1"

    def test_resolve_openvino_generic_npu_maps_to_concrete_id(self, prep_manager):
        """Generic NPU device label should stay generic for ORT-compatible provider options."""
        prep_manager._device_type = "NPU"
        prep_manager._device_id = "NPU"
        available = ["OpenVINOExecutionProvider"]
        with mock.patch("modules.inference.pipeline.preprocessing.config") as mock_cfg:
            mock_cfg.OV_CACHE_DIR = "/tmp/ov"
            _, options = prep_manager._resolve_providers(available)
            assert options[0]["device_type"] == "NPU"

    def test_resolve_provider_config_amd_auto_branch(self):
        """Test provider resolution for AMD when target_prep is AMD or AUTO."""
        from modules.inference.pipeline.preprocessing import provider as prep_provider

        with mock.patch(
            "modules.inference.pipeline.openvino_provider_dispatch.has_amd_provider",
            return_value=True,
        ):
            with mock.patch(
                "modules.inference.pipeline.openvino_provider_dispatch.amd_provider_config",
                return_value=(["ROCMExecutionProvider"], [{}]),
            ):
                res = prep_provider.resolve_provider_config_for_preprocessing(
                    "AMD",
                    "amd:0",
                    ["ROCMExecutionProvider"],
                    [],
                    "/tmp",
                    preprocess_threads=2,
                )
                assert res[0] == ["ROCMExecutionProvider"]

                res_auto = prep_provider.auto_provider_config_for_preprocessing(
                    ["ROCMExecutionProvider"],
                    [],
                    "/tmp",
                    2,
                    target_prep="AUTO",
                )
                assert res_auto[0] == ["ROCMExecutionProvider"]

    def test_resolve_openvino_or_cpu_disabled_or_missing(self):
        """Test _resolve_openvino_or_cpu returns CPU when missing or disabled."""
        from modules.inference.pipeline.preprocessing import provider as prep_provider

        res = prep_provider._resolve_openvino_or_cpu("GPU.0", ["CPUExecutionProvider"], [], "/tmp", 2)
        assert res == (["CPUExecutionProvider"], [{}])

        with mock.patch(
            "modules.inference.pipeline.openvino_resolver.is_openvino_family_disabled",
            return_value=True,
        ):
            res_disabled = prep_provider._resolve_openvino_or_cpu("GPU.0", ["OpenVINOExecutionProvider"], [], "/tmp", 2)
            assert res_disabled == (["CPUExecutionProvider"], [{}])

    def test_resolve_non_cuda_amd_preprocessing_cpu(self):
        """Test _resolve_non_cuda_amd_preprocessing returns CPU for CPU target."""
        from modules.inference.pipeline.preprocessing import provider as prep_provider

        res = prep_provider._resolve_non_cuda_amd_preprocessing(
            "CPU",
            "CPU",
            ["CPUExecutionProvider"],
            available_openvino_devices=[],
            ov_cache_dir="/tmp",
            preprocess_threads=2,
        )
        assert res == (["CPUExecutionProvider"], [{}])

    def test_resolve_openvino_prefers_concrete_device_when_generic_and_dotted_exist(self, prep_manager):
        """When runtime reports both generic and dotted IDs, prefer exact generic-family match."""
        prep_manager._device_type = "NPU"
        prep_manager._device_id = "NPU"
        available = ["OpenVINOExecutionProvider"]

        mock_core = mock.MagicMock()
        mock_core.available_devices = ["NPU", "NPU.0"]

        with (
            mock.patch("openvino.Core", return_value=mock_core),
            mock.patch("modules.inference.pipeline.preprocessing.config") as mock_cfg,
        ):
            mock_cfg.OV_CACHE_DIR = "/tmp/ov"
            _, options = prep_manager._resolve_providers(available)
            assert options[0]["device_type"] == "NPU"

    def test_resolve_openvino_dotted_request_falls_back_to_generic_when_dotted_missing(self, prep_manager):
        """If requested dotted ID is unavailable and only generic family is reported, use generic family token."""
        prep_manager._device_type = "NPU"
        prep_manager._device_id = "NPU.0"
        available = ["OpenVINOExecutionProvider"]

        mock_core = mock.MagicMock()
        mock_core.available_devices = ["CPU", "NPU"]

        with (
            mock.patch("openvino.Core", return_value=mock_core),
            mock.patch("modules.inference.pipeline.preprocessing.config") as mock_cfg,
        ):
            mock_cfg.OV_CACHE_DIR = "/tmp/ov"
            _, options = prep_manager._resolve_providers(available)
            assert options[0]["device_type"] == "NPU"

    def test_resolve_openvino_multi_npu_slot_uses_load_config_device_id(self, prep_manager):
        """Explicit NPU slot selection should survive ORT normalization through OpenVINO load_config."""
        prep_manager._device_type = "NPU"
        prep_manager._device_id = "NPU.1"
        available = ["OpenVINOExecutionProvider"]

        mock_core = mock.MagicMock()
        mock_core.available_devices = ["NPU.0", "NPU.1"]

        with (
            mock.patch("openvino.Core", return_value=mock_core),
            mock.patch("modules.inference.pipeline.preprocessing.config") as mock_cfg,
        ):
            mock_cfg.OV_CACHE_DIR = "/tmp/ov"
            _, options = prep_manager._resolve_providers(available)
            assert options[0]["device_type"] == "NPU"
            assert options[0]["load_config"] == '{"NPU":{"DEVICE_ID":"1"}}'

    def test_resolve_openvino_generic_gpu_binds_concrete_device_type(self, prep_manager):
        """Generic GPU runtime IDs should resolve to a concrete OpenVINO GPU device."""
        prep_manager._device_type = "GPU"
        prep_manager._device_id = "GPU"
        available = ["OpenVINOExecutionProvider"]

        mock_core = mock.MagicMock()
        mock_core.available_devices = ["CPU", "GPU.0"]

        with (
            mock.patch("openvino.Core", return_value=mock_core),
            mock.patch("modules.inference.pipeline.preprocessing.config") as mock_cfg,
        ):
            mock_cfg.OV_CACHE_DIR = "/tmp/ov"
            _, options = prep_manager._resolve_providers(available)
            assert options[0]["device_type"] == "GPU.0"

    def test_resolve_openvino_binds_generic_device_type_when_device_query_fails(self, prep_manager):
        """If OpenVINO cannot enumerate devices, preserve generic family token."""
        prep_manager._device_type = "NPU"
        prep_manager._device_id = "NPU"
        available = ["OpenVINOExecutionProvider"]

        with (
            mock.patch("openvino.Core", side_effect=RuntimeError("query failed")),
            mock.patch("modules.inference.pipeline.preprocessing.config") as mock_cfg,
        ):
            mock_cfg.OV_CACHE_DIR = "/tmp/ov"
            _, options = prep_manager._resolve_providers(available)
            assert options[0]["device_type"] == "NPU"

    def test_resolve_openvino_falls_back_to_cpu_when_provider_missing(self, prep_manager):
        """Intel preprocess targets should gracefully fall back to CPU when OpenVINO is unavailable."""
        prep_manager._device_type = "NPU"
        prep_manager._device_id = "NPU.0"
        available = ["CPUExecutionProvider"]

        providers, options = prep_manager._resolve_providers(available)
        assert providers == ["CPUExecutionProvider"]
        assert options == [{}]

    def test_resolve_openvino_uses_cpu_when_family_circuit_breaker_is_open(self, prep_manager):
        """After a recorded OpenVINO init failure, provider resolution should avoid OpenVINO for that family."""
        prep_manager._device_type = "NPU"
        prep_manager._device_id = "NPU.0"
        openvino_resolver.mark_openvino_family_unavailable("NPU.0")

        providers, options = prep_manager._resolve_providers(["OpenVINOExecutionProvider", "CPUExecutionProvider"])
        assert providers == ["CPUExecutionProvider"]
        assert options == [{}]

    def test_allocate_openvino_device_family_disabled(self, prep_manager):
        openvino_resolver.mark_openvino_family_unavailable("NPU.0")
        assert prep_manager._allocate_openvino_device("NPU.0") == "CPU"


class TestOpenvinoRuntimeReload:
    """Tests for Intel-path ONNX Runtime recovery logic."""

    def test_ensure_openvino_onnxruntime_skips_non_openvino_targets(self):
        with mock.patch("modules.inference.pipeline.preprocessing._reload_onnxruntime_from_intel_path") as mock_reload:
            preprocessing._ensure_openvino_onnxruntime("CPU")
            mock_reload.assert_not_called()

    def test_ensure_openvino_onnxruntime_reloads_when_provider_missing(self):
        mock_ort = mock.MagicMock()
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

        with (
            mock.patch("modules.inference.pipeline.preprocessing.ort", mock_ort),
            mock.patch(
                "modules.inference.pipeline.preprocessing._reload_onnxruntime_from_intel_path",
                return_value=True,
            ) as mock_reload,
        ):
            preprocessing._ensure_openvino_onnxruntime("NPU")
            mock_reload.assert_called_once()
