"""Preprocessing pipeline tests (split from test_preprocessing.py)."""

import logging
from unittest import mock

import pytest

from modules.inference.pipeline import openvino_resolver, preprocessing
from modules.inference.pipeline.preprocessing import PreprocessingManager
from modules.inference.pipeline.preprocessing import helpers as preprocessing_helpers

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


class TestSeparatorInit:
    """Tests for separator initialization and patching."""

    def test_apply_onnx_optimizations_success(self):
        class MockSession:
            """Mock ORT Session."""

            is_patched = False

            def __init__(self, *args, **kwargs):
                pass

        mock_ort = mock.MagicMock()
        mock_ort.InferenceSession = MockSession
        with mock.patch("modules.inference.pipeline.preprocessing.ort", mock_ort):
            preprocessing.apply_onnx_optimizations()
            assert MockSession.is_patched is True

    def test_log_openvino_cpu_fallback(self, prep_manager):
        prep_manager._device_id = "NPU"
        openvino_resolver.clear_openvino_disabled_families()
        session = mock.MagicMock()
        session.get_providers.return_value = ["CPUExecutionProvider"]

        openvino_resolver._log_openvino_cpu_fallback(session, {"device_type": "NPU"})

        # NPU should be disabled due to CPU fallback
        assert openvino_resolver.is_openvino_family_disabled("NPU") is True

    def test_log_openvino_cpu_fallback_accepts_openvino(self, caplog):
        session = mock.MagicMock()
        session.get_providers.return_value = [
            "OpenVINOExecutionProvider",
            "CPUExecutionProvider",
        ]

        with caplog.at_level(logging.WARNING):
            preprocessing_helpers._log_openvino_cpu_fallback(session, {"device_type": "NPU"})

        assert "fell back" not in caplog.text

    def test_load_separator_model_retries_openvino_device_before_cpu(self, prep_manager):
        prep_manager._device_type = "NPU"
        prep_manager._device_id = "NPU.0"

        separator = mock.MagicMock()
        separator.load_model.side_effect = [RuntimeError("openvino init failed"), None]

        with mock.patch(
            "modules.inference.pipeline.preprocessing._get_available_openvino_devices",
            return_value=["NPU", "GPU.0"],
        ):
            prep_manager._load_separator_model(separator)

        assert separator.onnx_execution_provider == [
            "OpenVINOExecutionProvider",
            "CPUExecutionProvider",
        ]
        assert separator.load_model.call_count == 2

    def test_load_separator_model_falls_back_to_cpu_after_openvino_retries_fail(self, prep_manager):
        prep_manager._device_type = "NPU"
        prep_manager._device_id = "NPU.0"

        separator = mock.MagicMock()
        separator.load_model.side_effect = [
            RuntimeError("npu.0 failed"),
            RuntimeError("npu generic failed"),
            RuntimeError("gpu.0 failed"),
            None,
        ]

        with mock.patch(
            "modules.inference.pipeline.preprocessing._get_available_openvino_devices",
            return_value=["NPU", "GPU.0"],
        ):
            prep_manager._load_separator_model(separator)

        assert separator.onnx_execution_provider == ["CPUExecutionProvider"]
        assert separator.load_model.call_count == 4

    def test_load_separator_model_retries_heuristic_openvino_candidates_when_enumeration_fails(self, prep_manager):
        prep_manager._device_type = "NPU"
        prep_manager._device_id = "NPU"

        separator = mock.MagicMock()
        separator.load_model.side_effect = [RuntimeError("npu failed"), None]

        with mock.patch(
            "modules.inference.pipeline.preprocessing._get_available_openvino_devices",
            return_value=[],
        ):
            prep_manager._load_separator_model(separator)

        assert separator.onnx_execution_provider == [
            "OpenVINOExecutionProvider",
            "CPUExecutionProvider",
        ]
        assert separator.load_model.call_count == 2

    def test_load_separator_model_opens_global_circuit_breaker_on_openvino_loader_failure(self, prep_manager):
        prep_manager._device_type = "NPU"
        prep_manager._device_id = "NPU.0"

        separator = mock.MagicMock()
        separator.load_model.side_effect = [
            RuntimeError("INTEL_OPENVINO_DIR is set but OpenVINO library wasn't able to be loaded."),
            None,
        ]

        with mock.patch(
            "modules.inference.pipeline.preprocessing._get_available_openvino_devices",
            return_value=["NPU", "GPU.0"],
        ):
            prep_manager._load_separator_model(separator)

        assert separator.onnx_execution_provider == ["CPUExecutionProvider"]
        assert separator.load_model.call_count == 2
        assert openvino_resolver.is_openvino_family_disabled("NPU") is True
        assert openvino_resolver.is_openvino_family_disabled("GPU") is True

    def test_init_separator_success(self, prep_manager):
        mock_ort = mock.MagicMock()
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
        mock_ort.__version__ = "1.24.1"

        with mock.patch("modules.inference.pipeline.preprocessing.ort", mock_ort):
            with mock.patch("modules.inference.pipeline.preprocessing._lazy_import_separator") as mock_imp:
                mock_sep_cls = mock.MagicMock()
                mock_imp.return_value = mock_sep_cls

                sep = prep_manager._init_separator()
                assert sep is not None
                assert prep_manager.separator is not None
                mock_sep_cls.assert_called_once()

    def test_init_separator_failure(self, prep_manager):
        with mock.patch("modules.inference.pipeline.preprocessing._lazy_import_separator") as mock_imp:
            mock_sep_cls = mock.MagicMock()
            mock_imp.return_value = mock_sep_cls
            mock_sep_inst = mock_sep_cls.return_value
            mock_sep_inst.load_model.side_effect = Exception("Fail")

            mock_ort = mock.MagicMock()
            mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

            with mock.patch("modules.inference.pipeline.preprocessing.ort", mock_ort):
                with pytest.raises(Exception):
                    prep_manager._init_separator()
                assert prep_manager.separator is None

    def test_init_separator_no_ort(self, prep_manager):
        with mock.patch("modules.inference.pipeline.preprocessing.ort", None):
            with pytest.raises(ImportError):
                prep_manager._init_separator()

    def test_init_separator_no_sep(self, prep_manager):
        with mock.patch(
            "modules.inference.pipeline.preprocessing._lazy_import_separator",
            return_value=None,
        ):
            with pytest.raises(ImportError):
                prep_manager._init_separator()
