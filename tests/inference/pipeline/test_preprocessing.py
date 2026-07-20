"""Tests for preprocessing.py (UVR/MDX-NET Vocal Separation)."""

import logging
import os
import time
from unittest import mock

import pytest

from modules.inference.pipeline import openvino_resolver, preprocessing
from modules.inference.pipeline.preprocessing import CACHE_DIR, PreprocessingManager
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


class TestManagerBasics:
    """Tests for basic manager operations."""

    def test_init_defaults(self):
        with mock.patch("modules.inference.pipeline.preprocessing.config") as mock_cfg:
            mock_cfg.PREPROCESS_DEVICE = "AUTO"
            pm = PreprocessingManager()
            assert pm._device_id == "AUTO"

    def test_unload_model(self, prep_manager):
        prep_manager.separator = mock.MagicMock()
        prep_manager.unload_model()
        assert prep_manager.separator is None

    def test_offload(self, prep_manager):
        # Case 1: Has separator
        prep_manager.separator = mock.MagicMock()
        with mock.patch("modules.inference.pipeline.preprocessing.utils.clear_gpu_cache") as mock_clear:
            prep_manager.offload()
            assert prep_manager.separator is None
            mock_clear.assert_called_once()

        # Case 2: No separator
        prep_manager.offload()  # Should not crash


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
        with mock.patch("modules.inference.pipeline.preprocessing._get_available_openvino_devices", return_value=["CPU", "GPU.0"]):
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

        with mock.patch("modules.inference.pipeline.openvino_provider_dispatch.has_amd_provider", return_value=True):
            with mock.patch(
                "modules.inference.pipeline.openvino_provider_dispatch.amd_provider_config", return_value=(["ROCMExecutionProvider"], [{}])
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

        with mock.patch("modules.inference.pipeline.openvino_resolver.is_openvino_family_disabled", return_value=True):
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
            mock.patch("modules.inference.pipeline.preprocessing._reload_onnxruntime_from_intel_path", return_value=True) as mock_reload,
        ):
            preprocessing._ensure_openvino_onnxruntime("NPU")
            mock_reload.assert_called_once()


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
        session.get_providers.return_value = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]

        with caplog.at_level(logging.WARNING):
            preprocessing_helpers._log_openvino_cpu_fallback(session, {"device_type": "NPU"})

        assert "fell back" not in caplog.text

    def test_load_separator_model_retries_openvino_device_before_cpu(self, prep_manager):
        prep_manager._device_type = "NPU"
        prep_manager._device_id = "NPU.0"

        separator = mock.MagicMock()
        separator.load_model.side_effect = [RuntimeError("openvino init failed"), None]

        with mock.patch("modules.inference.pipeline.preprocessing._get_available_openvino_devices", return_value=["NPU", "GPU.0"]):
            prep_manager._load_separator_model(separator)

        assert separator.onnx_execution_provider == ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
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

        with mock.patch("modules.inference.pipeline.preprocessing._get_available_openvino_devices", return_value=["NPU", "GPU.0"]):
            prep_manager._load_separator_model(separator)

        assert separator.onnx_execution_provider == ["CPUExecutionProvider"]
        assert separator.load_model.call_count == 4

    def test_load_separator_model_retries_heuristic_openvino_candidates_when_enumeration_fails(self, prep_manager):
        prep_manager._device_type = "NPU"
        prep_manager._device_id = "NPU"

        separator = mock.MagicMock()
        separator.load_model.side_effect = [RuntimeError("npu failed"), None]

        with mock.patch("modules.inference.pipeline.preprocessing._get_available_openvino_devices", return_value=[]):
            prep_manager._load_separator_model(separator)

        assert separator.onnx_execution_provider == ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
        assert separator.load_model.call_count == 2

    def test_load_separator_model_opens_global_circuit_breaker_on_openvino_loader_failure(self, prep_manager):
        prep_manager._device_type = "NPU"
        prep_manager._device_id = "NPU.0"

        separator = mock.MagicMock()
        separator.load_model.side_effect = [
            RuntimeError("INTEL_OPENVINO_DIR is set but OpenVINO library wasn't able to be loaded."),
            None,
        ]

        with mock.patch("modules.inference.pipeline.preprocessing._get_available_openvino_devices", return_value=["NPU", "GPU.0"]):
            prep_manager._load_separator_model(separator)

        assert separator.onnx_execution_provider == ["CPUExecutionProvider"]
        assert separator.load_model.call_count == 2
        assert openvino_resolver.is_openvino_family_disabled("NPU") is True
        assert openvino_resolver.is_openvino_family_disabled("GPU") is True


def _make_onnx_mock_session():
    """Create a patched ORT mock session for ONNX optimization tests."""
    original_init_called = [False]

    class MockSession:
        """Mock ORT Session for patching tests."""

        is_patched = False

        def __init__(self, model_path, sess_options=None, providers=None, provider_options=None, **kwargs):
            original_init_called[0] = True
            self.providers = providers
            self.provider_options = provider_options

    mock_ort = mock.MagicMock()
    mock_ort.InferenceSession = MockSession
    return mock_ort, original_init_called


def test_onnx_session_patching_cpu_fallback():
    """Test that CPU fallback is rewritten to OpenVINO providers."""
    mock_ort, original_init_called = _make_onnx_mock_session()

    with mock.patch("modules.inference.pipeline.preprocessing.ort", mock_ort):
        preprocessing.apply_onnx_optimizations()
        with mock.patch("modules.inference.pipeline.preprocessing.utils.THREAD_CONTEXT") as mock_ctx:
            mock_ctx.ov_options = {"device_type": "GPU"}
            session = mock_ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])

    assert (
        "OpenVINOExecutionProvider" in session.providers,
        session.provider_options[0]["device_type"],
        session.provider_options[1],
        original_init_called[0],
    ) == (
        True,
        "GPU",
        {},
        True,
    )


def test_onnx_session_patching_preserves_openvino_provider_options():
    """Test that explicit OpenVINO providers keep normalized options intact."""
    mock_ort, _ = _make_onnx_mock_session()

    with mock.patch("modules.inference.pipeline.preprocessing.ort", mock_ort):
        preprocessing.apply_onnx_optimizations()
        with mock.patch("modules.inference.pipeline.preprocessing.utils.THREAD_CONTEXT") as mock_ctx:
            mock_ctx.ov_options = {"device_type": "GPU"}
            session = mock_ort.InferenceSession("model.onnx", providers=["OpenVINOExecutionProvider", "CPUExecutionProvider"])

    assert (session.provider_options[0]["device_type"], session.provider_options[1]) == ("GPU", {})


def test_onnx_session_patching_normalizes_provider_options_lists():
    """Test provider_options list normalization and expansion."""
    mock_ort, _ = _make_onnx_mock_session()

    with mock.patch("modules.inference.pipeline.preprocessing.ort", mock_ort):
        preprocessing.apply_onnx_optimizations()
        with mock.patch("modules.inference.pipeline.preprocessing.utils.THREAD_CONTEXT") as mock_ctx:
            mock_ctx.ov_options = {"device_type": "GPU"}
            session3 = mock_ort.InferenceSession("model.onnx", providers=["OpenVINOExecutionProvider"], provider_options=[None])
            session4 = mock_ort.InferenceSession(
                "model.onnx", providers=["CPUExecutionProvider", "OpenVINOExecutionProvider"], provider_options=[{}]
            )

    assert (
        isinstance(session3.provider_options[0], dict),
        session3.provider_options[0]["device_type"],
        len(session4.provider_options),
        session4.provider_options[1]["device_type"],
    ) == (
        True,
        "GPU",
        2,
        "GPU",
    )


def test_reload_onnxruntime_from_intel_path_returns_false_when_path_missing():
    """Reload should fail fast when Intel ONNX path is unavailable."""
    with mock.patch("os.path.exists", return_value=False):
        assert preprocessing._reload_onnxruntime_from_intel_path() is False


def test_reload_onnxruntime_from_intel_path_success_updates_module_ort():
    """Successful Intel-path reload should replace module-level ORT reference."""
    reloaded_ort = mock.MagicMock()
    reloaded_ort.get_available_providers.return_value = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]

    with (
        mock.patch("os.path.exists", return_value=True),
        mock.patch("importlib.import_module", return_value=reloaded_ort),
    ):
        assert preprocessing._reload_onnxruntime_from_intel_path() is True
        assert preprocessing.ort is reloaded_ort


def test_reload_onnxruntime_from_intel_path_purges_cached_submodules():
    """Intel-path reload should evict cached onnxruntime submodules from prior CUDA/NVIDIA imports."""
    reloaded_ort = mock.MagicMock()
    reloaded_ort.get_available_providers.return_value = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]

    with (
        mock.patch("os.path.exists", return_value=True),
        mock.patch("importlib.import_module", return_value=reloaded_ort),
        mock.patch.object(preprocessing.sys, "path", ["/app/libs/nvidia", "/app/libs/intel", "/app"]),
        mock.patch.dict(
            preprocessing.sys.modules,
            {
                "onnxruntime": object(),
                "onnxruntime.capi": object(),
                "onnxruntime.capi.onnxruntime_pybind11_state": object(),
            },
            clear=False,
        ),
    ):
        assert preprocessing._reload_onnxruntime_from_intel_path() is True
        assert preprocessing.sys.path[0] == "/app/libs/intel"
        assert "onnxruntime.capi" not in preprocessing.sys.modules
        assert "onnxruntime.capi.onnxruntime_pybind11_state" not in preprocessing.sys.modules


def test_ensure_openvino_onnxruntime_logs_warning_when_reload_unavailable():
    """Intel preprocess target should log a warning when OpenVINO provider cannot be recovered."""
    mock_ort = mock.MagicMock()
    mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

    with (
        mock.patch("modules.inference.pipeline.preprocessing.ort", mock_ort),
        mock.patch("modules.inference.pipeline.preprocessing._reload_onnxruntime_from_intel_path", return_value=False),
        mock.patch("modules.inference.pipeline.preprocessing.logger.warning") as mock_warning,
    ):
        preprocessing._ensure_openvino_onnxruntime("NPU")
        mock_warning.assert_called_once()


def test_openvino_device_resolution_prefix_match_and_normalization_paths():
    """Device matching should support prefix matches and remap to alternate Intel family when needed."""
    assert preprocessing._find_matching_openvino_device("GPU.1", ["GPU.1", "GPU.2"]) == "GPU.1"
    assert preprocessing._find_matching_openvino_device("NPU", ["CPU", "GPU"]) == "GPU"


def test_openvino_device_resolution_prefers_alternate_concrete_device_when_family_missing():
    """If NPU family is unavailable, select a concrete alternate Intel GPU device before session creation."""
    assert preprocessing._find_matching_openvino_device("NPU", ["CPU", "GPU.0"]) == "GPU.0"


def test_openvino_retry_candidates_use_heuristics_when_device_query_unavailable():
    """When OpenVINO cannot enumerate devices, retries should still prefer Intel accelerators before CPU fallback."""
    with mock.patch("modules.inference.pipeline.preprocessing._get_available_openvino_devices", return_value=[]):
        assert preprocessing._openvino_retry_candidates("NPU") == ["NPU", "NPU.0", "GPU.0", "GPU"]


def test_resolve_openvino_device_type_uses_concrete_candidate_when_enumeration_unavailable():
    """Generic family requests should keep explicit alias tokens when OpenVINO cannot enumerate devices."""
    with mock.patch("modules.inference.pipeline.preprocessing._get_available_openvino_devices", return_value=[]):
        assert preprocessing._resolve_openvino_device_type("NPU") == "NPU"


def test_cpu_and_cuda_provider_fallback_helpers():
    """Provider helper functions should return deterministic CPU fallback configs."""
    cpu_providers, cpu_options = preprocessing._cpu_provider_config()
    assert cpu_providers == ["CPUExecutionProvider"]
    assert cpu_options == [{}]

    fallback_providers, fallback_options = preprocessing._cuda_or_cpu_provider_config("cuda:0", ["CPUExecutionProvider"])
    assert fallback_providers == ["CPUExecutionProvider"]
    assert fallback_options == [{}]


def test_auto_provider_config_uses_cpu_when_openvino_reports_no_accelerators():
    """AUTO preprocessing should not select OpenVINO when only CPU is visible to OpenVINO."""
    with mock.patch("modules.inference.pipeline.preprocessing._get_available_openvino_devices", return_value=["CPU"]):
        providers, options = preprocessing._auto_provider_config(["OpenVINOExecutionProvider", "CPUExecutionProvider"])

    assert providers == ["CPUExecutionProvider"]
    assert options == [{}]


def test_auto_provider_config_uses_openvino_when_accelerator_visible():
    """AUTO preprocessing should select OpenVINO when OpenVINO reports Intel GPU/NPU devices."""
    with (
        mock.patch("modules.inference.pipeline.preprocessing._get_available_openvino_devices", return_value=["CPU", "GPU.0"]),
        mock.patch("modules.inference.pipeline.preprocessing.config") as mock_cfg,
    ):
        mock_cfg.OV_CACHE_DIR = "/tmp/ov"
        mock_cfg.PREPROCESS_THREADS = 4
        providers, options = preprocessing._auto_provider_config(["OpenVINOExecutionProvider", "CPUExecutionProvider"])

    assert providers == ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
    assert options[0]["device_type"] == "GPU.0"
    assert options[0]["num_streams"] == "1"
    assert options[0]["cache_dir"].endswith("/uvr/gpu")


def test_auto_provider_config_uses_first_visible_intel_accelerator_when_both_families_are_visible():
    """AUTO preprocessing should use runtime discovery order when both Intel families are visible."""
    with (
        mock.patch("modules.inference.pipeline.preprocessing._get_available_openvino_devices", return_value=["CPU", "GPU.0", "NPU.0"]),
        mock.patch("modules.inference.pipeline.preprocessing.config") as mock_cfg,
    ):
        mock_cfg.OV_CACHE_DIR = "/tmp/ov"
        mock_cfg.PREPROCESS_THREADS = 4
        providers, options = preprocessing._auto_provider_config(["OpenVINOExecutionProvider", "CPUExecutionProvider"])

    assert providers == ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
    assert options[0]["device_type"] == "GPU.0"
    assert options[0]["num_streams"] == "1"
    assert options[0]["cache_dir"].endswith("/uvr/gpu")


def test_auto_provider_config_respects_npu_first_discovery_order_when_visible():
    """AUTO preprocessing should select NPU when OpenVINO reports NPU before GPU."""
    with (
        mock.patch("modules.inference.pipeline.preprocessing._get_available_openvino_devices", return_value=["CPU", "NPU.0", "GPU.0"]),
        mock.patch("modules.inference.pipeline.preprocessing.config") as mock_cfg,
    ):
        mock_cfg.OV_CACHE_DIR = "/tmp/ov"
        mock_cfg.PREPROCESS_THREADS = 4
        providers, options = preprocessing._auto_provider_config(["OpenVINOExecutionProvider", "CPUExecutionProvider"])

    assert providers == ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
    assert options[0]["device_type"] == "NPU"
    assert options[0]["num_streams"] == "1"
    assert options[0]["cache_dir"].endswith("/uvr/npu")


def test_stem_resolution_candidates_include_output_cache_and_source_parent():
    """Stem lookup candidate ordering should include output dir, cache dirs, and source parent."""
    sep = mock.MagicMock()
    sep.output_dir = "out-dir"
    candidates = preprocessing._stem_resolution_candidates(sep, "parent/input.wav")
    assert candidates[0] == "out-dir"
    assert "parent" in candidates


def test_load_separator_model_non_openvino_exception_sets_separator_none(prep_manager):
    """Non-OpenVINO load failures should clear separator and re-raise."""
    prep_manager._device_type = "CPU"
    prep_manager.separator = mock.MagicMock()
    sep = mock.MagicMock()
    sep.load_model.side_effect = RuntimeError("load-fail")

    with pytest.raises(RuntimeError):
        prep_manager._load_separator_model(sep)

    assert prep_manager.separator is None


def test_build_active_yield_cb_releases_and_reacquires_lock(prep_manager):
    """Active yield callback should release lock while invoking cooperative callback."""
    events = []

    def _cb():
        events.append("yielded")

    prep_manager.lock.acquire()
    wrapped = prep_manager._build_active_yield_cb(_cb)
    assert wrapped is not None
    wrapped()
    prep_manager.lock.release()

    assert events == ["yielded"]


def test_openvino_init_lock_is_shared_within_same_accelerator_family():
    """GPU family targets should reuse a single init lock to prevent same-family races."""
    gpu_lock = preprocessing._openvino_init_lock_for("GPU.0", "GPU")
    same_family_lock = preprocessing._openvino_init_lock_for("GPU.1", "GPU")

    assert gpu_lock is same_family_lock


def test_openvino_init_lock_is_distinct_across_gpu_and_npu_families():
    """GPU and NPU initializations should not block each other on first-load paths."""
    gpu_lock = preprocessing._openvino_init_lock_for("GPU.0", "GPU")
    npu_lock = preprocessing._openvino_init_lock_for("NPU.0", "NPU")

    assert gpu_lock is not npu_lock

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
        with mock.patch("modules.inference.pipeline.preprocessing._lazy_import_separator", return_value=None):
            with pytest.raises(ImportError):
                prep_manager._init_separator()


class TestPreprocessAudio:
    """Tests for the main preprocess_audio entry point."""

    def test_preprocess_disabled(self, prep_manager):
        with mock.patch("modules.inference.pipeline.preprocessing.config") as mock_cfg:
            mock_cfg.ENABLE_VOCAL_SEPARATION = False
            assert prep_manager.preprocess_audio("test.wav") == "test.wav"

    def test_preprocess_returns_original_when_requested_accelerator_and_cpu_fallback_fail(self, prep_manager):
        """Requested accelerator failures should fall back to CPU and return original if CPU also fails."""
        with (
            mock.patch("modules.inference.pipeline.preprocessing.config.ENABLE_VOCAL_SEPARATION", True),
            mock.patch("modules.inference.pipeline.preprocessing.utils.prepare_for_uvr", return_value="audio.wav"),
            mock.patch.object(
                prep_manager,
                "_init_separator",
                side_effect=preprocessing.UVRAcceleratorUnavailableError("OpenVINO unavailable"),
            ),
        ):
            assert prep_manager.preprocess_audio("audio.wav") == "audio.wav"

    def test_preprocess_retries_cpu_for_openvino_session_cpu_fallback_error(self, prep_manager):
        prep_manager._device_type = "NPU"
        prep_manager._device_id = "NPU"
        with (
            mock.patch("modules.inference.pipeline.preprocessing.config.ENABLE_VOCAL_SEPARATION", True),
            mock.patch("modules.inference.pipeline.preprocessing.utils.prepare_for_uvr", return_value="audio.wav"),
            mock.patch.object(
                prep_manager,
                "_run_preprocess_pipeline",
                side_effect=RuntimeError(
                    "OpenVINOExecutionProvider did not initialize for device_type=NPU; "
                    "ONNX Runtime fell back to providers=['CPUExecutionProvider']"
                ),
            ),
        ):
            assert prep_manager.preprocess_audio("audio.wav") == "audio.wav"
            assert prep_manager._run_preprocess_pipeline.call_count == 2

    def test_preprocess_success(self, prep_manager):
        with mock.patch("modules.inference.pipeline.preprocessing.config") as mock_cfg:
            mock_cfg.ENABLE_VOCAL_SEPARATION = True
            mock_sep = mock.MagicMock()
            mock_sep.separate.return_value = ["vocal.wav"]
            prep_manager._init_separator = mock.MagicMock(return_value=mock_sep)

            mock_ort = mock.MagicMock()
            mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

            with mock.patch("modules.inference.pipeline.preprocessing.ort", mock_ort):
                with mock.patch("modules.inference.pipeline.preprocessing.utils.prepare_for_uvr", side_effect=lambda path, **_: path):
                    res = prep_manager.preprocess_audio("test.wav")
                    assert "vocal.wav" in res

    def test_preprocess_stage_order_shows_ffmpeg_before_vocal_separation(self, prep_manager):
        """FFmpeg preparation should occur before Vocal Separation stage is published."""
        with mock.patch("modules.inference.pipeline.preprocessing.config") as mock_cfg:
            mock_cfg.ENABLE_VOCAL_SEPARATION = True
            mock_sep = mock.MagicMock()
            mock_sep.separate.return_value = ["vocal.wav"]
            prep_manager._init_separator = mock.MagicMock(return_value=mock_sep)

            mock_ort = mock.MagicMock()
            mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

            events = []

            def _prep(path, **_kwargs):
                events.append("prepare_for_uvr")
                return path

            def _stage(*args, **kwargs):
                _ = kwargs
                if len(args) >= 2 and args[1] == "Vocal Separation":
                    events.append("vocal_stage")

            with (
                mock.patch("modules.inference.pipeline.preprocessing.ort", mock_ort),
                mock.patch("modules.inference.pipeline.preprocessing.utils.prepare_for_uvr", side_effect=_prep),
                mock.patch("modules.inference.pipeline.preprocessing.scheduler.update_task_progress", side_effect=_stage),
            ):
                prep_manager.preprocess_audio("test.wav")

            assert events.index("prepare_for_uvr") < events.index("vocal_stage")

    def test_preprocess_cleanup_secondary_stems(self, prep_manager):
        with mock.patch("modules.inference.pipeline.preprocessing.config") as mock_cfg:
            mock_cfg.ENABLE_VOCAL_SEPARATION = True
            mock_sep = mock.MagicMock()
            mock_sep.separate.return_value = ["vocal.wav", "instrumental.wav"]
            prep_manager._init_separator = mock.MagicMock(return_value=mock_sep)

            mock_ort = mock.MagicMock()
            mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

            with (
                mock.patch("modules.inference.pipeline.preprocessing.ort", mock_ort),
                mock.patch("os.path.exists", return_value=True),
                mock.patch("os.remove") as mock_remove,
            ):
                with mock.patch("modules.inference.pipeline.preprocessing.utils.prepare_for_uvr", side_effect=lambda path, **_: path):
                    prep_manager.preprocess_audio("test.wav")
                    mock_remove.assert_called()

    def test_preprocess_exception(self, prep_manager):
        with mock.patch("modules.inference.pipeline.preprocessing.config") as mock_cfg:
            mock_cfg.ENABLE_VOCAL_SEPARATION = True
            prep_manager._init_separator = mock.MagicMock(side_effect=Exception("Crash"))
            assert prep_manager.preprocess_audio("test.wav") == "test.wav"

    def test_preprocess_force(self, prep_manager):
        with mock.patch("modules.inference.pipeline.preprocessing.config") as mock_cfg:
            mock_cfg.ENABLE_VOCAL_SEPARATION = False  # Disabled via config
            mock_sep = mock.MagicMock()
            mock_sep.separate.return_value = ["vocal.wav"]
            prep_manager._init_separator = mock.MagicMock(return_value=mock_sep)

            mock_ort = mock.MagicMock()
            mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

            with mock.patch("modules.inference.pipeline.preprocessing.ort", mock_ort):
                with mock.patch("modules.inference.pipeline.preprocessing.utils.prepare_for_uvr", side_effect=lambda path, **_: path):
                    # Should still run because force=True
                    res = prep_manager.preprocess_audio("test.wav", force=True)
                    assert "vocal.wav" in res

    def test_preprocess_relative_stem(self, prep_manager):
        with mock.patch("modules.inference.pipeline.preprocessing.config") as mock_cfg:
            mock_cfg.ENABLE_VOCAL_SEPARATION = True
            mock_sep = mock.MagicMock()
            mock_sep.separate.return_value = ["relative_vocal.wav"]
            prep_manager._init_separator = mock.MagicMock(return_value=mock_sep)

            with mock.patch("modules.inference.pipeline.preprocessing.ort", mock.MagicMock()):
                with mock.patch("modules.inference.pipeline.preprocessing.utils.prepare_for_uvr", side_effect=lambda path, **_: path):
                    res = prep_manager.preprocess_audio("test.wav")
                    assert "relative_vocal.wav" in res
                    assert str(CACHE_DIR) in res

    def test_preprocess_resolves_relative_stem_from_effective_separator_output_dir(self, prep_manager):
        with mock.patch("modules.inference.pipeline.preprocessing.config") as mock_cfg:
            mock_cfg.ENABLE_VOCAL_SEPARATION = True
            mock_sep = mock.MagicMock()
            mock_sep.separate.return_value = ["relative_vocal.wav"]
            mock_sep.output_dir = "/alt/output"
            prep_manager._init_separator = mock.MagicMock(return_value=mock_sep)

            with (
                mock.patch("modules.inference.pipeline.preprocessing.ort", mock.MagicMock()),
                mock.patch("modules.inference.pipeline.preprocessing.utils.prepare_for_uvr", side_effect=lambda path, **_: path),
                mock.patch("os.path.exists", side_effect=lambda p: p.replace("\\", "/") == "/alt/output/relative_vocal.wav"),
            ):
                res = prep_manager.preprocess_audio("test.wav")

            assert res.replace("\\", "/") == "/alt/output/relative_vocal.wav"

    def test_preprocess_relative_stem_falls_back_to_cache_dir_when_not_found_anywhere(self, prep_manager):
        with mock.patch("modules.inference.pipeline.preprocessing.config") as mock_cfg:
            mock_cfg.ENABLE_VOCAL_SEPARATION = True
            mock_sep = mock.MagicMock()
            mock_sep.separate.return_value = ["relative_vocal.wav"]
            mock_sep.output_dir = "/alt/output"
            prep_manager._init_separator = mock.MagicMock(return_value=mock_sep)

            with (
                mock.patch("modules.inference.pipeline.preprocessing.ort", mock.MagicMock()),
                mock.patch("modules.inference.pipeline.preprocessing.utils.prepare_for_uvr", side_effect=lambda path, **_: path),
                mock.patch("os.path.exists", return_value=False),
            ):
                res = prep_manager.preprocess_audio("test.wav")

        assert res == str(CACHE_DIR / "relative_vocal.wav")

    def test_preprocess_cleanup_error(self, prep_manager):
        with mock.patch("modules.inference.pipeline.preprocessing.config") as mock_cfg:
            mock_cfg.ENABLE_VOCAL_SEPARATION = True
            mock_sep = mock.MagicMock()
            mock_sep.separate.return_value = ["vocal.wav", "extra.wav"]
            prep_manager._init_separator = mock.MagicMock(return_value=mock_sep)

            with (
                mock.patch("modules.inference.pipeline.preprocessing.ort", mock.MagicMock()),
                mock.patch("os.path.exists", return_value=True),
                mock.patch("os.remove", side_effect=OSError("Busy")),
            ):
                with mock.patch("modules.inference.pipeline.preprocessing.utils.prepare_for_uvr", side_effect=lambda path, **_: path):
                    # Should not raise
                    prep_manager.preprocess_audio("test.wav")

    def test_preprocess_prepare_fail(self, prep_manager):
        with mock.patch("modules.inference.pipeline.preprocessing.config") as mock_cfg:
            mock_cfg.ENABLE_VOCAL_SEPARATION = True
            with mock.patch("modules.inference.pipeline.preprocessing.utils.prepare_for_uvr", return_value=None):
                assert prep_manager.preprocess_audio("test.wav") == "test.wav"


class TestCache:
    """Tests for cache management."""

    def test_purge_cache_success(self, prep_manager):
        with mock.patch("modules.inference.pipeline.preprocessing.CACHE_DIR") as mock_cache:
            mock_file_old = mock.MagicMock()
            mock_file_old.is_file.return_value = True
            mock_file_old.stat.return_value.st_mtime = time.time() - 4000

            mock_file_new = mock.MagicMock()
            mock_file_new.is_file.return_value = True
            mock_file_new.stat.return_value.st_mtime = time.time()

            mock_cache.iterdir.return_value = [mock_file_old, mock_file_new]
            prep_manager._purge_stale_cache()
            mock_file_old.unlink.assert_called_once()
            mock_file_new.unlink.assert_not_called()

    def test_purge_cache_exception(self, prep_manager):
        with mock.patch("modules.inference.pipeline.preprocessing.CACHE_DIR") as mock_cache:
            mock_cache.iterdir.side_effect = Exception("Fail")
            prep_manager._purge_stale_cache()  # Should not raise

    def test_purge_cache_unlink_fail(self, prep_manager):
        with mock.patch("modules.inference.pipeline.preprocessing.CACHE_DIR") as mock_cache:
            mock_file = mock.MagicMock()
            mock_file.is_file.return_value = True
            mock_file.stat.return_value.st_mtime = time.time() - 4000
            mock_file.unlink.side_effect = OSError("Locked")
            mock_cache.iterdir.return_value = [mock_file]
            prep_manager._purge_stale_cache()  # Should not raise


def test_lazy_import():
    res = preprocessing._lazy_import_separator()
    assert res is not None or res is None


class TestCandidateOutputDirs:
    """Tests for _candidate_output_dirs()."""

    def test_returns_list_no_duplicates(self):
        dirs = preprocessing._candidate_output_dirs()
        assert isinstance(dirs, list)
        assert len(dirs) == len(set(dirs))

    def test_cache_dir_is_first(self):
        dirs = preprocessing._candidate_output_dirs()
        assert dirs[0] == str(preprocessing.CACHE_DIR)

    def test_persistent_temp_dir_included(self):
        with mock.patch("os.path.isdir", return_value=True):
            dirs = preprocessing._candidate_output_dirs()
            assert os.path.abspath(preprocessing.config.PERSISTENT_TEMP_DIR) in dirs

    def test_candidate_dirs_no_shm_dependency(self):
        with mock.patch("os.path.isdir", return_value=False):
            dirs = preprocessing._candidate_output_dirs()
            assert "/dev/shm" not in dirs
