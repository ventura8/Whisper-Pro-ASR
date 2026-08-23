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


def _make_onnx_mock_session():
    """Create a patched ORT mock session for ONNX optimization tests."""
    original_init_called = [False]

    class MockSession:
        """Mock ORT Session for patching tests."""

        is_patched = False

        def __init__(
            self,
            model_path,
            sess_options=None,
            providers=None,
            provider_options=None,
            **kwargs,
        ):
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
            session = mock_ort.InferenceSession(
                "model.onnx",
                providers=["OpenVINOExecutionProvider", "CPUExecutionProvider"],
            )

    assert (
        session.provider_options[0]["device_type"],
        session.provider_options[1],
    ) == ("GPU", {})


def test_onnx_session_patching_normalizes_provider_options_lists():
    """Test provider_options list normalization and expansion."""
    mock_ort, _ = _make_onnx_mock_session()

    with mock.patch("modules.inference.pipeline.preprocessing.ort", mock_ort):
        preprocessing.apply_onnx_optimizations()
        with mock.patch("modules.inference.pipeline.preprocessing.utils.THREAD_CONTEXT") as mock_ctx:
            mock_ctx.ov_options = {"device_type": "GPU"}
            session3 = mock_ort.InferenceSession(
                "model.onnx",
                providers=["OpenVINOExecutionProvider"],
                provider_options=[None],
            )
            session4 = mock_ort.InferenceSession(
                "model.onnx",
                providers=["CPUExecutionProvider", "OpenVINOExecutionProvider"],
                provider_options=[{}],
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
    reloaded_ort.get_available_providers.return_value = [
        "OpenVINOExecutionProvider",
        "CPUExecutionProvider",
    ]

    with (
        mock.patch("os.path.exists", return_value=True),
        mock.patch("importlib.import_module", return_value=reloaded_ort),
    ):
        assert preprocessing._reload_onnxruntime_from_intel_path() is True
        assert preprocessing.ort is reloaded_ort


def test_reload_onnxruntime_from_intel_path_purges_cached_submodules():
    """Intel-path reload should evict cached onnxruntime submodules from prior CUDA/NVIDIA imports."""
    reloaded_ort = mock.MagicMock()
    reloaded_ort.get_available_providers.return_value = [
        "OpenVINOExecutionProvider",
        "CPUExecutionProvider",
    ]

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
        mock.patch("modules.inference.pipeline.preprocessing.config") as mock_cfg,
        mock.patch("modules.inference.pipeline.preprocessing.ort", mock_ort),
        mock.patch(
            "modules.inference.pipeline.preprocessing._reload_onnxruntime_from_intel_path",
            return_value=False,
        ),
        mock.patch("modules.inference.pipeline.preprocessing.logger.warning") as mock_warning,
    ):
        mock_cfg.DEVICE = "CPU"
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
    with mock.patch(
        "modules.inference.pipeline.preprocessing._get_available_openvino_devices",
        return_value=[],
    ):
        assert preprocessing._openvino_retry_candidates("NPU") == [
            "NPU",
            "NPU.0",
            "GPU.0",
            "GPU",
        ]


def test_resolve_openvino_device_type_uses_concrete_candidate_when_enumeration_unavailable():
    """Generic family requests should keep explicit alias tokens when OpenVINO cannot enumerate devices."""
    with mock.patch(
        "modules.inference.pipeline.preprocessing._get_available_openvino_devices",
        return_value=[],
    ):
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
    with mock.patch(
        "modules.inference.pipeline.preprocessing._get_available_openvino_devices",
        return_value=["CPU"],
    ):
        providers, options = preprocessing._auto_provider_config(["OpenVINOExecutionProvider", "CPUExecutionProvider"])

    assert providers == ["CPUExecutionProvider"]
    assert options == [{}]


def test_auto_provider_config_uses_openvino_when_accelerator_visible():
    """AUTO preprocessing should select OpenVINO when OpenVINO reports Intel GPU/NPU devices."""
    with (
        mock.patch(
            "modules.inference.pipeline.preprocessing._get_available_openvino_devices",
            return_value=["CPU", "GPU.0"],
        ),
        mock.patch("modules.inference.pipeline.preprocessing.config") as mock_cfg,
    ):
        mock_cfg.OV_CACHE_DIR = "/tmp/ov"
        mock_cfg.PREPROCESS_THREADS = 4
        providers, options = preprocessing._auto_provider_config(["OpenVINOExecutionProvider", "CPUExecutionProvider"])

    assert providers == ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
    assert options[0]["device_type"] == "GPU.0"
    assert options[0]["num_streams"] == "1"
    assert "cache_dir" not in options[0]


def test_auto_provider_config_uses_first_visible_intel_accelerator_when_both_families_are_visible():
    """AUTO preprocessing should use runtime discovery order when both Intel families are visible."""
    with (
        mock.patch(
            "modules.inference.pipeline.preprocessing._get_available_openvino_devices",
            return_value=["CPU", "GPU.0", "NPU.0"],
        ),
        mock.patch("modules.inference.pipeline.preprocessing.config") as mock_cfg,
    ):
        mock_cfg.OV_CACHE_DIR = "/tmp/ov"
        mock_cfg.PREPROCESS_THREADS = 4
        providers, options = preprocessing._auto_provider_config(["OpenVINOExecutionProvider", "CPUExecutionProvider"])

    assert providers == ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
    assert options[0]["device_type"] == "GPU.0"
    assert options[0]["num_streams"] == "1"
    assert "cache_dir" not in options[0]


def test_auto_provider_config_respects_npu_first_discovery_order_when_visible():
    """AUTO preprocessing should select NPU when OpenVINO reports NPU before GPU."""
    with (
        mock.patch(
            "modules.inference.pipeline.preprocessing._get_available_openvino_devices",
            return_value=["CPU", "NPU.0", "GPU.0"],
        ),
        mock.patch("modules.inference.pipeline.preprocessing.config") as mock_cfg,
    ):
        mock_cfg.OV_CACHE_DIR = "/tmp/ov"
        mock_cfg.PREPROCESS_THREADS = 4
        providers, options = preprocessing._auto_provider_config(["OpenVINOExecutionProvider", "CPUExecutionProvider"])

    assert providers == ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
    assert options[0]["device_type"] == "NPU"
    assert options[0]["num_streams"] == "1"
    assert "cache_dir" not in options[0]


def test_block_openvino_alongside_cuda_redirects_openvino_result_to_cuda():
    """When ASR runs on CUDA, an OpenVINO-selected preprocessing provider must be
    redirected to CUDA (same vendor as ASR) rather than left as OpenVINO -- an
    OpenVINO GPU/NPU context alongside an active CUDA context has been observed
    to crash/hang natively."""
    from modules.inference.pipeline.preprocessing import provider as preprocessing_provider

    openvino_result = (["OpenVINOExecutionProvider", "CPUExecutionProvider"], [{"device_type": "GPU"}])
    with mock.patch("modules.inference.pipeline.preprocessing.provider.config") as mock_cfg:
        mock_cfg.DEVICE = "CUDA"
        providers, options = preprocessing_provider._block_openvino_alongside_cuda(
            openvino_result, "0", ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

    assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert options[0]["device_id"] == 0


def test_block_openvino_alongside_cuda_falls_back_to_cpu_when_cuda_unavailable():
    """The CUDA redirect itself must still fall back to CPU if CUDAExecutionProvider
    isn't actually available in onnxruntime, rather than claiming CUDA it can't use."""
    from modules.inference.pipeline.preprocessing import provider as preprocessing_provider

    openvino_result = (["OpenVINOExecutionProvider", "CPUExecutionProvider"], [{"device_type": "GPU"}])
    with mock.patch("modules.inference.pipeline.preprocessing.provider.config") as mock_cfg:
        mock_cfg.DEVICE = "CUDA"
        providers, options = preprocessing_provider._block_openvino_alongside_cuda(openvino_result, "0", ["CPUExecutionProvider"])

    assert providers == ["CPUExecutionProvider"]
    assert options == [{}]


def test_ensure_openvino_onnxruntime_does_not_reload_when_device_is_cuda():
    """_ensure_openvino_onnxruntime must skip the ONNX Runtime hot-reload entirely
    when ASR runs on CUDA, regardless of the requested preprocessing device_type --
    reloading would swap the process-wide `ort` module to the Intel-only build and
    poison CUDA-based execution providers for the rest of the process."""
    with (
        mock.patch("modules.inference.pipeline.preprocessing.config") as mock_cfg,
        mock.patch("modules.inference.pipeline.preprocessing._reload_onnxruntime_from_intel_path") as mock_reload,
    ):
        mock_cfg.DEVICE = "CUDA"
        preprocessing._ensure_openvino_onnxruntime("GPU")

    mock_reload.assert_not_called()


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
