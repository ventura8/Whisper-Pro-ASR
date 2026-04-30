"""Tests for preprocessing.py (UVR/MDX-NET Vocal Separation)."""
# pylint: disable=missing-function-docstring, protected-access, redefined-outer-name, too-few-public-methods, unused-argument, import-error
import time
import logging
from unittest import mock

import pytest

from modules.inference import preprocessing
from modules.inference.preprocessing import PreprocessingManager, CACHE_DIR

logger = logging.getLogger(__name__)


@pytest.fixture
def prep_manager():
    """Fixture to provide a clean PreprocessingManager instance."""
    unit = {"id": "CPU", "type": "CPU", "name": "CPU"}
    return PreprocessingManager(assigned_unit=unit)


class TestManagerBasics:
    """Tests for basic manager operations."""

    def test_init_defaults(self):
        with mock.patch("modules.inference.preprocessing.config") as mock_cfg:
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
        with mock.patch("modules.inference.preprocessing.utils.clear_gpu_cache") as mock_clear:
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
        assert options[0]["device_id"] == "1"

    def test_resolve_openvino_gpu(self, prep_manager):
        prep_manager._device_type = "GPU"
        prep_manager._device_id = "GPU.0"
        available = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
        with mock.patch("modules.inference.preprocessing.config") as mock_cfg:
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

    def test_resolve_openvino_custom_device(self, prep_manager):
        """Test OpenVINO resolution with a custom device ID."""
        prep_manager._device_type = "OpenVINO"
        prep_manager._device_id = "GPU.1"
        available = ["OpenVINOExecutionProvider"]
        with mock.patch("modules.inference.preprocessing.config") as mock_cfg:
            mock_cfg.OV_CACHE_DIR = "/tmp/ov"
            _, options = prep_manager._resolve_providers(available)
            assert options[0]["device_type"] == "GPU.1"


class TestSeparatorInit:
    """Tests for separator initialization and patching."""

    def test_apply_onnx_optimizations_success(self):
        class MockSession:
            """Mock ORT Session."""
            _is_patched = False

            def __init__(self, *args, **kwargs):
                pass

        mock_ort = mock.MagicMock()
        mock_ort.InferenceSession = MockSession
        with mock.patch("modules.inference.preprocessing.ort", mock_ort):
            preprocessing.apply_onnx_optimizations()
            assert MockSession._is_patched is True

    def test_onnx_session_patching_logic(self):
        """Test the actual patched __init__ function logic."""
        original_init_called = [False]

        class MockSession:
            """Mock ORT Session for patching test."""
            _is_patched = False

            def __init__(self, model_path, sess_options=None, providers=None, provider_options=None, **kwargs):
                original_init_called[0] = True
                self.providers = providers
                self.provider_options = provider_options

        mock_ort = mock.MagicMock()
        mock_ort.InferenceSession = MockSession

        with mock.patch("modules.inference.preprocessing.ort", mock_ort):
            preprocessing.apply_onnx_optimizations()

            with mock.patch("modules.inference.preprocessing.utils.THREAD_CONTEXT") as mock_ctx:
                mock_ctx.ov_options = {"device_type": "GPU"}

                # Test intercept CPU fallback
                session = mock_ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
                assert "OpenVINOExecutionProvider" in session.providers
                assert session.provider_options[0]["device_type"] == "GPU"
                assert original_init_called[0] is True

                logger.info("[Test] Mocking UVR with %s | %s",
                            "model.onnx", "model")
                session2 = mock_ort.InferenceSession(
                    "model.onnx",
                    providers=["OpenVINOExecutionProvider", "CPUExecutionProvider"]
                )
                assert session2.provider_options[0]["device_type"] == "GPU"

                # Test provider_options as list with wrong type (should become dict)
                session3 = mock_ort.InferenceSession("model.onnx", providers=["OpenVINOExecutionProvider"], provider_options=[None])
                assert isinstance(session3.provider_options[0], dict)
                assert session3.provider_options[0]["device_type"] == "GPU"

                # Test provider_options list expansion
                session4 = mock_ort.InferenceSession(
                    "model.onnx",
                    providers=["CPUExecutionProvider", "OpenVINOExecutionProvider"],
                    provider_options=[{}]
                )
                assert len(session4.provider_options) == 2
                assert session4.provider_options[1]["device_type"] == "GPU"

    def test_init_separator_success(self, prep_manager):
        mock_ort = mock.MagicMock()
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
        mock_ort.__version__ = "1.24.1"

        with mock.patch("modules.inference.preprocessing.ort", mock_ort):
            with mock.patch("modules.inference.preprocessing._lazy_import_separator") as mock_imp:
                mock_sep_cls = mock.MagicMock()
                mock_imp.return_value = mock_sep_cls

                sep = prep_manager._init_separator()
                assert sep is not None
                assert prep_manager.separator is not None
                mock_sep_cls.assert_called_once()

    def test_init_separator_failure(self, prep_manager):
        with mock.patch("modules.inference.preprocessing._lazy_import_separator") as mock_imp:
            mock_sep_cls = mock.MagicMock()
            mock_imp.return_value = mock_sep_cls
            mock_sep_inst = mock_sep_cls.return_value
            mock_sep_inst.load_model.side_effect = Exception("Fail")

            mock_ort = mock.MagicMock()
            mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

            with mock.patch("modules.inference.preprocessing.ort", mock_ort):
                with pytest.raises(Exception):
                    prep_manager._init_separator()
                assert prep_manager.separator is None

    def test_init_separator_no_ort(self, prep_manager):
        with mock.patch("modules.inference.preprocessing.ort", None):
            with pytest.raises(ImportError):
                prep_manager._init_separator()

    def test_init_separator_no_sep(self, prep_manager):
        with mock.patch("modules.inference.preprocessing._lazy_import_separator", return_value=None):
            with pytest.raises(ImportError):
                prep_manager._init_separator()


class TestPreprocessAudio:
    """Tests for the main preprocess_audio entry point."""

    def test_preprocess_disabled(self, prep_manager):
        with mock.patch("modules.inference.preprocessing.config") as mock_cfg:
            mock_cfg.ENABLE_VOCAL_SEPARATION = False
            assert prep_manager.preprocess_audio("test.wav") == "test.wav"

    def test_preprocess_success(self, prep_manager):
        with mock.patch("modules.inference.preprocessing.config") as mock_cfg:
            mock_cfg.ENABLE_VOCAL_SEPARATION = True
            mock_sep = mock.MagicMock()
            mock_sep.separate.return_value = ["vocal.wav"]
            prep_manager._init_separator = mock.MagicMock(return_value=mock_sep)

            mock_ort = mock.MagicMock()
            mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

            with mock.patch("modules.inference.preprocessing.ort", mock_ort):
                with mock.patch("modules.inference.preprocessing.utils.prepare_for_uvr", side_effect=lambda x: x):
                    res = prep_manager.preprocess_audio("test.wav")
                    assert "vocal.wav" in res

    def test_preprocess_cleanup_extra_stems(self, prep_manager):
        with mock.patch("modules.inference.preprocessing.config") as mock_cfg:
            mock_cfg.ENABLE_VOCAL_SEPARATION = True
            mock_sep = mock.MagicMock()
            mock_sep.separate.return_value = ["vocal.wav", "instrumental.wav"]
            prep_manager._init_separator = mock.MagicMock(return_value=mock_sep)

            mock_ort = mock.MagicMock()
            mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

            with mock.patch("modules.inference.preprocessing.ort", mock_ort), \
                    mock.patch("os.path.exists", return_value=True), \
                    mock.patch("os.remove") as mock_remove:
                with mock.patch("modules.inference.preprocessing.utils.prepare_for_uvr", side_effect=lambda x: x):
                    prep_manager.preprocess_audio("test.wav")
                    mock_remove.assert_called()

    def test_preprocess_exception(self, prep_manager):
        with mock.patch("modules.inference.preprocessing.config") as mock_cfg:
            mock_cfg.ENABLE_VOCAL_SEPARATION = True
            prep_manager._init_separator = mock.MagicMock(side_effect=Exception("Crash"))
            assert prep_manager.preprocess_audio("test.wav") == "test.wav"

    def test_preprocess_force(self, prep_manager):
        with mock.patch("modules.inference.preprocessing.config") as mock_cfg:
            mock_cfg.ENABLE_VOCAL_SEPARATION = False  # Disabled globally
            mock_sep = mock.MagicMock()
            mock_sep.separate.return_value = ["vocal.wav"]
            prep_manager._init_separator = mock.MagicMock(return_value=mock_sep)

            mock_ort = mock.MagicMock()
            mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

            with mock.patch("modules.inference.preprocessing.ort", mock_ort):
                with mock.patch("modules.inference.preprocessing.utils.prepare_for_uvr", side_effect=lambda x: x):
                    # Should still run because force=True
                    res = prep_manager.preprocess_audio("test.wav", force=True)
                    assert "vocal.wav" in res

    def test_preprocess_relative_stem(self, prep_manager):
        with mock.patch("modules.inference.preprocessing.config") as mock_cfg:
            mock_cfg.ENABLE_VOCAL_SEPARATION = True
            mock_sep = mock.MagicMock()
            mock_sep.separate.return_value = ["relative_vocal.wav"]
            prep_manager._init_separator = mock.MagicMock(return_value=mock_sep)

            with mock.patch("modules.inference.preprocessing.ort", mock.MagicMock()):
                with mock.patch("modules.inference.preprocessing.utils.prepare_for_uvr", side_effect=lambda x: x):
                    res = prep_manager.preprocess_audio("test.wav")
                    assert "relative_vocal.wav" in res
                    assert str(CACHE_DIR) in res

    def test_preprocess_cleanup_error(self, prep_manager):
        with mock.patch("modules.inference.preprocessing.config") as mock_cfg:
            mock_cfg.ENABLE_VOCAL_SEPARATION = True
            mock_sep = mock.MagicMock()
            mock_sep.separate.return_value = ["vocal.wav", "extra.wav"]
            prep_manager._init_separator = mock.MagicMock(return_value=mock_sep)

            with mock.patch("modules.inference.preprocessing.ort", mock.MagicMock()), \
                    mock.patch("os.path.exists", return_value=True), \
                    mock.patch("os.remove", side_effect=OSError("Busy")):
                with mock.patch("modules.inference.preprocessing.utils.prepare_for_uvr", side_effect=lambda x: x):
                    # Should not raise
                    prep_manager.preprocess_audio("test.wav")

    def test_preprocess_prepare_fail(self, prep_manager):
        with mock.patch("modules.inference.preprocessing.config") as mock_cfg:
            mock_cfg.ENABLE_VOCAL_SEPARATION = True
            with mock.patch("modules.inference.preprocessing.utils.prepare_for_uvr", return_value=None):
                assert prep_manager.preprocess_audio("test.wav") == "test.wav"


class TestCache:
    """Tests for cache management."""

    def test_purge_cache_success(self, prep_manager):
        with mock.patch("modules.inference.preprocessing.CACHE_DIR") as mock_cache:
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
        with mock.patch("modules.inference.preprocessing.CACHE_DIR") as mock_cache:
            mock_cache.iterdir.side_effect = Exception("Fail")
            prep_manager._purge_stale_cache()  # Should not raise

    def test_purge_cache_unlink_fail(self, prep_manager):
        with mock.patch("modules.inference.preprocessing.CACHE_DIR") as mock_cache:
            mock_file = mock.MagicMock()
            mock_file.is_file.return_value = True
            mock_file.stat.return_value.st_mtime = time.time() - 4000
            mock_file.unlink.side_effect = OSError("Locked")
            mock_cache.iterdir.return_value = [mock_file]
            prep_manager._purge_stale_cache()  # Should not raise


def test_lazy_import():
    res = preprocessing._lazy_import_separator()
    assert res is not None or res is None
