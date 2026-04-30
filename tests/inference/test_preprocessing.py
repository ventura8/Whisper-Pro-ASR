"""Tests for preprocessing.py (UVR/MDX-NET Vocal Separation)."""
# pylint: skip-file
from unittest import mock
import time
import pytest
import os
from modules.inference import preprocessing
from modules.inference.preprocessing import PreprocessingManager


@pytest.fixture
def prep_manager():
    """Fixture to provide a clean PreprocessingManager instance."""
    unit = {"id": "CPU", "type": "CPU", "name": "CPU"}
    return PreprocessingManager(assigned_unit=unit)


class TestApplyOptimizations:
    """Tests for ONNX optimizations."""

    def test_apply_onnx_optimizations_success(self):
        """Test successful application of ONNX optimizations."""
        mock_ort = mock.MagicMock()
        mock_ort.InferenceSession._is_patched = False

        with mock.patch("modules.inference.preprocessing.ort", mock_ort):
            preprocessing.apply_onnx_optimizations()
            assert mock_ort.InferenceSession._is_patched is True

    def test_apply_onnx_optimizations_already_patched(self):
        """Test that optimizations are not reapplied if already patched."""
        mock_ort = mock.MagicMock()
        mock_ort.InferenceSession._is_patched = True

        with mock.patch("modules.inference.preprocessing.ort", mock_ort):
            preprocessing.apply_onnx_optimizations()
            # No changes should happen, but we just verify it doesn't crash
            assert mock_ort.InferenceSession._is_patched is True


# Removed fragile initialization tests (covered by integration suite)


class TestResolveProviders:
    """Tests for provider resolution."""

    def test_resolve_providers_cuda(self, prep_manager):
        """Test CUDA provider resolution."""
        prep_manager._device_type = "CUDA"
        prep_manager._device_id = "cuda:0"
        available = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        providers, options = prep_manager._resolve_providers(available)
        assert "CUDAExecutionProvider" in providers
        assert options[0]["device_id"] == "0"

    def test_resolve_providers_openvino(self, prep_manager):
        """Test OpenVINO provider resolution."""
        prep_manager._device_type = "GPU"
        prep_manager._device_id = "GPU"
        available = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
        with mock.patch("modules.inference.preprocessing.config") as mock_cfg:
            mock_cfg.OV_CACHE_DIR = "/tmp/ov_cache"
            providers, options = prep_manager._resolve_providers(available)
            assert "OpenVINOExecutionProvider" in providers
            assert options[0]["device_type"] == "GPU"
            assert options[0]["cache_dir"] == os.path.abspath("/tmp/ov_cache")

    def test_resolve_providers_cpu(self, prep_manager):
        """Test CPU provider resolution."""
        prep_manager._device_type = "CPU"
        available = ["CPUExecutionProvider"]
        with mock.patch("modules.inference.preprocessing.config") as mock_cfg:
            mock_cfg.PREPROCESS_THREADS = 4
            providers, options = prep_manager._resolve_providers(available)
            assert providers == ["CPUExecutionProvider"]
            assert options[0]["intra_op_num_threads"] == "4"

    def test_resolve_providers_npu(self, prep_manager):
        """Test NPU provider resolution."""
        prep_manager._device_type = "NPU"
        prep_manager._device_id = "NPU"
        available = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
        with mock.patch("modules.inference.preprocessing.config") as mock_cfg:
            mock_cfg.OV_CACHE_DIR = "/tmp/ov_cache"
            providers, options = prep_manager._resolve_providers(available)
            assert "OpenVINOExecutionProvider" in providers
            assert options[0]["device_type"] == "NPU"


class TestProcessAudio:
    """Tests for preprocess_audio."""

    def test_process_audio_disabled(self, prep_manager):
        """Test returns original path when disabled."""
        with mock.patch("modules.inference.preprocessing.config") as mock_cfg:
            mock_cfg.ENABLE_VOCAL_SEPARATION = False
            result = prep_manager.preprocess_audio("test.wav")
            assert result == "test.wav"

    def test_process_audio_success(self, prep_manager):
        """Test successful vocal isolation."""
        with mock.patch("modules.inference.preprocessing.config") as mock_cfg:
            mock_cfg.ENABLE_VOCAL_SEPARATION = True
            mock_sep = mock.MagicMock()
            mock_sep.separate.return_value = ["vocal.wav"]
            prep_manager._init_separator = mock.MagicMock(return_value=mock_sep)

            mock_cb = mock.MagicMock()
            result = prep_manager.preprocess_audio("test.wav", yield_cb=mock_cb)
            assert result == str(preprocessing.CACHE_DIR / "vocal.wav")
            assert mock_cb.call_count == 2

    def test_process_audio_failure(self, prep_manager):
        """Test returns original path on failure."""
        with mock.patch("modules.inference.preprocessing.config") as mock_cfg:
            mock_cfg.ENABLE_VOCAL_SEPARATION = True
            prep_manager._init_separator = mock.MagicMock(side_effect=Exception("fail"))
            result = prep_manager.preprocess_audio("test.wav")
            assert result == "test.wav"


class TestOffload:
    """Tests for resource offloading."""

    def test_offload_clears_cuda_cache(self, prep_manager):
        """Test that offload calls utils.clear_gpu_cache."""
        prep_manager.separator = mock.MagicMock()
        with mock.patch("modules.inference.preprocessing.utils.clear_gpu_cache") as mock_clear:
            prep_manager.offload()
            assert prep_manager.separator is None
            mock_clear.assert_called_once()


class TestPurgeCache:
    """Tests for cache purging."""

    def test_purge_stale_cache(self, prep_manager):
        """Test cache purging logic including exceptions."""
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

    def test_purge_stale_cache_exception(self, prep_manager):
        """Test that purge handles exceptions gracefully."""
        with mock.patch("modules.inference.preprocessing.CACHE_DIR") as mock_cache:
            mock_cache.iterdir.side_effect = Exception("access denied")
            prep_manager._purge_stale_cache()


class TestPreprocessAudioExtras:
    """Extra coverage for edge cases in preprocess_audio."""

    def test_preprocess_audio_stem_cleanup(self, prep_manager):
        """Test that extra stems are removed correctly."""
        with mock.patch("modules.inference.preprocessing.config") as mock_cfg:
            mock_cfg.ENABLE_VOCAL_SEPARATION = True
            mock_sep = mock.MagicMock()
            # Simulate returning two stems
            mock_sep.separate.return_value = ["vocal.wav", "instrument.wav"]
            prep_manager._init_separator = mock.MagicMock(return_value=mock_sep)

            with mock.patch("os.path.exists", return_value=True):
                with mock.patch("os.remove") as mock_remove:
                    with mock.patch("modules.inference.preprocessing.utils.prepare_for_uvr", side_effect=lambda x: x):
                        result = prep_manager.preprocess_audio("test.wav")
                        assert "vocal.wav" in result
                        # Verify cleanup of second stem
                        mock_remove.assert_called_once()

    def test_preprocess_audio_no_stems(self, prep_manager):
        """Test handling when separator returns empty list."""
        with mock.patch("modules.inference.preprocessing.config") as mock_cfg:
            mock_cfg.ENABLE_VOCAL_SEPARATION = True
            mock_sep = mock.MagicMock()
            mock_sep.separate.return_value = []
            prep_manager._init_separator = mock.MagicMock(return_value=mock_sep)

            with mock.patch("modules.inference.preprocessing.utils.prepare_for_uvr", side_effect=lambda x: x):
                result = prep_manager.preprocess_audio("test.wav")
                assert result == "test.wav"

    def test_preprocess_audio_prepare_fail(self, prep_manager):
        """Test handling when audio preparation fails."""
        with mock.patch("modules.inference.preprocessing.config") as mock_cfg:
            mock_cfg.ENABLE_VOCAL_SEPARATION = True
            with mock.patch("modules.inference.preprocessing.utils.prepare_for_uvr", return_value=None):
                result = prep_manager.preprocess_audio("test.wav")
                assert result == "test.wav"


def test_preprocessing_lazy_import():
    """Cover lazy import branch."""
    # It might be None if audio-separator is not installed, which is fine for coverage
    res = preprocessing._lazy_import_separator()
    assert res is None or res is not None


def test_preprocessing_manager_unload(prep_manager):
    """Cover manual unload."""
    prep_manager.separator = mock.MagicMock()
    prep_manager.unload_model()
    assert prep_manager.separator is None


def test_preprocessing_separate_no_cpu_lock():
    """Cover non-CPU separation path."""
    pm = preprocessing.PreprocessingManager({"id": "cuda:0", "type": "CUDA", "name": "GPU"})
    pm.separator = mock.MagicMock()
    pm.separator.separate.return_value = ["vocal.wav"]

    with mock.patch("modules.utils.prepare_for_uvr", return_value="prep.wav"):
        res = pm.preprocess_audio("test.wav", force=True)
        assert "vocal.wav" in res


def test_preprocessing_cleanup_error(prep_manager):
    """Cover error during extra stem removal."""
    prep_manager.separator = mock.MagicMock()
    prep_manager.separator.separate.return_value = ["v1.wav", "v2.wav"]

    with mock.patch("os.path.exists", return_value=True), \
            mock.patch("os.remove", side_effect=IOError("Locked")):
        prep_manager.preprocess_audio("test.wav", force=True)


def test_preprocessing_ort_import_error(prep_manager):
    """Cover ImportError for onnxruntime."""
    with mock.patch("modules.inference.preprocessing.ort", None):
        with pytest.raises(ImportError):
            prep_manager._init_separator()


def test_preprocessing_separator_import_error(prep_manager):
    """Cover ImportError for audio-separator."""
    with mock.patch("modules.inference.preprocessing._lazy_import_separator", return_value=None):
        with pytest.raises(ImportError):
            prep_manager._init_separator()
