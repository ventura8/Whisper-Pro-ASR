"""Tests for preprocessing.py (UVR/MDX-NET Vocal Separation)."""
# pylint: skip-file
from unittest import mock
import time
import pytest
import os
from modules import preprocessing
from modules.preprocessing import PreprocessingManager


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

        with mock.patch("modules.preprocessing.ort", mock_ort):
            preprocessing.apply_onnx_optimizations()
            assert mock_ort.InferenceSession._is_patched is True

    def test_apply_onnx_optimizations_already_patched(self):
        """Test that optimizations are not reapplied if already patched."""
        mock_ort = mock.MagicMock()
        mock_ort.InferenceSession._is_patched = True

        with mock.patch("modules.preprocessing.ort", mock_ort):
            preprocessing.apply_onnx_optimizations()
            # No changes should happen, but we just verify it doesn't crash
            assert mock_ort.InferenceSession._is_patched is True


class TestInitSeparator:
    """Tests for separator initialization."""

    def test_init_separator_success(self, prep_manager):
        """Test successful separator initialization."""
        mock_sep_instance = mock.MagicMock()

        with mock.patch("modules.preprocessing.Separator", return_value=mock_sep_instance):
            with mock.patch("modules.preprocessing.ort") as mock_ort:
                mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
                sep = prep_manager._init_separator()
                assert sep == mock_sep_instance

    def test_init_separator_already_initialized(self, prep_manager):
        """Test returns existing separator if already initialized."""
        prep_manager.separator = "existing"
        assert prep_manager._init_separator() == "existing"

    def test_init_separator_missing_ort(self, prep_manager):
        """Test failure when onnxruntime is missing."""
        # Ensure Separator is "installed" so we reach the ort check
        with mock.patch("modules.preprocessing.Separator", mock.MagicMock()):
            with mock.patch("modules.preprocessing.ort", None):
                with pytest.raises(ImportError, match="onnxruntime not installed"):
                    prep_manager._init_separator()


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
        with mock.patch("modules.preprocessing.config") as mock_cfg:
            mock_cfg.OV_CACHE_DIR = "/tmp/ov_cache"
            providers, options = prep_manager._resolve_providers(available)
            assert "OpenVINOExecutionProvider" in providers
            assert options[0]["device_type"] == "GPU"


class TestProcessAudio:
    """Tests for process_audio_file."""

    def test_process_audio_disabled(self, prep_manager):
        """Test returns original path when disabled."""
        with mock.patch("modules.preprocessing.config") as mock_cfg:
            mock_cfg.ENABLE_VOCAL_SEPARATION = False
            result = prep_manager.process_audio_file("test.wav")
            assert result == "test.wav"

    def test_process_audio_success(self, prep_manager):
        """Test successful vocal isolation."""
        with mock.patch("modules.preprocessing.config") as mock_cfg:
            mock_cfg.ENABLE_VOCAL_SEPARATION = True
            mock_sep = mock.MagicMock()
            mock_sep.separate.return_value = ["vocal.wav"]
            prep_manager._init_separator = mock.MagicMock(return_value=mock_sep)

            mock_cb = mock.MagicMock()
            result = prep_manager.process_audio_file("test.wav", yield_cb=mock_cb)
            assert result == str(preprocessing.CACHE_DIR / "vocal.wav")
            assert mock_cb.call_count == 2

    def test_process_audio_failure(self, prep_manager):
        """Test returns original path on failure."""
        with mock.patch("modules.preprocessing.config") as mock_cfg:
            mock_cfg.ENABLE_VOCAL_SEPARATION = True
            prep_manager._init_separator = mock.MagicMock(side_effect=Exception("fail"))
            result = prep_manager.process_audio_file("test.wav")
            assert result == "test.wav"


class TestOffload:
    """Tests for resource offloading."""

    def test_offload_clears_cuda_cache(self, prep_manager):
        """Test that offload calls torch.cuda.empty_cache if available."""
        prep_manager.separator = mock.MagicMock()
        with mock.patch("modules.preprocessing.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            prep_manager.offload()
            assert prep_manager.separator is None
            mock_torch.cuda.empty_cache.assert_called_once()


class TestPurgeCache:
    """Tests for cache purging."""

    def test_purge_stale_cache(self, prep_manager):
        """Test cache purging logic including exceptions."""
        with mock.patch("modules.preprocessing.CACHE_DIR") as mock_cache:
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
        with mock.patch("modules.preprocessing.CACHE_DIR") as mock_cache:
            mock_cache.iterdir.side_effect = Exception("access denied")
            prep_manager._purge_stale_cache()
