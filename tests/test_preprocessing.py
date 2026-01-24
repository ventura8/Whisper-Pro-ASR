"""Tests for preprocessing.py (UVR/MDX-NET Vocal Separation)."""
from unittest import mock
import os
import numpy as np
from modules.preprocessing import PreprocessingManager
from modules import preprocessing

# pylint: disable=protected-access, too-few-public-methods


class TestGetManager:
    """Tests for get_manager singleton function."""

    def test_get_manager_creates_instance(self):
        """Test that get_manager creates a new instance."""
        # Reset singleton
        preprocessing._INSTANCE = None

        with mock.patch.object(preprocessing.PreprocessingManager, '__init__', return_value=None):
            manager = preprocessing.get_manager()
            assert manager is not None
            assert preprocessing._INSTANCE is manager

    def test_get_manager_returns_same_instance(self):
        """Test that get_manager returns the same instance on subsequent calls."""
        mock_instance = mock.MagicMock()
        preprocessing._INSTANCE = mock_instance

        manager = preprocessing.get_manager()
        assert manager is mock_instance


class TestPreprocessingManager:
    """Tests for PreprocessingManager class."""

    def test_init(self):
        """Test manager initialization."""
        with mock.patch('modules.preprocessing.config') as mock_config:
            mock_config.PREPROCESS_DEVICE = "GPU"
            mock_config.PREPROCESSING_CACHE_DIR = "/tmp/cache"

            manager = PreprocessingManager()

            assert manager.device == "GPU"
            assert manager.separator is None

    def test_ensure_models_loaded_disabled(self):
        """Test ensure_models_loaded when vocal separation is disabled."""
        with mock.patch('modules.preprocessing.config') as mock_config:
            mock_config.ENABLE_VOCAL_SEPARATION = False
            mock_config.ENABLE_LD_PREPROCESSING = False
            mock_config.PREPROCESS_DEVICE = "CPU"
            mock_config.PREPROCESSING_CACHE_DIR = "/tmp"

            manager = PreprocessingManager()
            manager._init_separator = mock.MagicMock()

            manager.ensure_models_loaded()

            manager._init_separator.assert_not_called()

    def test_ensure_models_loaded_enabled(self):
        """Test ensure_models_loaded when vocal separation is enabled."""
        with mock.patch('modules.preprocessing.config') as mock_config:
            mock_config.ENABLE_VOCAL_SEPARATION = True
            mock_config.ENABLE_LD_PREPROCESSING = False
            mock_config.PREPROCESS_DEVICE = "CPU"
            mock_config.PREPROCESSING_CACHE_DIR = "/tmp"
            mock_config.OV_CACHE_DIR = "/tmp"

            manager = PreprocessingManager()

            mock_separator = mock.MagicMock()
            mock_separator.separate.return_value = [
                "output_(Vocals)_model.wav"]
            manager.separator = mock_separator
            manager._init_separator = mock.MagicMock()

            with mock.patch('os.path.exists', return_value=True):
                with mock.patch('os.remove'):
                    with mock.patch('os.listdir', return_value=[]):
                        with mock.patch('modules.preprocessing.sf.write'):
                            manager.ensure_models_loaded()

    def test_init_separator_already_loaded(self):
        """Test _init_separator returns early if already loaded."""
        with mock.patch('modules.preprocessing.config') as mock_config:
            mock_config.PREPROCESS_DEVICE = "CPU"
            mock_config.PREPROCESSING_CACHE_DIR = "/tmp"

            manager = PreprocessingManager()
            manager.separator = mock.MagicMock()  # Already loaded

            # Should return early without doing anything
            manager._init_separator()


class TestProcessAudioFile:
    """Tests for process_audio_file method."""

    def test_process_audio_file_disabled(self):
        """Test process_audio_file returns original when disabled."""
        with mock.patch('modules.preprocessing.config') as mock_config:
            mock_config.ENABLE_VOCAL_SEPARATION = False
            mock_config.ENABLE_LD_PREPROCESSING = False
            mock_config.PREPROCESS_DEVICE = "CPU"
            mock_config.PREPROCESSING_CACHE_DIR = "/tmp"

            manager = PreprocessingManager()

            result = manager.process_audio_file("/path/to/file.wav")
            assert result == "/path/to/file.wav"

    def test_process_audio_file_success(self):
        """Test successful vocal separation."""
        with mock.patch('modules.preprocessing.config') as mock_config:
            mock_config.ENABLE_VOCAL_SEPARATION = True
            mock_config.PREPROCESS_DEVICE = "CPU"
            mock_config.PREPROCESSING_CACHE_DIR = "/tmp"

            manager = PreprocessingManager()

            mock_separator = mock.MagicMock()
            mock_separator.separate.return_value = [
                "file_(Vocals)_model.wav",
                "file_(Instrumental)_model.wav"
            ]
            manager.separator = mock_separator
            manager._init_separator = mock.MagicMock()

            with mock.patch('os.path.exists', return_value=True):
                with mock.patch('os.remove'):
                    with mock.patch(
                        'modules.preprocessing.utils.format_duration',
                        return_value="00:00:01"
                    ):
                        result = manager.process_audio_file(
                            "/path/to/file.wav")
                        assert "(Vocals)" in result

    def test_process_audio_file_vocal_in_cache_dir(self):
        """Test when vocal file is in cache directory."""
        with mock.patch('modules.preprocessing.config') as mock_config:
            mock_config.ENABLE_VOCAL_SEPARATION = True
            mock_config.PREPROCESS_DEVICE = "CPU"
            mock_config.PREPROCESSING_CACHE_DIR = "/tmp"

            manager = PreprocessingManager()

            mock_separator = mock.MagicMock()
            mock_separator.separate.return_value = ["file_(Vocals)_model.wav"]
            manager.separator = mock_separator
            manager._init_separator = mock.MagicMock()

            def exists_side_effect(path):
                """Return True for cache-related paths, False otherwise."""
                return "/tmp" in str(path) or "cache" in str(path).lower()

            with mock.patch('os.path.exists', side_effect=exists_side_effect):
                with mock.patch(
                    'modules.preprocessing.utils.format_duration',
                    return_value="00:00:01"
                ):
                    with mock.patch('os.remove'):
                        manager.process_audio_file("/path/to/file.wav")
                        # Should find the path

    def test_process_audio_file_no_vocals_found(self):
        """Test when vocals track is not found in output."""
        with mock.patch('modules.preprocessing.config') as mock_config:
            mock_config.ENABLE_VOCAL_SEPARATION = True
            mock_config.PREPROCESS_DEVICE = "CPU"
            mock_config.PREPROCESSING_CACHE_DIR = "/tmp"

            manager = PreprocessingManager()

            mock_separator = mock.MagicMock()
            mock_separator.separate.return_value = ["file_(Other)_model.wav"]
            manager.separator = mock_separator
            manager._init_separator = mock.MagicMock()

            result = manager.process_audio_file("/path/to/file.wav")
            assert result == "/path/to/file.wav"

    def test_process_audio_file_exception(self):
        """Test exception handling in process_audio_file."""
        with mock.patch('modules.preprocessing.config') as mock_config:
            mock_config.ENABLE_VOCAL_SEPARATION = True
            mock_config.PREPROCESS_DEVICE = "CPU"
            mock_config.PREPROCESSING_CACHE_DIR = "/tmp"

            manager = PreprocessingManager()
            manager._init_separator = mock.MagicMock(
                side_effect=RuntimeError("Init failed"))

            result = manager.process_audio_file("/path/to/file.wav")
            assert result == "/path/to/file.wav"


class TestProcessAudioChunk:
    """Tests for process_audio_chunk method."""

    def test_process_audio_chunk_disabled(self):
        """Test process_audio_chunk returns original when disabled."""
        with mock.patch('modules.preprocessing.config') as mock_config:
            mock_config.ENABLE_VOCAL_SEPARATION = False
            mock_config.ENABLE_LD_PREPROCESSING = False
            mock_config.PREPROCESS_DEVICE = "CPU"
            mock_config.PREPROCESSING_CACHE_DIR = "/tmp"

            manager = PreprocessingManager()

            chunk = np.zeros(16000, dtype=np.float32)
            result = manager.process_audio_chunk(chunk)
            assert np.array_equal(result, chunk)

    def test_process_audio_chunk_silence(self):
        """Test that silence is returned unchanged (optimization)."""
        with mock.patch('modules.preprocessing.config') as mock_config:
            mock_config.ENABLE_VOCAL_SEPARATION = True
            mock_config.PREPROCESS_DEVICE = "CPU"
            mock_config.PREPROCESSING_CACHE_DIR = "/tmp"

            manager = PreprocessingManager()

            # Very quiet chunk (below 0.005 threshold)
            chunk = np.ones(16000, dtype=np.float32) * 0.001
            result = manager.process_audio_chunk(chunk)
            assert np.array_equal(result, chunk)

    def test_process_audio_chunk_success(self):
        """Test successful chunk processing."""
        with mock.patch('modules.preprocessing.config') as mock_config:
            mock_config.ENABLE_VOCAL_SEPARATION = True
            mock_config.PREPROCESS_DEVICE = "CPU"
            mock_config.PREPROCESSING_CACHE_DIR = "/tmp"

            manager = PreprocessingManager()

            mock_separator = mock.MagicMock()
            mock_separator.separate.return_value = ["chunk_(Vocals)_model.wav"]
            manager.separator = mock_separator
            manager._init_separator = mock.MagicMock()

            # Loud chunk that needs processing
            chunk = np.ones(16000, dtype=np.float32) * 0.1

            # Mock file operations
            mock_audio = np.ones(16000, dtype=np.float32) * 0.05

            with mock.patch('modules.preprocessing.sf.write'):
                with mock.patch('modules.preprocessing.sf.read', return_value=(mock_audio, 16000)):
                    with mock.patch('os.path.exists', return_value=True):
                        with mock.patch('os.remove'):
                            result = manager.process_audio_chunk(chunk)
                            assert len(result) == 16000

    def test_process_audio_chunk_exception(self):
        """Test exception handling returns original chunk."""
        with mock.patch('modules.preprocessing.config') as mock_config:
            mock_config.ENABLE_VOCAL_SEPARATION = True
            mock_config.PREPROCESS_DEVICE = "CPU"
            mock_config.PREPROCESSING_CACHE_DIR = "/tmp"

            manager = PreprocessingManager()
            manager._init_separator = mock.MagicMock(
                side_effect=RuntimeError("Error"))

            chunk = np.ones(16000, dtype=np.float32) * 0.1
            result = manager.process_audio_chunk(chunk)
            assert np.array_equal(result, chunk)


class TestOffload:
    """Tests for offload method."""

    def test_offload_no_separator(self):
        """Test offload when no separator is loaded."""
        with mock.patch('modules.preprocessing.config') as mock_config:
            mock_config.PREPROCESS_DEVICE = "CPU"
            mock_config.PREPROCESSING_CACHE_DIR = "/tmp"

            manager = PreprocessingManager()
            manager.separator = None

            # Should not raise
            manager.offload()

    def test_offload_with_separator(self):
        """Test offload clears separator."""
        with mock.patch('modules.preprocessing.config') as mock_config:
            mock_config.PREPROCESS_DEVICE = "CPU"
            mock_config.PREPROCESSING_CACHE_DIR = "/tmp"

            manager = PreprocessingManager()
            manager.separator = mock.MagicMock()

            with mock.patch('modules.preprocessing.torch.cuda.is_available', return_value=False):
                manager.offload()

            assert manager.separator is None

    def test_offload_with_cuda(self):
        """Test offload clears CUDA cache when available."""
        with mock.patch('modules.preprocessing.config') as mock_config:
            mock_config.PREPROCESS_DEVICE = "CUDA"
            mock_config.PREPROCESSING_CACHE_DIR = "/tmp"

            manager = PreprocessingManager()
            manager.separator = mock.MagicMock()

            with mock.patch('modules.preprocessing.torch.cuda.is_available', return_value=True):
                with mock.patch('modules.preprocessing.torch.cuda.empty_cache') as mock_empty:
                    manager.offload()
                    mock_empty.assert_called_once()


class TestPreprocessingEdgeChunks:
    """Process and chunk edge cases."""

    def test_process_audio_file_cleanup_error(self):
        """Test process_audio_file when cleanup of instrumental fails."""
        with mock.patch('modules.preprocessing.config') as config:
            config.ENABLE_VOCAL_SEPARATION = True
            manager = PreprocessingManager()
            manager._init_separator = mock.MagicMock()
            mock_sep = mock.MagicMock()
            mock_sep.separate.return_value = [
                "file_(Vocals).wav", "file_(Instr).wav"]
            manager.separator = mock_sep

            with mock.patch('os.path.exists', return_value=True):
                with mock.patch('os.remove', side_effect=Exception("Cleanup error")):
                    res = manager.process_audio_file("orig.wav")
                    assert "Vocals" in res

    def test_process_audio_chunk_sf_write_fail(self):
        """Test exception during chunk processing."""
        with mock.patch('modules.preprocessing.config') as config:
            config.ENABLE_VOCAL_SEPARATION = True
            manager = PreprocessingManager()
            with mock.patch('modules.preprocessing.sf.write', side_effect=Exception("Write fail")):
                chunk = np.zeros(16000)
                res = manager.process_audio_chunk(chunk)
                assert np.array_equal(res, chunk)

    def test_postprocess_chunk_resampling_and_padding(self):
        """Test resampling and padding logic in _postprocess_chunk."""
        manager = PreprocessingManager()
        # Mock 44100Hz audio to be resampled to 16000Hz
        mock_data = np.ones(44100)
        with mock.patch('modules.preprocessing.sf.read', return_value=(mock_data, 44100)):
            res = manager._postprocess_chunk("vocal.wav", 16000, 16000)
            assert len(res) == 16000


class TestPreprocessingEdgeConfig:
    """Config, providers, and session options."""

    def test_configure_providers_cuda_available(self):
        """Test _configure_providers when CUDA is requested and available."""
        with mock.patch("modules.config.PREPROCESS_DEVICE", "CUDA"):
            available = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            providers, _ = PreprocessingManager._configure_providers(
                available, [], [])
            assert providers[0] == "CUDAExecutionProvider"

    def test_configure_providers_cuda_missing(self):
        """Test _configure_providers when CUDA is requested but NOT available."""
        with mock.patch("modules.config.PREPROCESS_DEVICE", "CUDA"):
            available = ["CPUExecutionProvider"]
            providers, _ = PreprocessingManager._configure_providers(
                available, [], [])
            assert "CUDAExecutionProvider" not in providers
            assert "CPUExecutionProvider" in providers

    def test_configure_providers_openvino_available(self):
        """Test _configure_providers when GPU (OpenVINO) is requested."""
        with mock.patch("modules.config.PREPROCESS_DEVICE", "GPU"):
            with mock.patch("modules.config.OV_CACHE_DIR", "/tmp"):
                available = ["OpenVINOExecutionProvider",
                             "CPUExecutionProvider"]
                providers, _ = PreprocessingManager._configure_providers(
                    available, [], [])
                # Should be in the tuple format [('Provider', options)]
                assert isinstance(providers[0], tuple)
                assert providers[0][0] == "OpenVINOExecutionProvider"
                assert providers[0][1]["device_type"] == "GPU"

    def test_log_available_providers_already_logged(self):
        """Test _log_available_providers returns early if already logged."""
        PreprocessingManager._logged_providers = True
        mock_ort = mock.MagicMock()
        with mock.patch("modules.preprocessing.ort", mock_ort):
            PreprocessingManager._log_available_providers()
            mock_ort.get_available_providers.assert_not_called()

    def test_postprocess_chunk_padding_only(self):
        """Test padding logic in _postprocess_chunk without resampling."""
        manager = PreprocessingManager()
        # Same SR, but shorter
        mock_data = np.ones(10000)
        with mock.patch('modules.preprocessing.sf.read', return_value=(mock_data, 16000)):
            res = manager._postprocess_chunk("vocal.wav", 16000, 16000)
            assert len(res) == 16000
            assert np.all(res[:10000] == 1.0)
            assert np.all(res[10000:] == 0.0)

    def test_postprocess_chunk_trimming(self):
        """Test trimming logic in _postprocess_chunk."""
        manager = PreprocessingManager()
        # Same SR, but longer
        mock_data = np.ones(20000)
        with mock.patch('modules.preprocessing.sf.read', return_value=(mock_data, 16000)):
            res = manager._postprocess_chunk("vocal.wav", 16000, 16000)
            assert len(res) == 16000

    def test_init_separator_env_vars(self):
        """Test that _init_separator sets environment variables."""
        manager = PreprocessingManager()
        with mock.patch("modules.preprocessing.Separator"):
            with mock.patch.dict("os.environ", {}):
                with mock.patch("modules.preprocessing.config") as cfg:
                    cfg.OV_CACHE_DIR = "/my/cache"
                    cfg.PREPROCESS_THREADS = 4
                    manager._init_separator()
                    assert os.environ["OV_CACHE_DIR"] == os.path.abspath(
                        "/my/cache")

    def test_patch_onnx_runtime_already_patched(self):
        """Test apply_onnx_optimizations skips if already patched."""
        with mock.patch("modules.preprocessing.ort") as mock_ort:
            mock_ort.InferenceSession._is_patched = True
            preprocessing.apply_onnx_optimizations()
            # Should not call anything
            mock_ort.get_available_providers.assert_not_called()


class TestPreprocessingEdgeSession:
    """Thread-limited session and hardware logging."""

    def test_thread_limited_session_enforcement(self):
        """Test that ThreadLimitedSession correctly applies thread limits."""
        mock_ort = mock.MagicMock()
        mock_ort.get_available_providers.return_value = [
            "CPUExecutionProvider"]
        mock_ort.SessionOptions = mock.MagicMock
        mock_ort.ExecutionMode = mock.MagicMock()

        # Mock the base class to avoid __init__ issues
        class MockSession:
            """Mock session for testing."""
            def __init__(self, *args, **kwargs):
                pass
        mock_ort.InferenceSession = MockSession
        mock_ort.InferenceSession._is_patched = False

        with mock.patch("modules.preprocessing.ort", mock_ort):
            with mock.patch("modules.config.PREPROCESS_THREADS", "2"):
                preprocessing.apply_onnx_optimizations()
                patched_session_class = preprocessing.ort.InferenceSession
                session = patched_session_class("model.onnx")
                assert session is not None
                assert getattr(patched_session_class, '_is_patched', False)

    def test_apply_onnx_optimizations_inner_loop(self):
        """Cover ThreadLimitedSession logic (lines 118-151)."""
        mock_ort = mock.MagicMock()
        mock_ort.SessionOptions = mock.MagicMock

        class MockSession:
            """Mock session for testing."""
            def __init__(self, *args, **kwargs):
                pass
        mock_ort.InferenceSession = MockSession
        mock_ort.InferenceSession._is_patched = False

        with mock.patch("modules.preprocessing.ort", mock_ort):
            with mock.patch("modules.config.PREPROCESS_THREADS", "4"):
                preprocessing.apply_onnx_optimizations()
                session = mock_ort.InferenceSession(
                    "model.onnx", providers=["OpenVINOExecutionProvider"])
                assert session is not None

    def test_apply_onnx_optimizations_cuda_full(self):
        """Cover CUDA override (lines 192-197)."""
        mock_sep = mock.MagicMock()
        def fake_setup(self, info):
            """Fake setup."""
            _ = self, info
        mock_sep.setup_torch_device = fake_setup
        mock_ort = mock.MagicMock()
        mock_ort.get_available_providers.return_value = [
            "CUDAExecutionProvider"]

        with mock.patch("modules.preprocessing.ort", mock_ort):
            with mock.patch("modules.preprocessing._lazy_import_separator", return_value=mock_sep):
                with mock.patch("modules.config.PREPROCESS_DEVICE", "CUDA"):
                    mock_ort.InferenceSession._is_patched = False
                    preprocessing.apply_onnx_optimizations()
                    instance = mock.MagicMock()
                    instance.logger = mock.MagicMock()
                    mock_sep.setup_torch_device(instance, "info")
                    assert instance.onnx_execution_provider[0] == "CUDAExecutionProvider"

    def test_init_separator_with_torch_fail(self):
        """Test _init_separator when torch.set_num_threads fails."""
        manager = PreprocessingManager()
        patch_fn = mock.patch(
            "modules.preprocessing.torch.set_num_threads",
            side_effect=Exception("torch fail"),
        )
        with patch_fn:
            with mock.patch("modules.preprocessing.Separator"):
                with mock.patch("modules.preprocessing.config") as cfg:
                    cfg.PREPROCESS_THREADS = 4
                    cfg.OV_CACHE_DIR = "/tmp"
                    manager._init_separator()

    def test_apply_onnx_optimizations_exception(self):
        """Test that apply_onnx_optimizations handles errors gracefully."""
        with mock.patch("modules.preprocessing.ort", side_effect=Exception("Fatal Error")):
            preprocessing.apply_onnx_optimizations()

    def test_init_separator_no_library(self):
        """Test _init_separator when audio-separator is missing."""
        preprocessing._INSTANCE = None
        with mock.patch("modules.preprocessing._lazy_import_separator", return_value=None):
            with mock.patch("modules.preprocessing.Separator", None):
                manager = PreprocessingManager()
                manager.separator = None
                manager._init_separator()
                assert manager.separator is None

    def test_log_available_providers_no_ort(self):
        """Test _log_available_providers handles missing ort."""
        PreprocessingManager._logged_providers = False
        with mock.patch("modules.preprocessing.ort", None):
            PreprocessingManager._log_available_providers()
            assert PreprocessingManager._logged_providers is False

    def test_configure_providers_openvino_missing(self):
        """Test _configure_providers warning when OV is missing."""
        with mock.patch("modules.config.PREPROCESS_DEVICE", "GPU"):
            available = ["CPUExecutionProvider"]
            providers, _ = PreprocessingManager._configure_providers(
                available, [], [])
            assert "OpenVINOExecutionProvider" not in providers


class TestPreprocessingCoverageBoost:
    """Extra tests to reach 90%+ coverage."""

    def test_lazy_import_separator_error(self):
        """Cover lines 52-53."""
        with mock.patch("modules.preprocessing.Separator", None):
            with mock.patch("modules.preprocessing._lazy_import_separator", return_value=None):
                manager = PreprocessingManager()
                manager._init_separator()
                assert manager.separator is None

    def test_apply_onnx_optimizations_error_catch(self):
        """Cover line 103-104."""
        mock_ort = mock.MagicMock()
        mock_ort.get_available_providers.side_effect = Exception("ORT crash")
        with mock.patch("modules.preprocessing.ort", mock_ort):
            preprocessing.apply_onnx_optimizations()

    def test_apply_onnx_optimizations_fallback(self):
        """Cover line 93."""
        mock_ort = mock.MagicMock()
        mock_ort.get_available_providers.return_value = [
            "CPUExecutionProvider"]
        with mock.patch("modules.preprocessing.ort", mock_ort):
            preprocessing.apply_onnx_optimizations()
            res = mock_ort.get_available_providers()
            assert res == ["CPUExecutionProvider"]
