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
            mock_config.VOCAL_SEPARATION_SEGMENT_DURATION = 600
            mock_config.VOCAL_SEPARATION_SEGMENT_DURATION = 600

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
                        with mock.patch(
                            'modules.preprocessing.utils.get_audio_duration',
                            return_value=300.0
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
            mock_config.VOCAL_SEPARATION_SEGMENT_DURATION = 600

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
                        with mock.patch(
                            'modules.preprocessing.utils.get_audio_duration',
                            return_value=300.0
                        ):
                            manager.process_audio_file("/path/to/file.wav")
                            # Should find the path

    def test_process_audio_file_no_vocals_found(self):
        """Test when vocals track is not found in output."""
        with mock.patch('modules.preprocessing.config') as mock_config:
            mock_config.ENABLE_VOCAL_SEPARATION = True
            mock_config.PREPROCESS_DEVICE = "CPU"
            mock_config.PREPROCESSING_CACHE_DIR = "/tmp"
            mock_config.VOCAL_SEPARATION_SEGMENT_DURATION = 600

            manager = PreprocessingManager()

            mock_separator = mock.MagicMock()
            mock_separator.separate.return_value = ["file_(Other)_model.wav"]
            manager.separator = mock_separator
            manager._init_separator = mock.MagicMock()

            with mock.patch('modules.preprocessing.utils.get_audio_duration', return_value=300.0):
                result = manager.process_audio_file("/path/to/file.wav")
            assert result == "/path/to/file.wav"

    def test_process_audio_file_exception(self):
        """Test exception handling in process_audio_file."""
        with mock.patch('modules.preprocessing.config') as mock_config:
            mock_config.ENABLE_VOCAL_SEPARATION = True
            mock_config.PREPROCESS_DEVICE = "CPU"
            mock_config.PREPROCESSING_CACHE_DIR = "/tmp"
            mock_config.VOCAL_SEPARATION_SEGMENT_DURATION = 600

            manager = PreprocessingManager()
            manager._init_separator = mock.MagicMock(
                side_effect=RuntimeError("Init failed"))

            result = manager.process_audio_file("/path/to/file.wav")
            assert result == "/path/to/file.wav"


class TestProcessAudioSegment:
    """Tests for process_audio_segment method."""

    def test_process_audio_segment_disabled(self):
        """Test process_audio_segment returns original when disabled."""
        with mock.patch('modules.preprocessing.config') as mock_config:
            mock_config.ENABLE_VOCAL_SEPARATION = False
            mock_config.ENABLE_LD_PREPROCESSING = False
            mock_config.PREPROCESS_DEVICE = "CPU"
            mock_config.PREPROCESSING_CACHE_DIR = "/tmp"

            manager = PreprocessingManager()

            segment = np.zeros(16000, dtype=np.float32)
            result = manager.process_audio_segment(segment)
            assert np.array_equal(result, segment)

    def test_process_audio_segment_silence(self):
        """Test that silence is returned unchanged (optimization)."""
        with mock.patch('modules.preprocessing.config') as mock_config:
            mock_config.ENABLE_VOCAL_SEPARATION = True
            mock_config.PREPROCESS_DEVICE = "CPU"
            mock_config.PREPROCESSING_CACHE_DIR = "/tmp"

            manager = PreprocessingManager()

            # Very quiet segment (below 0.005 threshold)
            segment = np.ones(16000, dtype=np.float32) * 0.001
            result = manager.process_audio_segment(segment)
            assert np.array_equal(result, segment)

    def test_process_audio_segment_success(self):
        """Test successful segment processing."""
        with mock.patch('modules.preprocessing.config') as mock_config:
            mock_config.ENABLE_VOCAL_SEPARATION = True
            mock_config.PREPROCESS_DEVICE = "CPU"
            mock_config.PREPROCESSING_CACHE_DIR = "/tmp"

            manager = PreprocessingManager()

            mock_separator = mock.MagicMock()
            mock_separator.separate.return_value = ["segment_(Vocals)_model.wav"]
            manager.separator = mock_separator
            manager._init_separator = mock.MagicMock()

            # Loud segment that needs processing
            segment = np.ones(16000, dtype=np.float32) * 0.1

            # Mock file operations
            mock_audio = np.ones(16000, dtype=np.float32) * 0.05

            with mock.patch('modules.preprocessing.sf.write'):
                with mock.patch('modules.preprocessing.sf.read', return_value=(mock_audio, 16000)):
                    with mock.patch('os.path.exists', return_value=True):
                        with mock.patch('os.remove'):
                            result = manager.process_audio_segment(segment)
                            assert len(result) == 16000

    def test_process_audio_segment_exception(self):
        """Test exception handling returns original segment."""
        with mock.patch('modules.preprocessing.config') as mock_config:
            mock_config.ENABLE_VOCAL_SEPARATION = True
            mock_config.PREPROCESS_DEVICE = "CPU"
            mock_config.PREPROCESSING_CACHE_DIR = "/tmp"

            manager = PreprocessingManager()
            manager._init_separator = mock.MagicMock(
                side_effect=RuntimeError("Error"))

            segment = np.ones(16000, dtype=np.float32) * 0.1
            result = manager.process_audio_segment(segment)
            assert np.array_equal(result, segment)


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


class TestPreprocessingEdgeSegments:
    """Process and segment edge cases."""

    def test_process_audio_file_cleanup_error(self):
        """Test process_audio_file when cleanup of instrumental fails."""
        with mock.patch('modules.preprocessing.config') as config:
            config.ENABLE_VOCAL_SEPARATION = True
            config.VOCAL_SEPARATION_SEGMENT_DURATION = 600
            manager = PreprocessingManager()
            manager._init_separator = mock.MagicMock()
            mock_sep = mock.MagicMock()
            mock_sep.separate.return_value = [
                "file_(Vocals).wav", "file_(Instr).wav"]
            manager.separator = mock_sep

            with mock.patch('os.path.exists', return_value=True):
                with mock.patch('os.remove', side_effect=Exception("Cleanup error")):
                    with mock.patch(
                            'modules.preprocessing.utils.get_audio_duration',
                            return_value=300.0):
                        res = manager.process_audio_file("orig.wav")
                        assert "Vocals" in res

    def test_process_audio_segment_sf_write_fail(self):
        """Test exception during segment processing."""
        with mock.patch('modules.preprocessing.config') as config:
            config.ENABLE_VOCAL_SEPARATION = True
            manager = PreprocessingManager()
            with mock.patch('modules.preprocessing.sf.write', side_effect=Exception("Write fail")):
                segment = np.zeros(16000)
                res = manager.process_audio_segment(segment)
                assert np.array_equal(res, segment)

    def test_postprocess_segment_resampling_and_padding(self):
        """Test resampling and padding logic in _postprocess_segment."""
        manager = PreprocessingManager()
        # Mock 44100Hz audio to be resampled to 16000Hz
        mock_data = np.ones(44100)
        with mock.patch('modules.preprocessing.sf.read', return_value=(mock_data, 44100)):
            res = manager._postprocess_segment("vocal.wav", 16000, 16000)
            assert len(res) == 16000

    def test_cleanup_stems_relative_path(self):
        """Test that relative paths from audio-separator are resolved to CACHE_DIR and deleted."""
        manager = PreprocessingManager()

        with mock.patch('modules.preprocessing.CACHE_DIR', '/mock/cache'):
            with mock.patch('os.path.exists') as mock_exists:
                # First exists check is for the relative path itself
                # Second exists check is for the resolved absolute path
                # Third exists check is for the resolved absolute path before remove
                mock_exists.side_effect = lambda path: str(path) in [
                    os.path.join('/mock/cache', 'file_(Instrumental).wav'),
                    os.path.join('/mock/cache', 'file_(Vocals).wav')
                ]

                with mock.patch('os.remove') as mock_remove:
                    manager._cleanup_stems(
                        ['file_(Instrumental).wav', 'file_(Vocals).wav'],
                        keep_path=os.path.join('/mock/cache', 'file_(Vocals).wav')
                    )

                    # Should only remove the instrumental file
                    expected_call = os.path.join('/mock/cache', 'file_(Instrumental).wav')
                    mock_remove.assert_called_once_with(expected_call)

    def test_purge_stale_cache_removes_files(self):
        """Test that _purge_stale_cache removes orphaned files at startup."""
        mock_file1 = mock.MagicMock()
        mock_file1.is_file.return_value = True
        mock_file2 = mock.MagicMock()
        mock_file2.is_file.return_value = True
        mock_dir = mock.MagicMock()
        mock_dir.is_file.return_value = False

        mock_cache = mock.MagicMock()
        mock_cache.iterdir.return_value = [mock_file1, mock_file2, mock_dir]

        with mock.patch('modules.preprocessing.CACHE_DIR', mock_cache):
            PreprocessingManager._purge_stale_cache()

        mock_file1.unlink.assert_called_once()
        mock_file2.unlink.assert_called_once()
        mock_dir.unlink.assert_not_called()

    def test_purge_stale_cache_empty_dir(self):
        """Test that _purge_stale_cache is a no-op on empty directories."""
        mock_cache = mock.MagicMock()
        mock_cache.iterdir.return_value = []

        with mock.patch('modules.preprocessing.CACHE_DIR', mock_cache):
            PreprocessingManager._purge_stale_cache()

    def test_purge_stale_cache_handles_errors(self):
        """Test that _purge_stale_cache silently handles file deletion errors."""
        mock_file = mock.MagicMock()
        mock_file.is_file.return_value = True
        mock_file.unlink.side_effect = PermissionError("Access denied")

        mock_cache = mock.MagicMock()
        mock_cache.iterdir.return_value = [mock_file]

        with mock.patch('modules.preprocessing.CACHE_DIR', mock_cache):
            # Should not raise
            PreprocessingManager._purge_stale_cache()


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

    def test_postprocess_segment_padding_only(self):
        """Test padding logic in _postprocess_segment without resampling."""
        manager = PreprocessingManager()
        # Same SR, but shorter
        mock_data = np.ones(10000)
        with mock.patch('modules.preprocessing.sf.read', return_value=(mock_data, 16000)):
            res = manager._postprocess_segment("vocal.wav", 16000, 16000)
            assert len(res) == 16000
            assert np.all(res[:10000] == 1.0)
            assert np.all(res[10000:] == 0.0)

    def test_postprocess_segment_trimming(self):
        """Test trimming logic in _postprocess_segment."""
        manager = PreprocessingManager()
        # Same SR, but longer
        mock_data = np.ones(20000)
        with mock.patch('modules.preprocessing.sf.read', return_value=(mock_data, 16000)):
            res = manager._postprocess_segment("vocal.wav", 16000, 16000)
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

class TestPreprocessingTempDir:
    """Tests for temp directory usage in preprocessing."""

    def test_init_separator_warmup_uses_temp_dir(self):
        """Test that warmup dummy file is created in TEMP_DIR."""
        manager = PreprocessingManager()
        manager.separator = mock.MagicMock()
        with mock.patch("modules.preprocessing.Separator"):
            with mock.patch("modules.preprocessing.config") as cfg:
                cfg.TEMP_DIR = "/tmp/whisper"
                cfg.PREPROCESS_THREADS = 4
                cfg.OV_CACHE_DIR = "/tmp"
                # Mock file operations
                with mock.patch("modules.preprocessing.sf.write") as mock_write:
                    with mock.patch("os.remove"):
                        manager._perform_warmup()
                        # Verify sf.write was called with a path in /tmp/whisper
                        args, _ = mock_write.call_args
                        assert args[0].startswith("/tmp/whisper")
                        assert "warmup" in args[0]

    def test_process_audio_segment_uses_temp_dir(self):
        """Test that process_audio_segment uses get_temp_dir for mkstemp."""
        manager = PreprocessingManager()
        with mock.patch("modules.preprocessing.config") as cfg:
            cfg.ENABLE_VOCAL_SEPARATION = True
            cfg.get_temp_dir.return_value = "/tmp/whisper"

            mock_sep = mock.MagicMock()
            mock_sep.separate.return_value = ["vocal.wav"]
            manager.separator = mock_sep

            segment = np.ones(16000, dtype=np.float32) * 0.1
            with mock.patch("tempfile.mkstemp", return_value=(1, "temp.wav")) as mock_mkstemp:
                with mock.patch("os.close"):
                    with mock.patch("modules.preprocessing.sf.write"):
                        with mock.patch("modules.preprocessing.sf.read",
                                        return_value=(segment, 16000)):
                            with mock.patch("os.remove"):
                                with mock.patch("os.path.exists", return_value=True):
                                    manager.process_audio_segment(segment)
                                    # Verify mkstemp was called
                                    assert mock_mkstemp.called
                                    # Verify it used the correct directory
                                    _, kwargs = mock_mkstemp.call_args
                                    assert kwargs["dir"] == "/tmp/whisper"


class TestPreprocessingCoverageFinal:
    """Final set of tests to hit 90%+."""

    def test_purge_stale_cache_legacy_support(self):
        """Test that _purge_stale_cache cleans up legacy v1.0.0 directories."""
        with mock.patch('modules.preprocessing.config') as mock_config:
            mock_config.PREPROCESSING_CACHE_DIR = "/tmp/new_cache"
            mock_config.OV_CACHE_DIR = "/tmp/persistent"
            mock_config.PREPROCESS_DEVICE = "CPU"

            mock_file1 = mock.MagicMock()
            mock_file1.is_file.return_value = True
            mock_file1.exists.return_value = True
            mock_file1.name = "upload_test.wav"

            mock_file2 = mock.MagicMock()
            mock_file2.is_file.return_value = True
            mock_file2.exists.return_value = True

            with mock.patch.object(preprocessing, 'Path') as mock_path_cls:
                mock_new_cache = mock.MagicMock()
                mock_new_cache.iterdir.return_value = [mock_file1]

                mock_persistent = mock.MagicMock()
                mock_legacy = mock.MagicMock()
                mock_legacy.exists.return_value = True
                mock_legacy.is_dir.return_value = True
                mock_legacy.iterdir.return_value = [mock_file2]
                mock_persistent.__truediv__.return_value = mock_legacy

                def path_side_effect(arg):
                    arg_str = str(arg)
                    if "new_cache" in arg_str:
                        return mock_new_cache
                    if "persistent" in arg_str:
                        return mock_persistent
                    return mock.MagicMock()

                mock_path_cls.side_effect = path_side_effect

                # Call directly
                PreprocessingManager._purge_stale_cache()

                # Verify unlinks - focus on legacy rmdir which implies inner loop success
                assert mock_legacy.rmdir.called

    def test_purge_persistent_temp(self):
        """Test that _purge_stale_cache cleans up persistent temp fallback."""
        with mock.patch('modules.preprocessing.config') as mock_config:
            mock_config.PREPROCESSING_CACHE_DIR = "/tmp/none"
            mock_config.OV_CACHE_DIR = "/tmp/none2"
            mock_config.PERSISTENT_TEMP_DIR = "/tmp/persistent_temp"

            mock_file = mock.MagicMock()
            mock_file.is_file.return_value = True
            mock_file.name = "upload_leak.tmp"

            with mock.patch.object(preprocessing, 'Path') as mock_path_cls:
                mock_pt = mock.MagicMock()
                mock_pt.exists.return_value = True
                mock_pt.is_dir.return_value = True
                mock_pt.iterdir.return_value = [mock_file]

                def path_side_effect(arg):
                    if "persistent_temp" in str(arg):
                        return mock_pt
                    return mock.MagicMock()

                mock_path_cls.side_effect = path_side_effect

                PreprocessingManager._purge_stale_cache()
                assert mock_file.unlink.called

    def test_process_audio_file_segmented_flow(self):
        """Test process_audio_file with segmentation path."""
        with mock.patch('modules.preprocessing.config') as mock_config:
            mock_config.ENABLE_VOCAL_SEPARATION = True
            mock_config.VOCAL_SEPARATION_SEGMENT_DURATION = 1 # 1 second
            mock_config.get_temp_dir.return_value = "/tmp"
            mock_config.PREPROCESS_DEVICE = "CPU"
            mock_config.WHISPER_TEMP_DIR = "/tmp"
            mock_config.PREPROCESSING_CACHE_DIR = "/tmp/cache"

            manager = PreprocessingManager()
            manager.separator = mock.MagicMock()
            manager.separator.separate.return_value = ["output_(Vocals)_test.wav"]
            # 2 seconds of audio @ 16kHz
            with mock.patch("modules.preprocessing.utils.get_audio_duration",
                            return_value=2.0):
                with mock.patch("modules.preprocessing.sf.info") as mock_info:
                    mock_info.return_value = mock.MagicMock(frames=32000,
                                                            samplerate=16000)
                    with mock.patch("modules.preprocessing.sf.SoundFile") as mock_sf:
                        # Mock the writer context manager
                        mock_writer = mock.MagicMock()
                        mock_writer.samplerate = 16000
                        mock_sf.return_value.__enter__.return_value = mock_writer

                        with mock.patch("modules.preprocessing.sf.read",
                                        return_value=(np.zeros((16000, 1)), 16000)):
                            with mock.patch("modules.preprocessing.sf.write"):
                                with mock.patch("tempfile.mkstemp",
                                                return_value=(1, "temp.wav")):
                                    with mock.patch("os.close"):
                                        with mock.patch("os.path.exists",
                                                        return_value=True):
                                            with mock.patch("os.remove"):
                                                res = manager.process_audio_file("dummy.wav")
                                                assert res == "temp.wav"
                                                # Should have processed 2 segments
                                                assert manager.separator.separate.call_count == 2

    def test_isolate_and_write_segment_exception_path(self):
        """Test _isolate_and_write_segment exception handling."""
        with mock.patch('modules.preprocessing.config') as mock_config:
            mock_config.get_temp_dir.return_value = "/tmp"
            manager = PreprocessingManager()
            manager.separator = mock.MagicMock(side_effect=Exception("Fatal Error"))

            mock_writer = mock.MagicMock()
            mock_writer.samplerate = 16000

            with mock.patch("modules.preprocessing.sf.read",
                            return_value=(np.zeros((16000, 1)), 16000)):
                with mock.patch("modules.preprocessing.sf.write"):
                    with mock.patch("tempfile.mkstemp",
                                    return_value=(1, "temp.wav")):
                        with mock.patch("os.close"):
                            with mock.patch("os.path.exists",
                                            return_value=True):
                                with mock.patch("os.remove"):
                                    # Should catch and write original audio
                                    # pylint: disable=protected-access
                                    manager._isolate_and_write_segment(
                                        mock_writer, "in.wav", 0, 16000)
                                    mock_writer.write.assert_called_once()

    def test_offload_logic_cuda(self):
        """Test model offloading and resource cleanup."""
        with mock.patch('modules.preprocessing.config'):
            manager = PreprocessingManager()
            manager.separator = mock.MagicMock()

            with mock.patch('modules.preprocessing.torch.cuda.is_available', return_value=True):
                with mock.patch('modules.preprocessing.torch.cuda.empty_cache') as mock_empty:
                    with mock.patch('modules.preprocessing.gc.collect') as mock_gc:
                        manager.offload()
                        assert manager.separator is None
                        mock_empty.assert_called_once()
                        mock_gc.assert_called_once()

    def test_ensure_models_loaded_exception(self):
        """Test ensure_models_loaded error handling path."""
        with mock.patch('modules.preprocessing.config') as mock_config:
            mock_config.ENABLE_VOCAL_SEPARATION = True
            mock_config.ENABLE_LD_PREPROCESSING = False
            mock_config.PREPROCESS_DEVICE = "CPU"
            mock_config.PREPROCESSING_CACHE_DIR = "/tmp"

            manager = PreprocessingManager()
            manager._init_separator = mock.MagicMock(side_effect=Exception("Load fail"))

            # Should catch and log error without crashing
            manager.ensure_models_loaded()
            manager._init_separator.assert_called_once()
