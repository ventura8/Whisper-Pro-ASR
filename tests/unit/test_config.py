"""Tests for modules/config.py"""
import os
import importlib
import tempfile
from unittest import mock
import modules.config as config_module


class TestConfig:
    """Test suite for config module."""

    def test_default_model_value(self):
        """Test that DEFAULT_MODEL is set correctly."""
        # Default model for Faster-Whisper
        assert "faster-whisper" in config_module.DEFAULT_WHISPER.lower() or \
               "whisper" in config_module.DEFAULT_WHISPER.lower()

    def test_model_id_from_env(self):
        """Test MODEL_ID reads from environment."""
        with mock.patch.dict(os.environ, {"ASR_MODEL": "/custom/model/path"}):
            importlib.reload(config_module)

            assert config_module.MODEL_ID == "/custom/model/path"

    def test_default_batch_size(self):
        """Test DEFAULT_BATCH_SIZE defaults to 1."""
        with mock.patch.dict(os.environ, {}, clear=True):
            importlib.reload(config_module)

            assert config_module.DEFAULT_BATCH_SIZE == 1

    def test_batch_size_from_env(self):
        """Test ASR_BATCH_SIZE from environment."""
        with mock.patch.dict(os.environ, {"ASR_BATCH_SIZE": "4"}):
            importlib.reload(config_module)

            assert config_module.DEFAULT_BATCH_SIZE == 4

    def test_default_beam_size(self):
        """Test DEFAULT_BEAM_SIZE defaults to 5."""
        with mock.patch.dict(os.environ, {}, clear=True):
            importlib.reload(config_module)

            assert config_module.DEFAULT_BEAM_SIZE == 5

    def test_beam_size_from_env(self):
        """Test ASR_BEAM_SIZE from environment."""
        with mock.patch.dict(os.environ, {"ASR_BEAM_SIZE": "1"}):
            importlib.reload(config_module)

            assert config_module.DEFAULT_BEAM_SIZE == 1

    def test_debug_mode_false_default(self):
        """Test DEBUG_MODE defaults to False."""
        with mock.patch.dict(os.environ, {}, clear=True):
            importlib.reload(config_module)

            assert config_module.DEBUG_MODE is False

    def test_debug_mode_true(self):
        """Test DEBUG_MODE is True when DEBUG=true."""
        with mock.patch.dict(os.environ, {"DEBUG": "true"}):
            importlib.reload(config_module)

            assert config_module.DEBUG_MODE is True

    def test_debug_mode_case_insensitive(self):
        """Test DEBUG_MODE handles case variations."""
        with mock.patch.dict(os.environ, {"DEBUG": "TRUE"}):
            importlib.reload(config_module)

            assert config_module.DEBUG_MODE is True

    def test_initial_steps_ratio(self):
        """Test INITIAL_STEPS_RATIO constant."""
        assert config_module.INITIAL_STEPS_RATIO == 2.8


class TestConfigEnv:
    """Config from environment and defaults."""

    def test_ov_cache_dir_default(self):
        """Test OV_CACHE_DIR defaults to './model_cache'."""
        with mock.patch.dict(os.environ, {}, clear=True):
            importlib.reload(config_module)

            assert config_module.OV_CACHE_DIR == "./model_cache"

    def test_ov_cache_dir_from_env(self):
        """Test OV_CACHE_DIR can be set via env."""
        with mock.patch.dict(os.environ, {"OV_CACHE_DIR": "/custom/cache"}):
            importlib.reload(config_module)

            assert config_module.OV_CACHE_DIR == "/custom/cache"

    def test_app_constants(self):
        """Test app name and version constants."""
        assert "Whisper" in config_module.APP_NAME
        assert config_module.VERSION == "1.0.4"

    def test_device_constant_exists(self):
        """Test DEVICE constant exists."""
        assert hasattr(config_module, 'DEVICE')
        assert config_module.DEVICE in ["CPU", "CUDA", "GPU", "NPU"]

    def test_asr_threads_default(self):
        """Test ASR_THREADS defaults to 4."""
        with mock.patch.dict(os.environ, {"CPU_CORE_LIMIT": "64"}, clear=True):
            importlib.reload(config_module)

            assert config_module.ASR_THREADS == 4

    def test_asr_threads_from_env(self):
        """Test ASR_THREADS can be set via env."""
        with mock.patch.dict(os.environ, {"ASR_THREADS": "8", "CPU_CORE_LIMIT": "64"}):
            importlib.reload(config_module)

            assert config_module.ASR_THREADS == 8

    def test_ffmpeg_threads_default(self):
        """Test FFMPEG_THREADS defaults to 1 when parallel prep is active."""
        with mock.patch.dict(os.environ, {"CPU_CORE_LIMIT": "64"}, clear=True):
            importlib.reload(config_module)

            # Parallel mode (default: 4) forces FFmpeg to 1
            assert config_module.FFMPEG_THREADS == 1

    def test_ffmpeg_threads_manual_zero(self):
        """Test FFMPEG_THREADS stays 0 if explicitly set and PREPROCESS_THREADS=1."""
        with mock.patch.dict(os.environ, {"FFMPEG_THREADS": "0", "ASR_PREPROCESS_THREADS": "1"}):
            importlib.reload(config_module)

            assert config_module.FFMPEG_THREADS == 0

    def test_thread_capping_to_cores(self):
        """Test that threads are capped to logical core count with priority."""
        env = {"ASR_THREADS": "64", "ASR_PREPROCESS_THREADS": "64", "CPU_CORE_LIMIT": "8"}
        with mock.patch.dict(os.environ, env):
            # Force CPU mode for test priority check
            with mock.patch("modules.config.DEVICE", "CPU"):
                importlib.reload(config_module)
                assert config_module.ASR_THREADS == 8
                # Prep is now allowed to use the full pool sequentially
                assert config_module.PREPROCESS_THREADS == 8

    def test_enable_vocal_separation_default(self):
        """Test ENABLE_VOCAL_SEPARATION defaults to False."""
        with mock.patch.dict(os.environ, {}, clear=True):
            importlib.reload(config_module)

            assert config_module.ENABLE_VOCAL_SEPARATION is False

    def test_enable_vocal_separation_false(self):
        """Test ENABLE_VOCAL_SEPARATION can be set to False."""
        with mock.patch.dict(os.environ, {"ENABLE_VOCAL_SEPARATION": "false"}):
            importlib.reload(config_module)

            assert config_module.ENABLE_VOCAL_SEPARATION is False

    def test_hallucination_phrases_exist(self):
        """Test HALLUCINATION_PHRASES list exists and is populated."""
        assert hasattr(config_module, 'HALLUCINATION_PHRASES')
        assert isinstance(config_module.HALLUCINATION_PHRASES, list)
        assert len(config_module.HALLUCINATION_PHRASES) > 0

    def test_compute_type_exists(self):
        """Test COMPUTE_TYPE exists."""
        assert hasattr(config_module, 'COMPUTE_TYPE')


class TestConfigHardware:
    """Config hardware detection and device overrides."""

    def test_hardware_detection_logic_cuda(self):
        """Test CUDA detection path."""
        with mock.patch.dict(os.environ, {"ASR_DEVICE": "AUTO"}):
            with mock.patch("ctranslate2.get_cuda_device_count", return_value=1):
                importlib.reload(config_module)
                assert config_module.DEVICE == "CUDA"

    def test_hardware_detection_logic_npu(self):
        """Test NPU detection path."""
        with mock.patch.dict(os.environ, {"ASR_DEVICE": "AUTO"}):
            with mock.patch("ctranslate2.get_cuda_device_count", return_value=0):
                mock_core = mock.MagicMock()
                mock_core.available_devices = ["NPU", "CPU"]
                mock_core.get_property.return_value = "Intel(R) AI Boost"
                with mock.patch("openvino.Core", return_value=mock_core):
                    importlib.reload(config_module)
                    # NPU detected means Preprocess=NPU but ASR=CPU (for Quality)
                    assert config_module.DEVICE == "CPU"
                    assert config_module.PREPROCESS_DEVICE == "NPU"

    def test_hardware_detection_logic_gpu_amd(self):
        """Test AMD/Intel GPU detection path (OpenVINO GPU)."""
        with mock.patch.dict(os.environ, {"ASR_DEVICE": "AUTO"}):
            with mock.patch("ctranslate2.get_cuda_device_count", return_value=0):
                mock_core = mock.MagicMock()
                mock_core.available_devices = ["GPU", "CPU"]
                mock_core.get_property.return_value = "Intel(R) Arc(TM) Graphics"
                with mock.patch("openvino.Core", return_value=mock_core):
                    importlib.reload(config_module)
                    # GPU detected means Preprocess=GPU but ASR=CPU (for Quality)
                    assert config_module.DEVICE == "CPU"
                    assert config_module.PREPROCESS_DEVICE == "GPU"

    def test_hardware_resource_pooling(self):
        """Test that HARDWARE_UNITS pool is correctly populated."""
        with mock.patch.dict(os.environ, {"ASR_DEVICE": "AUTO"}):
            with mock.patch("ctranslate2.get_cuda_device_count", return_value=0):
                mock_core = mock.MagicMock()
                mock_core.available_devices = ["GPU.0", "NPU.0", "CPU"]
                mock_core.get_property.return_value = "Intel Accelerator"
                with mock.patch("openvino.Core", return_value=mock_core):
                    importlib.reload(config_module)
                    # Should have 2 units (GPU and NPU)
                    assert len(config_module.HARDWARE_UNITS) == 2
                    assert any(u['type'] == 'GPU' for u in config_module.HARDWARE_UNITS)
                    assert any(u['type'] == 'NPU' for u in config_module.HARDWARE_UNITS)

    def test_hardware_unit_limits(self):
        """Test that MAX_*_UNITS correctly limits the pool."""
        with mock.patch.dict(os.environ, {
            "MAX_GPU_UNITS": "1",
            "MAX_NPU_UNITS": "0"
        }):
            with mock.patch("ctranslate2.get_cuda_device_count", return_value=0):
                mock_core = mock.MagicMock()
                mock_core.available_devices = ["GPU.0", "GPU.1", "NPU.0", "CPU"]
                mock_core.get_property.return_value = "Intel Accelerator"
                with mock.patch("openvino.Core", return_value=mock_core):
                    importlib.reload(config_module)
                    # Should only have 1 GPU and 0 NPUs
                    assert len([u for u in config_module.HARDWARE_UNITS if u['type'] == 'GPU']) == 1
                    assert len([u for u in config_module.HARDWARE_UNITS if u['type'] == 'NPU']) == 0

    def test_max_cpu_units_logic(self):
        """Test MAX_CPU_UNITS parsing and impact on CPU_PARALLEL_LIMIT."""
        with mock.patch.dict(os.environ, {"MAX_CPU_UNITS": "2"}):
            importlib.reload(config_module)
            assert config_module.MAX_CPU == 2
            assert config_module.CPU_PARALLEL_LIMIT == 2

    def test_cpu_parallel_limit_auto_scaling(self):
        """Test auto-scaling of CPU_PARALLEL_LIMIT when MAX_CPU is AUTO."""
        with mock.patch.dict(os.environ, {
            "MAX_CPU_UNITS": "AUTO",
            "ASR_THREADS": "2",
            "ASR_PREPROCESS_THREADS": "2",
            "CPU_CORE_LIMIT": "8"
        }):
            importlib.reload(config_module)
            # cores // max(threads) = 8 // 2 = 4
            assert config_module.CPU_PARALLEL_LIMIT == 4

    def test_hardware_property_exception(self):
        """Test that hardware names fall back on property exception."""
        with mock.patch.dict(os.environ, {"ASR_DEVICE": "AUTO"}):
            with mock.patch("ctranslate2.get_cuda_device_count", return_value=0):
                mock_core = mock.MagicMock()
                mock_core.available_devices = ["GPU"]
                mock_core.get_property.side_effect = Exception("Property fail")
                with mock.patch("openvino.Core", return_value=mock_core):
                    importlib.reload(config_module)
                    expected = ["CPU", "GPU", "NPU", "NVIDIA GPU"]
                    assert config_module.ASR_DEVICE_NAME in expected
                    assert config_module.PREPROCESS_DEVICE_NAME in expected

    def test_hardware_detection_logic_exception(self):
        """Test hardware detection handles exceptions gracefully."""
        with mock.patch.dict(os.environ, {"ASR_DEVICE": "AUTO"}):
            fail_patch = mock.patch(
                "ctranslate2.get_cuda_device_count",
                side_effect=Exception("Hard fail"),
            )
            with fail_patch:
                importlib.reload(config_module)
                assert config_module.DEVICE == "CPU"

    def test_hardware_detection_logic_manual_override(self):
        """Test manual ASR_DEVICE override path."""
        with mock.patch.dict(os.environ, {"ASR_DEVICE": "CPU", "ASR_PREPROCESS_DEVICE": "GPU"}):
            importlib.reload(config_module)
            assert config_module.DEVICE == "CPU"
            assert config_module.PREPROCESS_DEVICE == "GPU"

    def test_system_model_detection(self):
        """Test detection of baked-in system models."""
        with mock.patch("os.path.exists", side_effect=lambda p: "/app/system_models" in p):
            with mock.patch("os.listdir", return_value=["file1"]):
                importlib.reload(config_module)
                assert "/app/system_models" in config_module.MODEL_ID
                assert "/app/system_models" in config_module.UVR_MODEL_DIR

    def test_custom_compute_type(self):
        """Test custom ASR_COMPUTE_TYPE override."""
        with mock.patch.dict(os.environ, {"ASR_COMPUTE_TYPE": "FLOAT32"}):
            importlib.reload(config_module)
            assert config_module.COMPUTE_TYPE == "float32"

    def test_hallucination_phrases_content(self):
        """Test some common phrases exist."""
        assert "thank you for watching" in config_module.HALLUCINATION_PHRASES
        assert "vă mulțumim pentru vizionare" in config_module.HALLUCINATION_PHRASES

    def test_intel_engine_redirection(self):
        """Test that MODEL_ID redirects for INTEL-WHISPER."""
        env = {
            "ASR_ENGINE": "INTEL-WHISPER",
            "ASR_MODEL": "Systran/faster-whisper-large-v3"
        }
        with mock.patch.dict(os.environ, env):
            with mock.patch("os.path.exists", side_effect=lambda p: "whisper-openvino" in p):
                importlib.reload(config_module)
                assert "whisper-openvino" in config_module.MODEL_ID

    def test_intel_engine_hf_fallback(self):
        """Test that MODEL_ID falls back to HF for INTEL-WHISPER if local missing."""
        env = {
            "ASR_ENGINE": "INTEL-WHISPER",
            "ASR_MODEL": "Systran/faster-whisper-large-v3"
        }
        with mock.patch.dict(os.environ, env):
            with mock.patch("os.path.exists", return_value=False):
                importlib.reload(config_module)
                assert "OpenVINO" in config_module.MODEL_ID


class TestConfigSSD:
    """Tests for SSD optimization settings."""

    def test_temp_dir_default(self):
        """Test TEMP_DIR defaults to system temp."""
        with mock.patch.dict(os.environ, {}, clear=True):
            importlib.reload(config_module)
            assert config_module.TEMP_DIR == tempfile.gettempdir()

    def test_temp_dir_from_env(self):
        """Test TEMP_DIR from WHISPER_TEMP_DIR env."""
        with mock.patch.dict(os.environ, {"WHISPER_TEMP_DIR": "/tmp/whisper"}):
            with mock.patch("os.makedirs"):
                importlib.reload(config_module)
                assert config_module.TEMP_DIR == "/tmp/whisper"

    def test_temp_min_free_default(self):
        """Test TEMP_DIR_MIN_FREE_BYTES defaults to 512MB."""
        with mock.patch.dict(os.environ, {}, clear=True):
            importlib.reload(config_module)
            assert config_module.TEMP_DIR_MIN_FREE_BYTES == 512 * 1024 * 1024

    def test_temp_min_free_from_env(self):
        """Test TEMP_DIR_MIN_FREE_BYTES from env."""
        with mock.patch.dict(os.environ, {"WHISPER_TEMP_MIN_FREE_MB": "100"}):
            importlib.reload(config_module)
            assert config_module.TEMP_DIR_MIN_FREE_BYTES == 100 * 1024 * 1024

    def test_get_temp_dir_disk_usage_fail(self):
        """Test get_temp_dir fallback when disk_usage fails."""
        with mock.patch("shutil.disk_usage", side_effect=Exception("Disk fail")):
            # Should return PERSISTENT_TEMP_DIR
            assert config_module.get_temp_dir() == config_module.PERSISTENT_TEMP_DIR

    def test_validate_thread_concurrency_warning(self):
        """Test the over-provisioning warning in _validate_thread_concurrency."""
        env = {
            "ASR_THREADS": "8",
            "ASR_PREPROCESS_THREADS": "8",
            "FFMPEG_THREADS": "8",
            "CPU_CORE_LIMIT": "4"
        }
        with mock.patch.dict(os.environ, env):
            with mock.patch("logging.getLogger") as mock_get_logger:
                mock_logger = mock.MagicMock()
                mock_get_logger.return_value = mock_logger
                importlib.reload(config_module)
                # Should log a warning about over-provisioning
                warning_calls = mock_logger.warning.call_args_list
                assert any("OVER-PROVISIONING" in str(call) for call in warning_calls)

    def test_get_parallel_unit_limit_cuda_fail(self):
        """Test hardware unit limit fallback when ctranslate2 fails."""
        with mock.patch("ctranslate2.get_cuda_device_count", side_effect=Exception("No CUDA")):
            # Should return 0 or default
            res = config_module.get_parallel_limit("CUDA")
            assert res >= 0
