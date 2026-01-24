"""Tests for modules/config.py"""
# pylint: disable=import-outside-toplevel
import os
from unittest import mock


class TestConfig:
    """Test suite for config module."""

    def test_default_model_value(self):
        """Test that DEFAULT_MODEL is set correctly."""
        import modules.config as config_module
        # Default model for Faster-Whisper
        assert "faster-whisper" in config_module.DEFAULT_WHISPER.lower() or \
               "whisper" in config_module.DEFAULT_WHISPER.lower()

    def test_model_id_from_env(self):
        """Test MODEL_ID reads from environment."""
        with mock.patch.dict(os.environ, {"ASR_MODEL": "/custom/model/path"}):
            import importlib
            import modules.config as config_module
            importlib.reload(config_module)

            assert config_module.MODEL_ID == "/custom/model/path"

    def test_default_batch_size(self):
        """Test DEFAULT_BATCH_SIZE defaults to 1."""
        with mock.patch.dict(os.environ, {}, clear=True):
            import importlib
            import modules.config as config_module
            importlib.reload(config_module)

            assert config_module.DEFAULT_BATCH_SIZE == 1

    def test_batch_size_from_env(self):
        """Test ASR_BATCH_SIZE from environment."""
        with mock.patch.dict(os.environ, {"ASR_BATCH_SIZE": "4"}):
            import importlib
            import modules.config as config_module
            importlib.reload(config_module)

            assert config_module.DEFAULT_BATCH_SIZE == 4

    def test_default_beam_size(self):
        """Test DEFAULT_BEAM_SIZE defaults to 5."""
        with mock.patch.dict(os.environ, {}, clear=True):
            import importlib
            import modules.config as config_module
            importlib.reload(config_module)

            assert config_module.DEFAULT_BEAM_SIZE == 5

    def test_beam_size_from_env(self):
        """Test ASR_BEAM_SIZE from environment."""
        with mock.patch.dict(os.environ, {"ASR_BEAM_SIZE": "1"}):
            import importlib
            import modules.config as config_module
            importlib.reload(config_module)

            assert config_module.DEFAULT_BEAM_SIZE == 1

    def test_debug_mode_false_default(self):
        """Test DEBUG_MODE defaults to False."""
        with mock.patch.dict(os.environ, {}, clear=True):
            import importlib
            import modules.config as config_module
            importlib.reload(config_module)

            assert config_module.DEBUG_MODE is False

    def test_debug_mode_true(self):
        """Test DEBUG_MODE is True when DEBUG=true."""
        with mock.patch.dict(os.environ, {"DEBUG": "true"}):
            import importlib
            import modules.config as config_module
            importlib.reload(config_module)

            assert config_module.DEBUG_MODE is True

    def test_debug_mode_case_insensitive(self):
        """Test DEBUG_MODE handles case variations."""
        with mock.patch.dict(os.environ, {"DEBUG": "TRUE"}):
            import importlib
            import modules.config as config_module
            importlib.reload(config_module)

            assert config_module.DEBUG_MODE is True

    def test_initial_steps_ratio(self):
        """Test INITIAL_STEPS_RATIO constant."""
        import modules.config as config_module
        assert config_module.INITIAL_STEPS_RATIO == 2.8


class TestConfigEnv:
    """Config from environment and defaults."""

    def test_ov_cache_dir_default(self):
        """Test OV_CACHE_DIR defaults to './model_cache'."""
        with mock.patch.dict(os.environ, {}, clear=True):
            import importlib
            import modules.config as config_module
            importlib.reload(config_module)

            assert config_module.OV_CACHE_DIR == "./model_cache"

    def test_ov_cache_dir_from_env(self):
        """Test OV_CACHE_DIR can be set via env."""
        with mock.patch.dict(os.environ, {"OV_CACHE_DIR": "/custom/cache"}):
            import importlib
            import modules.config as config_module
            importlib.reload(config_module)

            assert config_module.OV_CACHE_DIR == "/custom/cache"

    def test_app_constants(self):
        """Test app name and version constants."""
        import modules.config as config_module
        assert "Whisper" in config_module.APP_NAME
        assert config_module.VERSION == "1.0.0"

    def test_device_constant_exists(self):
        """Test DEVICE constant exists."""
        import modules.config as config_module
        assert hasattr(config_module, 'DEVICE')
        assert config_module.DEVICE in ["CPU", "CUDA", "GPU", "NPU"]

    def test_asr_threads_default(self):
        """Test ASR_THREADS defaults to 4."""
        with mock.patch.dict(os.environ, {}, clear=True):
            import importlib
            import modules.config as config_module
            importlib.reload(config_module)

            assert config_module.ASR_THREADS == 4

    def test_asr_threads_from_env(self):
        """Test ASR_THREADS can be set via env."""
        with mock.patch.dict(os.environ, {"ASR_THREADS": "8"}):
            import importlib
            import modules.config as config_module
            importlib.reload(config_module)

            assert config_module.ASR_THREADS == 8

    def test_ffmpeg_threads_default(self):
        """Test FFMPEG_THREADS defaults to 0."""
        with mock.patch.dict(os.environ, {}, clear=True):
            import importlib
            import modules.config as config_module
            importlib.reload(config_module)

            assert config_module.FFMPEG_THREADS == 0

    def test_enable_vocal_separation_default(self):
        """Test ENABLE_VOCAL_SEPARATION defaults to False."""
        with mock.patch.dict(os.environ, {}, clear=True):
            import importlib
            import modules.config as config_module
            importlib.reload(config_module)

            assert config_module.ENABLE_VOCAL_SEPARATION is False

    def test_enable_vocal_separation_false(self):
        """Test ENABLE_VOCAL_SEPARATION can be set to False."""
        with mock.patch.dict(os.environ, {"ENABLE_VOCAL_SEPARATION": "false"}):
            import importlib
            import modules.config as config_module
            importlib.reload(config_module)

            assert config_module.ENABLE_VOCAL_SEPARATION is False

    def test_hallucination_phrases_exist(self):
        """Test HALLUCINATION_PHRASES list exists and is populated."""
        import modules.config as config_module
        assert hasattr(config_module, 'HALLUCINATION_PHRASES')
        assert isinstance(config_module.HALLUCINATION_PHRASES, list)
        assert len(config_module.HALLUCINATION_PHRASES) > 0

    def test_compute_type_exists(self):
        """Test COMPUTE_TYPE exists."""
        import modules.config as config_module
        assert hasattr(config_module, 'COMPUTE_TYPE')


class TestConfigHardware:
    """Config hardware detection and device overrides."""

    def test_hardware_detection_logic_cuda(self):
        """Test CUDA detection path."""
        with mock.patch.dict(os.environ, {"ASR_DEVICE": "AUTO"}):
            with mock.patch("ctranslate2.get_cuda_device_count", return_value=1):
                import importlib
                import modules.config as config_module
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
                    import importlib
                    import modules.config as config_module
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
                    import importlib
                    import modules.config as config_module
                    importlib.reload(config_module)
                    # GPU detected means Preprocess=GPU but ASR=CPU (for Quality)
                    assert config_module.DEVICE == "CPU"
                    assert config_module.PREPROCESS_DEVICE == "GPU"

    def test_hardware_detection_logic_exception(self):
        """Test hardware detection handles exceptions gracefully."""
        with mock.patch.dict(os.environ, {"ASR_DEVICE": "AUTO"}):
            fail_patch = mock.patch(
                "ctranslate2.get_cuda_device_count",
                side_effect=Exception("Hard fail"),
            )
            with fail_patch:
                import importlib
                import modules.config as config_module
                importlib.reload(config_module)
                assert config_module.DEVICE == "CPU"

    def test_hardware_detection_logic_manual_override(self):
        """Test manual ASR_DEVICE override path."""
        with mock.patch.dict(os.environ, {"ASR_DEVICE": "CPU", "ASR_PREPROCESS_DEVICE": "GPU"}):
            import importlib
            import modules.config as config_module
            importlib.reload(config_module)
            assert config_module.DEVICE == "CPU"
            assert config_module.PREPROCESS_DEVICE == "GPU"

    def test_system_model_detection(self):
        """Test detection of baked-in system models."""
        with mock.patch("os.path.exists", side_effect=lambda p: "/app/system_models" in p):
            with mock.patch("os.listdir", return_value=["file1"]):
                import importlib
                import modules.config as config_module
                importlib.reload(config_module)
                assert "/app/system_models" in config_module.MODEL_ID
                assert "/app/system_models" in config_module.UVR_MODEL_DIR

    def test_custom_compute_type(self):
        """Test custom ASR_COMPUTE_TYPE override."""
        with mock.patch.dict(os.environ, {"ASR_COMPUTE_TYPE": "FLOAT32"}):
            import importlib
            import modules.config as config_module
            importlib.reload(config_module)
            assert config_module.COMPUTE_TYPE == "float32"

    def test_hallucination_phrases_content(self):
        """Verify hallucination phrases include expected values."""
        import modules.config as config_module
        assert "thank you for watching" in config_module.HALLUCINATION_PHRASES
        assert "vă mulțumim pentru vizionare" in config_module.HALLUCINATION_PHRASES
