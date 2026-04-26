"""Tests for SSD Write Wear Optimization."""
# pylint: disable=import-outside-toplevel,reimported
import os
from unittest import mock
import tempfile
from modules import config

class TestSSDOptimization:
    """Test suite for SSD write wear optimization features."""

    def test_temp_dir_default(self):
        """Test that TEMP_DIR defaults to system temp when env var is absent."""
        with mock.patch.dict(os.environ, {}, clear=True):
            import importlib
            import modules.config as config_module
            importlib.reload(config_module)
            # It might be /tmp or whatever tempfile.gettempdir() returns
            assert config_module.TEMP_DIR == tempfile.gettempdir()

    def test_temp_dir_from_env(self):
        """Test that TEMP_DIR respects WHISPER_TEMP_DIR environment variable."""
        custom_temp = "/tmp/custom_whisper"
        with mock.patch.dict(os.environ, {"WHISPER_TEMP_DIR": custom_temp}):
            import importlib
            import modules.config as config_module
            # Mock os.makedirs to avoid actually creating the directory
            with mock.patch("os.makedirs"):
                importlib.reload(config_module)
                assert config_module.TEMP_DIR == custom_temp

    def test_preprocessing_cache_dir_derived_from_temp(self):
        """Test that PREPROCESSING_CACHE_DIR is derived from TEMP_DIR."""
        custom_temp = "/tmp/custom_whisper"
        with mock.patch.dict(os.environ, {"WHISPER_TEMP_DIR": custom_temp}):
            import importlib
            import modules.config as config_module
            # Mock disk_usage to return plenty of space
            with mock.patch("shutil.disk_usage") as mock_usage:
                mock_usage.return_value = mock.MagicMock(free=10 * 1024 * 1024 * 1024)
                with mock.patch("os.makedirs"):
                    importlib.reload(config_module)
                    assert config_module.PREPROCESSING_CACHE_DIR.startswith(custom_temp)

    def test_get_temp_dir_sufficient_space(self):
        """Test get_temp_dir returns TEMP_DIR when there is enough space."""
        with mock.patch("shutil.disk_usage") as mock_usage:
            # 1GB free
            mock_usage.return_value = mock.MagicMock(free=1024 * 1024 * 1024)

            with mock.patch("modules.config.TEMP_DIR", "/tmp/whisper"):
                res = config.get_temp_dir(required_bytes=100 * 1024 * 1024)  # 100MB
                assert res == "/tmp/whisper"

    def test_get_temp_dir_low_space_fallback(self):
        """Test get_temp_dir falls back to persistent temp when space is low."""
        with mock.patch("shutil.disk_usage") as mock_usage:
            # 100MB free, but we need 512MB (default min) or more
            mock_usage.return_value = mock.MagicMock(free=100 * 1024 * 1024)

            with mock.patch("modules.config.TEMP_DIR", "/tmp/whisper"):
                res = config.get_temp_dir(required_bytes=200 * 1024 * 1024)
                assert res == config.PERSISTENT_TEMP_DIR

    def test_get_temp_dir_error_fallback(self):
        """Test get_temp_dir falls back to persistent temp on OSError."""
        with mock.patch("shutil.disk_usage", side_effect=OSError("Drive not ready")):
            with mock.patch("modules.config.TEMP_DIR", "/tmp/whisper"):
                res = config.get_temp_dir()
                assert res == config.PERSISTENT_TEMP_DIR
