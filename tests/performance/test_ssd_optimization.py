"""Tests for SSD Write Wear Optimization."""

import importlib
import os
import tempfile
from collections import namedtuple
from unittest import mock

import pytest

import modules.core.config as config_module

pytestmark = pytest.mark.usefixtures("restore_config_after_reload")


class TestSSDOptimization:
    """Test suite for SSD write wear optimization features."""

    def test_temp_dir_default(self):
        """Test that TEMP_DIR defaults to system temp when env var is absent."""
        with mock.patch.dict(os.environ, {}, clear=True):
            importlib.reload(config_module)
            # It might be /tmp or whatever tempfile.gettempdir() returns
            assert config_module.TEMP_DIR == tempfile.gettempdir()

    def test_temp_dir_from_env(self):
        """Test that TEMP_DIR respects WHISPER_TEMP_DIR environment variable."""
        custom_temp = "/tmp/custom_whisper"
        with mock.patch.dict(os.environ, {"WHISPER_TEMP_DIR": custom_temp}):
            # Mock os.makedirs to avoid actually creating the directory
            with mock.patch("os.makedirs"):
                importlib.reload(config_module)
                assert config_module.TEMP_DIR == custom_temp

    def test_preprocessing_cache_dir_derived_from_temp(self):
        """Test that PREPROCESSING_CACHE_DIR is derived from TEMP_DIR."""
        custom_temp = "/tmp/custom_whisper"
        with mock.patch.dict(os.environ, {"WHISPER_TEMP_DIR": custom_temp}):
            # Mock disk_usage to return plenty of space
            with mock.patch("shutil.disk_usage") as mock_usage:
                mock_usage.return_value = mock.MagicMock(free=10 * 1024 * 1024 * 1024)
                with mock.patch("os.makedirs"):
                    importlib.reload(config_module)
                    assert config_module.PREPROCESSING_CACHE_DIR.startswith(custom_temp)

    def test_get_temp_dir_sufficient_space(self):
        """Test get_temp_dir returns TEMP_DIR when there is enough space."""
        with mock.patch("shutil.disk_usage") as mock_usage:
            # 4GB free (well above the 2048 MB minimum threshold)
            mock_usage.return_value = mock.MagicMock(free=4 * 1024 * 1024 * 1024)

            with mock.patch("modules.core.config.TEMP_DIR", "/tmp/whisper"):
                res = config_module.get_temp_dir(required_bytes=100 * 1024 * 1024)  # 100MB
                assert res == "/tmp/whisper"

    def test_get_temp_dir_low_space_fallback(self):
        """Test get_temp_dir falls back to persistent temp when space is low."""
        with mock.patch("shutil.disk_usage") as mock_usage:
            # 100MB free, but we need 512MB (default min) or more
            mock_usage.return_value = mock.MagicMock(free=100 * 1024 * 1024)

            with mock.patch("modules.core.config.TEMP_DIR", "/tmp/whisper"):
                res = config_module.get_temp_dir(required_bytes=200 * 1024 * 1024)
                assert res == config_module.PERSISTENT_TEMP_DIR

    def test_get_temp_dir_error_fallback(self):
        """Test get_temp_dir falls back to persistent temp on OSError."""
        with mock.patch("shutil.disk_usage", side_effect=OSError("Drive not ready")):
            with mock.patch("modules.core.config.TEMP_DIR", "/tmp/whisper"):
                res = config_module.get_temp_dir()
                assert res == config_module.PERSISTENT_TEMP_DIR


class TestTempDirNeverRaises:
    """get_temp_dir selects a directory; it must not refuse to answer.

    v1.3.0 added a RuntimeError for the case where neither directory clears the
    threshold. That threshold is max(min_free, 1.5x required) -- desired headroom, not
    the space the work needs -- and the function runs while resolving config on the
    request path, so a full disk became a hard failure on every request instead of a
    degraded one. Two tests had been failing since.
    """

    def test_returns_persistent_when_neither_has_the_preferred_headroom(self):
        with mock.patch("shutil.disk_usage") as usage:
            usage.return_value = mock.MagicMock(free=1024)  # 1 KB everywhere
            with mock.patch("modules.core.config.TEMP_DIR", "/tmp/whisper"):
                assert config_module.get_temp_dir(required_bytes=10 * 1024**3) == config_module.PERSISTENT_TEMP_DIR

    def test_says_so_in_the_log_rather_than_failing(self, caplog):
        import logging

        with mock.patch("shutil.disk_usage") as usage:
            usage.return_value = mock.MagicMock(free=1024)
            with mock.patch("modules.core.config.TEMP_DIR", "/tmp/whisper"):
                with caplog.at_level(logging.WARNING):
                    config_module.get_temp_dir(required_bytes=10 * 1024**3)

        assert any("Neither temp directory" in record.message for record in caplog.records)

    def test_zero_free_space_still_returns_a_directory(self):
        Usage = namedtuple("Usage", ["free"])
        with mock.patch("shutil.disk_usage", return_value=Usage(free=0)):
            with mock.patch("modules.core.config.TEMP_DIR", "/tmp/whisper"):
                assert config_module.get_temp_dir(required_bytes=1_000_000) == config_module.PERSISTENT_TEMP_DIR
