"""Defensive environment parsing in config_paths.

These helpers run while ``modules.core.config`` is being imported, which is what makes them
worth testing on their own: a ValueError here does not surface as a bad setting, it aborts
service startup before any log line can say which variable was at fault. An empty
``VAD_THRESHOLD=`` or ``WHISPER_TEMP_MIN_FREE_MB=`` in a .env file is an ordinary way to
write "leave it at the default", and must behave like one rather than taking the service
down on boot.
"""

from __future__ import annotations

import importlib
import os
import tempfile
from unittest import mock

import pytest

import modules.core.config as config_module
from modules.core import config_paths


class TestFloatEnv:
    """``float_env`` backs VAD_THRESHOLD, which used to be a bare float(os.environ.get(...))."""

    @pytest.mark.parametrize("raw", ["0.6", " 0.6 ", "1", "0", "-0.25", "1e-3"])
    def test_a_numeric_value_is_returned_as_a_float(self, raw: str):
        """Every ordinary numeric spelling still parses; the guard must not change values."""
        with mock.patch.dict(os.environ, {"VAD_THRESHOLD": raw}):
            assert config_paths.float_env("VAD_THRESHOLD", 0.5) == float(raw)

    @pytest.mark.parametrize("raw", ["", "   ", "\t"])
    def test_an_empty_value_means_the_default(self, raw: str):
        """`VAD_THRESHOLD=` in a .env file is how operators write "leave it alone"."""
        with mock.patch.dict(os.environ, {"VAD_THRESHOLD": raw}):
            assert config_paths.float_env("VAD_THRESHOLD", 0.5) == 0.5

    def test_an_absent_variable_means_the_default(self):
        """The overwhelmingly common case: the variable is simply not set."""
        with mock.patch.dict(os.environ, {}, clear=True):
            assert config_paths.float_env("VAD_THRESHOLD", 0.5) == 0.5

    @pytest.mark.parametrize("raw", ["abc", "0.5.1", "half", "None", "0,6"])
    def test_a_non_numeric_value_warns_and_falls_back_rather_than_raising(self, raw: str):
        """The whole point: startup continues, and the log names the variable and the value.

        A bare ``float(...)`` raised here, and because this runs at config import the process
        died with a traceback pointing at the assignment rather than at the setting.
        """
        with mock.patch.dict(os.environ, {"VAD_THRESHOLD": raw}):
            with mock.patch.object(config_paths.logger, "warning") as warned:
                assert config_paths.float_env("VAD_THRESHOLD", 0.5) == 0.5
        warned.assert_called_once()
        assert "VAD_THRESHOLD" in str(warned.call_args)

    def test_the_default_is_returned_unchanged_including_zero(self):
        """0.0 is a legitimate threshold and must not be confused with "unset"."""
        with mock.patch.dict(os.environ, {"VAD_THRESHOLD": "nonsense"}):
            with mock.patch.object(config_paths.logger, "warning"):
                assert config_paths.float_env("VAD_THRESHOLD", 0.0) == 0.0


class TestTempMinFreeMb:
    """The int helper float_env is modelled on; same contract, same reason."""

    def test_a_numeric_value_is_used(self):
        """A plain integer is honoured, so the setting still does what it says."""
        with mock.patch.dict(os.environ, {"WHISPER_TEMP_MIN_FREE_MB": "100"}):
            assert config_paths._temp_min_free_mb() == 100

    @pytest.mark.parametrize("raw", ["", "   "])
    def test_an_empty_value_means_the_default(self, raw: str):
        """Read during config import, so raising here would abort startup."""
        with mock.patch.dict(os.environ, {"WHISPER_TEMP_MIN_FREE_MB": raw}):
            assert config_paths._temp_min_free_mb() == config_paths.DEFAULT_TEMP_MIN_FREE_MB

    @pytest.mark.parametrize("raw", ["not-a-number", "2048MB", "1.5"])
    def test_a_malformed_value_warns_and_falls_back(self, raw: str):
        """ "1.5" is in the list deliberately: this is an int parse, and a float spelling is
        a plausible thing to write, so it must degrade rather than take the service down."""
        with mock.patch.dict(os.environ, {"WHISPER_TEMP_MIN_FREE_MB": raw}):
            with mock.patch.object(config_paths.logger, "warning") as warned:
                assert config_paths._temp_min_free_mb() == config_paths.DEFAULT_TEMP_MIN_FREE_MB
        warned.assert_called_once()


class TestPreprocessingCacheDir:
    """Where UVR's cache lands when the configured base is not writable."""

    def test_a_writable_base_gets_its_own_preprocessing_subdirectory(self, tmp_path):
        """A writable base gets its own preprocessing subdirectory."""
        result = config_paths.preprocessing_cache_dir(str(tmp_path))
        assert result == os.path.join(str(tmp_path), "preprocessing")
        assert os.path.isdir(result)

    @pytest.mark.parametrize("error", [PermissionError("read-only"), OSError("no space")])
    def test_an_unwritable_base_falls_back_to_the_system_temp_dir(self, error, tmp_path):
        """A read-only model_cache is an ordinary misconfiguration; it must not be fatal."""
        real_makedirs = os.makedirs
        calls = []

        def makedirs(path, *args, **kwargs):
            """Fail for the configured base, else create for real."""
            calls.append(path)
            if str(path).startswith(str(tmp_path)):
                raise error
            return real_makedirs(path, *args, **kwargs)

        with mock.patch("os.makedirs", side_effect=makedirs):
            result = config_paths.preprocessing_cache_dir(str(tmp_path))

        assert result == os.path.join(tempfile.gettempdir(), "preprocessing")

    def test_a_fallback_that_also_fails_still_returns_a_path(self, tmp_path):
        """Returning None here would surface as a TypeError deep inside ffmpeg argument
        building, naming neither the directory nor the permission that caused it."""
        with mock.patch("os.makedirs", side_effect=OSError("everything is read-only")):
            result = config_paths.preprocessing_cache_dir(str(tmp_path))
        assert result == os.path.join(tempfile.gettempdir(), "preprocessing")


@pytest.mark.usefixtures("restore_config_after_reload")
def test_config_reads_vad_threshold_through_the_guarded_parser():
    """The regression this guards: an unparseable VAD_THRESHOLD must not abort config import."""
    with mock.patch.dict(os.environ, {"VAD_THRESHOLD": ""}):
        importlib.reload(config_module)
        assert config_module.VAD_THRESHOLD == 0.5
