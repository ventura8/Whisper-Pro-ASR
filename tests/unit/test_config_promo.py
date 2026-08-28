"""Tests for subtitle promo configurations in modules/config.py"""

import importlib
import os
from unittest import mock

import pytest

import modules.core.config as config_module

pytestmark = pytest.mark.usefixtures("restore_config_after_reload")


def test_subtitle_promo_configs_defaults():
    """Subtitle promo settings should use the documented defaults."""
    with mock.patch.dict(os.environ, {}, clear=True):
        importlib.reload(config_module)
        assert config_module.SUBTITLE_PROMO_ENABLED is True
        assert config_module.SUBTITLE_PROMO_TEXT == "Made with Whisper Pro ASR"
        assert config_module.SUBTITLE_PROMO_DURATION == 3.0


def test_subtitle_promo_configs_custom_values():
    """Subtitle promo settings should read custom environment values."""
    env = {
        "SUBTITLE_PROMO_ENABLED": "false",
        "SUBTITLE_PROMO_TEXT": "Promo Test",
        "SUBTITLE_PROMO_DURATION": "5.5",
    }
    with mock.patch.dict(os.environ, env):
        importlib.reload(config_module)
        assert config_module.SUBTITLE_PROMO_ENABLED is False
        assert config_module.SUBTITLE_PROMO_TEXT == "Promo Test"
        assert config_module.SUBTITLE_PROMO_DURATION == 5.5


def test_subtitle_promo_configs_duration_fallback():
    """Invalid duration values should fall back to the default duration."""
    env = {"SUBTITLE_PROMO_DURATION": "invalid-float"}
    with mock.patch.dict(os.environ, env):
        importlib.reload(config_module)
        assert config_module.SUBTITLE_PROMO_DURATION == 3.0
