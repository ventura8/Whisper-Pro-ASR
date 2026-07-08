"""Tests for subtitle promo configurations in modules/config.py"""

import importlib
import os
from unittest import mock

import modules.core.config as config_module


def test_subtitle_promo_configs():
    """Test subtitle promo configurations parsing from environment."""
    # Test defaults
    with mock.patch.dict(os.environ, {}, clear=True):
        importlib.reload(config_module)
        assert config_module.SUBTITLE_PROMO_ENABLED is True
        assert config_module.SUBTITLE_PROMO_TEXT == "Made with Whisper Pro ASR"
        assert config_module.SUBTITLE_PROMO_DURATION == 3.0

    # Test custom
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

    # Test duration fallback
    env = {"SUBTITLE_PROMO_DURATION": "invalid-float"}
    with mock.patch.dict(os.environ, env):
        importlib.reload(config_module)
        assert config_module.SUBTITLE_PROMO_DURATION == 3.0
