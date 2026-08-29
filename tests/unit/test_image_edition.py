"""The dashboard must distinguish images: same version, very different accelerators.

VERSION stays a bare semver because API clients parse it; the edition rides alongside.
"""

import importlib
import os
from unittest import mock

import pytest

from modules.core import config as config_module

# Every test here reloads config with a patched environment. Without the restore fixture
# the last reload's VERSION, IMAGE_EDITION and VERSION_DISPLAY leak into whatever runs
# next, which is how an unrelated test starts asserting against an edition it never set.
pytestmark = pytest.mark.usefixtures("restore_config_after_reload")


def _reload(env):
    with mock.patch.dict(os.environ, env, clear=False):
        if "WHISPER_IMAGE_EDITION" not in env:
            os.environ.pop("WHISPER_IMAGE_EDITION", None)
        importlib.reload(config_module)
        return config_module.VERSION, config_module.IMAGE_EDITION, config_module.VERSION_DISPLAY


def test_edition_is_appended_for_display_only():
    """The edition rides alongside VERSION without contaminating it."""
    version, edition, display = _reload({"WHISPER_IMAGE_EDITION": "intel"})
    assert edition == "intel"
    assert display == f"{version} intel"
    assert version.count(".") == 2, "VERSION must stay a bare semver for API clients"


def test_full_edition_is_reported():
    """The full image reports its edition like any other."""
    _version, edition, display = _reload({"WHISPER_IMAGE_EDITION": "full"})
    assert edition == "full"
    assert display.endswith(" full")


def test_absent_edition_leaves_the_version_unchanged():
    """Running from a checkout rather than a shipped image must not show a stray space."""
    version, edition, display = _reload({})
    assert edition == ""
    assert display == version


def test_whitespace_is_trimmed():
    """A padded value in the environment must not reach the dashboard."""
    _version, edition, _display = _reload({"WHISPER_IMAGE_EDITION": "  amd  "})
    assert edition == "amd"
