"""The segment-first settings: their defaults, and that the environment reaches them.

They live in ``modules.core.config_segmentation`` and are re-exported by ``config``; both
modules are reloaded here, in that order, because the re-export is a plain assignment at
import time -- a reload of ``config`` alone would carry the previously parsed values.
"""

import importlib
import os
import re
from pathlib import Path
from unittest import mock

import pytest

import modules.core.config as config_module
from modules.core import config_segmentation

SETTINGS = [
    ("ASR_SEGMENT_FIRST", True, "false", False),
    ("SEGMENT_SPLIT_MIN_SILENCE_MS", 250, "400", 400),
    ("SEGMENT_SPLIT_PAD_MS", 200, "50", 50),
    ("SEGMENT_FIRST_MIN_REGIONS", 2, "5", 5),
    ("SEGMENT_CLIP_MERGE_GAP_SEC", 0.3, "0.75", 0.75),
    ("SEGMENT_RUN_MIN_SWITCH_SEC", 3.0, "4.5", 4.5),
    ("SEGMENT_RUN_MIN_SWITCH_SHARE", 0.2, "0.35", 0.35),
    ("ASR_SEGMENT_LANGUAGES", True, "0", False),
]


def _reload():
    importlib.reload(config_segmentation)
    importlib.reload(config_module)


@pytest.fixture(autouse=True)
def _both_modules_restored(restore_config_after_reload):
    """Reload *both* modules once the environment is back.

    The shared fixture reloads ``config`` alone, which re-exports whatever the last
    override left in ``config_segmentation`` -- so the final parameter here, which turns
    ``ASR_SEGMENT_LANGUAGES`` off, stayed off for every later test on the same xdist worker.
    """
    yield
    _reload()


@pytest.mark.parametrize(("name", "default", "_raw", "_parsed"), SETTINGS, ids=[s[0] for s in SETTINGS])
def test_the_default_when_the_environment_says_nothing(name, default, _raw, _parsed):
    """Every setting has the documented default, and config re-exports exactly it."""
    with mock.patch.dict(os.environ, {}, clear=True):
        _reload()
        assert getattr(config_segmentation, name) == default
        assert getattr(config_module, name) == default


@pytest.mark.parametrize(("name", "_default", "raw", "parsed"), SETTINGS, ids=[s[0] for s in SETTINGS])
def test_the_environment_overrides_it_and_config_follows(name, _default, raw, parsed):
    """An operator's value is parsed to the setting's own type and seen through both modules."""
    with mock.patch.dict(os.environ, {name: raw}):
        _reload()
        assert getattr(config_segmentation, name) == parsed
        assert getattr(config_module, name) == parsed


_COMPOSE_DEFAULT = re.compile(r"^\s*- (?P<name>[A-Z_]+)=\$\{(?P=name):-(?P<default>[^}]*)\}\s*$", re.MULTILINE)


def _compose_defaults() -> dict[str, str]:
    """``NAME -> default`` for every ``- NAME=${NAME:-default}`` line in docker-compose.yml."""
    text = (Path(__file__).resolve().parents[2] / "docker-compose.yml").read_text(encoding="utf-8")
    return {match["name"]: match["default"] for match in _COMPOSE_DEFAULT.finditer(text)}


@pytest.mark.parametrize(("name", "default", "_raw", "_parsed"), SETTINGS, ids=[s[0] for s in SETTINGS])
def test_compose_forwards_the_setting_with_the_same_default(name, default, _raw, _parsed):
    """A knob compose does not forward is a knob nobody can turn -- and a default repeated in
    compose that drifts from the code's would turn it the wrong way. Parsed through the
    setting's own type, so `3.0` and `3` mean the same thing."""
    defaults = _compose_defaults()
    assert name in defaults, f"docker-compose.yml does not forward {name}"
    if isinstance(default, bool):
        assert defaults[name] == str(default).lower()
    else:
        assert type(default)(defaults[name]) == default
