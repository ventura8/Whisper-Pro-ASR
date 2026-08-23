"""Shared restore helpers for tests that reload modules.core.config."""

from __future__ import annotations

import importlib
import os
from collections.abc import Generator

import pytest

import modules.core.config as config_module


@pytest.fixture
def restore_config_after_reload() -> Generator[None, None, None]:
    """Restore process env and reload config after env-driven importlib.reload.

    Reloading leaves module globals in the last env-driven state. Restoring is
    required so later pytest-xdist worker tests do not inherit polluted values
    (e.g. production SUBTITLE_PROMO_ENABLED=True).
    """
    original_environ = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(original_environ)
    importlib.reload(config_module)
