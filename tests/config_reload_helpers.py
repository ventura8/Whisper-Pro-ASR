"""Shared restore helpers for tests that reload modules.core.config."""

from __future__ import annotations

import contextlib
import importlib
import os
from collections.abc import Generator
from unittest import mock

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


@contextlib.contextmanager
def reloaded_config(env: dict, *, cuda_count: int = 0, openvino_devices: tuple[str, ...] = ("CPU",)):
    """Reload config with a fake machine underneath it, then yield the reloaded module.

    Hardware detection reaches for ctranslate2 and openvino at import time, so every test
    that wants to know what config *decides* has to say what it would have found. Shared
    because the setup is identical everywhere and pylint counts the repetition as
    duplicated code once it appears in more than one module.
    """
    core = mock.MagicMock()
    core.available_devices = list(openvino_devices)
    with (
        mock.patch.dict(os.environ, env),
        mock.patch("ctranslate2.get_cuda_device_count", return_value=cuda_count),
        mock.patch("openvino.Core", return_value=core),
    ):
        importlib.reload(config_module)
        yield config_module


def npu_cannot_execute():
    """Pin the NPU probe to its documented answer: it cannot run Whisper's dynamic IR.

    The probe reads the OpenVINO IR from disk, and config resolves the model path before
    probing, so without this the result depends on whether the machine running the tests
    has weights cached locally -- and a mocked ``openvino.Core`` reports an empty input
    list, which reads as "statically shaped" and flips the answer.
    """
    return mock.patch(
        "modules.core.device_probe.npu_can_execute",
        return_value=(False, "dynamic input shapes; the NPU plugin requires static upper bounds"),
    )
