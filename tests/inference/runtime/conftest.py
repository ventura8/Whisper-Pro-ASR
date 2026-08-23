"""Shared fixtures for tests/inference/runtime/*.

``reset_state`` was previously duplicated verbatim across
test_model_manager.py, test_model_manager_resource_lifecycle.py, and
test_concurrency_reentrancy.py.
"""

from collections.abc import Generator
from unittest import mock

import pytest

from modules.core import utils
from modules.inference import scheduler
from modules.inference.runtime import model_manager


@pytest.fixture(autouse=True)
def reset_state() -> Generator[None, None, None]:
    """Reset model_manager and scheduler global state before each test."""
    model_manager.MODEL_POOL.clear()
    model_manager.PREPROCESSOR_POOL.clear()

    # Mock HARDWARE_UNITS before creating SchedulerState
    with mock.patch("modules.core.config.HARDWARE_UNITS", [{"id": "CPU", "type": "CPU", "name": "CPU"}]):
        from modules.inference.scheduler import SchedulerState

        scheduler.STATE = SchedulerState()
        scheduler.STATE.engine_initialized = True

    utils.THREAD_CONTEXT.reset()

    yield

    model_manager.MODEL_POOL.clear()
    model_manager.PREPROCESSOR_POOL.clear()
    with mock.patch("modules.core.config.HARDWARE_UNITS", [{"id": "CPU", "type": "CPU", "name": "CPU"}]):
        from modules.inference.scheduler import SchedulerState

        scheduler.STATE = SchedulerState()
