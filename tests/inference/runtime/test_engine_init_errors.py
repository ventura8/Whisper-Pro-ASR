"""A missing engine dependency must say so, not report an empty pool.

Not every image ships every engine -- WhisperX is only in the `full` target -- so the
common failure is an ImportError inside init_unit. Reporting only the downstream
"engine pool is empty" symptom made that undiagnosable from an API response.
"""

from unittest import mock

import pytest

from modules.inference.runtime import model_manager

UNIT = {"id": "cuda:0", "type": "CUDA", "name": "NVIDIA GPU 0"}


@pytest.fixture(autouse=True)
def _clear_errors():
    model_manager.LAST_INIT_ERROR.clear()
    yield
    model_manager.LAST_INIT_ERROR.clear()


def test_missing_dependency_is_reported_as_an_unavailable_engine():
    with mock.patch("modules.core.config.engine_for_unit", return_value="WHISPERX"):
        with mock.patch("modules.inference.engines.engine_factory.create_engine", side_effect=ImportError("No module named 'whisperx'")):
            model_manager.init_unit(UNIT)

    message = model_manager.LAST_INIT_ERROR["cuda:0"]
    assert "WHISPERX" in message
    assert "not available in this image" in message
    assert "ASR_ENGINE" in message


def test_other_failures_name_the_engine_and_the_unit():
    with mock.patch("modules.core.config.engine_for_unit", return_value="FASTER-WHISPER"):
        with mock.patch("modules.inference.engines.engine_factory.create_engine", side_effect=RuntimeError("CUDA out of memory")):
            model_manager.init_unit(UNIT)

    message = model_manager.LAST_INIT_ERROR["cuda:0"]
    assert "FASTER-WHISPER" in message
    assert "NVIDIA GPU 0" in message
    assert "CUDA out of memory" in message


def test_a_successful_load_clears_a_previous_failure():
    """A stale message must not outlive the failure it described."""
    model_manager.LAST_INIT_ERROR["cuda:0"] = "old failure"

    with mock.patch("modules.core.config.engine_for_unit", return_value="FASTER-WHISPER"):
        with mock.patch("modules.inference.engines.engine_factory.create_engine", return_value=mock.MagicMock()):
            with mock.patch("modules.inference.pipeline.preprocessing.PreprocessingManager"):
                model_manager.init_unit(UNIT)

    assert "cuda:0" not in model_manager.LAST_INIT_ERROR
