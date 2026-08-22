"""Shared helper for modules/monitoring/telemetry.py tests, reused by
test_telemetry_loop.py and test_telemetry_liveness.py to avoid duplicating the
same get_service_stats() patch scaffolding in both files.
"""

from typing import Any
from unittest import mock

from modules.monitoring import telemetry


def get_service_stats_with_common_patches(
    mock_units: list[dict[str, str]] | None = None, model_loaded: bool = True, uvr_loaded: bool = True
) -> dict[str, Any]:
    from modules.inference.runtime import model_manager

    mock_units = mock_units or [
        {"id": "CPU", "type": "CPU", "name": "CPU"},
        {"id": "GPU", "type": "GPU", "name": "GPU"},
        {"id": "NPU", "type": "NPU", "name": "NPU"},
        {"id": "AUTO", "type": "AUTO", "name": "AUTO"},
    ]
    mock_preprocessor = mock.MagicMock()
    mock_preprocessor.separator = mock.MagicMock()

    with mock.patch("modules.core.config.HARDWARE_UNITS", mock_units):
        with mock.patch.dict(model_manager.MODEL_POOL, {"NPU": mock.MagicMock()}):
            with mock.patch.dict(model_manager.PREPROCESSOR_POOL, {"NPU": mock_preprocessor}):
                with mock.patch("modules.monitoring.history_manager.get_history_stats", return_value=([], {})):
                    with mock.patch("modules.monitoring.metrics_discovery.get_nvidia_metrics", return_value=[]):
                        with mock.patch(
                            "modules.inference.runtime.model_manager.is_engine_actually_loaded",
                            return_value=model_loaded,
                        ):
                            with mock.patch(
                                "modules.inference.runtime.model_manager.is_uvr_actually_loaded",
                                return_value=uvr_loaded,
                            ):
                                return telemetry.get_service_stats()
