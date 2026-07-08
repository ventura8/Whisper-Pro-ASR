# ruff: noqa: I001
"""Shared test fixtures and global mocks for the Whisper Pro ASR test suite."""

import argparse
import gc
import json
from unittest import mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Apply global mocks before any first-party project imports to prevent real module loading
from tests.mock_setup import mock_torch

import tests.conftest_bootstrap
import modules.inference.preprocessing as prep_module

# Now safe to import project modules
from modules.api import routes_asr, routes_detect, routes_system
from modules.inference import model_manager, scheduler
from modules.inference.scheduler import SchedulerState
from modules.inference.vad import reset_vad_state
from whisper_pro_asr import create_app

# Keep imported bootstrap alive to avoid unused import warning
_ = tests.conftest_bootstrap


def mock_tensor_with_shape(*shape):
    """Create a mock tensor with a specific shape attribute."""
    t = mock.MagicMock()
    t.shape = shape
    t.dtype = "float32"
    t.device = "cpu"
    return t


mock_torch.cat = lambda _t, dim=0: mock_tensor_with_shape(8, 512)
mock_torch.zeros = lambda *args, **kwargs: mock_tensor_with_shape(*args)
mock_torch.full = lambda size, _v, **kwargs: mock_tensor_with_shape(*size)


class FlaskCompatibleResponse:
    """Mock response wrapper compatible with Flask client assertions."""

    def __init__(self, response):
        self._response = response

    @property
    def status_code(self):
        """Get response status code."""
        return self._response.status_code

    @property
    def data(self):
        """Get response raw content data."""
        return self._response.content

    @property
    def content_type(self):
        """Get response content-type header."""
        return self._response.headers.get("content-type")

    @property
    def headers(self):
        """Get response headers dict."""
        return self._response.headers

    def get_json(self):
        """Get response body parsed as JSON."""
        return self._response.json()


class FlaskCompatibleClient:
    """Mock client wrapper compatible with Flask test assertions."""

    def __init__(self, app):
        self._client = TestClient(app, raise_server_exceptions=False)

    def get(self, *args, **kwargs):
        """Perform GET request and wrap response."""
        resp = self._client.get(*args, **kwargs)
        return FlaskCompatibleResponse(resp)

    def post(self, *args, **kwargs):
        """Perform POST request with payload rewriting and wrap response."""
        if "data" in kwargs:
            data = kwargs["data"]
            content_type = kwargs.get("content_type", "")
            if isinstance(data, dict):
                files = {}
                form_fields = {}
                for k, v in data.items():
                    if isinstance(v, tuple) and len(v) >= 2:
                        file_obj, filename = v[0], v[1]
                        mime_type = v[2] if len(v) >= 3 else "application/octet-stream"
                        content = file_obj.read() if hasattr(file_obj, "read") else file_obj
                        files[k] = (filename, content, mime_type)
                    else:
                        form_fields[k] = v
                if files:
                    kwargs["files"] = files
                    kwargs["data"] = form_fields
                elif "application/json" in content_type:
                    if isinstance(data, str):
                        kwargs["json"] = json.loads(data)
                    else:
                        kwargs["json"] = data
                    del kwargs["data"]
            elif isinstance(data, (str, bytes)) and "application/json" in content_type:
                try:
                    kwargs["json"] = json.loads(data)
                    del kwargs["data"]
                except json.JSONDecodeError:
                    pass
        if "content_type" in kwargs:
            del kwargs["content_type"]
        resp = self._client.post(*args, **kwargs)
        return FlaskCompatibleResponse(resp)


@pytest.fixture
def client():
    """FastAPI test client with full orchestration mocks."""
    with (
        mock.patch("modules.monitoring.dashboard.psutil", create=True) as mock_psu,
        mock.patch("modules.monitoring.dashboard.utils") as mock_utils,
    ):
        mock_psu.cpu_percent.return_value = 10.0
        mock_psu.cpu_count.return_value = 8
        mock_psu.virtual_memory.return_value = argparse.Namespace(
            percent=50.0, used=8 * (1024**3), total=16 * (1024**3)
        )

        mock_utils.get_system_telemetry.return_value = {
            "cpu_percent": 10.0,
            "app_cpu_percent": 5.0,
            "memory_percent": 50.0,
            "memory_used_gb": 8.0,
            "memory_total_gb": 16.0,
            "app_memory_gb": 1.0,
        }

        app = create_app(testing=True)
        yield FlaskCompatibleClient(app)


@pytest.fixture(autouse=True)
def reset_module_state():
    """Reset module-level state between tests to prevent test pollution."""
    # Force reset module state before test
    model_manager.MODEL_POOL = {}
    model_manager.PREPROCESSOR_POOL = {}
    prep_module.Separator = None
    prep_module.ort = None
    scheduler.STATE = SchedulerState()
    reset_vad_state()
    gc.collect()

    yield

    # Also reset after test
    model_manager.MODEL_POOL = {}
    model_manager.PREPROCESSOR_POOL = {}
    prep_module.Separator = None
    prep_module.ort = None
    scheduler.STATE = SchedulerState()
    reset_vad_state()
    gc.collect()


@pytest.fixture
def routes_app():
    """Create test FastAPI app with mocked model_manager for API routes testing."""
    with (
        mock.patch("modules.api.routes_asr.model_manager") as mock_mm_asr,
        mock.patch("modules.api.routes_detect.model_manager") as mock_mm_det,
        mock.patch("modules.inference.language_detection.run_voting_detection") as mock_ld,
    ):
        mock_mm_asr.is_engine_initialized.return_value = True
        mock_mm_det.is_engine_initialized.return_value = True

        mock_ld.return_value = {"confidence": 0.95, "detected_language": "en", "language": "en", "language_code": "en"}

        mock_mm_asr.run_transcription.return_value = {
            "text": "Hello world",
            "segments": [{"timestamp": (0.0, 1.0), "text": "Hello world"}],
        }

        test_app = FastAPI(title="Test App")
        test_app.include_router(routes_system.router)
        test_app.include_router(routes_asr.router)
        test_app.include_router(routes_detect.router)
        yield test_app


@pytest.fixture
def routes_client(request):
    """Create test client for routes_app."""
    app = request.getfixturevalue("routes_app")
    return FlaskCompatibleClient(app)
