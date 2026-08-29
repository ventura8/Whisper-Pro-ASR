"""Shared test fixtures and global mocks for the Whisper Pro ASR test suite."""

import argparse
import gc
import json
from typing import Any
from unittest import mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import modules.inference.pipeline.preprocessing as prep_module
import tests.conftest_bootstrap

# Now safe to import project modules
from modules.api.routes import asr as routes_asr
from modules.api.routes import detect as routes_detect
from modules.api.routes import system as routes_system
from modules.inference import scheduler
from modules.inference.pipeline.vad import reset_vad_state
from modules.inference.runtime import model_manager
from modules.inference.scheduler import SchedulerState

# Apply global mocks before any first-party project imports to prevent real module loading
from tests.mock_setup import mock_torch
from whisper_pro_asr import create_app

# Keep imported bootstrap alive to avoid unused import warning
_ = tests.conftest_bootstrap

# Register shared config-reload restore fixture for env-driven importlib.reload tests.
# class_progress: quiet mode shows file::TestClass once, then one dot per test.
pytest_plugins = ("tests.config_reload_helpers", "tests.class_progress")


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

    def json(self):
        """Get response JSON data."""
        return self._response.json()


def _is_json_content_type(content_type: str) -> bool:
    """Return whether the request content type should be treated as JSON."""
    return "application/json" in content_type


def _is_upload_field(value: Any) -> bool:
    """Return whether a field should be treated as a file upload tuple."""
    return isinstance(value, tuple) and len(value) >= 2


def _normalize_upload_field(value: tuple[Any, ...]) -> tuple[str, Any, str]:
    """Convert a Flask-style upload tuple into a test-client file payload."""
    file_obj, filename = value[0], value[1]
    mime_type = value[2] if len(value) >= 3 else "application/octet-stream"
    content = file_obj.read() if hasattr(file_obj, "read") else file_obj
    return filename, content, mime_type


def _split_form_and_files(data: dict[str, Any]) -> tuple[dict[str, tuple[str, Any, str]], dict[str, Any]]:
    """Split mixed form/file payloads into upload files and form fields."""
    files = {key: _normalize_upload_field(value) for key, value in data.items() if _is_upload_field(value)}
    form_fields = {key: value for key, value in data.items() if not _is_upload_field(value)}
    return files, form_fields


def _coerce_json_string_payload(kwargs: dict[str, Any], data: str | bytes) -> dict[str, Any]:
    """Convert a JSON string or bytes payload into kwargs['json']."""
    try:
        kwargs["json"] = json.loads(data)
        kwargs.pop("data", None)
    except (json.JSONDecodeError, UnicodeDecodeError):
        kwargs["content"] = data
        kwargs["content_type"] = kwargs.get("content_type", "application/json")
    return kwargs


def _coerce_post_json_payload(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Normalize JSON payloads carried in POST kwargs."""
    data = kwargs.get("data")
    content_type = kwargs.get("content_type", "")
    if not _is_json_content_type(content_type):
        return kwargs

    if isinstance(data, dict):
        kwargs["json"] = data
        del kwargs["data"]
        return kwargs

    if isinstance(data, (str, bytes)):
        return _coerce_json_string_payload(kwargs, data)
    return kwargs


def _coerce_post_file_payload(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Normalize upload payloads carried in POST kwargs."""
    data = kwargs.get("data")
    if not isinstance(data, dict):
        return kwargs

    files, form_fields = _split_form_and_files(data)
    if files:
        kwargs["files"] = files
        kwargs["data"] = form_fields
    return kwargs


def _rewrite_post_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Normalize POST kwargs for Flask-style tests."""
    kwargs = _coerce_post_file_payload(kwargs)
    kwargs = _coerce_post_json_payload(kwargs)
    kwargs.pop("content_type", None)
    return kwargs


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
        if "data" in kwargs or "content_type" in kwargs:
            kwargs = _rewrite_post_kwargs(kwargs)
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
        mock_psu.virtual_memory.return_value = argparse.Namespace(percent=50.0, used=8 * (1024**3), total=16 * (1024**3))

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
def disable_engine_isolation():
    """Keep engines and preprocessing in-process for the default suite.

    Isolation is a deployment default, but a test that merely exercises pool or
    scheduler logic should not pay to spawn a worker -- and with mocked engines the
    spawned process would import the real engine stack only to fail. Tests that mean to
    cover the out-of-process path opt back in explicitly (see
    tests/inference/engines/test_isolated_engine.py and the init_unit isolation tests).
    """
    with mock.patch("modules.core.config.ISOLATE_ENGINES", False):
        with mock.patch("modules.core.config.ISOLATE_PREPROCESSING", False):
            yield


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
        mock.patch("modules.api.routes.asr.model_manager") as mock_mm_asr,
        mock.patch("modules.api.routes.detect.model_manager") as mock_mm_det,
        mock.patch("modules.inference.pipeline.language_detection.run_voting_detection") as mock_ld,
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
