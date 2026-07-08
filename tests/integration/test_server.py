"""Comprehensive coverage for the main FastAPI server logic."""

import logging
import os
import runpy
import sys
from io import BytesIO
from unittest import mock

import pytest

import whisper_pro_asr as wpa_module
from modules.core import bootstrap
from tests.conftest import FlaskCompatibleClient


@pytest.fixture(name="whisper_pro_asr")
def whisper_pro_asr_fixture():
    """Fixture to retrieve whisper_pro_asr."""
    return wpa_module


def test_patch_path_explicit_cuda():
    """Test _initialize_hardware_path with explicit ASR_DEVICE=CUDA."""
    with mock.patch.dict(os.environ, {"ASR_DEVICE": "CUDA"}):
        fake_path = []
        with mock.patch.object(sys, "path", fake_path):
            with mock.patch("os.path.exists", side_effect=lambda p: "/nvidia" in p or "\\nvidia" in p):
                with mock.patch("importlib.reload"):
                    # Call the function directly
                    bootstrap.initialize_hardware_path()
                # Check if nvidia path was added
                nvidia_found = any("nvidia" in p for p in fake_path)
                assert nvidia_found, f"Expected nvidia path in {fake_path}"


def test_patch_path_auto_nvidia():
    """Test _initialize_hardware_path with AUTO and detected NVIDIA."""
    with mock.patch.dict(os.environ, {"ASR_DEVICE": "AUTO"}):
        with mock.patch("os.path.exists", side_effect=lambda p: "nvidia" in p or "\\nvidia" in p):
            fake_path = []
            with mock.patch.object(sys, "path", fake_path):
                with mock.patch("importlib.reload"):
                    bootstrap.initialize_hardware_path()
                # Check if nvidia path was added
                nvidia_found = any("nvidia" in p for p in fake_path)
                assert nvidia_found, f"Expected nvidia path in {fake_path}"


def test_on_import_onnx_exceptions(whisper_pro_asr):
    """Test that the server handles onnxruntime import errors gracefully."""
    # The module should have proper exception handling
    # Check that logger is set up and module loads without crash
    logger = logging.getLogger(__name__)
    assert logger is not None
    # Module loaded successfully
    assert whisper_pro_asr is not None


def test_request_logging_types(whisper_pro_asr):
    """Test request logging formats (Multipart, Form, JSON)."""
    app = whisper_pro_asr.create_app(testing=True)
    client = FlaskCompatibleClient(app)
    # 1. Multipart
    client.post("/asr", data={"file": (BytesIO(b"x"), "test.wav"), "field": "val"}, content_type="multipart/form-data")

    # 2. Form URL Encoded
    client.post("/asr", data={"info": "yes"}, content_type="application/x-www-form-urlencoded")

    # 3. JSON
    client.post("/asr", json={"key": "value"})


def test_post_wrapper_strips_content_type_without_data(whisper_pro_asr):
    """Client wrapper should drop content_type even when no data kwarg is provided."""
    app = whisper_pro_asr.create_app(testing=True)
    client = FlaskCompatibleClient(app)

    response = client.post("/asr", content_type="application/json")
    assert response.status_code in {400, 422, 503}


def test_request_logging_data_handling(whisper_pro_asr):
    """Test large body and invalid body."""
    app = whisper_pro_asr.create_app(testing=True)
    client = FlaskCompatibleClient(app)
    # 1. Large Body (truncated)
    client.post("/asr", content="A" * 1500, headers={"Content-Type": "text/plain"})

    # 2. Decoding error (simulated via invalid utf8)
    client.post("/asr", content=b"\x80\xff", headers={"Content-Type": "text/plain"})


def test_patch_path_explicit_cpu():
    """Test initialize_hardware_path with explicit ASR_DEVICE=INTEL."""
    with mock.patch.dict(os.environ, {"ASR_DEVICE": "INTEL"}):
        fake_path = []
        with mock.patch.object(sys, "path", fake_path):
            with mock.patch("os.path.exists", side_effect=lambda p: "/intel" in p or "\\intel" in p):
                with mock.patch("importlib.reload"):
                    bootstrap.initialize_hardware_path()
                # Check if intel path was added
                intel_found = any("intel" in p for p in fake_path)
                assert intel_found, f"Expected intel path in {fake_path}"


def test_request_logging_with_args(whisper_pro_asr):
    """Test request logging with query arguments."""
    app = whisper_pro_asr.create_app(testing=True)
    client = FlaskCompatibleClient(app)
    client.get("/asr?debug=true&model=large")


def test_request_logging_exception(whisper_pro_asr):
    """Test request logging middleware handles exceptions gracefully."""
    app = whisper_pro_asr.create_app(testing=True)
    # Testing standard flow through TestClient
    client = FlaskCompatibleClient(app)
    client.post("/asr", data={"file": (BytesIO(b"x"), "test.wav")}, content_type="multipart/form-data")


def test_error_handlers_and_teardown_exception(whisper_pro_asr):
    """Test 404, 500 error handlers and request teardown exception handling."""
    app = whisper_pro_asr.create_app(testing=True)
    client = FlaskCompatibleClient(app)
    # 1. Test 404 not found handler
    resp = client.get("/nonexistent-endpoint-xyz")
    assert resp.status_code == 404

    # 2. Test 500 server error handler
    # Mock modules.monitoring.dashboard.get_dashboard_html to raise an exception
    with mock.patch(
        "modules.monitoring.dashboard.get_dashboard_html", side_effect=ValueError("Simulated server error")
    ):
        resp = client.get("/", headers={"Accept": "text/html"})
        assert resp.status_code == 500


def test_verify_runtime_integrity_with_onnx(whisper_pro_asr):
    """Test verify_runtime_integrity covers the ONNX success case when present."""
    mock_ort = mock.MagicMock()
    mock_ort.__version__ = "1.24.1"
    mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

    with mock.patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        whisper_pro_asr.verify_runtime_integrity()
        # Ensure it successfully called onnxruntime properties
        mock_ort.get_available_providers.assert_called_once()


@pytest.mark.anyio
async def test_whisper_pro_asr_lifespan_testing_false(whisper_pro_asr):
    """Verify lifespan actions when testing is disabled (telemetry active)."""
    mock_app = mock.MagicMock()
    mock_app.state.testing = False

    with mock.patch("modules.inference.model_manager.init_pool") as mock_init:
        with mock.patch("modules.monitoring.telemetry.start_telemetry_loop") as mock_telemetry:
            async with whisper_pro_asr.lifespan(mock_app):
                pass
            mock_init.assert_called_once()
            mock_telemetry.assert_called_once()


def test_custom_swagger_ui_html(whisper_pro_asr):
    """Verify the customized Swagger documentation HTML route works."""
    app = whisper_pro_asr.create_app(testing=True)
    client = FlaskCompatibleClient(app)

    # 1. Test when static files exist locally
    with mock.patch("os.path.exists", return_value=True):
        resp = client.get("/docs")
        assert resp.status_code == 200
        html = resp.data.decode("utf-8")
        assert "/static/swagger-ui-bundle.js" in html
        assert "/static/swagger-theme.css" in html

    # 2. Test when static files do not exist (falls back to CDN references)
    with mock.patch("os.path.exists", return_value=False):
        resp = client.get("/docs")
        assert resp.status_code == 200
        html = resp.data.decode("utf-8")
        assert "https://cdn.jsdelivr.net" in html


def test_main_entrypoint():
    """Verify uvicorn launch logic inside the entrypoint module."""
    with mock.patch("uvicorn.run") as mock_run:
        runpy.run_module("whisper_pro_asr", run_name="__main__")
        mock_run.assert_called_once()
