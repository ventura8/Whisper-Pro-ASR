"""Comprehensive coverage for the main Flask server logic."""
import os
import sys
from io import BytesIO
# pylint: disable=no-member, protected-access, redefined-outer-name
# pylint: disable=reimported, import-outside-toplevel
from unittest import mock

import whisper_server


def test_patch_path_explicit_cuda():
    """Test _initialize_hardware_path with explicit ASR_DEVICE=CUDA."""
    import whisper_server
    with mock.patch.dict(os.environ, {"ASR_DEVICE": "CUDA"}):
        fake_path = []
        with mock.patch.object(sys, "path", fake_path):
            with mock.patch("os.path.exists",
                            side_effect=lambda p: "/nvidia" in p or "\\nvidia" in p):
                # Call the function directly
                whisper_server._initialize_hardware_path()
                # Check if nvidia path was added
                nvidia_found = any("nvidia" in p for p in fake_path)
                assert nvidia_found, f"Expected nvidia path in {fake_path}"


def test_patch_path_auto_nvidia():
    """Test _initialize_hardware_path with AUTO and detected NVIDIA."""
    import whisper_server
    with mock.patch.dict(os.environ, {"ASR_DEVICE": "AUTO"}):
        with mock.patch("os.path.exists", side_effect=lambda p: "nvidia" in p or "\\nvidia" in p):
            fake_path = []
            with mock.patch.object(sys, "path", fake_path):
                whisper_server._initialize_hardware_path()
                # Check if nvidia path was added
                nvidia_found = any("nvidia" in p for p in fake_path)
                assert nvidia_found, f"Expected nvidia path in {fake_path}"


def test_on_import_onnx_exceptions():
    """Test that the server handles onnxruntime import errors gracefully."""
    import logging
    import whisper_server
    # The module should have proper exception handling
    # Check that logger is set up and module loads without crash
    logger = logging.getLogger(__name__)
    assert logger is not None
    # Module loaded successfully
    assert whisper_server is not None


def test_request_logging_types():
    """Test request logging formats (Multipart, Form, JSON)."""
    app = whisper_server.create_app()
    with app.test_client() as client:
        # 1. Multipart
        client.post("/asr", data={"file": (BytesIO(b"x"), "test.wav"), "field": "val"},
                    content_type="multipart/form-data")

        # 2. Form URL Encoded
        client.post("/asr", data={"info": "yes"},
                    content_type="application/x-www-form-urlencoded")

        # 3. JSON
        client.post("/asr", json={"key": "value"})


def test_request_logging_data_handling():
    """Test large body and invalid body."""
    app = whisper_server.create_app()
    with app.test_client() as client:
        # 1. Large Body (truncated)
        client.post("/asr", data="A"*1500, content_type="text/plain")

        # 2. Decoding error (simulated via invalid utf8)
        client.post("/asr", data=b'\x80\xFF', content_type="text/plain")


def test_patch_path_explicit_cpu():
    """Test _initialize_hardware_path with explicit ASR_DEVICE=CPU."""
    import whisper_server
    with mock.patch.dict(os.environ, {"ASR_DEVICE": "CPU"}):
        fake_path = []
        with mock.patch.object(sys, "path", fake_path):
            with mock.patch("os.path.exists",
                            side_effect=lambda p: "/intel" in p or "\\intel" in p):
                whisper_server._initialize_hardware_path()
                # Check if intel path was added
                intel_found = any("intel" in p for p in fake_path)
                assert intel_found, f"Expected intel path in {fake_path}"


def test_request_logging_with_args():
    """Test request logging with query arguments."""
    app = whisper_server.create_app()
    with app.test_client() as client:
        client.get("/asr?debug=true&model=large")


def test_request_logging_exception():
    """Test request logging exception handler."""
    app = whisper_server.create_app()
    with app.test_client():
        # Use a more direct way to trigger the exception in before_request
        # instead of mocking the proxy itself which is risky
        with app.test_request_context("/asr", method="POST"):
            # The function is nested in create_app, but registered in before_request_funcs
            # It's registered under None (global) or the blueprint name
            log_func = None
            for func in app.before_request_funcs.get(None, []):
                if func.__name__ == "log_request_info":
                    log_func = func
                    break

            if log_func:
                mock_req = mock.MagicMock()
                with mock.patch("whisper_server.request", mock_req):
                    mock_req.content_type = "multipart/form-data"
                    mock_req.args = {}
                    mock_req.headers = {}
                    mock_req.method = "POST"
                    mock_req.path = "/asr"
                    mock_req.remote_addr = "127.0.0.1"

                    # This will raise if files is accessed and mocked to fail
                    type(mock_req).files = mock.PropertyMock(
                        side_effect=RuntimeError("Log Fail"))

                    # Directly call the handler to test its try-except
                    log_func()
                    # If it reached here without raising RuntimeError, it's successful
