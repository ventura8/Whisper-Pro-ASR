"""
Unit tests for security, CORS resolution, model allowlist, and access control.
"""

from pathlib import Path
from unittest import mock

import pytest
from fastapi import Request

from modules.api.support import security


def _create_mock_request(
    headers: dict[str, str] | None = None, client_host: str = "127.0.0.1", path: str = "/settings", method: str = "POST"
) -> Request:
    """Helper to create mock Request with headers and client info."""
    raw_headers = []
    if headers:
        for k, v in headers.items():
            raw_headers.append((k.lower().encode("latin-1"), v.encode("latin-1")))

    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "headers": raw_headers,
        "client": (client_host, 12345),
    }
    return Request(scope)


def test_cors_origins_default_empty():
    """Verify default CORS origins is empty when unconfigured."""
    with mock.patch.dict("os.environ", {}, clear=True):
        origins = security.get_cors_origins()
        assert origins == []


def test_cors_origins_allow_all():
    """Verify CORS wildcard when CORS_ALLOW_ALL is set."""
    with mock.patch.dict("os.environ", {"CORS_ALLOW_ALL": "true"}, clear=True):
        origins = security.get_cors_origins()
        assert origins == ["*"]


def test_cors_origins_explicit_list():
    """Verify parsing of comma-separated CORS origins."""
    with mock.patch.dict(
        "os.environ",
        {"CORS_ORIGINS": "http://localhost:3000, https://dashboard.local"},
        clear=True,
    ):
        origins = security.get_cors_origins()
        assert origins == ["http://localhost:3000", "https://dashboard.local"]


@pytest.mark.parametrize(
    "model_name",
    [
        "small",
        "base.en",
        "large-v3-turbo",
        "OpenVINO",
        "Systran/faster-whisper-large-v3",
        "openai/whisper-large-v3",
    ],
)
def test_is_valid_model_name_standard(model_name: str):
    """Verify standard Whisper model identifiers are allowed."""
    assert security.is_valid_model_name(model_name) is True


def test_is_valid_model_name_local_paths():
    """Verify approved local model directories are allowed."""
    assert security.is_valid_model_name("/app/system_models/whisper") is True
    assert security.is_valid_model_name("/models/whisper-openvino") is True
    assert security.is_valid_model_name("./model_cache/custom") is True


def test_is_valid_model_name_malicious_rejected():
    """Verify arbitrary external repo injection or path traversals are rejected."""
    assert security.is_valid_model_name("attacker/malicious-model") is False
    assert security.is_valid_model_name("../../etc/passwd") is False
    assert security.is_valid_model_name("") is False
    assert security.is_valid_model_name(None) is False


def test_is_valid_model_name_custom_allowlist():
    """Verify custom ALLOWED_MODELS env var allows additional models."""
    with mock.patch.dict("os.environ", {"ALLOWED_MODELS": "custom/enterprise-whisper,my-model"}, clear=True):
        assert security.is_valid_model_name("custom/enterprise-whisper") is True
        assert security.is_valid_model_name("my-model") is True
        assert security.is_valid_model_name("unauthorized/repo") is False


@pytest.mark.parametrize(
    ("device_name", "expected"),
    [
        ("CUDA", True),
        ("cpu", True),
        ("AUTO", True),
        ("npu", True),
        ("AMD", True),
        ("GPU", True),
        ("invalid_device", False),
        ("", False),
    ],
)
def test_is_valid_device(device_name: str, expected: bool):
    """Verify device validation against supported hardware."""
    assert security.is_valid_device(device_name) is expected


def test_auth_token_extraction():
    """Verify extraction of API key from X-API-Key and Authorization Bearer."""
    req1 = _create_mock_request({"X-API-Key": "secret123"})
    assert security.extract_auth_token(req1) == "secret123"

    req2 = _create_mock_request({"Authorization": "Bearer token456"})
    assert security.extract_auth_token(req2) == "token456"

    req3 = _create_mock_request({})
    assert security.extract_auth_token(req3) == ""


def test_verify_admin_request_valid_key():
    """Verify admin auth success when correct key is provided."""
    with mock.patch.dict("os.environ", {"ADMIN_API_KEY": "admin_pass"}, clear=True):
        valid_req = _create_mock_request({"X-API-Key": "admin_pass"})
        assert security.verify_admin_request(valid_req) is None


def test_verify_admin_request_invalid_key():
    """Verify admin auth failure when invalid key is provided."""
    with mock.patch.dict("os.environ", {"ADMIN_API_KEY": "admin_pass"}, clear=True):
        invalid_req = _create_mock_request({"X-API-Key": "wrong_pass"})
        err = security.verify_admin_request(invalid_req)
        assert err is not None
        assert err[1] == 401


def test_verify_admin_request_missing_key():
    """Verify admin auth failure when key is omitted."""
    with mock.patch.dict("os.environ", {"ADMIN_API_KEY": "admin_pass"}, clear=True):
        missing_req = _create_mock_request({})
        err = security.verify_admin_request(missing_req)
        assert err is not None
        assert err[1] == 401


def test_verify_admin_request_unauthenticated_mode_csrf():
    """Verify anti-CSRF checks in unauthenticated mode."""
    with mock.patch.dict("os.environ", {}, clear=True):
        # Trusted local origin
        trusted_req = _create_mock_request({"Origin": "http://localhost:9000", "Host": "localhost:9000"})
        assert security.verify_admin_request(trusted_req) is None

        # Untrusted cross-origin request
        untrusted_req = _create_mock_request({"Origin": "https://malicious-site.com", "Host": "service.internal:9000"})
        err = security.verify_admin_request(untrusted_req)
        assert err is not None
        assert err[1] == 403


@pytest.mark.parametrize(
    ("provided", "expected", "result"),
    [
        ("", "", True),
        ("key", "", True),
        ("", "expected", False),
        ("wrong", "expected", False),
        ("expected", "expected", True),
    ],
)
def test_verify_auth_token_branches(provided: str, expected: str, result: bool):
    """Verify constant time token verification branches."""
    assert security.verify_auth_token(provided, expected) is result


def test_verify_api_request_no_key():
    """Verify unauthenticated API request succeeds when API_KEY is unset."""
    with mock.patch.dict("os.environ", {}, clear=True):
        req = _create_mock_request({})
        assert security.verify_api_request(req) is None


def test_verify_api_request_with_key():
    """Verify API request auth when API_KEY is set."""
    with mock.patch.dict("os.environ", {"API_KEY": "test_key"}, clear=True):
        # Valid key via X-API-Key
        valid_req = _create_mock_request({"X-API-Key": "test_key"})
        assert security.verify_api_request(valid_req) is None

        # Valid key via Authorization Bearer
        bearer_req = _create_mock_request({"Authorization": "Bearer test_key"})
        assert security.verify_api_request(bearer_req) is None

        # Invalid key
        invalid_req = _create_mock_request({"X-API-Key": "test_wrong"})
        err = security.verify_api_request(invalid_req)
        assert err is not None
        assert err[1] == 401


def test_verify_csrf_origin_wildcard_cors():
    """Wildcard CORS must not skip CSRF checks on administrative routes."""
    with mock.patch.dict("os.environ", {"CORS_ALLOW_ALL": "true"}, clear=True):
        req = _create_mock_request({"Origin": "https://anywhere.com", "Host": "service.internal:9000"})
        err = security.verify_admin_request(req)
        assert err is not None
        assert err[1] == 403

        same_host = _create_mock_request({"Origin": "http://service.internal:9000", "Host": "service.internal:9000"})
        assert security.verify_admin_request(same_host) is None


def test_verify_csrf_referer_branches():
    """Verify referer validation against trusted/untrusted origins."""
    with mock.patch.dict("os.environ", {"CORS_ORIGINS": "https://allowed.domain.com"}, clear=True):
        # Referer in allowed_origins
        req_allowed = _create_mock_request({"Referer": "https://allowed.domain.com/page", "Host": "internal:9000"})
        assert security.verify_admin_request(req_allowed) is None

        # Untrusted referer
        req_bad = _create_mock_request({"Referer": "https://evil.com/phish", "Host": "internal:9000"})
        err = security.verify_admin_request(req_bad)
        assert err is not None
        assert err[1] == 403


def test_verify_csrf_localhost_requires_matching_host_port():
    """Unauthenticated admin CSRF must not trust arbitrary localhost ports."""
    with mock.patch.dict("os.environ", {}, clear=True):
        trusted = _create_mock_request({"Origin": "http://localhost:9000", "Host": "localhost:9000"})
        assert security.verify_admin_request(trusted) is None
        untrusted = _create_mock_request({"Origin": "http://localhost:9999", "Host": "localhost:9000"})
        err = security.verify_admin_request(untrusted)
        assert err is not None
        assert err[1] == 403


def test_verify_api_path_request_protects_documented_surfaces():
    """API_KEY must be enforced on documented general API paths."""
    with mock.patch.dict("os.environ", {"API_KEY": "test_key"}, clear=True):
        missing = _create_mock_request({}, path="/asr")
        err = security.verify_api_path_request(missing)
        assert err is not None
        assert err[1] == 401

        valid = _create_mock_request({"X-API-Key": "test_key"}, path="/status", method="GET")
        assert security.verify_api_path_request(valid) is None

        public = _create_mock_request({}, path="/docs", method="GET")
        assert security.verify_api_path_request(public) is None


def _logged_audit_text(mock_info: mock.MagicMock) -> str:
    return " ".join(str(part) for part in mock_info.call_args[0])


def test_audit_log_admin_action_includes_client_metadata():
    """Audit logs should include peer IP, forwarded IP, user agent, and details."""
    req = _create_mock_request({"X-Forwarded-For": "10.0.0.1", "User-Agent": "SecurityTest/1.0"})
    with mock.patch.object(security.logger, "info") as mock_info:
        security.audit_log_admin_action(req, "test_action", "test_details")
    mock_info.assert_called_once()
    logged_text = _logged_audit_text(mock_info)
    for token in ("test_action", "127.0.0.1", "10.0.0.1", "SecurityTest/1.0", "test_details"):
        assert token in logged_text


def test_audit_log_admin_action_defaults_when_headers_missing():
    """Audit logs should fall back to loopback peer IP, unknown forwarded IP, and unknown user agent."""
    req = _create_mock_request({})
    with mock.patch.object(security.logger, "info") as mock_info:
        security.audit_log_admin_action(req, "simple_action")
    mock_info.assert_called_once()
    logged = mock_info.call_args[0]
    assert "simple_action" in logged
    assert "127.0.0.1" in logged
    assert "unknown" in logged


def test_verify_csrf_origin_requires_origin_or_referer():
    """Unauthenticated admin requests must include Origin or Referer."""
    with mock.patch.dict("os.environ", {}, clear=True):
        missing = _create_mock_request({"Host": "localhost:9000"})
        err = security.verify_admin_request(missing)
        assert err is not None
        assert err[1] == 403
        assert "Origin or Referer header required" in err[0]


def test_log_unauthenticated_exposure_warning_when_keys_unset():
    """Local-mode startup must warn that management data is unauthenticated."""
    with mock.patch.dict("os.environ", {}, clear=True):
        with mock.patch.object(security.logger, "warning") as mock_warn:
            security.log_unauthenticated_exposure_warning()
    mock_warn.assert_called_once_with(security.UNAUTHENTICATED_EXPOSURE_WARNING)


def test_log_unauthenticated_exposure_warning_silent_when_api_key_set():
    """Configured API_KEY must suppress the unauthenticated-exposure warning."""
    with mock.patch.dict("os.environ", {"API_KEY": "secret"}, clear=True):
        with mock.patch.object(security.logger, "warning") as mock_warn:
            security.log_unauthenticated_exposure_warning()
    mock_warn.assert_not_called()


def test_log_unauthenticated_exposure_warning_silent_when_admin_key_set():
    """A dedicated ADMIN_API_KEY also means authentication is configured."""
    with mock.patch.dict("os.environ", {"ADMIN_API_KEY": "admin-secret"}, clear=True):
        with mock.patch.object(security.logger, "warning") as mock_warn:
            security.log_unauthenticated_exposure_warning()
    mock_warn.assert_not_called()


def test_compose_publishes_localhost_only_by_default():
    """Default compose host publish must not expose unauthenticated data on the LAN."""
    compose_path = Path(__file__).resolve().parents[2] / "docker-compose.yml"
    text = compose_path.read_text(encoding="utf-8")
    assert '"127.0.0.1:9000:9000"' in text
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert stripped != '- "9000:9000"'
