"""
Security and Access Control Utilities for Whisper Pro ASR
"""

import logging
import os
import re
import secrets
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# Standard known-safe Whisper models
STANDARD_MODEL_NAMES = {
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large",
    "large-v1",
    "large-v2",
    "large-v3",
    "large-v3-turbo",
    "turbo",
    "distil-large-v2",
    "distil-large-v3",
    "distil-medium.en",
    "distil-small.en",
    "openvino",
}

# Regex pattern for trusted HuggingFace Whisper model repositories
SAFE_HF_REPO_PATTERN = re.compile(
    r"^(Systran/faster-whisper-[a-zA-Z0-9_.-]+|"
    r"openai/whisper-[a-zA-Z0-9_.-]+|"
    r"guillaumekln/faster-whisper-[a-zA-Z0-9_.-]+|"
    r"deepdml/faster-whisper-[a-zA-Z0-9_.-]+)$",
    re.IGNORECASE,
)

# Supported hardware device strings
SUPPORTED_DEVICES = {"AUTO", "CUDA", "CPU", "NPU", "GPU", "AMD"}


def _is_wildcard_cors_enabled() -> bool:
    allow_all = os.environ.get("CORS_ALLOW_ALL", "false").strip().lower() in ("true", "1", "yes")
    cors_env = os.environ.get("CORS_ORIGINS", os.environ.get("CORS_ALLOWED_ORIGINS", "")).strip()
    return allow_all or cors_env == "*"


def get_cors_origins() -> list[str]:
    """
    Resolve configured CORS origins from environment variables.
    Defaults to empty list (disabling wildcard cross-origin access) unless explicitly configured.
    """
    if _is_wildcard_cors_enabled():
        logger.warning("[Security] CORS wildcard (*) is enabled. Sensitive endpoints may be accessible cross-origin.")
        return ["*"]

    cors_env = os.environ.get("CORS_ORIGINS", os.environ.get("CORS_ALLOWED_ORIGINS", "")).strip()
    if not cors_env:
        return []

    origins = [origin.strip() for origin in cors_env.split(",") if origin.strip()]
    logger.info("[Security] CORS allowed origins configured: %s", origins)
    return origins


def get_api_key() -> str:
    """Retrieve global API key from environment."""
    return os.environ.get("API_KEY", os.environ.get("WHISPER_API_KEY", "")).strip()


def get_admin_api_key() -> str:
    """Retrieve admin-level API key from environment."""
    admin_key = os.environ.get("ADMIN_API_KEY", "").strip()
    if admin_key:
        return admin_key
    return get_api_key()


def get_custom_allowed_models() -> set[str]:
    """Retrieve custom allowed model identifiers from environment."""
    custom_models_env = os.environ.get("ALLOWED_MODELS", "").strip()
    if not custom_models_env:
        return set()
    return {m.strip().lower() for m in custom_models_env.split(",") if m.strip()}


def _is_standard_or_custom_model(cleaned: str) -> bool:
    lowered = cleaned.lower()
    return lowered in STANDARD_MODEL_NAMES or lowered in get_custom_allowed_models()


def _clean_model_candidate(model_name: Optional[str]) -> Optional[str]:
    if not isinstance(model_name, str):
        return None
    cleaned = model_name.strip()
    if not cleaned or ".." in cleaned:
        return None
    return cleaned


def is_valid_model_name(model_name: Optional[str]) -> bool:
    """
    Validate that a requested model identifier is safe and permitted.
    Prevents model supply chain injection and unauthorized model downloads.
    """
    cleaned = _clean_model_candidate(model_name)
    if cleaned is None:
        return False

    if _is_standard_or_custom_model(cleaned):
        return True

    if bool(SAFE_HF_REPO_PATTERN.match(cleaned)):
        return True

    return _is_valid_local_model_path(cleaned)


def _is_valid_local_model_path(path: str) -> bool:
    """Check if path is an authorized local system model path."""
    allowed_roots = (
        Path("/app/system_models"),
        Path("/models"),
        Path("model_cache"),
        Path("./model_cache"),
    )

    try:
        candidate = Path(path).expanduser().resolve(strict=False)
    except tuple([OSError, RuntimeError]):
        candidate = Path(path).expanduser().absolute()

    for root in allowed_roots:
        try:
            root_resolved = root.resolve(strict=False)
        except tuple([OSError, RuntimeError]):
            root_resolved = root.absolute()

        try:
            # `relative_to` enforces commonpath containment, preventing symlink escapes.
            candidate.relative_to(root_resolved)
            return True
        except ValueError:
            continue

    return False


def is_valid_device(device_name: Optional[str]) -> bool:
    """Validate hardware device identifier."""
    if not device_name or not isinstance(device_name, str):
        return False
    return device_name.strip().upper() in SUPPORTED_DEVICES


def extract_auth_token(request: Request) -> str:
    """Extract auth token from X-API-Key or Authorization Bearer header."""
    header_key = request.headers.get("X-API-Key", "").strip()
    if header_key:
        return header_key

    auth_header = request.headers.get("Authorization", "").strip()
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()

    return ""


def verify_auth_token(provided_key: str, expected_key: str) -> bool:
    """Constant-time token verification to mitigate timing attacks."""
    if not expected_key:
        return True
    if not provided_key:
        return False
    return secrets.compare_digest(provided_key, expected_key)


def verify_api_request(request: Request) -> Optional[tuple[str, int]]:
    """
    Verify API request against API_KEY if configured.
    Returns None if authorized, or (error_message, status_code) on failure.
    """
    expected_key = get_api_key()
    if not expected_key:
        return None

    provided_key = extract_auth_token(request)
    if not verify_auth_token(provided_key, expected_key):
        return ("Unauthorized: Valid API key required", 401)

    return None


UNAUTHENTICATED_EXPOSURE_WARNING = (
    "[Security] No API_KEY/ADMIN_API_KEY configured. Management and transcription "
    "data are reachable without authentication on this process bind address. "
    "Set API_KEY before publishing the service beyond localhost."
)


def log_unauthenticated_exposure_warning() -> None:
    """Warn at startup when optional auth is unset (local-mode data exposure)."""
    if get_admin_api_key():
        return
    logger.warning(UNAUTHENTICATED_EXPOSURE_WARNING)


API_KEY_PROTECTED_PATHS = frozenset(
    {
        "/status",
        "/system/stats",
        "/history",
        "/system/history",
        "/analytics",
        "/system/analytics",
        "/asr",
        "/v1/audio/transcriptions",
        "/v1/audio/translations",
        "/detect-language",
        "/detectlang",
    }
)


def verify_api_path_request(request: Request) -> Optional[tuple[str, int]]:
    """Apply API_KEY checks only to documented general API surfaces."""
    if request.method == "OPTIONS" or request.url.path not in API_KEY_PROTECTED_PATHS:
        return None
    return verify_api_request(request)


def api_key_rejection_response(request: Request) -> Optional[JSONResponse]:
    """Return a 401 JSON response when a protected API path is unauthorized."""
    auth_err = verify_api_path_request(request)
    if not auth_err:
        return None
    message, status_code = auth_err
    return JSONResponse(content={"error": message}, status_code=status_code)


def verify_admin_request(request: Request) -> Optional[tuple[str, int]]:
    """
    Verify administrative request.
    If ADMIN_API_KEY is configured, checks token.
    If no key is configured, verifies anti-CSRF headers / origin.
    """
    admin_key = get_admin_api_key()
    if admin_key:
        provided_key = extract_auth_token(request)
        if not verify_auth_token(provided_key, admin_key):
            return ("Unauthorized: Valid Admin API key required", 401)
        return None

    return _verify_csrf_origin(request)


def _is_untrusted_header(header_val: str, req_host: str, allowed_origins: list[str], is_origin: bool) -> bool:
    if not header_val:
        return False
    checker = _is_origin_trusted if is_origin else _is_referer_trusted
    return not checker(header_val, req_host, allowed_origins)


def _check_csrf_header(header_val: str, header_type: str, req_host: str, allowed_origins: list[str]) -> Optional[tuple[str, int]]:
    is_origin = header_type == "origin"
    if _is_untrusted_header(header_val, req_host, allowed_origins, is_origin):
        logger.warning("[Security] Rejected cross-origin admin request from %s: %s", header_type, header_val)
        return ("Forbidden: Untrusted cross-origin request", 403)
    return None


def _verify_csrf_origin(request: Request) -> Optional[tuple[str, int]]:
    """
    Anti-CSRF origin verification for unauthenticated administrative endpoints.
    Protects against cross-origin state modification from untrusted websites.
    """
    origin = request.headers.get("origin", "").strip()
    referer = request.headers.get("referer", "").strip()

    if not origin and not referer:
        logger.warning("[Security] Rejected admin request with no Origin or Referer")
        return ("Forbidden: Origin or Referer header required", 403)

    allowed_origins = get_cors_origins()
    req_host = request.headers.get("host", "").strip().lower()
    return _check_csrf_header(origin, "origin", req_host, allowed_origins) or _check_csrf_header(
        referer, "referer", req_host, allowed_origins
    )


def _is_origin_trusted(origin: str, req_host: str, allowed_origins: list[str]) -> bool:
    """Check if origin matches request host or explicit CORS allowlist."""
    if origin in allowed_origins:
        return True
    try:
        parsed = urlparse(origin)
        return bool(parsed.netloc.lower() == req_host.lower())
    except (ValueError, AttributeError):
        return False


def _is_referer_trusted(referer: str, req_host: str, allowed_origins: list[str]) -> bool:
    """Check if referer matches request host or explicit CORS allowlist."""
    try:
        parsed = urlparse(referer)
        origin_from_ref = f"{parsed.scheme}://{parsed.netloc}"
        if origin_from_ref in allowed_origins:
            return True
        return bool(parsed.netloc.lower() == req_host.lower())
    except (ValueError, AttributeError):
        return False


def audit_log_admin_action(request: Request, action: str, details: str = "") -> None:
    """Log an administrative action for security auditing."""
    peer_ip = request.client.host if request.client else "unknown"
    forwarded_for = request.headers.get("X-Forwarded-For", "").strip() or "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    details_str = f" | {details}" if details else ""
    logger.info(
        "[Security Audit] Admin action: '%s' | ClientPeer: %s | X-Forwarded-For: %s | UA: %s%s",
        action,
        peer_ip,
        forwarded_for,
        user_agent,
        details_str,
    )
