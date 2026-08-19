"""Helpers for validating and logging approved local media paths."""

import logging
import os
from typing import Optional

from modules.core import config, utils

_MEDIA_PATH_EXTENSIONS = (
    ".mkv",
    ".mp4",
    ".avi",
    ".wav",
    ".m4a",
    ".mp3",
    ".flac",
    ".ts",
    ".mov",
    ".webm",
    ".wmv",
    ".mpg",
    ".mpeg",
)


def get_approved_roots() -> list[str]:
    """Return canonical roots allowed for local-path optimization."""

    roots = [
        os.path.realpath(config.TEMP_DIR),
        os.path.realpath(config.PERSISTENT_DIR),
        os.path.realpath(os.getcwd()),
    ]
    for root in config.APPROVED_ROOTS:
        roots.append(os.path.realpath(root))
    return roots


def is_path_approved(normalized_path: str, approved_roots: list[str]) -> bool:
    """Return True when path is equal to or nested under an approved root."""

    for root in approved_roots:
        if normalized_path == root or normalized_path.startswith(os.path.join(root, "")):
            return True
    return False


def log_local_path_optimization(logger: logging.Logger, normalized_path: str):
    """Emit optimization log once per request for the resolved local path."""

    already_logged = getattr(utils.THREAD_CONTEXT, "optimized_local_path_logged", None)
    if already_logged != normalized_path:
        logger.info("[System] Optimization: Using Local Path -> %s", normalized_path)
        utils.THREAD_CONTEXT.optimized_local_path_logged = normalized_path


def looks_like_media_path(value: str) -> bool:
    """Return True when a string looks like an absolute media file path."""
    clean = value.strip().strip('"').strip("'")
    if not clean.startswith("/"):
        return False
    lower = clean.lower()
    return any(lower.endswith(ext) for ext in _MEDIA_PATH_EXTENSIONS)


def extract_path_from_mapping_keys(data: dict) -> Optional[str]:
    """Recover Bazarr JSON bodies that encode the media path as the object key."""
    if not isinstance(data, dict):
        return None
    for key in data:
        if isinstance(key, str) and looks_like_media_path(key):
            return key.strip().strip('"').strip("'")
    return None


def _strip_media_path_keys(params: dict) -> dict:
    normalized = {}
    for key, value in params.items():
        if isinstance(key, str) and looks_like_media_path(key):
            continue
        normalized[key] = value
    return normalized


def normalize_bazarr_request_params(params: dict) -> dict:
    """Promote path-as-key Bazarr payloads to local_path and drop duplicate keys."""
    if not isinstance(params, dict):
        return {}
    path = extract_path_from_mapping_keys(params)
    normalized = _strip_media_path_keys(params)
    if path:
        normalized["local_path"] = path
    return normalized
