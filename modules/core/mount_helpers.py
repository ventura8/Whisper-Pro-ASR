"""Filesystem mount discovery and writable directory resolution helpers."""

import logging
import os
import re
import shutil
import tempfile

logger = logging.getLogger(__name__)


def is_path_writable(path: str) -> bool:
    """Return True when a directory allows create+delete operations."""
    try:
        with tempfile.TemporaryFile(dir=path):
            pass
        return True
    except (PermissionError, OSError):
        return False


def _try_candidate_dir(candidate: str) -> bool:
    if not candidate:
        return False
    try:
        os.makedirs(candidate, exist_ok=True)
    except (PermissionError, OSError):
        return False
    return is_path_writable(candidate)


def _ensure_fallback_writable(label: str, fallback: str) -> str:
    try:
        os.makedirs(fallback, exist_ok=True)
    except (PermissionError, OSError) as exc:
        raise RuntimeError(f"[Config] {label} fallback directory {fallback} could not be created: {exc}") from exc

    if not is_path_writable(fallback):
        raise RuntimeError(f"[Config] {label} fallback directory {fallback} is not writable.")

    logger.warning("[Config] %s directory is not writable. Falling back to %s", label, fallback)
    return fallback


def resolve_writable_dir(label: str, candidates: list[str], fallback: str) -> str:
    """Pick the first writable directory from candidates, else use fallback."""
    for candidate in candidates:
        if _try_candidate_dir(candidate):
            return candidate

    return _ensure_fallback_writable(label, fallback)


def get_custom_mount_points() -> list[str]:
    """Discover custom mount points from /proc/mounts to automatically approve volumes."""
    if not os.path.exists("/proc/mounts"):
        return []
    try:
        system_roots = _system_mount_roots()
        return _read_custom_mount_points(system_roots)
    except (FileNotFoundError, PermissionError, OSError, ValueError, IndexError):
        return []


def _system_mount_roots() -> set[str]:
    return {
        "/",
        "/proc",
        "/sys",
        "/dev",
        "/run",
        "/boot",
        "/lib",
        "/lib64",
        "/bin",
        "/sbin",
        "/usr",
        "/var",
        "/etc",
        "/root",
        "/home",
        tempfile.gettempdir(),
        "/sys/firmware",
    }


def _read_custom_mount_points(system_roots: set[str]) -> list[str]:
    mounts = []
    with open("/proc/mounts", "r", encoding="utf-8") as f:
        for line in f:
            mount_point = _extract_mount_point(line)
            if mount_point and _is_custom_mount_point(mount_point, system_roots):
                mounts.append(mount_point)
    return mounts


def _extract_mount_point(line: str) -> str | None:
    parts = line.split()
    if len(parts) >= 2:
        raw = parts[1]
        # Decode octal escapes written by the kernel (e.g. \040 for space)
        return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), raw)
    return None


def _is_custom_mount_point(mount_point: str, system_roots: set[str]) -> bool:
    if mount_point in system_roots:
        return False
    if any(mount_point.startswith(root + "/") for root in system_roots):
        return False
    if mount_point.endswith(("/hosts", "/hostname", "/resolv.conf")):
        return False
    return True


def resolve_temp_dir(temp_dir: str, persistent_dir: str, min_free_bytes: int, required_bytes: int = 0) -> str:
    """Return the best temp directory based on available tmpfs disk space."""
    resolved_temp = os.path.abspath(temp_dir)
    resolved_persistent = os.path.abspath(persistent_dir)
    headroom_bytes = int(required_bytes * 1.5) if required_bytes > 0 else 0
    threshold = max(min_free_bytes, headroom_bytes)
    try:
        free = shutil.disk_usage(resolved_temp).free
        if free < threshold:
            logger.debug(
                "[Config] tmpfs free space (%d MB) below threshold (%d MB) — falling back to persistent temp dir.",
                free // (1024 * 1024),
                threshold // (1024 * 1024),
            )
            persistent_free = shutil.disk_usage(resolved_persistent).free
            if persistent_free < threshold:
                # Warn, do not raise. This function only *selects* a directory, and it is
                # called while resolving config on the request path -- raising here turned
                # a capacity warning into a hard failure for every request. The threshold
                # is also max(min_free, 1.5x required): it is desired headroom, not the
                # space the work actually needs, so being under it does not mean the write
                # would fail. Let the write fail with a real ENOSPC if it truly cannot fit,
                # which says what was actually being written.
                logger.warning(
                    "[Config] Neither temp directory has the preferred %d MB free (tmpfs %d MB, persistent %d MB); "
                    "using the persistent directory anyway.",
                    threshold // (1024 * 1024),
                    free // (1024 * 1024),
                    persistent_free // (1024 * 1024),
                )
            return resolved_persistent
    except OSError:
        return resolved_persistent
    return resolved_temp
