"""Runtime directory resolution for Whisper Pro ASR.

Every writable location the service needs -- the model cache, transient audio, persistent
state and logs -- is resolved here, each with a fallback for the case where the configured
path is not writable. Split out of ``config.py`` to keep that module inside the project's
module-length limit.

Resolution runs on demand rather than at this module's import, because ``config`` is
reloaded in tests to pick up a changed environment; a value computed once at import here
would survive that reload and report the previous run's directories.
"""

from __future__ import annotations

import logging
import math
import os
import tempfile

from modules.core.mount_helpers import get_custom_mount_points, resolve_writable_dir

logger = logging.getLogger(__name__)

#: Default cache location, relative to the working directory the container starts in.
LOCAL_CACHE = "./model_cache"

#: Free space the temp directory must keep, when the environment does not say otherwise.
DEFAULT_TEMP_MIN_FREE_MB = 2048


def float_env(name: str, default: float, *, minimum: float | None = None, maximum: float | None = None) -> float:
    """Return an environment value as a float, falling back on anything unusable.

    Same reasoning as :func:`_temp_min_free_mb`: these are read while ``config`` is being
    imported, so a ValueError does not surface as a bad setting -- it aborts startup before
    any log line explains which variable was at fault. An empty ``VAD_THRESHOLD=`` in a .env
    file is an ordinary way to write "leave it at the default", and must behave like one.

    ``minimum``/``maximum`` extend that to values that parse but cannot mean anything. A
    Silero VAD threshold is a probability: ``VAD_THRESHOLD=50`` (a plausible way to write
    "50%") parses fine and then rejects every frame as non-speech, so the service returns
    empty transcripts for valid audio with nothing in the log to say why. NaN is rejected
    for the same reason -- every comparison against it is False, which reads as silence too.
    """
    raw = os.environ.get(name, "")
    if not str(raw).strip():
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning("[Config] %s=%r is not a number; using the default %s.", name, raw, default)
        return default
    if not _within_range(value, minimum, maximum):
        logger.warning("[Config] %s=%r is outside the usable range [%s, %s]; using the default %s.", name, raw, minimum, maximum, default)
        return default
    return value


def _within_range(value: float, minimum: float | None, maximum: float | None) -> bool:
    """Whether a parsed number is finite and inside whichever bounds were given."""
    if not math.isfinite(value):
        return False
    if minimum is not None and value < minimum:
        return False
    return maximum is None or value <= maximum


def _temp_min_free_mb() -> int:
    """Megabytes of headroom required in the temp directory, from the environment.

    A malformed value falls back to the default rather than raising. This is read during
    ``config`` import, so a ValueError here aborted startup outright -- and an empty
    ``WHISPER_TEMP_MIN_FREE_MB=`` in a .env file is enough to produce one, which is a
    thoroughly ordinary way to write "leave it at the default".
    """
    raw = os.environ.get("WHISPER_TEMP_MIN_FREE_MB", "")
    if not raw.strip():
        return DEFAULT_TEMP_MIN_FREE_MB
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning("WHISPER_TEMP_MIN_FREE_MB=%r is not a number; using the default of %d MB.", raw, DEFAULT_TEMP_MIN_FREE_MB)
        return DEFAULT_TEMP_MIN_FREE_MB


def _state_dir_candidates(persistent_dir: str) -> list[str]:
    """Return the ordered candidates for the state directory."""
    state_dir_env = os.environ.get("WHISPER_STATE_DIR")
    if state_dir_env:
        return [state_dir_env, persistent_dir, "./test_state"]
    return [persistent_dir, "./test_state"]


def _persistent_temp_dir(ov_cache_dir: str) -> str:
    path = os.path.abspath(os.path.join(ov_cache_dir, "temp"))
    try:
        os.makedirs(path, exist_ok=True)
    except (PermissionError, OSError):
        return os.path.abspath(tempfile.gettempdir())
    return path


def _temp_dir() -> str:
    path = os.environ.get("WHISPER_TEMP_DIR", tempfile.gettempdir())
    try:
        os.makedirs(path, exist_ok=True)
    except (PermissionError, OSError):
        return tempfile.gettempdir()
    return path


def resolve_runtime_dirs() -> dict[str, object]:
    """Resolve every runtime directory, returning them for the caller to publish.

    Returned rather than assigned so ``config`` keeps ownership of its own public names,
    and so a reload recomputes all of them together.
    """
    runtime_fallback_root = os.path.join(tempfile.gettempdir(), "whisper-runtime")
    ov_cache_dir = resolve_writable_dir(
        "OV cache",
        [os.environ.get("OV_CACHE_DIR", LOCAL_CACHE), LOCAL_CACHE],
        os.path.join(runtime_fallback_root, "model_cache"),
    )

    # Falls back inside model_cache (a persistent bind mount) rather than tmpfs, so task
    # history survives container recreation even when /app/data is unwritable.
    persistent_fallback = os.path.abspath(os.path.join(ov_cache_dir, ".state"))
    persistent_dir = resolve_writable_dir(
        "Persistent state",
        [os.environ.get("WHISPER_PERSISTENT_DIR", "/app/data")],
        persistent_fallback,
    )
    state_dir = resolve_writable_dir("State", _state_dir_candidates(persistent_dir), persistent_fallback)
    log_dir = resolve_writable_dir(
        "Log",
        [os.environ.get("WHISPER_LOG_DIR", state_dir), state_dir],
        os.path.join(runtime_fallback_root, "logs"),
    )

    approved_roots = [p.strip() for p in os.environ.get("WHISPER_APPROVED_ROOTS", "").split(",") if p.strip()]
    approved_roots.extend(_auto_approved_mount_roots())

    return {
        "RUNTIME_FALLBACK_ROOT": runtime_fallback_root,
        "OV_CACHE_DIR": ov_cache_dir,
        "PERSISTENT_FALLBACK": persistent_fallback,
        "PERSISTENT_DIR": persistent_dir,
        "STATE_DIR": state_dir,
        "LOG_DIR": log_dir,
        "APPROVED_ROOTS": approved_roots,
        "TEMP_DIR": _temp_dir(),
        "TEMP_DIR_MIN_FREE_BYTES": _temp_min_free_mb() * 1024 * 1024,
        "PERSISTENT_TEMP_DIR": _persistent_temp_dir(ov_cache_dir),
    }


def _auto_approved_mount_roots() -> list[str]:
    """Bind mounts to admit as approved roots, and a log line saying which.

    APPROVED_ROOTS decides which caller-supplied ``local_path`` values the API will read from
    disk, so extending it with every bind mount grants request-driven read access to whatever
    the operator happened to mount -- including directories mounted for an unrelated reason.
    Mounting a media library so Bazarr can send paths into it is the intended case and the
    reason this defaults to on; mounting a config directory is not, and previously there was
    no way to tell the two apart and no record that the grant had happened at all.

    ``WHISPER_AUTO_APPROVE_MOUNTS=false`` turns it off, leaving WHISPER_APPROVED_ROOTS as the
    only source. The default stays on because flipping it silently breaks every deployment
    that relies on local-path optimisation, which is most of them.
    """
    if os.environ.get("WHISPER_AUTO_APPROVE_MOUNTS", "true").strip().lower() in ("0", "false", "no", "off"):
        return []
    roots = get_custom_mount_points()
    if roots:
        # Logged rather than assumed: an implicit authorisation nobody can see is one nobody
        # can review, and this is the line that makes it reviewable.
        logger.info(
            "[Config] Auto-approved %d bind mount(s) for local-path requests: %s "
            "(set WHISPER_AUTO_APPROVE_MOUNTS=false to require WHISPER_APPROVED_ROOTS instead)",
            len(roots),
            ", ".join(roots),
        )
    return roots


def preprocessing_cache_dir(base: str) -> str:
    """Return a writable ``preprocessing`` directory under ``base``, or under the temp dir."""
    path = os.path.join(base, "preprocessing")
    try:
        os.makedirs(path, exist_ok=True)
        return path
    except (PermissionError, OSError):
        path = os.path.join(tempfile.gettempdir(), "preprocessing")
        try:
            os.makedirs(path, exist_ok=True)
        except OSError:
            pass
        return path
