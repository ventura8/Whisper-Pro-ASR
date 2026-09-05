"""Source-path resolution and display-name helpers for API routes.

Split out of request_utils.py to stay under the 500-line Python file limit.
Contains: approved-roots-gated local path resolution, the synchronous upload
handler (used by prepare_source_path's fallback path), and the dashboard
display-name resolution logic (local_path / video_file / uploaded audio_file
priority chain). request_utils.py imports from here; nothing in this module
imports back from request_utils.py, avoiding a circular dependency.
"""

import logging
import os
import uuid
from typing import Optional

from modules.api.support.local_path import (
    get_approved_roots,
    is_path_approved,
    log_local_path_optimization,
)
from modules.core import config, utils

logger = logging.getLogger(__name__)


def _remove_path_if_exists(path: str):
    if os.path.exists(path):
        os.remove(path)


def _ensure_non_empty_file(tmp_path: str) -> int:
    file_size = os.path.getsize(tmp_path)
    if file_size == 0:
        raise ValueError("Remote data stream is empty (0 bytes received).")
    return file_size


def resolve_local_path(raw_path):
    """Check if the provided path exists locally."""
    clean_path = raw_path.strip().strip('"').strip("'")
    candidates = [clean_path, clean_path.replace("+", " ")]

    approved_roots = get_approved_roots()

    for p in candidates:
        if not p:
            continue
        normalized_p = os.path.realpath(p)
        if not is_path_approved(normalized_p, approved_roots):
            logger.warning("[System] Path not in approved roots (volume not mounted?): %s", p)
            return None

        if os.path.exists(normalized_p):
            log_local_path_optimization(logger, normalized_p)
            return normalized_p
    return None


def shutil_copy_file_in_chunks(src, dst):
    """Helper to copy file stream in chunks to avoid high RAM spikes."""
    while True:
        chunk = src.read(1024 * 1024)  # 1MB chunk
        if not chunk:
            break
        dst.write(chunk)


def _write_upload_sync(audio_file, tmp_path: str):
    if hasattr(audio_file, "file") and audio_file.file:
        try:
            audio_file.file.seek(0)
        except tuple([Exception]):
            pass
        with open(tmp_path, "wb") as f:
            shutil_copy_file_in_chunks(audio_file.file, f)
    else:
        content = audio_file.read() if hasattr(audio_file, "read") else audio_file
        with open(tmp_path, "wb") as f:
            f.write(content)


def _validate_upload_sync(tmp_path: str):
    _ensure_non_empty_file(tmp_path)


def _track_successful_upload(tmp_path: str):
    file_size = os.path.getsize(tmp_path)
    utils.track_file(tmp_path)
    logger.info("[System] Remote source ingestion successful: %d bytes", file_size)


def _cleanup_temp_upload_on_error(tmp_path: Optional[str]):
    if not tmp_path:
        return
    try:
        _remove_path_if_exists(tmp_path)
    except FileNotFoundError:
        pass


def _valid_candidate_ext(candidate: Optional[str]) -> Optional[str]:
    if not candidate:
        return None
    ext = os.path.splitext(candidate.strip().strip('"').strip("'"))[1]
    return ext if ext and len(ext) <= 6 else None


def _extract_ext(original_filename: str, local_path: Optional[str]) -> str:
    if getattr(utils.THREAD_CONTEXT, "input_flags", None):
        return ".raw"
    return _valid_candidate_ext(original_filename) or _valid_candidate_ext(local_path) or ".tmp"


extract_ext = _extract_ext


def _build_upload_tmp_path(original_filename: str, local_path: Optional[str] = None) -> str:
    ext = _extract_ext(original_filename, local_path)
    return os.path.join(config.get_temp_dir(), f"upload_{uuid.uuid4().hex}{ext}")


def handle_upload(audio_file):
    """Handle binary file upload."""
    if not audio_file:
        return None, None, None

    original_filename = getattr(audio_file, "filename", "uploaded_file") or "uploaded_file"
    logger.info("[System] Ingesting remote data: %s", original_filename)
    tmp_path = None
    try:
        tmp_path = _build_upload_tmp_path(original_filename)
        _write_upload_sync(audio_file, tmp_path)
        _validate_upload_sync(tmp_path)
        _track_successful_upload(tmp_path)
        return tmp_path, tmp_path, original_filename
    except Exception:
        _cleanup_temp_upload_on_error(tmp_path)
        raise


_GENERIC_UPLOAD_NAMES = frozenset({"audio_file", "file", "blob"})


def _filtered_upload_display_name(raw_name) -> Optional[str]:
    """Basename a client-supplied upload filename, rejecting Bazarr/generic field-name
    echoes (e.g. 'audio_file') that are not real filenames -- see _GENERIC_UPLOAD_NAMES."""
    if not raw_name:
        return None
    base = os.path.basename(str(raw_name).strip())
    return base if base and base not in _GENERIC_UPLOAD_NAMES else None


def _basename_from_path(path: str) -> str:
    # Bazarr can run on Windows and send Windows-style paths (e.g. C:\media\ep.avi);
    # os.path.basename on Linux only splits on '/', so without this a Windows path
    # would display as the full path instead of just the filename.
    normalized_path = path.strip().strip('"').strip("'").replace("\\", "/")
    return os.path.basename(normalized_path)


def _display_name_from_path(path: str) -> Optional[str]:
    if not path:
        return None
    base = _basename_from_path(path)
    return base or None


def _resolve_upload_display_name(
    display_name: Optional[str],
    original_filename: Optional[str],
    video_file: Optional[str],
    fallback: str,
) -> str:
    """Shared fallback chain for both the pre-materialized and fresh-upload paths:
    an already-known display name, then the client's own (non-generic) filename,
    then Bazarr's video_file caller metadata, then whatever fallback the caller has."""
    return display_name or _filtered_upload_display_name(original_filename) or _display_name_from_path(video_file or "") or fallback


def _resolve_pre_materialized_upload(audio_file, display_name: Optional[str], video_file: Optional[str] = None) -> Optional[tuple]:
    if not isinstance(audio_file, str):
        return None
    resolved_p = resolve_local_path(audio_file)
    if not resolved_p:
        return None
    original_filename = getattr(audio_file, "original_filename", None)
    resolved_name = _resolve_upload_display_name(display_name, original_filename, video_file, os.path.basename(resolved_p))
    logger.info("[System] Using pre-materialized upload: %s", os.path.basename(resolved_p))
    return resolved_p, resolved_p, resolved_name


def _prepare_audio_file_path(audio_file, display_name: Optional[str], video_file: Optional[str] = None) -> Optional[tuple]:
    pre_materialized = _resolve_pre_materialized_upload(audio_file, display_name, video_file)
    if pre_materialized:
        return pre_materialized
    tmp_path, temp_path, original_filename = handle_upload(audio_file)
    if tmp_path:
        resolved_name = _resolve_upload_display_name(display_name, original_filename, video_file, original_filename)
        return tmp_path, temp_path, resolved_name
    return None


def _derive_display_name_from_path(local_path: Optional[str]) -> Optional[str]:
    if not local_path:
        return None
    return _basename_from_path(local_path)


def _resolve_local_source(local_path: Optional[str], display_name: Optional[str]) -> Optional[tuple]:
    if not local_path:
        return None
    resolved = resolve_local_path(local_path)
    if not resolved:
        return None
    utils.THREAD_CONTEXT.input_flags = None
    return resolved, None, display_name


def prepare_source_path(
    local_path: Optional[str] = None,
    audio_file=None,
    video_file: Optional[str] = None,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Resolve input media - 1. Local path mapping, 2. Upload fallback.

    audio_file is intentionally untyped: it is either a real FastAPI/Starlette
    UploadFile (fresh upload) or a MaterializedUploadPath str (already-materialized
    upload, tagged with the original filename), matching this module's existing
    convention for the same parameter on _prepare_audio_file_path/handle_upload."""
    display_name = _derive_display_name_from_path(local_path)
    local_resolution = _resolve_local_source(local_path, display_name)
    if local_resolution:
        return local_resolution

    if audio_file:
        res = _prepare_audio_file_path(audio_file, display_name, video_file)
        if res:
            return res

    if local_path:
        raise ValueError(f"Path not accessible: {local_path} (Volumes unmapped and no audio data attached)")

    return None, None, None


def _display_name_from_upload_file(audio_file) -> Optional[str]:
    return _filtered_upload_display_name(getattr(audio_file, "filename", None))


def _resolve_early_display_name_from_local_path(local_path) -> Optional[str]:
    if not local_path or not isinstance(local_path, str):
        return None
    return _display_name_from_path(local_path)


def _resolve_early_display_name_from_video_file(video_file) -> Optional[str]:
    if not video_file or not isinstance(video_file, str):
        return None
    return _display_name_from_path(video_file)


def _resolve_early_display_name_from_audio_file(audio_file) -> Optional[str]:
    if not audio_file:
        return None
    if isinstance(audio_file, str):
        return _filtered_upload_display_name(getattr(audio_file, "original_filename", None))
    return _display_name_from_upload_file(audio_file)


def get_display_name_early(local_path=None, audio_file=None, video_file=None):
    """Extract a descriptive filename for the dashboard before processing starts.

    Bazarr's real client uploads audio with a generic field-echoed filename (e.g.
    'audio_file', see _GENERIC_UPLOAD_NAMES) and separately sends `video_file` as the
    real media path purely for display/logging -- prefer that over "Unknown Media"
    when the upload itself carries no usable name, matching the priority the history
    view already uses (see history_helpers._extract_best_filename)."""
    return (
        _resolve_early_display_name_from_local_path(local_path)
        or _resolve_early_display_name_from_video_file(video_file)
        or _resolve_early_display_name_from_audio_file(audio_file)
        or "Unknown Media"
    )
