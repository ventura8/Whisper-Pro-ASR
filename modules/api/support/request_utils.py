"""
Private utilities and helpers for API routes.
"""

import logging
import os
import traceback
from typing import Optional

from fastapi import Request

from modules.api.support.audio_standardization import get_clean_wav_or_error
from modules.api.support.local_path import extract_path_from_mapping_keys
from modules.api.support.source_resolution import (
    _build_upload_tmp_path,
    extract_ext,
    get_display_name_early,
    handle_upload,
    prepare_source_path,
    resolve_local_path,
    shutil_copy_file_in_chunks,
)
from modules.api.support.upload_extraction import (
    _is_valid_upload_file,
    extract_uploaded_file,
)
from modules.core import utils
from modules.inference.runtime import model_manager

# Re-exported for existing callers/tests that import these from request_utils
# (e.g. modules.api.routes.asr / detect, and their test suites) -- the actual
# implementations live in source_resolution.py to keep this file under the
# 500-line limit.
__all__ = [
    "extract_ext",
    "get_display_name_early",
    "handle_upload",
    "prepare_source_path",
    "resolve_local_path",
    "shutil_copy_file_in_chunks",
]

logger = logging.getLogger(__name__)

_route_public_api = (get_clean_wav_or_error,)


async def materialize_upload_file(upload_file, local_path=None):
    """Save an UploadFile to disk in the async context to avoid cross-thread SpooledTemporaryFile issues.

    Returns the disk path and the original filename, or (None, None) if no file.
    """
    if not _is_valid_upload_file(upload_file):
        return None, None

    original_filename = _resolve_original_filename(upload_file, local_path)
    tmp_path = _build_upload_tmp_path(original_filename, local_path)

    write_result = await _write_upload_to_disk(upload_file, tmp_path)
    if not write_result["success"]:
        return None, None
    if not _validate_or_discard_materialized_file(tmp_path, write_result["used_sync"]):
        return None, None
    _track_materialized_upload(tmp_path)
    return tmp_path, original_filename


def _validate_or_discard_materialized_file(tmp_path: str, used_sync: bool) -> bool:
    try:
        _validate_materialized_file_or_cleanup(tmp_path, used_sync)
        return True
    except ValueError:
        if used_sync and not os.path.exists(tmp_path):
            return False
        raise


def _resolve_original_filename(upload_file, local_path: Optional[str] = None) -> str:
    name = getattr(upload_file, "filename", None)
    if name and os.path.splitext(name)[1]:
        return name
    if local_path:
        return os.path.basename(local_path.strip().strip('"').strip("'"))
    return name or "uploaded_file"


def _validate_materialized_file_or_cleanup(tmp_path: str, used_sync: bool):
    try:
        _validate_materialized_file(tmp_path, used_sync)
    except ValueError as e:
        _remove_path_if_exists(tmp_path)
        raise e


async def _write_upload_to_disk(upload_file, tmp_path: str) -> dict:
    """Writes uploaded file to disk. Returns (success, used_sync_fallback)."""
    if await _write_upload_to_disk_async(upload_file, tmp_path):
        return {"success": True, "used_sync": False}
    if _write_upload_to_disk_sync_fallback(upload_file, tmp_path):
        return {"success": True, "used_sync": True}
    _remove_path_if_exists(tmp_path)
    return {"success": False, "used_sync": False, "error": "write_failed"}


async def _write_upload_to_disk_async(upload_file, tmp_path: str) -> bool:
    try:
        await upload_file.seek(0)
        with open(tmp_path, "wb") as f:
            while True:
                chunk = await upload_file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        return True
    except (AttributeError, TypeError, OSError, ValueError, RuntimeError):
        return False


def _write_upload_to_disk_sync_fallback(upload_file, tmp_path: str) -> bool:
    try:
        if not (hasattr(upload_file, "file") and upload_file.file):
            return False
        upload_file.file.seek(0)
        with open(tmp_path, "wb") as f:
            shutil_copy_file_in_chunks(upload_file.file, f)
        return True
    except (AttributeError, OSError, ValueError, RuntimeError):
        return False


def _validate_materialized_file(tmp_path: str, used_sync: bool):
    _validate_materialized_sync_write(tmp_path, used_sync)
    _ensure_non_empty_file(tmp_path)


def _validate_materialized_sync_write(tmp_path: str, used_sync: bool):
    if used_sync and not os.path.exists(tmp_path):
        raise ValueError("Remote data stream is empty (0 bytes received).")


def _ensure_non_empty_file(tmp_path: str) -> int:
    file_size = os.path.getsize(tmp_path)
    if file_size == 0:
        raise ValueError("Remote data stream is empty (0 bytes received).")
    return file_size


def _remove_path_if_exists(path: str):
    if os.path.exists(path):
        os.remove(path)


def _track_materialized_upload(tmp_path: str):
    file_size = os.path.getsize(tmp_path)
    utils.track_file(tmp_path)
    logger.info(
        "[System] Upload materialized to disk: %s (%d bytes)",
        os.path.basename(tmp_path),
        file_size,
    )


def _resolve_video_file_as_path(form_data, request) -> str | None:
    """Resolve Bazarr's `video_file` caller-metadata field the same approved-roots-gated
    way as local_path (see resolve_local_path). Now that we replicate Bazarr's own
    audio-track selection and delay correction server-side (see
    utils.get_stream_alignment_directives), reading the mapped source directly here --
    like the existing local_path zero-copy optimization -- is preferable to using
    Bazarr's already re-encoded upload: it operates on the original, full-quality
    source instead of Bazarr's lossy resampled copy, and skips the upload/materialization
    entirely. Takes priority over any uploaded audio_file whenever it resolves."""
    video_file = extract_video_file(form_data, request)
    return resolve_local_path(video_file) if video_file else None


def _record_audio_source_mode(mode_message: str) -> None:
    """Log immediately (visible in the main/file log right away) and stash on
    THREAD_CONTEXT so it can also be re-logged once task registration begins --
    this function runs in the async route handler, before early_task_registration(),
    so LogBufferHandler (dashboard per-task "Execution Logs") can't capture it yet:
    it keys its buffer by task_id/registration_thread_id, neither of which exists
    until the worker thread enters early_task_registration(). worker_context/
    apply_worker_context_from_dict carries this value into that later point (see
    asr.py/detect.py's _perform_*_task, which re-log it right after entering)."""
    logger.info("[System] %s", mode_message)
    utils.THREAD_CONTEXT.audio_source_mode = mode_message


async def resolve_and_materialize_upload(local_path, audio_file, file, form_data, request):
    """Extract local path, uploaded file, and materialize the upload to disk."""
    resolved_local_path = extract_local_path(local_path, form_data, request)
    uploaded_file = extract_uploaded_file(audio_file, file, form_data)
    _setup_input_flags(request, form_data)

    # If the mapped local path is readable inside the container, prefer it and
    # skip upload materialization entirely (zero-copy Bazarr flow).
    optimized_local_path = resolve_local_path(resolved_local_path) if resolved_local_path else None
    if optimized_local_path:
        # local_path resolves to the original media container, not the (now-bypassed)
        # upload -- Bazarr's encode=false/raw_pcm hints describe the upload's raw PCM
        # format, and applying them here would make FFmpeg misinterpret a real
        # container as headerless PCM.
        utils.THREAD_CONTEXT.input_flags = None
        _record_audio_source_mode(f"Audio source: MAPPED PATH (local_path) -> {optimized_local_path}")
        return optimized_local_path, None

    optimized_video_path = _resolve_video_file_as_path(form_data, request)
    if optimized_video_path:
        # video_file resolves to the original media container, not the (now-bypassed)
        # upload -- Bazarr's encode=false/raw_pcm hints describe the upload's raw PCM
        # format, and applying them here would make FFmpeg misinterpret a real
        # container as headerless PCM.
        utils.THREAD_CONTEXT.input_flags = None
        _record_audio_source_mode(f"Audio source: MAPPED PATH (video_file) -> {optimized_video_path}")
        return optimized_video_path, None

    uploaded_file = await _materialize_if_needed(uploaded_file, local_path=resolved_local_path)
    if uploaded_file:
        _record_audio_source_mode(f"Audio source: UPLOADED AUDIO -> materialized to {uploaded_file}")

    return resolved_local_path, uploaded_file


def _is_raw_pcm_requested(request, form_data) -> bool:
    for key in ("raw_pcm", "is_pcm"):
        val = request.query_params.get(key) or form_data.get(key)
        if val and str(val).strip().lower() in ("true", "1", "yes"):
            return True
    return False


def _setup_input_flags(request, form_data):
    encode_val = request.query_params.get("encode")
    if encode_val in (None, ""):
        encode_val = form_data.get("encode")
    raw_pcm = _is_raw_pcm_requested(request, form_data)
    if str(encode_val).lower() == "false" or raw_pcm:
        utils.THREAD_CONTEXT.input_flags = ["-f", "s16le", "-ar", "16000", "-ac", "1"]
    else:
        utils.THREAD_CONTEXT.input_flags = None


class MaterializedUploadPath(str):
    """A materialized upload's on-disk temp path, tagged with the client's original
    filename. `_build_upload_tmp_path` always names the temp file `upload_<uuid>.ext`
    (never the client's real name), so without this the original filename is gone by
    the time display-name resolution (get_display_name_early, prepare_source_path)
    runs -- both would otherwise show the random temp basename in the dashboard/history
    instead of the file the client actually sent."""

    original_filename: Optional[str] = None


async def _materialize_if_needed(uploaded_file, local_path: Optional[str] = None) -> Optional[str]:
    materialized_path, original_filename = await materialize_upload_file(uploaded_file, local_path=local_path)
    if not materialized_path:
        return None
    tagged_path = MaterializedUploadPath(materialized_path)
    tagged_path.original_filename = original_filename
    return tagged_path


def apply_worker_context_from_dict(worker_context: dict) -> None:
    """Apply a captured worker_context dict onto the current thread's THREAD_CONTEXT.
    Shared by asr.py/detect.py's own _apply_*_worker_context wrappers and
    detect_coalescing.py's coalesced-follower flow."""
    utils.THREAD_CONTEXT.request_json = worker_context["request_json"]
    utils.THREAD_CONTEXT.endpoint = worker_context["endpoint"]
    utils.THREAD_CONTEXT.caller_info = worker_context["caller_info"]
    utils.THREAD_CONTEXT.input_flags = worker_context.get("input_flags")
    utils.THREAD_CONTEXT.audio_source_mode = worker_context.get("audio_source_mode")


def log_audio_source_mode(worker_context: dict) -> None:
    """Re-log the audio-source decision (made earlier, before task registration
    existed) now that a task_id/registration_thread_id exists -- only now can
    LogBufferHandler capture it into the dashboard's per-task "Execution Logs".
    Call this right after entering early_task_registration()."""
    mode = worker_context.get("audio_source_mode")
    if mode:
        logger.info("[System] %s", mode)


def cleanup_files(*args):
    """Securely remove temporary processing assets, including tracked ones."""
    to_remove = set(args) | set(utils.get_tracked_files())
    for f_path in to_remove:
        if f_path and os.path.exists(f_path):
            try:
                os.remove(f_path)
                logger.debug("[System] Cleaned up: %s", f_path)
            except tuple([Exception]):
                pass
    # Reset tracking
    utils.get_tracked_files().clear()


def handle_error(err, context="ASR"):
    """Centralized error handling for routes."""
    status_code = 500
    if isinstance(err, ValueError):
        logger.warning("[%s] Invalid parameter: %s", context, err)
        status_code = 400
    elif isinstance(err, FileNotFoundError):
        status_code = 404
    else:
        logger.error("%s CRITICAL: %s\n%s", context.upper(), err, traceback.format_exc())

    msg = str(err) if status_code != 500 else f"Service Error: {str(err)}"
    return msg, status_code


def initialize_task_context(local_path=None, audio_file=None, is_priority=False, video_file=None):
    """Shared initialization logic for transcription and detection tasks."""
    source_path, upload_temp, display_name = prepare_source_path(local_path, audio_file, video_file)
    if display_name:
        utils.THREAD_CONTEXT.filename = display_name
        model_manager.update_task_metadata(filename=display_name)
    if source_path:
        utils.THREAD_CONTEXT.source_path = source_path
        model_manager.update_task_metadata(source_path=source_path)

    if not source_path:
        return None, None, ("No audio source provided", 400)

    model_manager.update_task_progress(0, "Analyzing Media")
    duration = utils.get_audio_duration(source_path)
    model_manager.update_task_metadata(video_duration=duration)

    if is_priority:
        model_manager.wait_for_priority()
    return source_path, upload_temp, None


async def parse_form_data(request: Request) -> dict:
    """Parse form data or JSON body from request safely."""
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        return await _parse_json_body(request)
    return await _parse_multipart_form(request)


async def _parse_json_body(request: Request) -> dict:
    try:
        body = await request.json()
        if isinstance(body, dict):
            return body
    except tuple([Exception]):
        pass
    return {}


async def _parse_multipart_form(request: Request) -> dict:
    form_data = {}
    try:
        form = await request.form()
        for k, v in form.items():
            form_data[k] = v
    except tuple([Exception]):
        pass
    return form_data


def extract_local_path(local_path: str | None, form_data: dict, request: Request) -> str | None:
    """Extract local path parameter from form data and query params."""
    # video_file is intentionally excluded: it's Bazarr caller metadata sent for
    # logging/display only (see whisperai.py's pass_video_name option) and must
    # never be resolved as a local filesystem path -- see the wire-format
    # contract asserted in tests/integration/test_bazarr_wire_format.py.
    candidates = [
        local_path,
        form_data.get("local_path"),
        form_data.get("file_path"),
        form_data.get("original_path"),
        form_data.get("file"),
        form_data.get("audio_file"),
        request.query_params.get("local_path"),
        request.query_params.get("file_path"),
        request.query_params.get("original_path"),
        request.query_params.get("file"),
        request.query_params.get("audio_file"),
        extract_path_from_mapping_keys(form_data),
    ]
    for val in candidates:
        if val and isinstance(val, str):
            return val
    return None


def extract_video_file(form_data: dict, request: Request) -> str | None:
    """Extract Bazarr's `video_file` caller-metadata field. Historically display/logging
    only, but this value may also be passed to _resolve_video_file_as_path, which
    resolves it through the same approved-roots gate as local_path (see
    resolve_local_path) and, when it resolves, uses it directly as the transcription
    source -- see extract_local_path's exclusion comment for why *this* function
    itself never resolves it as a filesystem path."""
    for val in (form_data.get("video_file"), request.query_params.get("video_file")):
        if val and isinstance(val, str):
            return val
    return None
