"""
Private utilities and helpers for API routes.
"""

import logging
import os
import time
import traceback
import uuid
from typing import Optional

from fastapi import Request

from modules.api.support.local_path import get_approved_roots, is_path_approved, log_local_path_optimization
from modules.api.support.upload_extraction import _is_valid_upload_file, extract_uploaded_file
from modules.core import config, utils
from modules.inference.runtime import model_manager

logger = logging.getLogger(__name__)


async def materialize_upload_file(upload_file, is_raw_pcm=False, local_path=None):
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
    if not _validate_or_discard_materialized_file(tmp_path, is_raw_pcm, write_result["used_sync"]):
        return None, None
    _track_materialized_upload(tmp_path)
    return tmp_path, original_filename


def _validate_or_discard_materialized_file(tmp_path: str, is_raw_pcm: bool, used_sync: bool) -> bool:
    try:
        _validate_materialized_file_or_cleanup(tmp_path, is_raw_pcm, used_sync)
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


def _validate_materialized_file_or_cleanup(tmp_path: str, is_raw_pcm: bool, used_sync: bool):
    try:
        _validate_materialized_file(tmp_path, is_raw_pcm, used_sync)
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


def _validate_materialized_file(tmp_path: str, is_raw_pcm: bool, used_sync: bool):
    del is_raw_pcm
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
    logger.info("[System] Upload materialized to disk: %s (%d bytes)", os.path.basename(tmp_path), file_size)


async def resolve_and_materialize_upload(local_path, audio_file, file, form_data, request):
    """Extract local path, uploaded file, and materialize the upload to disk."""
    resolved_local_path = extract_local_path(local_path, form_data, request)
    uploaded_file = extract_uploaded_file(audio_file, file, form_data)

    # If the mapped local path is readable inside the container, prefer it and
    # skip upload materialization entirely (zero-copy Bazarr flow).
    optimized_local_path = resolve_local_path(resolved_local_path) if resolved_local_path else None
    if optimized_local_path:
        utils.THREAD_CONTEXT.input_flags = None
        return optimized_local_path, None

    _setup_input_flags(request, form_data)

    uploaded_file = await _materialize_if_needed(uploaded_file, local_path=resolved_local_path)

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


async def _materialize_if_needed(uploaded_file, local_path: Optional[str] = None) -> Optional[str]:
    is_raw_pcm = utils.THREAD_CONTEXT.input_flags is not None
    materialized_path, _ = await materialize_upload_file(uploaded_file, is_raw_pcm=is_raw_pcm, local_path=local_path)
    if materialized_path:
        return materialized_path
    return None


def prepare_source_path(local_path=None, audio_file=None):
    """Resolve input media - 1. Local path mapping, 2. Upload fallback."""
    display_name = _derive_display_name_from_path(local_path)
    local_resolution = _resolve_local_source(local_path, display_name)
    if local_resolution:
        return local_resolution

    if audio_file:
        res = _prepare_audio_file_path(audio_file, display_name)
        if res:
            return res

    if local_path:
        raise ValueError(f"Path not accessible: {local_path} (Volumes unmapped and no audio data attached)")

    return None, None, None


def _derive_display_name_from_path(local_path) -> Optional[str]:
    if not local_path:
        return None
    return os.path.basename(local_path.strip().strip('"').strip("'"))


def _resolve_local_source(local_path, display_name: Optional[str]) -> Optional[tuple]:
    if not local_path:
        return None
    resolved = resolve_local_path(local_path)
    if not resolved:
        return None
    return resolved, None, display_name


def _prepare_audio_file_path(audio_file, display_name: Optional[str]) -> Optional[tuple]:
    pre_materialized = _resolve_pre_materialized_upload(audio_file, display_name)
    if pre_materialized:
        return pre_materialized
    tmp_path, temp_path, original_filename = handle_upload(audio_file)
    if tmp_path:
        resolved_name = display_name if display_name else original_filename
        return tmp_path, temp_path, resolved_name
    return None


def _resolve_pre_materialized_upload(audio_file, display_name: Optional[str]) -> Optional[tuple]:
    if not isinstance(audio_file, str):
        return None
    resolved_p = resolve_local_path(audio_file)
    if not resolved_p:
        return None
    resolved_name = display_name if display_name else os.path.basename(resolved_p)
    logger.info("[System] Using pre-materialized upload: %s", os.path.basename(resolved_p))
    return resolved_p, resolved_p, resolved_name


def get_display_name_early(local_path=None, audio_file=None):
    """Extract a descriptive filename for the dashboard before processing starts."""
    if local_path:
        return os.path.basename(local_path.strip().strip('"').strip("'"))

    if not audio_file:
        return "Unknown Media"

    if isinstance(audio_file, str):
        return os.path.basename(audio_file)

    return getattr(audio_file, "filename", None) or "Unknown Media"


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


def _extract_ext(original_filename: str, local_path: Optional[str]) -> str:
    ref_path = original_filename or local_path or ""
    ext = os.path.splitext(ref_path.strip().strip('"').strip("'"))[1]
    return ext if ext and len(ext) <= 6 else ".tmp"


def _build_upload_tmp_path(original_filename: str, local_path: Optional[str] = None) -> str:
    ext = _extract_ext(original_filename, local_path)
    return os.path.join(config.get_temp_dir(), f"upload_{uuid.uuid4().hex}{ext}")


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


def shutil_copy_file_in_chunks(src, dst):
    """Helper to copy file stream in chunks to avoid high RAM spikes."""
    while True:
        chunk = src.read(1024 * 1024)  # 1MB chunk
        if not chunk:
            break
        dst.write(chunk)


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


def _run_convert_to_wav(source_path: str, input_flags) -> tuple[Optional[str], Optional[tuple]]:
    try:
        clean_wav = utils.convert_to_wav(source_path, input_flags=input_flags)
        if not clean_wav:
            return None, ("FFmpeg conversion failed - invalid media format", 400)
        return clean_wav, None
    except (RuntimeError, ValueError, OSError) as e:
        logger.warning("[Prep] FFmpeg conversion failed for %s: %s", source_path, e)
        return None, ("FFmpeg conversion failed - invalid media format", 400)


def get_clean_wav_or_error(source_path, input_flags=None):
    """Normalize input media to 16kHz mono WAV."""
    flags = input_flags if input_flags is not None else getattr(utils.THREAD_CONTEXT, "input_flags", None)
    model_manager.update_task_progress(0, "Standardizing Audio")
    logger.info("[Prep] Normalizing audio stream (FFmpeg)...")
    start = time.time()

    if not flags and _is_file_corrupted(source_path):
        return None, ("Input file is corrupted (only null bytes).", 400)

    clean_wav, err = _run_convert_to_wav(source_path, flags)
    if err:
        return None, err

    logger.info("[Prep] Standardization completed in %s", utils.format_duration(time.time() - start))
    return clean_wav, None


def _is_file_corrupted(source_path: str) -> bool:
    try:
        if _contains_only_null_header(source_path):
            return True
    except tuple([Exception]):
        pass
    return False


def _contains_only_null_header(source_path: str) -> bool:
    if not os.path.exists(source_path):
        return False
    with open(source_path, "rb") as f:
        header = f.read(1024)
    return len(header) > 0 and all(b == 0 for b in header)


def initialize_task_context(local_path=None, audio_file=None, is_priority=False):
    """Shared initialization logic for transcription and detection tasks."""
    source_path, upload_temp, display_name = prepare_source_path(local_path, audio_file)
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
    candidates = [
        local_path,
        form_data.get("local_path"),
        form_data.get("video_file"),
        form_data.get("file_path"),
        form_data.get("original_path"),
        form_data.get("file"),
        form_data.get("audio_file"),
        request.query_params.get("local_path"),
        request.query_params.get("video_file"),
        request.query_params.get("file_path"),
        request.query_params.get("original_path"),
        request.query_params.get("file"),
        request.query_params.get("audio_file"),
    ]
    for val in candidates:
        if val and isinstance(val, str):
            return val
    return None
