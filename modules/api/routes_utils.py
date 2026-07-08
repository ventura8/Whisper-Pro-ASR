"""
Private utilities and helpers for API routes.
"""

import logging
import os
import time
import traceback
import uuid

from fastapi import Request, UploadFile
from starlette.datastructures import UploadFile as StarletteUploadFile

from modules.core import config, utils
from modules.inference import model_manager

logger = logging.getLogger(__name__)


async def materialize_upload_file(upload_file, is_raw_pcm=False):
    """Save an UploadFile to disk in the async context to avoid cross-thread SpooledTemporaryFile issues.

    Returns the disk path and the original filename, or (None, None) if no file.
    """
    if not upload_file or not isinstance(upload_file, (UploadFile, StarletteUploadFile)):
        return None, None

    original_filename = getattr(upload_file, "filename", "uploaded_file") or "uploaded_file"
    ext = os.path.splitext(original_filename)[1] if original_filename else ".tmp"
    if len(ext) > 6:
        ext = ".tmp"

    upload_dir = config.get_temp_dir()
    tmp_path = os.path.join(upload_dir, f"upload_{uuid.uuid4().hex}{ext}")
    used_sync_fallback = False

    try:
        await upload_file.seek(0)
        with open(tmp_path, "wb") as f:
            while True:
                chunk = await upload_file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    except (AttributeError, TypeError, OSError, ValueError, RuntimeError):
        # Fallback: try sync read from the underlying file object
        try:
            if hasattr(upload_file, "file") and upload_file.file:
                used_sync_fallback = True
                upload_file.file.seek(0)
                with open(tmp_path, "wb") as f:
                    shutil_copy_file_in_chunks(upload_file.file, f)
            else:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                return None, None
        except (AttributeError, OSError, ValueError, RuntimeError):
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return None, None

    if used_sync_fallback and not os.path.exists(tmp_path):
        return None, None

    if used_sync_fallback and os.path.getsize(tmp_path) == 0:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise ValueError("Remote data stream is empty (0 bytes received).")

    f_size = os.path.getsize(tmp_path)

    if f_size == 0:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise ValueError("Remote data stream is empty (0 bytes received).")

    if f_size > 1024 and not is_raw_pcm:
        with open(tmp_path, "rb") as f:
            header = f.read(1024)
            if header and all(b == 0 for b in header):
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
                raise ValueError("Input file is corrupted (contains only null bytes).")

    utils.track_file(tmp_path)
    logger.info("[System] Upload materialized to disk: %s (%d bytes)", os.path.basename(tmp_path), f_size)
    return tmp_path, original_filename


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

    encode_val = request.query_params.get("encode") or form_data.get("encode") or "true"
    is_raw_pcm = str(encode_val).lower() == "false"
    if is_raw_pcm:
        utils.THREAD_CONTEXT.input_flags = ["-f", "s16le", "-ar", "16000", "-ac", "1"]
    else:
        utils.THREAD_CONTEXT.input_flags = None

    materialized_path, _ = await materialize_upload_file(uploaded_file, is_raw_pcm=is_raw_pcm)
    if materialized_path:
        uploaded_file = materialized_path
    elif uploaded_file:
        uploaded_file = None

    return resolved_local_path, uploaded_file


def prepare_source_path(local_path=None, audio_file=None):
    """Resolve input media - 1. Local path mapping, 2. Upload fallback."""
    display_name = None
    if local_path:
        display_name = os.path.basename(local_path.strip().strip('"').strip("'"))
        p = resolve_local_path(local_path)
        if p:
            return p, None, display_name

    if audio_file:
        # If audio_file is already a pre-materialized disk path (string),
        # use it directly instead of going through handle_upload.
        if isinstance(audio_file, str):
            resolved_p = resolve_local_path(audio_file)
            if resolved_p:
                resolved_name = display_name if display_name else os.path.basename(resolved_p)
                logger.info("[System] Using pre-materialized upload: %s", os.path.basename(resolved_p))
                return resolved_p, resolved_p, resolved_name

        tmp_path, temp_path, original_filename = handle_upload(audio_file)
        if tmp_path:
            # If a local path (representing the original client-side file name) was provided,
            # we prefer its basename over any generic/fallback filename resolved from upload.
            resolved_name = display_name if display_name else original_filename
            return tmp_path, temp_path, resolved_name

    if local_path:
        raise ValueError(f"Path not accessible: {local_path} (Volumes unmapped and no audio data attached)")

    return None, None, None


def get_display_name_early(local_path=None, audio_file=None):
    """Extract a descriptive filename for the dashboard before processing starts."""
    if local_path:
        return os.path.basename(local_path.strip().strip('"').strip("'"))

    if audio_file and isinstance(audio_file, str):
        return os.path.basename(audio_file)

    if audio_file and hasattr(audio_file, "filename") and audio_file.filename:
        return audio_file.filename

    return "Unknown Media"


def resolve_local_path(raw_path):
    """Check if the provided path exists locally."""
    clean_path = raw_path.strip().strip('"').strip("'")
    candidates = [clean_path, clean_path.replace("+", " ")]

    approved_roots = [
        os.path.realpath(config.TEMP_DIR),
        os.path.realpath(config.PERSISTENT_DIR),
        os.path.realpath(os.getcwd()),
    ]
    for r in config.APPROVED_ROOTS:
        approved_roots.append(os.path.realpath(r))

    for p in candidates:
        if p:
            normalized_p = os.path.realpath(p)

            # Check if normalized path stays under any configured roots
            is_valid = False
            for root in approved_roots:
                if normalized_p == root or normalized_p.startswith(os.path.join(root, "")):
                    is_valid = True
                    break

            if not is_valid:
                logger.warning(
                    "[System] Path not in approved roots (volume not mounted?): %s",
                    p,
                )
                return None

            if os.path.exists(normalized_p):
                already_logged = getattr(utils.THREAD_CONTEXT, "optimized_local_path_logged", None)
                if already_logged != normalized_p:
                    logger.info("[System] Optimization: Using Local Path -> %s", normalized_p)
                    utils.THREAD_CONTEXT.optimized_local_path_logged = normalized_p
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
        ext = os.path.splitext(original_filename)[1] if original_filename else ".tmp"
        if len(ext) > 6:
            ext = ".tmp"

        upload_dir = config.get_temp_dir()
        tmp_path = os.path.join(upload_dir, f"upload_{uuid.uuid4().hex}{ext}")

        # Save the UploadFile contents to disk
        if hasattr(audio_file, "file") and audio_file.file:
            # Under FastAPI, audio_file is an UploadFile
            # Reset pointer to start just in case
            try:
                audio_file.file.seek(0)
            except tuple([Exception]):
                pass
            with open(tmp_path, "wb") as f:
                shutil_copy_file_in_chunks(audio_file.file, f)
        else:
            # Handle fallback for raw binary streams in tests
            content = audio_file.read() if hasattr(audio_file, "read") else audio_file
            with open(tmp_path, "wb") as f:
                f.write(content)

        f_size = os.path.getsize(tmp_path)

        if f_size == 0:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise ValueError("Remote data stream is empty (0 bytes received).")

        if f_size > 1024:
            with open(tmp_path, "rb") as f:
                header = f.read(1024)
                if header and all(b == 0 for b in header):
                    if os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except FileNotFoundError:
                            pass
                    raise ValueError("Input file is corrupted (contains only null bytes).")

        utils.track_file(tmp_path)
        logger.info("[System] Remote source ingestion successful: %d bytes", f_size)
        return tmp_path, tmp_path, original_filename
    except Exception:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except FileNotFoundError:
                pass
        raise


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


def get_clean_wav_or_error(source_path, input_flags=None):
    """Normalize input media to 16kHz mono WAV."""
    if input_flags is None:
        input_flags = getattr(utils.THREAD_CONTEXT, "input_flags", None)

    model_manager.update_task_progress(0, "Standardizing Audio")
    logger.info("[Prep] Normalizing audio stream (FFmpeg)...")
    start = time.time()

    if not input_flags:
        try:
            if os.path.exists(source_path):
                with open(source_path, "rb") as f:
                    header = f.read(1024)
                    if len(header) > 0 and all(b == 0 for b in header):
                        return None, ("Input file is corrupted (only null bytes).", 400)
        except tuple([Exception]):
            pass

    clean_wav = utils.convert_to_wav(source_path, input_flags=input_flags)
    if not clean_wav:
        return None, ("FFmpeg conversion failed - invalid media format", 400)

    logger.info("[Prep] Standardization completed in %s", utils.format_duration(time.time() - start))
    return clean_wav, None


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
    form_data = {}
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
            if isinstance(body, dict):
                form_data.update(body)
        except tuple([Exception]):
            pass
    else:
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


def extract_uploaded_file(audio_file: UploadFile | None, file: UploadFile | None, form_data: dict) -> UploadFile | None:
    """Extract uploaded file from route parameters and form data."""
    uploaded_file = audio_file or file
    if uploaded_file and not isinstance(uploaded_file, (UploadFile, StarletteUploadFile)):
        uploaded_file = None
    if not uploaded_file:
        uploaded_file = form_data.get("audio_file") or form_data.get("file") or form_data.get("video_file")
        if uploaded_file and not isinstance(uploaded_file, (UploadFile, StarletteUploadFile)):
            uploaded_file = None

    if not uploaded_file:
        for _, v in form_data.items():
            if isinstance(v, (UploadFile, StarletteUploadFile)):
                uploaded_file = v
                break
    return uploaded_file
