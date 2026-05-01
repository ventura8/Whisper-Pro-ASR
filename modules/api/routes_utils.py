"""
Private utilities and helpers for API routes.
"""
import logging
import os
import uuid
import traceback
import time
from flask import request  # pylint: disable=import-error
from modules import config
from modules import utils
from modules.inference import model_manager

logger = logging.getLogger(__name__)


def prepare_source_path():
    """Resolve input media - 1. Local path mapping, 2. Upload fallback."""
    raw_path = get_raw_path()
    if raw_path:
        p = resolve_local_path(raw_path)
        if p:
            return p, None, os.path.basename(p)

    tmp_path, temp_path, original_filename = handle_upload()
    if tmp_path:
        return tmp_path, temp_path, original_filename

    if raw_path:
        raise ValueError(
            f"Path not accessible: {raw_path} (Volumes unmapped and no audio data attached)")

    return None, None, None


def get_raw_path():
    """Extract raw path from request arguments or form data."""
    path_keys = ['video_file', 'local_path', 'file_path', 'original_path']
    for key in path_keys:
        val = request.args.get(key) or request.form.get(key)
        if val:
            return val
    return None


def get_display_name_early():
    """Extract a descriptive filename for the dashboard before processing starts."""
    # 1. Local path keys
    raw = get_raw_path()
    if raw:
        return os.path.basename(raw.strip().strip('"').strip("'"))

    # 2. Upload keys
    audio_file = request.files.get('audio_file') or request.files.get('file')
    if audio_file:
        return audio_file.filename or "Uploaded Media"

    return "Unknown Media"


def resolve_local_path(raw_path):
    """Check if the provided path exists locally."""
    clean_path = raw_path.strip().strip('"').strip("'")
    candidates = [clean_path, clean_path.replace('+', ' ')]

    for p in candidates:
        if p and os.path.exists(p):
            logger.info("[System] Optimization: Using Local Path -> %s", p)
            return p
    return None


def handle_upload():
    """Handle binary file upload."""
    audio_file = request.files.get('audio_file') or request.files.get('file')
    if not audio_file:
        return None, None, None

    original_filename = audio_file.filename or "uploaded_file"
    logger.info("[System] Ingesting remote data: %s", original_filename)
    tmp_path = None
    try:
        if hasattr(audio_file.stream, 'seek'):
            try:
                audio_file.stream.seek(0)
            except Exception:  # pylint: disable=broad-exception-caught
                pass

        ext = os.path.splitext(audio_file.filename)[1] if audio_file.filename else ".tmp"
        if len(ext) > 6:
            ext = ".tmp"

        expected_size = request.content_length or 0
        upload_dir = config.get_temp_dir(required_bytes=expected_size)
        tmp_path = os.path.join(upload_dir, f"upload_{uuid.uuid4().hex}{ext}")

        audio_file.save(tmp_path)
        f_size = os.path.getsize(tmp_path)

        if f_size == 0:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise ValueError("Remote data stream is empty (0 bytes received).")

        if f_size > 1024:
            with open(tmp_path, 'rb') as f:
                header = f.read(1024)
                if all(b == 0 for b in header):
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                    raise ValueError("Input file is corrupted (contains only null bytes).")

        utils.track_file(tmp_path)
        logger.info("[System] Remote source ingestion successful: %d bytes", f_size)
        return tmp_path, tmp_path, original_filename
    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise e


def cleanup_files(*args):
    """Securely remove temporary processing assets, including tracked ones."""
    to_remove = set(args) | set(utils.get_tracked_files())
    for f_path in to_remove:
        if f_path and os.path.exists(f_path):
            try:
                os.remove(f_path)
                logger.debug("[System] Cleaned up: %s", f_path)
            except Exception:  # pylint: disable=broad-exception-caught
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
        logger.error("%s CRITICAL: %s\\n%s", context.upper(), err, traceback.format_exc())

    msg = str(err) if status_code != 500 else f"Service Error: {str(err)}"
    return msg, status_code


def get_clean_wav_or_error(source_path):
    """Normalize input media to 16kHz mono WAV."""
    model_manager.update_task_progress(0, "Standardizing Audio")
    logger.info("[Prep] Normalizing audio stream (FFmpeg)...")
    start = time.time()

    try:
        if os.path.exists(source_path):
            with open(source_path, 'rb') as f:
                header = f.read(1024)
                if len(header) > 0 and all(b == 0 for b in header):
                    return None, ("Input file is corrupted (only null bytes).", 400)
    except Exception:  # pylint: disable=broad-exception-caught
        pass

    clean_wav = utils.convert_to_wav(source_path)
    if not clean_wav:
        return None, ("FFmpeg conversion failed - invalid media format", 400)

    logger.info("[Prep] Standardization completed in %s",
                utils.format_duration(time.time() - start))
    return clean_wav, None


def initialize_task_context(is_priority=False):
    """Shared initialization logic for transcription and detection tasks."""
    source_path, upload_temp, display_name = prepare_source_path()
    if display_name:
        utils.THREAD_CONTEXT.filename = display_name
        model_manager.update_task_metadata(filename=display_name)

    if not source_path:
        return None, None, ("No audio source provided", 400)

    model_manager.update_task_progress(0, "Analyzing Media")
    duration = utils.get_audio_duration(source_path)
    model_manager.update_task_metadata(video_duration=duration)

    if is_priority:
        model_manager.wait_for_priority()
    return source_path, upload_temp, None
