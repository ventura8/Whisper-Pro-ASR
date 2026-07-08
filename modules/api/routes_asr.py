"""
ASR Transcription Routes for Whisper Pro ASR
"""

import json
import logging
import os
import time
import urllib.parse
from typing import Optional

import anyio
from fastapi import APIRouter, File, Query, Request, Response, UploadFile
from fastapi.responses import JSONResponse

from modules.api import routes_utils
from modules.core import config, utils
from modules.inference import language_detection, model_manager
from modules.inference.concurrency import _check_preemption

router = APIRouter(tags=["Transcription"])
logger = logging.getLogger(__name__)


@router.get("/asr")
def transcribe_status():
    """Check ASR engine status."""
    status_str = "ready" if model_manager.is_engine_initialized() else "not_ready"
    return Response(content=status_str, media_type="text/plain")


@router.post("/asr")
@router.post("/v1/audio/transcriptions")
@router.post("/v1/audio/translations")
async def transcribe(
    request: Request,
    local_path: Optional[str] = Query(None),
    audio_file: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None),
):
    """
    High-Precision Audio Transcription (ASR)
    ---
    Convert speech to text with hardware acceleration (OpenAI Compatible).
    """
    utils.THREAD_CONTEXT.reset()
    if not model_manager.is_engine_initialized():
        return JSONResponse(content={"error": "Model not loaded"}, status_code=503)

    model_manager.increment_active_session()
    try:
        # 1. Parse request parameters dynamically
        form_data = await routes_utils.parse_form_data(request)
        params = await get_request_params(request, form_data)

        # Resolve local_path and upload_file
        resolved_local_path, uploaded_file = await routes_utils.resolve_and_materialize_upload(
            local_path, audio_file, file, form_data, request
        )

        utils.THREAD_CONTEXT.caller_info = {
            "ip": request.client.host if request.client else "127.0.0.1",
            "user_agent": request.headers.get("User-Agent", "Unknown"),
        }
        sanitized_params = params.copy()
        if "hf_token" in sanitized_params:
            sanitized_params["hf_token"] = "[MASKED]"
        utils.THREAD_CONTEXT.request_json = sanitized_params
        utils.THREAD_CONTEXT.endpoint = request.url.path

        start_time = time.time()
        filename = routes_utils.get_display_name_early(resolved_local_path, uploaded_file)
        task_type = "Translation" if params.get("task") == "translate" else "Transcription"
        logger.info(
            "    Task: %s | Format: %s | Lang: %s",
            task_type.upper(),
            params.get("output_format", "srt").upper(),
            params.get("language") or "auto-detect",
        )
        with model_manager.early_task_registration(task_type=task_type, filename=filename):
            # Run transcription setup in worker thread
            result, source_path, err = await anyio.to_thread.run_sync(
                _perform_transcription_task, params, task_type, resolved_local_path, uploaded_file
            )
            if err:
                model_manager.update_task_metadata(status="failed")
                msg, code = err
                return JSONResponse(content={"error": msg}, status_code=code)

            return build_response(result, params, {"total": time.time() - start_time}, source_path, start_time)
    except tuple([Exception]) as e:
        msg, code = routes_utils.handle_error(e)
        return JSONResponse(content={"error": msg}, status_code=code)
    finally:
        await anyio.to_thread.run_sync(routes_utils.cleanup_files)
        model_manager.decrement_active_session()


def _perform_transcription_task(params, task_type, local_path, audio_file):
    """Inner logic for transcription execution."""
    try:
        source_path, _, err = routes_utils.initialize_task_context(local_path, audio_file, is_priority=False)
        if err:
            return None, None, err

        model_manager.update_task_progress(None, f"{task_type} Initializing")
        _check_preemption()

        # --- Stage 1: FFmpeg normalization ---
        clean_wav = None
        if not config.ENABLE_VOCAL_SEPARATION:
            clean_wav, err = routes_utils.get_clean_wav_or_error(source_path)
            if err:
                return None, source_path, err

        # --- Stage 2: Language detection on clean audio (post-FFmpeg, pre-vocal separation) ---
        detection_target = clean_wav if clean_wav else source_path
        lang = _detect_lang_if_needed(params.get("language"), detection_target)

        # --- Stage 3: Yield to detect language tasks before starting vocal separation ---
        _check_preemption()

        # --- Stage 4: run_transcription (vocal separation -> inference) ---
        target_audio = clean_wav if clean_wav else source_path
        result = model_manager.run_transcription(
            target_audio,
            lang,
            params["task"],
            diarize=params.get("diarize", False),
            min_speakers=params.get("min_speakers"),
            max_speakers=params.get("max_speakers"),
            hf_token=params.get("hf_token"),
            initial_prompt=params.get("initial_prompt"),
            vad_filter=params.get("vad_filter", True),
            word_timestamps=params.get("word_timestamps", False),
            batch_size=params.get("batch_size"),
        )

        if result:
            model_manager.update_task_metadata(result=result)
        return result, source_path, None
    except Exception as e:
        model_manager.update_task_metadata(result={"error": str(e)})
        raise e


async def get_request_params(request: Request, form_data: dict):
    """Extract parameters from request."""
    query_params = dict(request.query_params)
    params = {
        "task": query_params.get("task") or form_data.get("task") or "transcribe",
        "language": query_params.get("language")
        or form_data.get("source_lang")
        or query_params.get("source_lang")
        or form_data.get("language"),
        "output_format": (
            query_params.get("response_format")
            or query_params.get("output")
            or form_data.get("output")
            or form_data.get("response_format")
            or "srt"
        ),
    }
    if "/translations" in request.url.path:
        params["task"] = "translate"

    try:
        bs = query_params.get("batch_size") or form_data.get("batch_size")
        params["batch_size"] = int(bs) if bs else config.DEFAULT_BATCH_SIZE
    except (ValueError, TypeError):
        params["batch_size"] = config.DEFAULT_BATCH_SIZE

    # Speaker Diarization parameters
    diarize_val = query_params.get("diarize") or form_data.get("diarize") or "false"
    params["diarize"] = str(diarize_val).lower() == "true"

    try:
        min_spk = query_params.get("min_speakers") or form_data.get("min_speakers")
        params["min_speakers"] = int(min_spk) if min_spk else None
    except (ValueError, TypeError):
        params["min_speakers"] = None

    try:
        max_spk = query_params.get("max_speakers") or form_data.get("max_speakers")
        params["max_speakers"] = int(max_spk) if max_spk else None
    except (ValueError, TypeError):
        params["max_speakers"] = None

    params["hf_token"] = query_params.get("hf_token") or form_data.get("hf_token") or request.headers.get("X-HF-Token")

    # New ASR parameters
    params["initial_prompt"] = query_params.get("initial_prompt") or form_data.get("initial_prompt")

    highlight_val = (
        query_params.get("subtitle_highlight_words")
        or query_params.get("SUBTITLE_HIGHLIGHT_WORDS")
        or form_data.get("subtitle_highlight_words")
        or form_data.get("SUBTITLE_HIGHLIGHT_WORDS")
        or os.environ.get("SUBTITLE_HIGHLIGHT_WORDS", "false")
    )
    params["subtitle_highlight_words"] = str(highlight_val).lower() == "true"

    vad_val = query_params.get("vad_filter") or form_data.get("vad_filter") or "true"
    params["vad_filter"] = str(vad_val).lower() == "true"

    word_ts_val = query_params.get("word_timestamps") or form_data.get("word_timestamps") or "false"
    params["word_timestamps"] = str(word_ts_val).lower() == "true"
    if params["subtitle_highlight_words"]:
        params["word_timestamps"] = True

    try:
        mlw = query_params.get("max_line_width") or form_data.get("max_line_width")
        params["max_line_width"] = int(mlw) if mlw else None
    except (ValueError, TypeError):
        params["max_line_width"] = None

    try:
        mlc = query_params.get("max_line_count") or form_data.get("max_line_count")
        params["max_line_count"] = int(mlc) if mlc else None
    except (ValueError, TypeError):
        params["max_line_count"] = None

    return params


def _detect_lang_if_needed(lang, path):
    """Run voting language detection on the provided audio path if no language was specified.

    This is called on the post-FFmpeg normalised WAV (when available) so that
    the detector operates on clean, standardised audio rather than the raw
    source upload, improving identification accuracy before vocal separation begins.
    """
    if not lang:
        _check_preemption()
        model_manager.update_task_progress(0, "Language Detection")
        logger.info("[LD] Running language detection on post-FFmpeg audio: %s", os.path.basename(path))
        res = language_detection.run_voting_detection(path, model_manager)
        loggable = {k: v for k, v in res.items() if k != "logs"}
        logger.info("LD Response JSON: %s", json.dumps(loggable, ensure_ascii=False, indent=None))
        return res.get("detected_language")
    return lang


def build_response(result, params, stats, path, start):
    """Format final response."""
    if not isinstance(result, dict) or not result:
        return JSONResponse(content={"error": "Transcription failed: no result produced"}, status_code=500)

    stats["total"] = time.time() - start
    stats["video_dur"] = result.get("video_duration_sec", 0.0)
    stats["language"] = result.get("language")

    perf = result.get("performance", {})
    task_type = "TRANSLATE" if params.get("task") == "translate" else "TRANSCRIBE"
    logger.info(
        "ASR Completed | Type: %s | Lang: %s | Total: %s | Queue: %.2fs | Isolation: %.2fs | Inference: %.2fs",
        task_type,
        stats["language"],
        utils.format_duration(stats["total"]),
        perf.get("queue_sec", 0),
        perf.get("isolation_sec", 0),
        perf.get("inference_sec", 0),
    )

    supported_formats = {"json", "vtt", "txt", "tsv", "srt"}
    fmt = params.get("output_format", "srt").lower()
    if fmt not in supported_formats:
        fmt = "srt"

    if fmt == "json":
        return JSONResponse(content=result)

    content = ""
    if fmt == "vtt":
        content = utils.generate_vtt(
            result,
            max_line_width=params.get("max_line_width"),
            max_line_count=params.get("max_line_count"),
            highlight_words=params.get("subtitle_highlight_words", False),
        )
    elif fmt == "txt":
        content = utils.generate_txt(result)
    elif fmt == "tsv":
        content = utils.generate_tsv(result)
    else:
        content = utils.generate_srt(
            result,
            max_line_width=params.get("max_line_width"),
            max_line_count=params.get("max_line_count"),
            highlight_words=params.get("subtitle_highlight_words", False),
        )

    subtitle_lang = result.get("language") or params.get("language") or "en"

    fname = os.path.splitext(os.path.basename(path))[0]
    ascii_filename = f"{fname}.{subtitle_lang}-ai.{fmt}".encode("ascii", "ignore").decode("ascii").replace('"', "")
    if not ascii_filename.strip():
        ascii_filename = f"transcription.{subtitle_lang}-ai.{fmt}"
    quoted_filename = urllib.parse.quote(f"{fname}.{subtitle_lang}-ai.{fmt}")

    headers = {"Content-Disposition": f"attachment; filename=\"{ascii_filename}\"; filename*=UTF-8''{quoted_filename}"}
    return Response(content=content, media_type="text/plain", headers=headers)
