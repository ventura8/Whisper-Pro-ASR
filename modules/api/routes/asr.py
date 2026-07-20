"""
ASR Transcription Routes for Whisper Pro ASR
"""

import json
import logging
import os
import time
import urllib.parse
from typing import Any, Optional, TypedDict

import anyio
from fastapi import APIRouter, File, Query, Request, Response, UploadFile
from fastapi.responses import JSONResponse

from modules.api.support import request_utils as routes_utils
from modules.core import config, utils
from modules.inference.pipeline import language_detection
from modules.inference.runtime import model_manager
from modules.inference.runtime.concurrency import _check_preemption

router = APIRouter(tags=["Transcription"])
logger = logging.getLogger(__name__)

type ApiError = tuple[str, int]
type QueryParams = dict[str, str]
type FormData = dict[str, Any]


class RequestParams(TypedDict, total=False):
    """Normalized request parameters accepted by transcription routes."""

    task: str
    language: Optional[str]
    output_format: str
    batch_size: Optional[int]
    diarize: bool
    min_speakers: Optional[int]
    max_speakers: Optional[int]
    hf_token: Optional[str]
    initial_prompt: Optional[str]
    subtitle_highlight_words: bool
    vad_filter: bool
    word_timestamps: bool
    max_line_width: Optional[int]
    max_line_count: Optional[int]


class TranscriptionPerformance(TypedDict, total=False):
    """Performance metrics emitted for a transcription request."""

    queue_sec: float
    isolation_sec: float
    inference_sec: float


class TranscriptionResult(TypedDict, total=False):
    """Structured transcription response payload metadata."""

    language: str
    video_duration_sec: float
    performance: TranscriptionPerformance


@router.get("/asr")
def transcribe_status() -> Response:
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
) -> Response:
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
        form_data = await routes_utils.parse_form_data(request)
        params = await get_request_params(request, form_data)
        resolved_local_path, uploaded_file = await routes_utils.resolve_and_materialize_upload(
            local_path, audio_file, file, form_data, request
        )

        _setup_thread_context(request, params)
        start_time = time.time()
        filename = routes_utils.get_display_name_early(resolved_local_path, uploaded_file)
        task_type = _resolve_task_type(params)
        _log_task_start(task_type, params)
        with model_manager.early_task_registration(task_type=task_type, filename=filename):
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


def _setup_thread_context(request: Request, params: RequestParams) -> None:
    utils.THREAD_CONTEXT.caller_info = {
        "ip": request.client.host if request.client else "127.0.0.1",
        "user_agent": request.headers.get("User-Agent", "Unknown"),
    }
    sanitized_params = params.copy()
    if "hf_token" in sanitized_params:
        sanitized_params["hf_token"] = "".join(["[", "MASKED", "]"])
    utils.THREAD_CONTEXT.request_json = sanitized_params
    utils.THREAD_CONTEXT.endpoint = request.url.path
    utils.THREAD_CONTEXT.clean_audio = params.get("clean_audio")


def _perform_transcription_task(
    params: RequestParams,
    task_type: str,
    local_path: Optional[str],
    audio_file: Optional[UploadFile],
) -> tuple[Optional[TranscriptionResult], Optional[str], Optional[ApiError]]:
    """Inner logic for transcription execution."""
    try:
        source_path, init_err = _initialize_transcription_context(local_path, audio_file, task_type)
        if init_err:
            return None, None, init_err

        clean_wav, err = _get_transcription_source(source_path)
        if err:
            return None, source_path, err

        lang = _detect_lang_for_transcription(params.get("language"), source_path, clean_wav)
        result = _run_transcription(params, source_path, clean_wav, lang)

        if result:
            model_manager.update_task_metadata(result=result)
        return result, source_path, None
    except Exception as e:
        model_manager.update_task_metadata(result={"error": str(e)})
        raise e


def _get_transcription_source(source_path: str) -> tuple[Optional[str], Optional[ApiError]]:
    return routes_utils.get_clean_wav_or_error(source_path)


def _initialize_transcription_context(
    local_path: Optional[str],
    audio_file: Optional[UploadFile],
    task_type: str,
) -> tuple[Optional[str], Optional[ApiError]]:
    source_path, _, err = routes_utils.initialize_task_context(local_path, audio_file, is_priority=False)
    if err:
        return None, err
    model_manager.update_task_progress(None, f"{task_type} Initializing")
    _check_preemption()
    return source_path, None


def _detect_lang_for_transcription(
    lang: Optional[str],
    source_path: str,
    clean_wav: Optional[str],
) -> Optional[str]:
    detection_target = clean_wav if clean_wav else source_path
    return _detect_lang_if_needed(lang, detection_target)


def _run_transcription(
    params: RequestParams,
    source_path: str,
    clean_wav: Optional[str],
    lang: Optional[str],
) -> TranscriptionResult:
    _check_preemption()
    target_audio = clean_wav if clean_wav else source_path
    return model_manager.run_transcription(
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


async def get_request_params(request: Request, form_data: FormData) -> RequestParams:
    """Extract parameters from request."""
    query_params: QueryParams = dict(request.query_params)
    params = _build_base_request_params(request, query_params, form_data)
    _apply_batch_and_diarization_params(params, query_params, form_data, request)
    _apply_prompt_and_format_flags(params, query_params, form_data)
    _apply_subtitle_layout_params(params, query_params, form_data)
    return params


def _build_base_request_params(
    request: Request,
    query_params: QueryParams,
    form_data: FormData,
) -> RequestParams:
    params = {
        "task": _pick_first(query_params, form_data, ["task"], "transcribe"),
        "language": _pick_first(query_params, form_data, ["language", "source_lang"]),
        "output_format": _pick_first(query_params, form_data, ["response_format", "output"], "srt"),
    }
    if "/translations" in request.url.path:
        params["task"] = "translate"
    return params


def _apply_batch_and_diarization_params(
    params: RequestParams,
    query_params: QueryParams,
    form_data: FormData,
    request: Request,
) -> None:
    params["batch_size"] = _parse_int_param(query_params, form_data, "batch_size", config.DEFAULT_BATCH_SIZE)
    params["diarize"] = _parse_bool_param(query_params, form_data, "diarize", False)
    params["min_speakers"] = _parse_int_param(query_params, form_data, "min_speakers", None)
    params["max_speakers"] = _parse_int_param(query_params, form_data, "max_speakers", None)
    params["hf_token"] = form_data.get("hf_token") or request.headers.get("X-HF-Token")


def _apply_prompt_and_format_flags(
    params: RequestParams,
    query_params: QueryParams,
    form_data: FormData,
) -> None:
    params["initial_prompt"] = _pick_first(query_params, form_data, ["initial_prompt"])
    params["subtitle_highlight_words"] = _parse_subtitle_highlight(query_params, form_data)
    params["vad_filter"] = _parse_bool_param(query_params, form_data, "vad_filter", True)
    params["word_timestamps"] = _parse_bool_param(query_params, form_data, "word_timestamps", False)
    clean_audio = _parse_bool_param(query_params, form_data, "clean_audio", None)
    if clean_audio is None:
        clean_audio = _parse_bool_param(query_params, form_data, "vocal_separation", None)
    if clean_audio is None:
        clean_audio = _parse_bool_param(query_params, form_data, "enable_vocal_separation", None)
    params["clean_audio"] = clean_audio
    if params["subtitle_highlight_words"]:
        params["word_timestamps"] = True


def _apply_subtitle_layout_params(
    params: RequestParams,
    query_params: QueryParams,
    form_data: FormData,
) -> None:
    params["max_line_width"] = _parse_int_param(query_params, form_data, "max_line_width", None)
    params["max_line_count"] = _parse_int_param(query_params, form_data, "max_line_count", None)


def _parse_subtitle_highlight(query_params: QueryParams, form_data: FormData) -> bool:
    val = (
        query_params.get("subtitle_highlight_words")
        or query_params.get("SUBTITLE_HIGHLIGHT_WORDS")
        or form_data.get("subtitle_highlight_words")
        or form_data.get("SUBTITLE_HIGHLIGHT_WORDS")
        or os.environ.get("SUBTITLE_HIGHLIGHT_WORDS", "false")
    )
    return str(val).lower() == "true"


def _pick_first(
    query_params: QueryParams,
    form_data: FormData,
    keys: list[str],
    default: Optional[str] = None,
) -> Optional[str]:
    for key in keys:
        value = query_params.get(key)
        if value not in (None, ""):
            return value
        value = form_data.get(key)
        if value not in (None, ""):
            return value
    return default


def _parse_bool_param(query_params: QueryParams, form_data: FormData, key: str, default: Optional[bool]) -> Optional[bool]:
    val = query_params.get(key)
    if val in (None, ""):
        val = form_data.get(key)
    if val in (None, ""):
        return default
    return str(val).lower() in ("true", "1", "yes")


def _parse_int_param(
    query_params: QueryParams,
    form_data: FormData,
    key: str,
    default: Optional[int],
) -> Optional[int]:
    val = query_params.get(key) or form_data.get(key)
    if not val:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _detect_lang_if_needed(lang: Optional[str], path: str) -> Optional[str]:
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


def build_response(
    result: Optional[TranscriptionResult],
    params: RequestParams,
    stats: dict[str, float],
    path: str,
    start: float,
) -> Response:
    """Format final response."""
    if not isinstance(result, dict) or not result:
        return JSONResponse(content={"error": "Transcription failed: no result produced"}, status_code=500)

    _fill_response_stats(stats, result, start)
    _log_completion(params, stats, result)
    fmt = _normalize_output_format(params.get("output_format", "srt"))

    if fmt == "json":
        return JSONResponse(content=result)

    content = _generate_formatted_content(result, params, fmt)
    return _build_file_attachment_response(content, result, params, path, fmt)


def _fill_response_stats(stats: dict[str, float], result: TranscriptionResult, start: float) -> None:
    stats["total"] = time.time() - start
    stats["video_dur"] = result.get("video_duration_sec", 0.0)
    stats["language"] = result.get("language")


def _log_completion(params: RequestParams, stats: dict[str, float], result: TranscriptionResult) -> None:
    perf = result.get("performance", {})
    task_type = "TRANSLATE" if params.get("task") == "translate" else "TRANSCRIBE"
    video_dur = stats.get("video_dur", 0.0)
    total_dur = stats.get("total", 0.0)
    speed = video_dur / total_dur if total_dur > 0 else 0.0

    logger.info(
        "ASR Completed | Type: %s | Lang: %s | Video: %s | Total: %s | Speed: %.2fx | Queue: %s | Isolation: %s | Inference: %s",
        task_type,
        stats["language"],
        utils.format_duration(video_dur),
        utils.format_duration(total_dur),
        speed,
        utils.format_duration(perf.get("queue_sec", 0)),
        utils.format_duration(perf.get("isolation_sec", 0)),
        utils.format_duration(perf.get("inference_sec", 0)),
    )


def _normalize_output_format(output_format: str) -> str:
    fmt = str(output_format or "srt").lower()
    if fmt in {"json", "vtt", "txt", "tsv", "srt"}:
        return fmt
    return "srt"


def _resolve_task_type(params: RequestParams) -> str:
    return "Translation" if params.get("task") == "translate" else "Transcription"


def _log_task_start(task_type: str, params: RequestParams) -> None:
    logger.info(
        "    Task: %s | Format: %s | Lang: %s",
        task_type.upper(),
        params.get("output_format", "srt").upper(),
        params.get("language") or "auto-detect",
    )


def _generate_formatted_content(result: dict[str, Any], params: RequestParams, fmt: str) -> str:
    if fmt == "vtt":
        return utils.generate_vtt(
            result,
            max_line_width=params.get("max_line_width"),
            max_line_count=params.get("max_line_count"),
            highlight_words=params.get("subtitle_highlight_words", False),
        )
    if fmt == "txt":
        return utils.generate_txt(result)
    if fmt == "tsv":
        return utils.generate_tsv(result)
    return utils.generate_srt(
        result,
        max_line_width=params.get("max_line_width"),
        max_line_count=params.get("max_line_count"),
        highlight_words=params.get("subtitle_highlight_words", False),
    )


def _build_file_attachment_response(content: str, result: dict, params: dict, path: str, fmt: str) -> Response:
    subtitle_lang = result.get("language") or params.get("language") or "en"
    fname = os.path.splitext(os.path.basename(path))[0]
    ascii_filename = f"{fname}.{subtitle_lang}-ai.{fmt}".encode("ascii", "ignore").decode("ascii").replace('"', "")
    if not ascii_filename.strip():
        ascii_filename = f"transcription.{subtitle_lang}-ai.{fmt}"
    quoted_filename = urllib.parse.quote(f"{fname}.{subtitle_lang}-ai.{fmt}")
    headers = {"Content-Disposition": f"attachment; filename=\"{ascii_filename}\"; filename*=UTF-8''{quoted_filename}"}
    return Response(content=content, media_type="text/plain", headers=headers)
