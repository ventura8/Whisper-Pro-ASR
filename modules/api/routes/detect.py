"""
Language Detection Routes for Whisper Pro ASR
"""

import json
import logging
import time
from typing import Optional

import anyio
from fastapi import APIRouter, File, Query, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile as StarletteUploadFile

from modules.api.routes import detect_coalescing
from modules.api.routes.detect_coalescing import CoalescedDetectResult, DetectResponsePayload
from modules.api.support import request_utils as routes_utils
from modules.api.support.local_path import normalize_bazarr_request_params
from modules.core import config, utils
from modules.inference.pipeline import language_detection
from modules.inference.runtime import model_manager

router = APIRouter(tags=["Identification"])
logger = logging.getLogger(__name__)


@router.post("/detect-language")
@router.post("/detectlang")
async def detect_language(
    request: Request,
    local_path: Optional[str] = Query(None),
    audio_file: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None),
):
    """
    Automated Language Identification
    ---
    Identify the primary language of an audio stream.
    """
    utils.THREAD_CONTEXT.reset()
    if not model_manager.is_engine_initialized():
        return Response("Model not loaded", status_code=503)

    try:
        # 1. Parse request parameters dynamically
        form_data = await routes_utils.parse_form_data(request)

        resolved_local_path, uploaded_file = await routes_utils.resolve_and_materialize_upload(
            local_path, audio_file, file, form_data, request
        )

        video_file = routes_utils.extract_video_file(form_data, request)
        worker_context = _build_detect_worker_context(request, form_data, resolved_local_path, video_file)
        routes_utils.apply_worker_context_from_dict(worker_context)

        start_time = time.time()
        filename = routes_utils.get_display_name_early(resolved_local_path, uploaded_file, video_file)
        dedupe_key = detect_coalescing.build_dedupe_key(resolved_local_path, uploaded_file) if config.ENABLE_LD_REQUEST_COALESCING else None

        if dedupe_key:
            return await detect_coalescing.handle_coalesced_detect(
                dedupe_key,
                filename,
                resolved_local_path,
                uploaded_file,
                start_time,
                worker_context=worker_context,
                run_detection_internal=_run_detection_internal,
            )

        return await _run_detection_without_dedupe(resolved_local_path, uploaded_file, filename, start_time, worker_context=worker_context)
    except tuple([Exception]) as e:
        msg, code = routes_utils.handle_error(e, "LD")
        return JSONResponse(content={"error": msg}, status_code=code)


def _build_detect_worker_context(
    request: Request,
    form_data: dict,
    resolved_local_path: Optional[str] = None,
    video_file: Optional[str] = None,
) -> dict:
    params = normalize_bazarr_request_params(_build_request_params(request, form_data))
    if resolved_local_path:
        params["local_path"] = resolved_local_path
    return {
        "caller_info": _get_caller_info(request),
        "request_json": _mask_sensitive_params(params),
        "endpoint": request.url.path,
        "input_flags": getattr(utils.THREAD_CONTEXT, "input_flags", None),
        "audio_source_mode": getattr(utils.THREAD_CONTEXT, "audio_source_mode", None),
        "video_file": video_file,
    }


def _build_request_params(request: Request, form_data: dict) -> dict:
    """Dump form fields into a display/audit dict, skipping file uploads.

    `await request.form()` (Starlette) yields plain `starlette.datastructures.UploadFile`
    instances, not the `fastapi.UploadFile` subclass -- checking against the subclass
    here always failed, so uploads fell through to `str(v)` and leaked a raw
    'UploadFile(filename=...)' repr into the audit payload instead of being skipped."""
    params = dict(request.query_params)
    for k, v in form_data.items():
        if not isinstance(v, StarletteUploadFile):
            params[k] = str(v)
    return params


def _mask_sensitive_params(params: dict) -> dict:
    sanitized = params.copy()
    for key in ("hf_token", "api_key"):
        if key in sanitized:
            sanitized[key] = "[MASKED]"
    return sanitized


def _get_caller_info(request: Request) -> dict:
    ip = "127.0.0.1"
    if request.client:
        ip = request.client.host
    return {
        "ip": ip,
        "user_agent": request.headers.get("User-Agent", "Unknown"),
    }


async def _run_detection_without_dedupe(
    resolved_local_path,
    uploaded_file,
    filename,
    start_time,
    *,
    worker_context,
) -> DetectResponsePayload:
    """Run a single detect-language request without coalescing."""
    response, _ = await _run_detection_internal(resolved_local_path, uploaded_file, filename, start_time, worker_context=worker_context)
    return response


async def _run_detection_internal(
    resolved_local_path,
    uploaded_file,
    filename,
    start_time,
    *,
    worker_context,
) -> tuple[DetectResponsePayload, CoalescedDetectResult]:
    """Run detection and return both the HTTP response and raw (result, err) tuple."""
    model_manager.increment_active_session()

    try:
        # Run the entire priority task including registration inside the thread pool
        # to avoid blocking the FastAPI event loop thread on priority sequential lock.
        result, err = await anyio.to_thread.run_sync(
            lambda: _perform_detect_language_task(
                resolved_local_path,
                uploaded_file,
                filename,
                start_time,
                worker_context=worker_context,
            )
        )
        if err:
            msg, code = err
            return JSONResponse(content={"error": msg}, status_code=code), (None, err)
        return result, (result, None)
    except tuple([Exception]) as e:
        msg, code = routes_utils.handle_error(e, "LD")
        return JSONResponse(content={"error": msg}, status_code=code), (None, (msg, code))
    finally:
        await anyio.to_thread.run_sync(routes_utils.cleanup_files)
        model_manager.decrement_active_session()


def _perform_detect_language_task(
    resolved_local_path,
    uploaded_file,
    filename,
    start_time,
    *,
    worker_context,
):
    """
    Orchestrates the language detection sequence in a background worker thread.

    This runs inside the thread pool to avoid blocking the FastAPI event loop
    when acquiring priority task locks.
    """
    routes_utils.apply_worker_context_from_dict(worker_context)
    with model_manager.early_task_registration(task_type="Language Detection", filename=filename, is_priority=True):
        routes_utils.log_audio_source_mode(worker_context)
        source_path, _, err = routes_utils.initialize_task_context(
            resolved_local_path, uploaded_file, True, video_file=worker_context.get("video_file")
        )
        if err:
            msg, code = err
            model_manager.record_task_failure(msg, code, context="LD")
            return None, err

        model_manager.update_task_progress(None, "Analyzing Stream")

        try:
            result = language_detection.run_voting_detection(source_path, model_manager, start_time)
        except tuple([Exception]) as e:
            msg, code = routes_utils.handle_error(e, "LD")
            model_manager.record_task_failure(msg, code, context="LD")
            return None, (msg, code)

        _log_detection_result(result, start_time)
        model_manager.update_task_metadata(result=result)
        return result, None


def _log_detection_result(result, start_time):
    """Log identification details."""
    elapsed = time.time() - start_time
    detected_lang = result.get("detected_language", "unknown")
    detected_conf = result.get("confidence", 0) * 100
    perf = result.get("performance") or {}

    q_sec = perf.get("queue_sec", 0.0)
    iso_sec = perf.get("isolation_sec", 0.0)
    inf_sec = perf.get("inference_sec", 0.0)
    logger.info(
        "[LD] Completed | Language: %s (%.1f%%) | Total: %.2fs | Queue: %.2fs | Isolation: %.2fs | Inference: %.2fs",
        detected_lang,
        detected_conf,
        elapsed,
        q_sec,
        iso_sec,
        inf_sec,
    )

    candidates = _get_candidates_dict(result)
    top_3 = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:3]
    cand_str = ", ".join([f"{k}:{v * 100:.1f}%" for k, v in top_3])

    q_dur = utils.format_duration(q_sec)
    m_dur = utils.format_duration(perf.get("montage_sec", 0))
    s_dur = utils.format_duration(iso_sec)
    i_dur = utils.format_duration(inf_sec)
    perf_str = f"Queue:{q_dur} | Montage:{m_dur} | Isolation:{s_dur} | Inference:{i_dur}"

    logger.info(
        "LD Completed | Lang: %s (%.1f%%) | Segments: %d | Rank: %s | Phases: %s | Total: %s",
        detected_lang,
        detected_conf,
        result.get("segments_processed", 1),
        cand_str,
        perf_str,
        utils.format_duration(elapsed),
    )

    loggable = {k: v for k, v in result.items() if k != "logs"}
    logger.info("LD Response JSON: %s", json.dumps(loggable, ensure_ascii=False, indent=None))


def _get_candidates_dict(result: dict) -> dict:
    candidates = _select_candidate_source(result)
    return _normalize_candidates(candidates)


def _select_candidate_source(result: dict):
    voting_details = result.get("voting_details")
    if voting_details:
        return voting_details
    all_probabilities = result.get("all_probabilities")
    if all_probabilities:
        return all_probabilities
    return {}


def _normalize_candidates(candidates) -> dict:
    if isinstance(candidates, dict):
        return candidates
    if not isinstance(candidates, list):
        return {}

    normalized = {}
    for item in candidates:
        pair = _to_candidate_pair(item)
        if pair is None:
            return {}
        key, value = pair
        normalized[key] = value
    return normalized


def _to_candidate_pair(item):
    if not isinstance(item, (list, tuple)):
        return None
    if len(item) != 2:
        return None
    return item[0], item[1]
