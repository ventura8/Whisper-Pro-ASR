"""
Language Detection Routes for Whisper Pro ASR
"""

import asyncio
import concurrent.futures
import json
import logging
import os
import threading
import time
from typing import Optional, TypedDict

import anyio
from fastapi import APIRouter, File, Query, Request, Response, UploadFile
from fastapi.responses import JSONResponse

from modules.api.support import request_utils as routes_utils
from modules.api.support.local_path import normalize_bazarr_request_params
from modules.core import config, utils
from modules.inference.pipeline import language_detection
from modules.inference.runtime import model_manager

router = APIRouter(tags=["Identification"])
logger = logging.getLogger(__name__)

type DetectError = tuple[str, int]
type DetectResponsePayload = dict[str, object] | Response
type CoalescedDetectResult = tuple[DetectResponsePayload | None, DetectError | None]


class DetectRequestContext(TypedDict):
    """Request context used while processing language-detection jobs."""

    resolved_local_path: Optional[str]
    uploaded_file: Optional[UploadFile]
    filename: str
    start_time: float
    worker_context: dict


_INFLIGHT_DETECT_LOCK = threading.Lock()
_INFLIGHT_DETECT_BY_PATH: dict[str, concurrent.futures.Future[CoalescedDetectResult]] = {}


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

        worker_context = _build_detect_worker_context(request, form_data, resolved_local_path)
        _apply_detect_worker_context(worker_context)

        start_time = time.time()
        filename = routes_utils.get_display_name_early(resolved_local_path, uploaded_file)
        dedupe_key = _build_dedupe_key(resolved_local_path, uploaded_file) if config.ENABLE_LD_REQUEST_COALESCING else None

        if dedupe_key:
            return await _handle_coalesced_detect(
                dedupe_key, filename, resolved_local_path, uploaded_file, start_time, worker_context=worker_context
            )

        return await _run_detection_without_dedupe(resolved_local_path, uploaded_file, filename, start_time, worker_context=worker_context)
    except tuple([Exception]) as e:
        msg, code = routes_utils.handle_error(e, "LD")
        return JSONResponse(content={"error": msg}, status_code=code)


def _build_detect_worker_context(
    request: Request,
    form_data: dict,
    resolved_local_path: Optional[str] = None,
) -> dict:
    params = normalize_bazarr_request_params(_build_request_params(request, form_data))
    if resolved_local_path:
        params["local_path"] = resolved_local_path
    return {
        "caller_info": _get_caller_info(request),
        "request_json": _mask_sensitive_params(params),
        "endpoint": request.url.path,
        "input_flags": getattr(utils.THREAD_CONTEXT, "input_flags", None),
    }


def _apply_detect_worker_context(worker_context: dict) -> None:
    utils.THREAD_CONTEXT.request_json = worker_context["request_json"]
    utils.THREAD_CONTEXT.endpoint = worker_context["endpoint"]
    utils.THREAD_CONTEXT.caller_info = worker_context["caller_info"]
    utils.THREAD_CONTEXT.input_flags = worker_context.get("input_flags")


def _apply_worker_context_from_dict(worker_context: dict) -> None:
    utils.THREAD_CONTEXT.request_json = worker_context["request_json"]
    utils.THREAD_CONTEXT.endpoint = worker_context["endpoint"]
    utils.THREAD_CONTEXT.caller_info = worker_context["caller_info"]
    utils.THREAD_CONTEXT.input_flags = worker_context.get("input_flags")


def _build_request_params(request: Request, form_data: dict) -> dict:
    params = dict(request.query_params)
    for k, v in form_data.items():
        if not isinstance(v, UploadFile):
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


async def _handle_coalesced_detect(
    dedupe_key: str,
    filename: str,
    resolved_local_path: Optional[str],
    uploaded_file: Optional[UploadFile],
    start_time: float,
    *,
    worker_context: dict,
) -> DetectResponsePayload:
    is_leader = False
    with _INFLIGHT_DETECT_LOCK:
        shared_future = _INFLIGHT_DETECT_BY_PATH.get(dedupe_key)
        if shared_future is None:
            shared_future = concurrent.futures.Future[CoalescedDetectResult]()
            _INFLIGHT_DETECT_BY_PATH[dedupe_key] = shared_future
            is_leader = True

    if not is_leader:
        logger.info(
            "[LD] Coalescing duplicate detect-language request for %s; waiting for in-flight result.",
            filename,
        )
        return await _await_shared_result_with_dashboard_task(shared_future, dedupe_key, filename, worker_context=worker_context)

    return await _run_leader_detection(
        shared_future,
        dedupe_key,
        {
            "resolved_local_path": resolved_local_path,
            "uploaded_file": uploaded_file,
            "filename": filename,
            "start_time": start_time,
            "worker_context": worker_context,
        },
    )


def _build_dedupe_key(resolved_local_path: Optional[str], uploaded_file: Optional[UploadFile]) -> Optional[str]:
    """Build a stable key for local-path detection requests that can be safely coalesced."""
    if uploaded_file is not None or not resolved_local_path:
        return None
    normalized = os.path.abspath(os.path.normpath(resolved_local_path))
    return f"local_path::{normalized}"


async def _await_shared_result(shared_future: concurrent.futures.Future[CoalescedDetectResult]) -> DetectResponsePayload:
    """Wait for a leader request and return the same response payload."""
    try:
        result, err = await asyncio.wrap_future(shared_future)
    except tuple([Exception]) as e:
        msg, code = routes_utils.handle_error(e, "LD")
        return JSONResponse(content={"error": msg}, status_code=code)

    if err:
        msg, code = err
        return JSONResponse(content={"error": msg}, status_code=code)
    return result


async def _await_shared_result_with_dashboard_task(
    shared_future: concurrent.futures.Future[CoalescedDetectResult],
    dedupe_key: str,
    filename: str,
    *,
    worker_context: dict,
) -> DetectResponsePayload:
    """Represent coalesced followers in task telemetry while waiting for leader output."""
    return await anyio.to_thread.run_sync(
        lambda: _await_shared_result_with_dashboard_task_sync(
            shared_future,
            dedupe_key,
            filename,
            worker_context=worker_context,
        )
    )


def _record_ld_failure(msg: str, code: int) -> None:
    """Persist LD failure metadata for dashboard history."""
    model_manager.record_task_failure(msg, code, context="LD")


def _json_response_failure(response: JSONResponse) -> tuple[str, int]:
    """Extract error message/code from an LD error response payload."""
    code = response.status_code
    try:
        payload = json.loads(response.body)
        if isinstance(payload, dict) and payload.get("error"):
            return str(payload["error"]), code
    except tuple([Exception]):
        pass
    return "Language detection failed", code


def _await_shared_result_with_dashboard_task_sync(
    shared_future: concurrent.futures.Future[CoalescedDetectResult],
    dedupe_key: str,
    filename: str,
    *,
    worker_context: dict,
) -> DetectResponsePayload:
    """Worker-thread follower flow to avoid blocking the event loop on registration."""
    _apply_worker_context_from_dict(worker_context)
    with model_manager.early_task_registration(
        task_type="Language Detection (Coalesced)",
        filename=filename,
        is_priority=False,
    ):
        model_manager.update_task_metadata(
            stage="Coalesced Request (Waiting for Leader)",
            status="queued",
            coalesced=True,
            coalesced_key=dedupe_key,
        )
        try:
            result, err = shared_future.result()
        except tuple([Exception]) as e:
            msg, code = routes_utils.handle_error(e, "LD")
            _record_ld_failure(msg, code)
            return JSONResponse(content={"error": msg}, status_code=code)

        if err:
            msg, code = err
            _record_ld_failure(msg, code)
            return JSONResponse(content={"error": msg}, status_code=code)

        if isinstance(result, JSONResponse):
            if result.status_code >= 400:
                fail_msg, fail_code = _json_response_failure(result)
                _record_ld_failure(fail_msg, fail_code)
            return result

        model_manager.update_task_metadata(
            status="completed",
            stage="Coalesced Request (Reused Leader Result)",
            result=result,
            progress=100,
        )
        return result


async def _run_leader_detection(
    shared_future: concurrent.futures.Future[CoalescedDetectResult],
    dedupe_key: str,
    request_context: DetectRequestContext,
) -> DetectResponsePayload:
    """Execute the canonical detect-language task for a dedupe key."""
    try:
        response, result_tuple = await _run_detection_internal(
            request_context["resolved_local_path"],
            request_context["uploaded_file"],
            request_context["filename"],
            request_context["start_time"],
            worker_context=request_context["worker_context"],
        )
        _safe_set_future_result(shared_future, result_tuple)
        return response
    except BaseException as e:
        _safe_set_future_exception(shared_future, e)
        if isinstance(e, Exception):
            msg, code = routes_utils.handle_error(e, "LD")
            return JSONResponse(content={"error": msg}, status_code=code)
        raise
    finally:
        _safe_set_future_exception(shared_future, RuntimeError("Leader exited early without setting a result."))
        with _INFLIGHT_DETECT_LOCK:
            _INFLIGHT_DETECT_BY_PATH.pop(dedupe_key, None)


def _safe_set_future_result(
    shared_future: concurrent.futures.Future[CoalescedDetectResult],
    result: CoalescedDetectResult,
) -> None:
    if not shared_future.done():
        shared_future.set_result(result)


def _safe_set_future_exception(
    shared_future: concurrent.futures.Future[CoalescedDetectResult],
    exc: BaseException,
) -> None:
    if not shared_future.done():
        shared_future.set_exception(exc)


async def _run_detection_without_dedupe(
    resolved_local_path,
    uploaded_file,
    filename,
    start_time,
    *,
    worker_context,
):
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
):
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
    _apply_worker_context_from_dict(worker_context)
    with model_manager.early_task_registration(task_type="Language Detection", filename=filename, is_priority=True):
        source_path, _, err = routes_utils.initialize_task_context(resolved_local_path, uploaded_file, True)
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
