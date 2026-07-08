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
from typing import Optional

import anyio
from fastapi import APIRouter, File, Query, Request, Response, UploadFile
from fastapi.responses import JSONResponse

from modules.api import routes_utils
from modules.core import config, utils
from modules.inference import language_detection, model_manager

router = APIRouter(tags=["Identification"])
logger = logging.getLogger(__name__)

_INFLIGHT_DETECT_LOCK = threading.Lock()
_INFLIGHT_DETECT_BY_PATH: dict[str, concurrent.futures.Future] = {}


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

        # Setup contextvars request metadata
        params = {
            **dict(request.query_params),
            **{k: str(v) for k, v in form_data.items() if not isinstance(v, UploadFile)},
        }
        sanitized_params = params.copy()
        for sensitive_key in ("hf_token", "api_key"):
            if sensitive_key in sanitized_params:
                sanitized_params[sensitive_key] = "[MASKED]"
        utils.THREAD_CONTEXT.request_json = sanitized_params
        utils.THREAD_CONTEXT.endpoint = request.url.path
        utils.THREAD_CONTEXT.caller_info = {
            "ip": request.client.host if request.client else "127.0.0.1",
            "user_agent": request.headers.get("User-Agent", "Unknown"),
        }

        start_time = time.time()
        filename = routes_utils.get_display_name_early(resolved_local_path, uploaded_file)
        dedupe_key = (
            _build_dedupe_key(resolved_local_path, uploaded_file) if config.ENABLE_LD_REQUEST_COALESCING else None
        )

        if dedupe_key:
            is_leader = False
            with _INFLIGHT_DETECT_LOCK:
                shared_future = _INFLIGHT_DETECT_BY_PATH.get(dedupe_key)
                if shared_future is None:
                    shared_future = concurrent.futures.Future()
                    _INFLIGHT_DETECT_BY_PATH[dedupe_key] = shared_future
                    is_leader = True

            if not is_leader:
                logger.info(
                    "[LD] Coalescing duplicate detect-language request for %s; waiting for in-flight result.",
                    filename,
                )
                return await _await_shared_result_with_dashboard_task(shared_future, dedupe_key, filename)

            return await _run_leader_detection(
                shared_future,
                dedupe_key,
                {
                    "resolved_local_path": resolved_local_path,
                    "uploaded_file": uploaded_file,
                    "filename": filename,
                    "start_time": start_time,
                },
            )

        return await _run_detection_without_dedupe(resolved_local_path, uploaded_file, filename, start_time)
    except tuple([Exception]) as e:
        msg, code = routes_utils.handle_error(e, "LD")
        return JSONResponse(content={"error": msg}, status_code=code)


def _build_dedupe_key(resolved_local_path: Optional[str], uploaded_file: Optional[UploadFile]) -> Optional[str]:
    """Build a stable key for local-path detection requests that can be safely coalesced."""
    if uploaded_file is not None or not resolved_local_path:
        return None
    normalized = os.path.abspath(os.path.normpath(resolved_local_path))
    return f"local_path::{normalized}"


async def _await_shared_result(shared_future: concurrent.futures.Future):
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
    shared_future: concurrent.futures.Future,
    dedupe_key: str,
    filename: str,
):
    """Represent coalesced followers in task telemetry while waiting for leader output."""
    return await anyio.to_thread.run_sync(
        _await_shared_result_with_dashboard_task_sync,
        shared_future,
        dedupe_key,
        filename,
    )


def _await_shared_result_with_dashboard_task_sync(
    shared_future: concurrent.futures.Future,
    dedupe_key: str,
    filename: str,
):
    """Worker-thread follower flow to avoid blocking the event loop on registration."""
    with model_manager.early_task_registration(
        task_type="Language Detection (Coalesced)",
        filename=filename,
        is_priority=True,
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
            return JSONResponse(content={"error": msg}, status_code=code)

        if err:
            msg, code = err
            return JSONResponse(content={"error": msg}, status_code=code)

        if isinstance(result, JSONResponse):
            if result.status_code >= 400:
                model_manager.update_task_metadata(status="failed")
            return result

        model_manager.update_task_metadata(
            status="completed",
            stage="Coalesced Request (Reused Leader Result)",
            result=result,
            progress=100,
        )
        return result


async def _run_leader_detection(
    shared_future: concurrent.futures.Future,
    dedupe_key: str,
    request_context: dict,
):
    """Execute the canonical detect-language task for a dedupe key."""
    try:
        response, result_tuple = await _run_detection_internal(
            request_context["resolved_local_path"],
            request_context["uploaded_file"],
            request_context["filename"],
            request_context["start_time"],
        )
        if not shared_future.done():
            shared_future.set_result(result_tuple)
        return response
    except BaseException as e:
        if not shared_future.done():
            shared_future.set_exception(e)
        if isinstance(e, Exception):
            msg, code = routes_utils.handle_error(e, "LD")
            return JSONResponse(content={"error": msg}, status_code=code)
        raise
    finally:
        if not shared_future.done():
            shared_future.set_exception(RuntimeError("Leader exited early without setting a result."))
        with _INFLIGHT_DETECT_LOCK:
            _INFLIGHT_DETECT_BY_PATH.pop(dedupe_key, None)


async def _run_detection_without_dedupe(resolved_local_path, uploaded_file, filename, start_time):
    """Run a single detect-language request without coalescing."""
    response, _ = await _run_detection_internal(resolved_local_path, uploaded_file, filename, start_time)
    return response


async def _run_detection_internal(resolved_local_path, uploaded_file, filename, start_time):
    """Run detection and return both the HTTP response and raw (result, err) tuple."""
    model_manager.increment_active_session()

    try:
        # Run the entire priority task including registration inside the thread pool
        # to avoid blocking the FastAPI event loop thread on priority sequential lock.
        result, err = await anyio.to_thread.run_sync(
            _perform_detect_language_task, resolved_local_path, uploaded_file, filename, start_time
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


def _perform_detect_language_task(resolved_local_path, uploaded_file, filename, start_time):
    """
    Orchestrates the language detection sequence in a background worker thread.

    This runs inside the thread pool to avoid blocking the FastAPI event loop
    when acquiring priority task locks.
    """
    with model_manager.early_task_registration(task_type="Language Detection", filename=filename, is_priority=True):
        source_path, _, err = routes_utils.initialize_task_context(resolved_local_path, uploaded_file, True)
        if err:
            model_manager.update_task_metadata(status="failed")
            return None, err

        model_manager.update_task_progress(None, "Analyzing Stream")

        result = language_detection.run_voting_detection(source_path, model_manager, start_time)

        _log_detection_result(result, start_time)
        model_manager.update_task_metadata(result=result)
        return result, None


def _log_detection_result(result, start_time):
    """Log identification details."""
    elapsed = time.time() - start_time
    detected_lang = result.get("detected_language", "unknown")
    detected_conf = result.get("confidence", 0) * 100

    candidates = result.get("voting_details") or result.get("all_probabilities") or {}
    if isinstance(candidates, list):
        try:
            candidates = dict(candidates)
        except (TypeError, ValueError):
            candidates = {}
    scans = result.get("segments_processed", 1)
    top_3 = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:3]
    cand_str = ", ".join([f"{k}:{v * 100:.1f}%" for k, v in top_3])

    perf = result.get("performance", {})
    q_dur = utils.format_duration(perf.get("queue_sec", 0))
    m_dur = utils.format_duration(perf.get("montage_sec", 0))
    s_dur = utils.format_duration(perf.get("isolation_sec", 0))
    i_dur = utils.format_duration(perf.get("inference_sec", 0))
    perf_str = f"Queue:{q_dur} | Montage:{m_dur} | Isolation:{s_dur} | Inference:{i_dur}"

    logger.info(
        "LD Completed | Lang: %s (%.1f%%) | Segments: %d | Rank: %s | Phases: %s | Total: %s",
        detected_lang,
        detected_conf,
        scans,
        cand_str,
        perf_str,
        utils.format_duration(elapsed),
    )

    # Log the full JSON response so it appears in the logs
    loggable = {k: v for k, v in result.items() if k != "logs"}
    logger.info("LD Response JSON: %s", json.dumps(loggable, ensure_ascii=False, indent=None))
