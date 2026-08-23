"""Language-detection request coalescing (dedupe) for detect.py.

Split out of detect.py to stay under the 500-line Python file limit. Multiple
concurrent detect-language requests for the same local_path are collapsed into
one real detection run; followers await the leader's result instead of
redoing the (expensive) work.
"""

import asyncio
import concurrent.futures
import json
import logging
import os
import threading
from typing import Awaitable, Callable, Optional, TypedDict

import anyio
from fastapi import Response, UploadFile
from fastapi.responses import JSONResponse

from modules.api.support import request_utils as routes_utils
from modules.inference.runtime import model_manager

logger = logging.getLogger(__name__)

type DetectError = tuple[str, int]
type DetectResponsePayload = dict[str, object] | Response
type CoalescedDetectResult = tuple[DetectResponsePayload | None, DetectError | None]
type RunDetectionInternal = Callable[..., Awaitable[tuple[DetectResponsePayload, CoalescedDetectResult]]]


class DetectRequestContext(TypedDict):
    """Request context used while processing language-detection jobs."""

    resolved_local_path: Optional[str]
    uploaded_file: Optional[UploadFile]
    filename: str
    start_time: float
    worker_context: dict


_INFLIGHT_DETECT_LOCK = threading.Lock()
_INFLIGHT_DETECT_BY_PATH: dict[str, concurrent.futures.Future[CoalescedDetectResult]] = {}


async def handle_coalesced_detect(
    dedupe_key: str,
    filename: str,
    resolved_local_path: Optional[str],
    uploaded_file: Optional[UploadFile],
    start_time: float,
    *,
    worker_context: dict,
    run_detection_internal: RunDetectionInternal,
) -> DetectResponsePayload:
    """Coalesce duplicate detect-language requests for the same dedupe_key: the first
    caller becomes the leader and runs the real detection via run_detection_internal;
    subsequent callers become followers and await the leader's shared result instead."""
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
        run_detection_internal=run_detection_internal,
    )


def build_dedupe_key(resolved_local_path: Optional[str], uploaded_file: Optional[UploadFile]) -> Optional[str]:
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
    routes_utils.apply_worker_context_from_dict(worker_context)
    with model_manager.early_task_registration(
        task_type="Language Detection (Coalesced)",
        filename=filename,
        is_priority=False,
    ):
        routes_utils.log_audio_source_mode(worker_context)
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
    *,
    run_detection_internal: RunDetectionInternal,
) -> DetectResponsePayload:
    """Execute the canonical detect-language task for a dedupe key."""
    try:
        response, result_tuple = await run_detection_internal(
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
