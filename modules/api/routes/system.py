"""
System and Diagnostic Routes for Whisper Pro ASR
"""

import json
import logging
import os
from typing import Optional

import anyio
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse

from modules.api.support import security
from modules.core import config, engine_registry, logging_setup, utils
from modules.inference.runtime import model_manager
from modules.monitoring import dashboard, history_manager, telemetry_manager

router = APIRouter(tags=["System"])
logger = logging.getLogger(__name__)


@router.get("/")
def root(request: Request):
    """
    Service Health Check / Dashboard
    ---
    Returns JSON health status or HTML dashboard depending on Accept header.
    """
    if "text/html" in request.headers.get("accept", ""):
        return HTMLResponse(content=dashboard.get_dashboard_html())

    logger.info("[System] Health check (JSON): OK")
    return {
        "message": "Whisper ASR Webservice is working",
        "status": "healthy",
        "app": config.APP_NAME,
        "version": config.VERSION,
        "dashboard": f"{request.base_url}dashboard",
    }


@router.get("/status")
@router.get("/system/stats")
def status():
    """
    Hardware and Model Diagnostics
    ---
    Returns CPU, Memory, GPU/NPU utilization and active session counts.
    """
    stats = dashboard.get_status_data()
    _normalize_status_aliases(stats)
    _update_engine_metadata(stats)

    logger.debug("[System] Status check: %d active, %d queued", stats.get("active_sessions", 0), stats.get("queued_sessions", 0))
    return stats


def _normalize_status_aliases(stats: dict):
    has_system = "system" in stats
    has_telemetry = "telemetry" in stats
    if has_system == has_telemetry:
        _ensure_default_status_maps(stats, has_system)
        return
    _mirror_missing_status_alias(stats, has_system)


def _mirror_missing_status_alias(stats: dict, has_system: bool):
    if has_system:
        stats["telemetry"] = stats["system"]
    else:
        stats["system"] = stats["telemetry"]


def _ensure_default_status_maps(stats: dict, has_system: bool):
    if not has_system:
        stats["telemetry"] = {}
        stats["system"] = {}


def _update_engine_metadata(stats: dict):
    if "engines" not in stats or not isinstance(stats.get("engines"), dict):
        stats["engines"] = {}

    resolution = getattr(config, "ASR_ENGINE_RESOLUTION", None)
    if resolution is None:
        resolution = getattr(config, "asr_engine_resolution", f"explicit -> {config.ASR_ENGINE}")

    engine_meta = {
        "selected": config.ASR_ENGINE,
        "source": getattr(config, "ASR_ENGINE_SOURCE", "explicit"),
        "resolution": resolution,
        "supported": engine_registry.supported_engines(),
    }
    stats["engines"].update(engine_meta)
    stats["asr_engine"] = engine_meta["selected"]
    stats["supported_asr_engines"] = engine_meta["supported"]


@router.get("/history")
@router.get("/system/history")
def get_history():
    """
    Retrieve full task history
    ---
    Get the list of recently completed and active tasks.
    """
    return history_manager.get_history()


@router.post("/system/history/clear")
def clear_history(request: Request):
    """
    Purge all task history
    ---
    Clear all task records from the history manager.
    """
    auth_err = security.verify_admin_request(request)
    if auth_err:
        return JSONResponse(content={"error": auth_err[0]}, status_code=auth_err[1])

    security.audit_log_admin_action(request, "clear_history")
    history_manager.clear_history()
    return {"status": "success", "message": "History cleared"}


@router.post("/system/telemetry/clear")
def clear_telemetry(request: Request):
    """
    Purge all telemetry history
    ---
    Clear all telemetry snapshots from the database.
    """
    auth_err = security.verify_admin_request(request)
    if auth_err:
        return JSONResponse(content={"error": auth_err[0]}, status_code=auth_err[1])

    try:
        security.audit_log_admin_action(request, "clear_telemetry")
        telemetry_manager.clear_telemetry_history()
        return {"status": "success", "message": "Telemetry cleared"}
    except OSError as e:
        logger.error("[System] Failed to clear telemetry history: %s", e)
        raise HTTPException(status_code=500, detail="Failed to clear telemetry") from e


@router.post("/system/cleanup")
def trigger_cleanup(request: Request):
    """
    Manually trigger temporary asset cleanup
    ---
    Force removal of old temporary audio files.
    """
    auth_err = security.verify_admin_request(request)
    if auth_err:
        return JSONResponse(content={"error": auth_err[0]}, status_code=auth_err[1])

    security.audit_log_admin_action(request, "trigger_cleanup")
    utils.purge_temporary_assets()
    utils.cleanup_old_files(config.LOG_DIR, days=config.LOG_RETENTION_DAYS)
    return {"status": "success", "message": "Cleanup triggered"}


@router.get("/dashboard")
def render_dashboard():
    """
    Direct Dashboard Access
    ---
    View the HTML monitoring dashboard.
    """
    return HTMLResponse(content=dashboard.get_dashboard_html())


@router.get("/logs/download")
def download_logs(request: Request):
    """
    System Log Export
    ---
    Download the system log file.
    """
    auth_err = security.verify_admin_request(request)
    if auth_err:
        return JSONResponse(content={"error": auth_err[0]}, status_code=auth_err[1])

    log_name = "whisper_pro.log"
    log_path = _resolve_log_path(log_name)

    if not log_path:
        return JSONResponse(content={"error": "Log file not found"}, status_code=404)

    _flush_log_handlers()

    try:
        security.audit_log_admin_action(request, "download_logs")
        with open(log_path, "rb") as f:
            content = f.read()
        headers = {
            "Content-Disposition": f'attachment; filename="{log_name}"',
            "Cache-Control": "no-cache, no-store, must-revalidate, post-check=0, pre-check=0",
            "Pragma": "no-cache",
            "Expires": "0",
        }
        return Response(content=content, media_type="text/plain", headers=headers)
    except (RuntimeError, OSError, ValueError, KeyError, AttributeError, TypeError) as e:
        logger.error("[System] Log download error: %s", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)


def _resolve_log_path(log_name: str) -> Optional[str]:
    path = os.path.join(config.LOG_DIR, log_name)
    if os.path.exists(path):
        return path
    path = os.path.join(config.TEMP_DIR, log_name)
    if os.path.exists(path):
        return path
    logger.error("[System] Log download failed: File not found at %s", path)
    return None


def _flush_log_handlers():
    try:
        for handler in logging.root.handlers:
            handler.flush()
    except (RuntimeError, OSError, ValueError, KeyError, AttributeError, TypeError) as e:
        logger.debug("[System] Minor error during log flush: %s", e)


@router.get("/settings")
@router.get("/system/settings")
def get_settings(request: Request):
    """View current service settings (admin-protected)."""
    auth_err = security.verify_admin_request(request)
    if auth_err:
        return JSONResponse(content={"error": auth_err[0]}, status_code=auth_err[1])
    return {
        "ASR_MODEL": config.ASR_MODEL,
        "ASR_DEVICE": config.ASR_DEVICE,
        "ASR_ENGINE": config.ASR_ENGINE,
        "TELEMETRY_RETENTION_HOURS": int(os.environ.get("TELEMETRY_RETENTION_HOURS", 24)),
        "LOG_RETENTION_DAYS": int(os.environ.get("LOG_RETENTION_DAYS", config.LOG_RETENTION_DAYS)),
    }


@router.post("/settings")
@router.post("/system/settings")
async def update_settings(request: Request):
    """
    Dynamic Service Configuration
    ---
    Update service settings at runtime.
    """
    auth_err = security.verify_admin_request(request)
    if auth_err:
        return JSONResponse(content={"error": auth_err[0]}, status_code=auth_err[1])

    try:
        return await _update_settings_impl(request)
    except (RuntimeError, OSError, ValueError, KeyError, AttributeError, TypeError) as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


async def _update_settings_impl(request: Request):
    data, early_err = await _parse_and_validate_settings_payload(request)
    if early_err:
        return early_err
    security.audit_log_admin_action(request, "update_settings", f"keys={list(data.keys())}")
    model_changed = await anyio.to_thread.run_sync(_apply_settings_updates, data)
    if model_changed:
        await anyio.to_thread.run_sync(model_manager.load_model)
    return {"status": "success", "message": "Settings updated"}


async def _parse_and_validate_settings_payload(request: Request) -> tuple[dict, Optional[JSONResponse]]:
    data, parse_err = await _parse_settings_payload(request)
    if parse_err:
        return {}, parse_err
    empty_err = _validate_non_empty_settings_payload(data)
    if empty_err:
        return {}, empty_err
    validation_err = _validate_settings_data(data)
    if validation_err:
        return {}, validation_err
    return data, None


async def _parse_settings_payload(request: Request) -> tuple[dict, Optional[JSONResponse]]:
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            return {}, JSONResponse(content={"error": "Malformed JSON"}, status_code=400)
        return payload, None
    except (json.JSONDecodeError, ValueError):
        return {}, JSONResponse(content={"error": "Malformed JSON"}, status_code=400)


def _validate_non_empty_settings_payload(data: dict) -> Optional[JSONResponse]:
    if data:
        return None
    return JSONResponse(content={"error": "No data provided"}, status_code=400)


def _validate_model_and_device(data: dict) -> Optional[JSONResponse]:
    if "ASR_MODEL" in data and not security.is_valid_model_name(str(data["ASR_MODEL"])):
        return JSONResponse(content={"error": "Invalid or disallowed ASR_MODEL"}, status_code=400)
    if "ASR_DEVICE" in data and not security.is_valid_device(str(data["ASR_DEVICE"])):
        return JSONResponse(content={"error": "Invalid ASR_DEVICE"}, status_code=400)
    return None


def _validate_retention_ranges(data: dict) -> Optional[JSONResponse]:
    if _is_invalid_positive_int_setting(data, "telemetry_retention_hours", min_val=1, max_val=720):
        return JSONResponse(
            content={"error": "telemetry_retention_hours must be an integer between 1 and 720"},
            status_code=400,
        )
    if _is_invalid_positive_int_setting(data, "log_retention_days", min_val=1, max_val=90):
        return JSONResponse(
            content={"error": "log_retention_days must be an integer between 1 and 90"},
            status_code=400,
        )
    return None


def _validate_settings_data(data: dict) -> Optional[JSONResponse]:
    model_err = _validate_model_and_device(data)
    if model_err:
        return model_err
    return _validate_retention_ranges(data)


def _is_invalid_positive_int_setting(data: dict, key: str, min_val: int = 1, max_val: int = 1000) -> bool:
    if key not in data:
        return False
    raw_value = data[key]
    if isinstance(raw_value, (bool, float)):
        return True
    try:
        val = int(raw_value)
        return val < min_val or val > max_val
    except (ValueError, TypeError):
        return True


def _apply_settings_updates(data: dict) -> bool:
    model_changed = False
    model_changed = _update_model_setting(data, "ASR_MODEL", model_changed)
    model_changed = _update_model_setting(data, "ASR_DEVICE", model_changed)
    _update_telemetry_retention_setting(data)
    _update_log_retention_setting(data)

    return model_changed


def _update_model_setting(data: dict, env_key: str, model_changed: bool) -> bool:
    if env_key not in data:
        return model_changed
    old_val = os.environ.get(env_key)
    new_val = str(data[env_key])
    if old_val != new_val:
        model_changed = True
    config.update_env(env_key, data[env_key])
    logger.info("[Settings] %s updated to %s", env_key.replace("_", " "), data[env_key])
    return model_changed


def _update_telemetry_retention_setting(data: dict):
    if "telemetry_retention_hours" in data:
        val = int(data["telemetry_retention_hours"])
        config.update_env("TELEMETRY_RETENTION_HOURS", val)
        logger.info("[Settings] Telemetry retention updated to %sh", val)


def _update_log_retention_setting(data: dict):
    if "log_retention_days" in data:
        val = int(data["log_retention_days"])
        config.update_env("LOG_RETENTION_DAYS", val)
        logger.info("[Settings] Log retention updated to %sd", val)
        logging_setup.update_log_retention(val)


@router.get("/analytics")
@router.get("/system/analytics")
def get_analytics(request: Request):
    """
    Retrieve service usage analytics
    ---
    Get cumulative and daily breakdown of tasks and durations.
    """
    if "text/html" in request.headers.get("accept", ""):
        return HTMLResponse(content=dashboard.get_analytics_html())

    data = history_manager.get_analytics_data()
    return data


@router.get("/help")
def help_endpoint(request: Request):
    """
    API Discovery
    """
    return {
        "app": config.APP_NAME,
        "version": config.VERSION,
        "endpoints": sorted(security.API_KEY_PROTECTED_PATHS | {"/dashboard", "/logs/download", "/settings"}),
        "docs": f"{request.base_url}docs",
    }
