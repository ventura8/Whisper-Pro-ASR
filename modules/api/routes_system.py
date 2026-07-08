"""
System and Diagnostic Routes for Whisper Pro ASR
"""

import json
import logging
import os

import anyio
from fastapi import APIRouter, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse

from modules.core import config, engine_registry, logging_setup, utils
from modules.inference import model_manager
from modules.monitoring import dashboard, history_manager

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
    # Add 'telemetry' alias for test compatibility if one is missing
    if "system" in stats and "telemetry" not in stats:
        stats["telemetry"] = stats["system"]
    elif "telemetry" in stats and "system" not in stats:
        stats["system"] = stats["telemetry"]
    elif "system" not in stats and "telemetry" not in stats:
        # Fallback for minimal mocks in tests
        stats["telemetry"] = {}
        stats["system"] = {}

    # Add 'engines' for test compatibility
    if "engines" not in stats or not isinstance(stats.get("engines"), dict):
        stats["engines"] = {}

    engine_meta = {
        "selected": config.ASR_ENGINE,
        "source": getattr(config, "ASR_ENGINE_SOURCE", "explicit"),
        "resolution": getattr(
            config,
            "ASR_ENGINE_RESOLUTION",
            getattr(config, "asr_engine_resolution", f"explicit -> {config.ASR_ENGINE}"),
        ),
        "supported": engine_registry.supported_engines(),
    }
    stats["engines"].update(engine_meta)
    stats["asr_engine"] = engine_meta["selected"]
    stats["supported_asr_engines"] = engine_meta["supported"]

    logger.debug(
        "[System] Status check: %d active, %d queued", stats.get("active_sessions", 0), stats.get("queued_sessions", 0)
    )
    return stats


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
def clear_history():
    """
    Purge all task history
    ---
    Clear all task records from the history manager.
    """
    history_manager.clear_history()
    return {"status": "success", "message": "History cleared"}


@router.post("/system/cleanup")
def trigger_cleanup():
    """
    Manually trigger temporary asset cleanup
    ---
    Force removal of old temporary audio files.
    """
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
def download_logs():
    """
    System Log Export
    ---
    Download the system log file.
    """
    log_dir = config.LOG_DIR
    log_name = "whisper_pro.log"
    log_path = os.path.join(log_dir, log_name)

    # Fallback to TEMP_DIR for tests or environments without explicit log volumes
    if not os.path.exists(log_path):
        log_path = os.path.join(config.TEMP_DIR, log_name)
        log_dir = config.TEMP_DIR

    if not os.path.exists(log_path):
        logger.error("[System] Log download failed: File not found at %s", log_path)
        return JSONResponse(content={"error": "Log file not found"}, status_code=404)

    # FORCE FLUSH all logging handlers to ensure the latest logs are on disk
    try:
        for handler in logging.root.handlers:
            handler.flush()
    except (RuntimeError, OSError, ValueError, KeyError, AttributeError, TypeError) as e:
        logger.debug("[System] Minor error during log flush: %s", e)

    try:
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


@router.get("/settings")
@router.get("/system/settings")
def get_settings():
    """View current service settings."""
    return {
        "ASR_MODEL": config.ASR_MODEL,
        "ASR_DEVICE": config.ASR_DEVICE,
        "ASR_ENGINE": config.ASR_ENGINE,
        "TELEMETRY_RETENTION_HOURS": int(os.environ.get("TELEMETRY_RETENTION_HOURS", 24)),
    }


@router.post("/settings")
@router.post("/system/settings")
async def update_settings(request: Request):
    """
    Dynamic Service Configuration
    ---
    Update service settings at runtime.
    """
    try:
        try:
            data = await request.json()
        except (json.JSONDecodeError, ValueError):
            return JSONResponse(content={"error": "Malformed JSON"}, status_code=400)

        if not data:
            return JSONResponse(content={"error": "No data provided"}, status_code=400)

        # Validate settings first
        validated_telemetry_retention = None
        if "telemetry_retention_hours" in data:
            try:
                val = int(data["telemetry_retention_hours"])
                if val < 1:
                    raise ValueError
                validated_telemetry_retention = val
            except (ValueError, TypeError):
                return JSONResponse(
                    content={"error": "telemetry_retention_hours must be a positive integer"}, status_code=400
                )

        validated_log_retention = None
        if "log_retention_days" in data:
            try:
                val = int(data["log_retention_days"])
                if val < 1:
                    raise ValueError
                validated_log_retention = val
            except (ValueError, TypeError):
                return JSONResponse(content={"error": "log_retention_days must be a positive integer"}, status_code=400)

        model_changed = False
        if "ASR_MODEL" in data:
            old_model = os.environ.get("ASR_MODEL")
            if old_model != str(data["ASR_MODEL"]):
                model_changed = True
            config.update_env("ASR_MODEL", data["ASR_MODEL"])
            logger.info("[Settings] ASR Model updated to %s", data["ASR_MODEL"])

        if "ASR_DEVICE" in data:
            old_device = os.environ.get("ASR_DEVICE")
            if old_device != str(data["ASR_DEVICE"]):
                model_changed = True
            config.update_env("ASR_DEVICE", data["ASR_DEVICE"])
            logger.info("[Settings] ASR Device updated to %s", data["ASR_DEVICE"])

        if validated_telemetry_retention is not None:
            config.update_env("TELEMETRY_RETENTION_HOURS", validated_telemetry_retention)
            logger.info("[Settings] Telemetry retention updated to %sh", validated_telemetry_retention)

        if validated_log_retention is not None:
            config.update_env("LOG_RETENTION_DAYS", validated_log_retention)
            logger.info("[Settings] Log retention updated to %sd", validated_log_retention)
            logging_setup.update_log_retention(validated_log_retention)

        # Reload model pool with new settings without blocking the event loop only if model settings changed
        if model_changed:
            await anyio.to_thread.run_sync(model_manager.load_model)

        return {"status": "success", "message": "Settings updated"}
    except (RuntimeError, OSError, ValueError, KeyError, AttributeError, TypeError) as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


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
        "endpoints": [
            "/status",
            "/asr",
            "/v1/audio/transcriptions",
            "/v1/audio/translations",
            "/detect-language",
            "/detectlang",
            "/dashboard",
            "/logs/download",
            "/settings",
            "/analytics",
        ],
        "docs": f"{request.base_url}docs",
    }
