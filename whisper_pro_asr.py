# ruff: noqa: I001
"""
Whisper Pro ASR - Enterprise Transcription Service
Main entry point for the Whisper Pro ASR FastAPI application.
"""

import importlib
import logging
import os
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# CRITICAL: Bootstrap hardware path before ANY other first-party imports
from modules.core import bootstrap

from modules.api import routes_asr, routes_detect, routes_system
from modules.core import config, logging_setup, utils
from modules.inference import model_manager

# Initialize global logger
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    """Manage application startup and shutdown lifecycle hooks."""
    model_manager.init_pool()

    # Dynamic import to break cyclic dependency
    telemetry = importlib.import_module("modules.monitoring.telemetry")
    if not getattr(fastapi_app.state, "testing", False):
        telemetry.start_telemetry_loop()

    yield
    # Cleanup on shutdown if needed


def create_app(testing=False):
    """Enterprise FastAPI Factory."""
    logging_setup.setup_logging()
    logging_setup.log_banner()
    utils.cleanup_old_files(config.LOG_DIR, days=config.LOG_RETENTION_DAYS)
    logger.debug("Using bootstrap configuration: %s", bootstrap)

    if config.VERIFY_RUNTIME:
        verify_runtime_integrity()

    fastapi_app = FastAPI(
        title="Whisper Pro ASR API",
        description="Enterprise-grade Whisper Automatic Speech Recognition web service",
        version=config.VERSION,
        lifespan=lifespan,
        docs_url=None,  # Disabled native docs to override with customized Swagger
        redoc_url=None,
    )

    fastapi_app.state.testing = testing

    # Configure CORS
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Configure Request Lifecycle Logging & Storage Hygiene Middleware
    @fastapi_app.middleware("http")
    async def request_lifecycle_middleware(request: Request, call_next):
        request.state.start_time = time.time()
        client_ip = request.headers.get("X-Forwarded-For", request.client.host if request.client else "127.0.0.1")
        content_length = request.headers.get("content-length", "0")
        body_log = f" | Body: {content_length} bytes" if content_length != "0" else ""

        log_func = logger.debug if request.url.path in ["/status", "/"] else logger.info
        log_func(">>> %s %s [Source: %s]%s", request.method, request.url.path, client_ip, body_log)

        # Scope tracked files per request using a contextvar and request.state
        tracked_list = []
        token = utils.REQUEST_TRACKED_FILES_VAR.set(tracked_list)
        request.state.tracked_files = tracked_list

        response = None
        try:
            response = await call_next(request)
        finally:
            utils.cleanup_tracked_files(request)
            utils.REQUEST_TRACKED_FILES_VAR.reset(token)
            duration = time.time() - request.state.start_time
            status_code = response.status_code if response is not None else 500
            log_func = logger.debug if request.url.path in ["/status", "/"] else logger.info
            log_func(
                "<<< %s %s [%d] | Latency: %s",
                request.method,
                request.url.path,
                status_code,
                utils.format_duration(duration),
            )

        return response

    # Mount static assets
    fastapi_app.mount("/static", StaticFiles(directory="static"), name="static")

    # Custom Swagger Endpoint for themed documentation
    @fastapi_app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        swagger_js = (
            "/static/swagger-ui-bundle.js"
            if os.path.exists("static/swagger-ui-bundle.js")
            else "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"
        )
        swagger_css = (
            "/static/swagger-ui.css"
            if os.path.exists("static/swagger-ui.css")
            else "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css"
        )
        swagger_fav = (
            "/static/favicon.png"
            if os.path.exists("static/favicon.png")
            else "https://fastapi.tiangolo.com/img/favicon.png"
        )

        res = get_swagger_ui_html(
            openapi_url=fastapi_app.openapi_url,
            title=fastapi_app.title + " - Swagger UI",
            swagger_js_url=swagger_js,
            swagger_css_url=swagger_css,
            swagger_favicon_url=swagger_fav,
        )
        html = res.body.decode("utf-8")
        if os.path.exists("static/swagger-theme.css"):
            theme_link = '<link rel="stylesheet" type="text/css" href="/static/swagger-theme.css">'
            html = html.replace("</head>", f"  {theme_link}\n</head>")
        return HTMLResponse(content=html)

    # Register API Routers
    fastapi_app.include_router(routes_system.router)
    fastapi_app.include_router(routes_asr.router)
    fastapi_app.include_router(routes_detect.router)

    return fastapi_app


def verify_runtime_integrity():
    """Safety check for critical AI backends."""
    try:
        ort = importlib.import_module("onnxruntime")
        logger.info("[System] Runtime: ONNX %s | Providers: %s", ort.__version__, ort.get_available_providers())
    except (ImportError, AttributeError):
        logger.warning("[System] ONNX Runtime not detected - Check hardware path patching!")


app = create_app()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000, log_config=None)
