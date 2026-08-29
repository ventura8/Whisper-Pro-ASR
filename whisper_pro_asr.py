"""
Whisper Pro ASR - Enterprise Transcription Service
Main entry point for the Whisper Pro ASR FastAPI application.

Spawn-safety: ``multiprocessing`` ``spawn`` re-imports this file as
``__mp_main__`` before the WhisperX worker target runs. The guarded block below
must not execute in that path, or the main ``torch`` stack lands in the child
``sys.modules`` before isolation activates.
"""

# Spawn bootstrap re-imports this file as __mp_main__; skip app imports there.
if __name__ != "__mp_main__":
    import importlib
    import logging
    import os
    import time
    from contextlib import asynccontextmanager
    from typing import AsyncGenerator, Awaitable, Callable
    from urllib.parse import quote

    import uvicorn
    from fastapi import FastAPI, Request, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.openapi.docs import get_swagger_ui_html
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    from starlette.datastructures import QueryParams

    from modules.api.routes import asr as routes_asr
    from modules.api.routes import detect as routes_detect
    from modules.api.routes import system as routes_system
    from modules.api.support import security

    # CRITICAL: Bootstrap hardware path before ANY other first-party imports
    from modules.core import bootstrap, config, logging_setup, model_provisioning, utils
    from modules.inference.runtime import model_manager

    # Initialize global logger
    logger = logging.getLogger(__name__)

    def _warm_up_torch_stack() -> None:
        """Import torch/torchvision once, single-threaded, before any request thread does.

        UVR preprocessing hot-reloads onnxruntime's sys.modules entry per-request
        (see modules/inference/pipeline/openvino_resolver.py); torch/torchvision's
        own first import racing with that reload from a request-handling thread
        has been observed to leave torchvision partially initialized ("operator
        torchvision::nms does not exist" / circular-import errors). Importing them
        here removes the race by ensuring the first import always happens before
        the server accepts any requests.

        Always attempted regardless of ENABLE_VOCAL_SEPARATION: a request can
        activate vocal isolation per-call via `clean_audio=true` even when the
        global default is off (see asr.py's `_apply_prompt_and_format_flags`),
        which would otherwise hit the exact same unprotected first-import race
        this warm-up exists to prevent.
        """
        try:
            torch = importlib.import_module("torch")
            torchvision = importlib.import_module("torchvision")
        except ImportError:
            logger.debug("[Startup] torch/torchvision not installed; skipping warm-up.")
            return
        except (RuntimeError, AttributeError, OSError) as exc:
            # A broken/partial torch install (e.g. mismatched CUDA build, corrupted
            # wheel, or a shared-library load failure surfacing as OSError) can raise
            # these instead of ImportError. Warm-up is a best-effort optimization, not
            # a hard dependency -- log and let startup continue rather than crash the
            # whole app over it.
            logger.warning("[Startup] torch/torchvision warm-up failed; continuing without it: %s", exc)
            return
        logger.debug("[Startup] Warmed up torch %s / torchvision %s", torch.__version__, torchvision.__version__)

    @asynccontextmanager
    async def lifespan(fastapi_app: FastAPI) -> AsyncGenerator[None, None]:
        """Manage application startup and shutdown lifecycle hooks."""
        _warm_up_torch_stack()
        model_manager.init_pool()

        # Models are not baked into the image; fetch them into the persistent cache
        # volume in the background so the server binds immediately. Tasks submitted
        # before this finishes wait in the scheduler queue rather than failing.
        if not getattr(fastapi_app.state, "testing", False):
            model_provisioning.start_background_provisioning(config)

        # Dynamic import to break cyclic dependency
        telemetry = importlib.import_module("modules.monitoring.telemetry")
        if not getattr(fastapi_app.state, "testing", False):
            telemetry.start_telemetry_loop()

        yield
        # Cleanup on shutdown if needed

    def sanitize_query_params(query_params: QueryParams | None) -> str:
        """Sanitize query parameters for safe logging."""
        if not query_params:
            return ""

        safe_fields = {"beam_size", "task", "language", "vad_filter", "word_timestamps", "clean_audio"}
        items = []
        for k, v in query_params.multi_items():
            if k.lower() in safe_fields:
                items.append(f"{quote(k)}={quote(v)}")
        return "&".join(items)

    def create_app(testing: bool = False) -> FastAPI:
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
        cors_origins = security.get_cors_origins()
        if cors_origins:
            fastapi_app.add_middleware(
                CORSMiddleware,
                allow_origins=cors_origins,
                allow_credentials=False,
                allow_methods=["*"],
                allow_headers=["*"],
            )

        def _optional_request_log_suffixes(request: Request) -> tuple[str, str, str]:
            """Build the optional ' | Body:'/' | Params:'/' | Content-Type:' log
            suffixes, each empty when the corresponding header/query data is absent."""
            content_length = request.headers.get("content-length", "0")
            body_log = f" | Body: {content_length} bytes" if content_length != "0" else ""
            query_str = sanitize_query_params(request.query_params)
            params_log = f" | Params: {query_str}" if query_str else ""
            content_type = request.headers.get("content-type", "")
            ct_log = f" | Content-Type: {content_type}" if content_type else ""
            return body_log, params_log, ct_log

        def _log_request_start(request: Request) -> Callable[..., None]:
            """Log the incoming request line and return the log function used
            (debug for noisy health-check paths, info otherwise) so the completion
            log below is logged at the same level."""
            client_ip = request.headers.get("X-Forwarded-For", request.client.host if request.client else "127.0.0.1")
            body_log, params_log, ct_log = _optional_request_log_suffixes(request)

            log_func = logger.debug if request.url.path in ("/status", "/") else logger.info
            log_func(">>> %s %s [Source: %s]%s%s%s", request.method, request.url.path, client_ip, body_log, params_log, ct_log)
            return log_func

        def _log_request_complete(request: Request, response: Response | None, log_func: Callable[..., None]) -> None:
            """Log the completion line using the log function chosen at request start."""
            duration = time.time() - request.state.start_time
            status_code = response.status_code if response is not None else 500
            log_func(
                "<<< %s %s [%d] | Latency: %s",
                request.method,
                request.url.path,
                status_code,
                utils.format_duration(duration),
            )

        # Configure Request Lifecycle Logging & Storage Hygiene Middleware
        @fastapi_app.middleware("http")
        async def request_lifecycle_middleware(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
            request.state.start_time = time.time()
            log_func = _log_request_start(request)

            # Scope tracked files per request using a contextvar and request.state
            tracked_list = []
            token = utils.REQUEST_TRACKED_FILES_VAR.set(tracked_list)
            request.state.tracked_files = tracked_list

            response = None
            try:
                auth_response = security.api_key_rejection_response(request)
                response = auth_response if auth_response is not None else await call_next(request)
            finally:
                utils.cleanup_tracked_files(request)
                utils.REQUEST_TRACKED_FILES_VAR.reset(token)
                _log_request_complete(request, response, log_func)

            return response

        # Mount static assets
        fastapi_app.mount("/static", StaticFiles(directory="static"), name="static")

        # Custom Swagger Endpoint for themed documentation
        @fastapi_app.get("/docs", include_in_schema=False)
        async def custom_swagger_ui_html() -> HTMLResponse:
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
            swagger_fav = "/static/favicon.png" if os.path.exists("static/favicon.png") else "https://fastapi.tiangolo.com/img/favicon.png"

            res = get_swagger_ui_html(
                openapi_url=fastapi_app.openapi_url or "",
                title=fastapi_app.title + " - Swagger UI",
                swagger_js_url=swagger_js,
                swagger_css_url=swagger_css,
                swagger_favicon_url=swagger_fav,
            )
            html = bytes(res.body).decode("utf-8")
            if os.path.exists("static/swagger-theme.css"):
                theme_link = '<link rel="stylesheet" type="text/css" href="/static/swagger-theme.css">'
                html = html.replace("</head>", f"  {theme_link}\n</head>")
            return HTMLResponse(content=html)

        # Register API Routers
        fastapi_app.include_router(routes_system.router)
        fastapi_app.include_router(routes_asr.router)
        fastapi_app.include_router(routes_detect.router)

        return fastapi_app

    def verify_runtime_integrity() -> None:
        """Safety check for critical AI backends."""
        try:
            ort = importlib.import_module("onnxruntime")
            logger.info("[System] Runtime: ONNX %s | Providers: %s", ort.__version__, ort.get_available_providers())
        except (ImportError, AttributeError):
            logger.warning("[System] ONNX Runtime not detected - Check hardware path patching!")

    app = create_app()

    if __name__ == "__main__":
        uvicorn.run(app, host=config.HOST, port=9000, log_config=None)
