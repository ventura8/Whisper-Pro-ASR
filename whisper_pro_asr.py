"""
Whisper Pro ASR - Enterprise Transcription Service
Main entry point for the Whisper Pro ASR Flask application.
"""
import importlib
import logging
import time
from flask import Flask, request, jsonify
from flasgger import Swagger  # pylint: disable=import-error

# CRITICAL: Bootstrap hardware path before ANY other first-party imports
from modules import bootstrap  # pylint: disable=unused-import

# First-party imports (now safe due to bootstrap auto-initialization)
from modules import config, logging_setup, utils
from modules.inference import model_manager
from modules.api import routes_system, routes_asr, routes_detect

# Initialize global logger
logger = logging.getLogger(__name__)


def _register_error_handlers(flask_app):
    """Register system-level error handlers."""
    @flask_app.errorhandler(404)
    def not_found(_):
        return jsonify({"error": "Endpoint not found"}), 404

    @flask_app.errorhandler(500)
    def server_error(err):
        logger.error("[System] Unhandled Server Error: %s", err)
        return jsonify({"error": "Internal server error"}), 500


def _setup_request_lifecycle(flask_app):
    """Inject telemetry and CORS into the request lifecycle."""
    @flask_app.before_request
    def start_timer():
        request.start_time = time.time()
        client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
        body_log = f" | Body: {len(request.data)} bytes" if request.data else ""

        log_func = logger.debug if request.path in ["/status", "/"] else logger.info
        log_func(">>> %s %s [Source: %s]%s",
                 request.method, request.path, client_ip, body_log)

    @flask_app.after_request
    def log_request_done(response):
        duration = time.time() - getattr(request, 'start_time', time.time())
        log_func = logger.debug if request.path in ["/status", "/"] else logger.info
        log_func("<<< %s %s [%d] | Latency: %s",
                 request.method, request.path, response.status_code,
                 utils.format_duration(duration))

        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response

    @flask_app.teardown_request
    def teardown_cleanup(exception=None):
        """Final catch-all to ensure storage hygiene after every request."""
        if exception:
            logger.debug("[System] Request teardown with exception: %s", exception)
        utils.cleanup_tracked_files()


def _setup_swagger(flask_app):
    """Configure Swagger Documentation."""
    swagger_config = {
        "headers": [],
        "specs": [
            {
                "endpoint": 'apispec_1',
                "route": '/apispec_1.json',
                "rule_filter": lambda rule: True,
                "model_filter": lambda tag: True,
            }
        ],
        "static_url_path": "/flasgger_static",
        "swagger_ui": True,
        "specs_route": "/docs"
    }
    flask_app.config['SWAGGER'] = {
        'title': 'Whisper Pro ASR API',
        'uiversion': 3,
        'openapi': '3.0.1'
    }
    Swagger(flask_app, config=swagger_config)


def create_app():
    """Enterprise Flask Factory."""
    logging_setup.setup_logging()
    logging_setup.log_banner()

    if config.VERIFY_RUNTIME:
        _verify_runtime_integrity()

    flask_app = Flask(__name__)
    _setup_swagger(flask_app)

    flask_app.register_blueprint(routes_system.bp)
    flask_app.register_blueprint(routes_asr.bp)
    flask_app.register_blueprint(routes_detect.bp)

    _register_error_handlers(flask_app)
    _setup_request_lifecycle(flask_app)

    model_manager.init_pool()

    # Dynamic import to break cyclic dependency
    telemetry = importlib.import_module("modules.monitoring.telemetry")
    if not flask_app.config.get('TESTING'):
        telemetry.start_telemetry_loop()

    return flask_app


def _verify_runtime_integrity():
    """Safety check for critical AI backends."""
    try:
        ort = importlib.import_module("onnxruntime")
        logger.info("[System] Runtime: ONNX %s | Providers: %s",
                    ort.__version__, ort.get_available_providers())
    except (ImportError, AttributeError):
        logger.warning("[System] ONNX Runtime not detected - Check hardware path patching!")


if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=9000, debug=config.DEBUG_MODE, use_reloader=False)
