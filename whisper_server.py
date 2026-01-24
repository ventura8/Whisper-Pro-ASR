"""
Whisper Pro ASR - Mission-Critical Transcription Service

This is the primary runtime entry point for the Whisper Pro ASR service.
It manages dynamic hardware context resolution (Intel NPU vs NVIDIA GPU),
orchesTrates the Flask application lifecycle, and triggers eager AI 
model loading to ensure zero-latency first requests.
"""
import os
import sys
import logging
import importlib
import time

from flasgger import Swagger  # pylint: disable=import-error
from flask import Flask, request  # pylint: disable=import-error

from modules import config, logging_setup, model_manager, routes, utils

# --- [DYNAMIC HARDWARE CONTEXT LOADER] ---


def _initialize_hardware_path():
    """Resolve and inject the appropriate hardware library context."""
    use_nvidia, context_reason = _detect_hardware()
    boot_logger = _setup_boot_logger()

    # Inject the prioritized library set into the Python search path
    lib_root = "/app/libs"
    target_lib = os.path.join(lib_root, "nvidia" if use_nvidia else "intel")

    if os.path.exists(target_lib):
        if target_lib not in sys.path:
            sys.path.insert(0, target_lib)

        # Force reload if onnxruntime was already imported
        if "onnxruntime" in sys.modules:
            importlib.reload(sys.modules["onnxruntime"])

        boot_logger.info("[System] Context: %s -> Path: %s",
                         context_reason, target_lib)


def _detect_hardware():
    """Probe system environment for available hardware acceleration."""
    def env_map(key):
        return os.environ.get(key, "AUTO").upper()

    req_asr = env_map("ASR_DEVICE")
    req_prep = env_map("ASR_PREPROCESS_DEVICE")

    # Priority 1: User-defined hardware overrides
    if "CUDA" in [req_asr, req_prep]:
        return True, "Explicit CUDA override"
    if any(d in ["NPU", "GPU", "CPU"] for d in [req_asr, req_prep]):
        return False, f"Explicit Intel/Generic override ({req_asr}/{req_prep})"

    # Priority 2: Automated hardware-assisted probing
    if os.path.exists("/proc/driver/nvidia/version"):
        return True, "Detected NVIDIA Silicon (/proc)"

    try:
        # Use a lightweight probe via ctranslate2 if available in base path
        import ctranslate2 # pylint: disable=import-outside-toplevel, import-error
        if ctranslate2.get_cuda_device_count() > 0:
            return True, f"Detected {ctranslate2.get_cuda_device_count()} NVIDIA GPU(s) (Probe)"
    except Exception: # pylint: disable=broad-exception-caught
        pass

    return False, "Assuming Intel/CPU Silicon"


def _setup_boot_logger():
    """Initialize a thread-safe logger for the bootstrap phase."""
    boot_logger = logging.getLogger("Bootstrap")
    if not boot_logger.handlers:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        boot_logger.addHandler(sh)
        boot_logger.setLevel(logging.INFO)
        boot_logger.propagate = False
    return boot_logger


# Execute path patching before any heavy AI modules are initialized
_initialize_hardware_path()

# pylint: disable=wrong-import-position


# Initialize global logger
logger = logging.getLogger(__name__)

# Verification of runtime integrity
try:
    import onnxruntime as ort
    logger.debug(
        "[System] Inference Engine: ONNX Runtime loaded from %s", ort.__file__)
except ImportError:
    logger.warning(
        "[System] CRITICAL: No inference runtime found in current path!")
except Exception as e:  # pylint: disable=broad-exception-caught
    logger.error("[System] Failed to verify inference runtime: %s", e)


# --- [APPLICATION FACTORY] ---

def create_app():
    """Configure and initialize the Whisper Pro Flask environment."""
    flask_app = Flask(__name__)
    flask_app.url_map.strict_slashes = False

    # Mount API Endpoints
    flask_app.register_blueprint(routes.bp)

    # Configure OpenAPI/Swagger UI Documentation
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

    swagger_template = {
        "swagger": "2.0",
        "info": {
            "title": config.APP_NAME,
            "description": (
                "Enterprise-grade Whisper ASR specialized for Intel NPU/OpenVINO and NVIDIA CUDA."
            ),
            "version": config.VERSION,
            "contact": {
                "name": "Whisper Pro ASR",
                "url": "https://github.com/ventura8/Whisper-Pro-ASR",
            },
        },
    }
    Swagger(flask_app, config=swagger_config, template=swagger_template)

    # Initialize Service Infrastructure
    logging_setup.log_banner()
    model_manager.load_model()

    # --- TELEMETRY & AUDIT ---

    @flask_app.before_request
    def log_request_info():
        """Capture detailed metadata for every incoming request."""
        request.start_time = time.time()

        # Identity extraction
        client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)

        # Payload analysis (Safe truncation and formatting)
        body_log = ""
        try:
            if request.content_type:
                if "multipart/form-data" in request.content_type:
                    file_info = [f"{k}:<{v.filename}>" for k,
                                 v in request.files.items(multi=True)]
                    body_log = f" | Files: {file_info}"
                elif request.is_json:
                    body_log = f" | JSON: {request.get_json(silent=True)}"
                else:
                    raw_data = request.get_data()
                    if raw_data:
                        snippet = raw_data[:500].decode(
                            'utf-8', errors='replace')
                        body_log = f" | Raw: {snippet}{'...' if len(raw_data) > 500 else ''}"
        except Exception as e:  # pylint: disable=broad-exception-caught
            body_log = f" | Metadata Error: {str(e)}"

        logger.info(">>> %s %s [Source: %s]%s",
                    request.method, request.path, client_ip, body_log)

    @flask_app.after_request
    def log_request_done(response):
        """Finalize telemetry and inject CORS permits."""
        duration = time.time() - getattr(request, 'start_time', time.time())

        logger.info("<<< %s %s [%d] | Latency: %s",
                    request.method, request.path, response.status_code,
                    utils.format_duration(duration))

        # Modern CORS policy
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers',
                             'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods',
                             'GET,PUT,POST,DELETE,OPTIONS')
        return response

    return flask_app


# --- [ENTRY POINT] ---

if __name__ == '__main__':  # pragma: no cover
    app = create_app()
    logger.info(
        "[System] Starting Production Service (v%s) on port 9000", config.VERSION)

    # use_reloader=False is mandatory to prevent double-GPU/NPU reservation desync
    app.run(host='0.0.0.0', port=9000,
            debug=config.DEBUG_MODE,
            use_reloader=False)
else:
    # Gunicorn/WSGI defer app creation
    pass
