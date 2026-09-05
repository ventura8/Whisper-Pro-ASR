"""
Logging Configuration and Performance Diagnostics

This module initializes the global logging system, applies noisy warning filters,
and builds the interactive hardware-diagnostic banner displayed at startup.
"""

import importlib
import logging
import os
import sys
import threading
from logging.handlers import TimedRotatingFileHandler

from modules.core import config, utils
from modules.core.constants import INTEL_ENV_KEYS

# Thread-safe global log buffer for dashboard visibility
# Key: thread_id, Value: list of strings
TASK_LOGS: dict[int | str, list[str]] = {}
TASK_LOGS_LOCK = threading.Lock()

# --- [GLOBAL LOGGING CONFIGURATION] ---
LOG_LEVEL = logging.INFO


def setup_logging():
    """Initialize the global logging system with task-aware filters."""
    log_level = _resolve_log_level()
    _configure_root_logger(log_level)
    _reapply_filters_to_root_handlers()
    _ensure_log_buffer_attached()
    _ensure_file_handler_attached()


def _resolve_log_level():
    sys.modules[__name__].LOG_LEVEL = logging.DEBUG if config.DEBUG_MODE else logging.INFO
    return sys.modules[__name__].LOG_LEVEL


def _configure_root_logger(log_level):
    logging.basicConfig(level=log_level, format="%(asctime)s %(task_ctx)s %(message)s", force=True, stream=sys.stdout)


def _reapply_filters_to_root_handlers():
    for handler in logging.root.handlers[:]:
        _apply_standard_filters(handler)


def _ensure_log_buffer_attached():
    if log_buffer not in logging.root.handlers:
        logging.root.addHandler(log_buffer)
    _apply_standard_filters(log_buffer)


def _ensure_file_handler_attached():
    fh = get_file_handler()
    if fh and fh not in logging.root.handlers:
        logging.root.addHandler(fh)


def _apply_standard_filters(handler):
    """Apply the standard suite of filters to a given handler."""
    # Remove existing instances to avoid duplicates
    existing_types = [type(f) for f in handler.filters]
    if ContextualFilter not in existing_types:
        handler.addFilter(ContextualFilter())
    if IgnoreSpecificWarnings not in existing_types:
        handler.addFilter(IgnoreSpecificWarnings())
    if WerkzeugStatusFilter not in existing_types:
        handler.addFilter(WerkzeugStatusFilter())


# Suppress noisy library-level logging for transformers and optimum
try:
    import optimum.utils.logging
    import transformers

    transformers.utils.logging.set_verbosity_error()
    optimum.utils.logging.set_verbosity_error()
except ImportError:
    pass


class ContextualFilter(logging.Filter):
    """
    Injects thread-local context (e.g. filename) into every log record.
    """

    def filter(self, record):
        """Inject filename and step info into log record."""
        # Allow pre-existing task_ctx to persist (useful for logo/banner)
        if hasattr(record, "task_ctx"):
            return True

        filename = getattr(utils.THREAD_CONTEXT, "filename", None) or "System"
        step_info = getattr(utils.THREAD_CONTEXT, "step_info", None)

        if _should_omit_system_context(filename, step_info):
            record.task_ctx = ""
            return True

        record.task_ctx = _build_task_context(filename, step_info)
        return True

    def __repr__(self):
        return "ContextualFilter()"


def _should_omit_system_context(filename: str, step_info) -> bool:
    return filename == "System" and not step_info


def _build_task_context(filename: str, step_info) -> str:
    ctx = f"[{filename}]"
    if step_info:
        ctx += f" {step_info}"
    return ctx


class IgnoreSpecificWarnings(logging.Filter):
    """
    Custom logging filter to suppress known noisy warnings from AI backends.
    """

    def filter(self, record):
        """Return False to suppress specific warning messages, True otherwise."""
        msg = record.getMessage().lower()

        # Suppress common but non-critical AI backend messages
        suppressions = [
            "default values have been modified",  # generation_config defaults
            "a custom logits processor of type",  # processor registration
            "chunk_length_s",  # experimental feature warnings
            "device set to use cpu",  # CPU fallback notices
            "will use cpu instead",  # CPU fallback notices
            "this is a development server",  # Flask dev mode warning
        ]

        if any(substring in msg for substring in suppressions):
            return False

        return True

    def __repr__(self):
        return "IgnoreSpecificWarnings()"


class WerkzeugStatusFilter(logging.Filter):
    """
    Demotes repetitive /status polling access logs to DEBUG level.
    """

    def filter(self, record):
        """Change log level for /status requests."""
        msg = record.getMessage()
        if "GET /status " in msg:
            # If not in debug mode, just drop the record entirely to stop spam
            if logging.getLogger().level > logging.DEBUG:
                return False
            record.levelno = logging.DEBUG
            record.levelname = "DEBUG"
        return True

    def __repr__(self):
        return "WerkzeugStatusFilter()"


class LogBufferHandler(logging.Handler):
    """
    Captures log records into a global thread-keyed buffer for dashboard visibility.
    """

    def emit(self, record):
        try:
            task_id = getattr(utils.THREAD_CONTEXT, "task_id", None)
            thread_id = getattr(utils.THREAD_CONTEXT, "registration_thread_id", None) or threading.get_ident()
            with TASK_LOGS_LOCK:
                target_key = _resolve_log_buffer_target_key(task_id, thread_id)
                if target_key is not None:
                    TASK_LOGS[target_key].append(self.format(record))
        except (AttributeError, ValueError, TypeError):
            pass


def _resolve_log_buffer_target_key(task_id, thread_id):
    if task_id and task_id in TASK_LOGS:
        return task_id
    if thread_id in TASK_LOGS:
        return thread_id
    return None


# Apply filters to the root logger and specific third-party modules
log_buffer = LogBufferHandler()
log_buffer.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))

for hand in logging.root.handlers:
    _apply_standard_filters(hand)

logging.root.addHandler(log_buffer)

# --- [PERSISTENT FILE LOGGING] ---
_FILE_HANDLER_HOLDER = [None]


def get_file_handler():
    """Idempotent factory for the persistent file handler."""
    if _FILE_HANDLER_HOLDER[0]:
        return _FILE_HANDLER_HOLDER[0]

    log_file = os.path.join(config.LOG_DIR, "whisper_pro.log")
    try:
        os.makedirs(config.LOG_DIR, exist_ok=True)
        # Retention is configurable via environment, defaults to 7 days
        retention_days = int(os.environ.get("LOG_RETENTION_DAYS", 7))
        handler = TimedRotatingFileHandler(log_file, when="D", interval=1, backupCount=retention_days, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(task_ctx)s [%(levelname)s] %(message)s"))
        _apply_standard_filters(handler)
        _FILE_HANDLER_HOLDER[0] = handler
        return _FILE_HANDLER_HOLDER[0]
    except (OSError, AttributeError, ValueError) as e:
        print(f"Failed to initialize file logging: {e}")
        return None


def update_log_retention(days):
    """Dynamically update the backup count of the file handler."""
    try:
        fh = get_file_handler()
        if fh:
            fh.backupCount = int(days)
            logger.info("[Logging] Dynamic log retention updated to %d days", fh.backupCount)
    except tuple([Exception]) as e:
        logger.error("[Logging] Failed to dynamically update log retention: %s", e)


# Initial attachment (at import time)
_INIT_HANDLER = get_file_handler()
if _INIT_HANDLER:
    logging.root.addHandler(_INIT_HANDLER)

LOGGERS_TO_FILTER = ["transformers", "optimum", "optimum.intel", "openvino", "werkzeug", "uvicorn.access"]

for logger_name in LOGGERS_TO_FILTER:
    _logger = logging.getLogger(logger_name)
    _logger.addFilter(IgnoreSpecificWarnings())
    if logger_name in ["werkzeug", "uvicorn.access"]:
        _logger.addFilter(WerkzeugStatusFilter())
    _logger.propagate = True

logger = logging.getLogger(__name__)

# Alignment constants for banner output
_LABEL_WIDTH = 30


def _format_prop_value(val):
    """Standardize the representation of OpenVINO hardware property values."""
    if isinstance(val, (list, tuple)):
        return ", ".join(map(str, val))
    if isinstance(val, bool):
        return "Yes" if val else "No"
    return str(val)


def _extract_hardware_properties(core, real_device):
    """Iterate and format all supported OpenVINO physical device properties."""
    info_lines = []
    supported_props = core.get_property(real_device, "SUPPORTED_PROPERTIES")
    skip_props = {
        "SUPPORTED_PROPERTIES",
        "FULL_DEVICE_NAME",
        "DEVICE_ID",
        "CACHING_PROPERTIES",
        "SUPPORTED_CONFIG_KEYS",
    }

    for prop_key in supported_props:
        if prop_key in skip_props:
            continue
        line = _read_hardware_property_line(core, real_device, prop_key)
        if line:
            info_lines.append(line)
    return info_lines


def _read_hardware_property_line(core, real_device, prop_key):
    label = _humanize_property_label(prop_key)
    try:
        val = core.get_property(real_device, prop_key)
        return _format_hardware_property_line(label, val)
    except tuple([Exception]):
        return None


def _humanize_property_label(prop_key: str) -> str:
    label = prop_key
    for prefix in ["DEVICE_", "NPU_", "GPU_", "CPU_", "Intel_"]:
        label = label.replace(prefix, "")
    return label.title().replace("_", " ")


def _format_hardware_property_line(label: str, val):
    if val is None:
        return None
    val_str = _format_prop_value(val)
    if not val_str or val_str.lower() == "none":
        return None
    return f"  {label:<{_LABEL_WIDTH}}: {val_str}"


def _get_device_properties(device_alias):
    """
    Query intensive hardware properties from the OpenVINO runtime for diagnostics.
    """
    device_full_name = device_alias
    try:
        ov = importlib.import_module("openvino")
        core = ov.Core()

        real_device = _resolve_real_openvino_device(core.available_devices, device_alias)
        if not real_device:
            return device_full_name, []

        device_full_name = _resolve_device_full_name(core, real_device)
        info_lines = _extract_hardware_properties(core, real_device)
        info_lines.sort()
        return device_full_name, info_lines
    except tuple([Exception]):
        return device_full_name, []


def _resolve_real_openvino_device(available_devices, device_alias):
    if device_alias in available_devices:
        return device_alias
    for dev in available_devices:
        if dev.startswith(device_alias):
            return dev
    return None


def _resolve_device_full_name(core, real_device):
    try:
        return core.get_property(real_device, "FULL_DEVICE_NAME")
    except tuple([Exception]):
        return real_device


def _get_real_model_name():
    """Retrieve the human-readable Whisper model ID."""
    model_id = config.MODEL_ID

    # Resolve Intel-Whisper baked paths
    if config.ASR_ENGINE == "INTEL-WHISPER":
        if model_id in [config.OV_MODEL_BAKED, config.OV_MODEL_LEGACY]:
            return "OpenVINO/whisper-large-v3-fp16-ov"

    # Resolve Faster-Whisper baked paths
    if model_id == config.SYS_WHISPER_PATH:
        return "Systran/faster-whisper-large-v3"

    return model_id


def _get_vocal_separator_model_display():
    """Format the vocal separator model name for display."""
    if not config.ENABLE_VOCAL_SEPARATION:
        return "N/A (disabled)"
    return config.VOCAL_SEPARATION_MODEL


def _get_intel_runtime_env_lines():
    lines = []
    for env_key in INTEL_ENV_KEYS:
        env_value = os.environ.get(env_key)
        lines.append(f"  {env_key:<{_LABEL_WIDTH}}: {env_value if env_value else '<unset>'}")
    return lines


def _get_openvino_available_devices_line():
    try:
        ov = importlib.import_module("openvino")
        core = ov.Core()
        devices = ", ".join(core.available_devices) or "<none>"
        return f"  {'OpenVINO devices':<{_LABEL_WIDTH}}: {devices}"
    except tuple([Exception]) as exc:
        return f"  {'OpenVINO devices':<{_LABEL_WIDTH}}: unavailable ({exc})"


def _get_openvino_target_probe_lines():
    if os.environ.get("INTEL_DEEP_OV_PROBE", "false").lower() != "true":
        return []

    probe_targets = ["GPU", "GPU.0", "NPU", "NPU.0"]
    lines = ["  [OPENVINO TARGET PROBE]"]
    try:
        ov = importlib.import_module("openvino")
        core = ov.Core()
        for target in probe_targets:
            try:
                full_name = core.get_property(target, "FULL_DEVICE_NAME")
                lines.append(f"  {target:<{_LABEL_WIDTH}}: {full_name}")
            except tuple([Exception]) as exc:
                lines.append(f"  {target:<{_LABEL_WIDTH}}: unavailable ({exc})")
    except tuple([Exception]) as exc:
        lines.append(f"  {'OpenVINO probe':<{_LABEL_WIDTH}}: unavailable ({exc})")
    return lines


def _banner_logo():
    """Return the ASCII art banner logo."""
    return r"""
   _      _     _
  | |    | |   (_)
  | |    | |_  _ ___ _ __   ___ _ __
  | |/\| | '_ \| / __| '_ \ / _ \ '__|
  \  /\  / | | | \__ \ |_) |  __/ |
   \/  \/|_| |_|_|___/ .__/ \___|_|
                     | |
                     |_|
   _____               _____  _____  _____
  |  __ \             /  _  \/  ___||  __ \
  | |__) | __ ___    |  /_\  \___ \| |__) |
  |  ___/ '__/ _ \   |  ___  |___ \|  _  /
  | |   | | | (_) |  |  | |  /____/| | \ \
  |_|   |_|  \___/   |_| |_| \____/|_|  \_\
    """


def _model_and_cache_status():
    """Assess local model availability and OpenVINO kernel cache state."""
    model_status = "Locally Found (Fast)" if os.path.exists(config.MODEL_ID) else "Hugging Face (Download/Cache)"
    cache_dir = config.OV_CACHE_DIR
    cache_status = "FOUND (Optimized Load)" if os.path.exists(cache_dir) and os.listdir(cache_dir) else "MISSING (Full Initialization)"
    return model_status, cache_status


def _threads_str():
    """Retrieve current thread allocation settings."""
    return f"ASR={config.ASR_THREADS} | Preprocess={config.PREPROCESS_THREADS} | FFmpeg={config.FFMPEG_THREADS}"


def _unique_device_props(asr_props, prep_props):
    """Consolidate and deduplicate hardware properties."""
    seen = []
    for prop in asr_props + prep_props:
        if prop not in seen:
            seen.append(prop)
    return seen[:10]


def _banner_config_lines(cfg):
    """Build the configuration details block for the startup banner."""
    w = _LABEL_WIDTH
    if config.ASR_ENGINE == "INTEL-WHISPER":
        asr_runtime_val = f"OpenVINO ({config.DEVICE})"
    else:
        asr_runtime_val = f"{config.ASR_ENGINE_DEVICE.upper()} (Compute: {config.ASR_ENGINE_COMPUTE_TYPE})"

    preprocess_val = f"Vocals={config.ENABLE_VOCAL_SEPARATION} | LD-Pre={config.ENABLE_LD_PREPROCESSING}"
    lines = [
        "================================================================",
        f"      {config.APP_NAME} {config.VERSION_DISPLAY}",
        "================================================================",
        "  [ENGINE CONFIG]",
        f"  {'Whisper Model ID':<{w}}: {_get_real_model_name()}",
        f"  {'Vocal Separator Model ID':<{w}}: {_get_vocal_separator_model_display()}",
        f"  {'ASR Engine':<{w}}: {config.ASR_ENGINE}",
        f"  {'ASR Engine Source':<{w}}: {getattr(config, 'ASR_ENGINE_SOURCE', 'explicit')}",
        f"  {'Beam Size':<{w}}: {config.DEFAULT_BEAM_SIZE}",
        f"  {'Threads':<{w}}: {cfg['threads']}",
        f"  {'Preprocess Flags':<{w}}: {preprocess_val}",
        "",
        "  [HARDWARE INFO]",
        f"  {'Pipeline target':<{w}}: {cfg['asr_display']}",
        f"  {'ASR Runtime':<{w}}: {asr_runtime_val}",
        f"  {'Preprocess Device':<{w}}: {cfg['prep_display']}",
        f"  {'Resource Pool':<{w}}: {cfg['resource_pool']}",
        "",
    ]
    lines.append("  [INTEL RUNTIME ENV]")
    lines.extend(cfg.get("intel_env", []))
    lines.append("")
    lines.append(cfg.get("openvino_devices", f"  {'OpenVINO devices':<{w}}: <unavailable>"))
    lines.append("")
    lines.extend(cfg.get("openvino_probe", []))
    if cfg.get("openvino_probe"):
        lines.append("")
    if cfg["unique_props"]:
        lines.append("  [DEVICE PROPERTIES]")
        lines.extend(cfg["unique_props"])
        lines.append("")

    lines.extend(
        [
            f"  {'Model Source':<{w}}: {cfg['model_status']}",
            f"  {'Binary Cache Status':<{w}}: {cfg['cache_status']}",
            "================================================================",
        ]
    )
    return lines


def log_banner():
    """Generate and log the high-impact startup banner."""
    _log_banner_logo()
    cfg = _build_banner_config()
    _log_banner_config_lines(cfg)


def _log_banner_logo():
    logger.info("%sWhisper Pro ASR Startup%s", "\033[96m", "\033[0m", extra={"task_ctx": ""})
    for logo_line in _banner_logo().split("\n"):
        if logo_line.strip():
            logger.info("%s%s%s", "\033[96m", logo_line, "\033[0m", extra={"task_ctx": ""})


def _build_banner_config() -> dict:
    model_status, cache_status = _model_and_cache_status()
    asr_full, asr_props = _get_device_properties(config.DEVICE)
    prep_full, prep_props = _get_device_properties(config.PREPROCESS_DEVICE)
    return {
        "model_status": model_status,
        "cache_status": cache_status,
        "threads": _threads_str(),
        "asr_display": asr_full or config.ASR_DEVICE_NAME,
        "prep_display": prep_full or config.PREPROCESS_DEVICE_NAME,
        "resource_pool": ", ".join([u["id"] for u in config.HARDWARE_UNITS]),
        "unique_props": _unique_device_props(asr_props, prep_props),
        "intel_env": _get_intel_runtime_env_lines(),
        "openvino_devices": _get_openvino_available_devices_line(),
        "openvino_probe": _get_openvino_target_probe_lines(),
    }


def _log_banner_config_lines(cfg: dict):
    for line in _banner_config_lines(cfg):
        if line.startswith("==="):
            logger.info("%s", line, extra={"task_ctx": ""})
        else:
            logger.info("%s", line)
