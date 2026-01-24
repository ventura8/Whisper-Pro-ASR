"""
Logging Configuration and Performance Diagnostics

This module initializes the global logging system, applies noisy warning filters,
and builds the interactive hardware-diagnostic banner displayed at startup.
"""
import os
import logging
import sys
from . import config

# --- [GLOBAL LOGGING CONFIGURATION] ---
LOG_LEVEL = logging.DEBUG if config.DEBUG_MODE else logging.INFO
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(message)s',
                    force=True, stream=sys.stdout)

# Suppress noisy library-level logging for transformers and optimum
try:
    import transformers
    import optimum.utils.logging
    transformers.utils.logging.set_verbosity_error()
    optimum.utils.logging.set_verbosity_error()
except ImportError:  # pragma: no cover
    pass


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
            "a custom logits processor of type",   # processor registration
            "chunk_length_s",                      # experimental feature warnings
            "device set to use cpu",               # CPU fallback notices
            "will use cpu instead",                # CPU fallback notices
            "this is a development server"         # Flask dev mode warning
        ]

        if any(substring in msg for substring in suppressions):
            return False

        return True

    def __repr__(self):
        return "IgnoreSpecificWarnings()"


# Apply filters to the root logger and specific third-party modules
for handler in logging.root.handlers:
    handler.addFilter(IgnoreSpecificWarnings())

LOGGERS_TO_FILTER = [
    "transformers",
    "optimum",
    "optimum.intel",
    "openvino",
    "werkzeug"
]

for logger_name in LOGGERS_TO_FILTER:
    _logger = logging.getLogger(logger_name)
    _logger.addFilter(IgnoreSpecificWarnings())
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
        "SUPPORTED_PROPERTIES", "FULL_DEVICE_NAME", "DEVICE_ID",
        "CACHING_PROPERTIES", "SUPPORTED_CONFIG_KEYS"
    }

    for prop_key in supported_props:
        if prop_key in skip_props:
            continue

        # Human-readable labels (e.g. DEVICE_MANUFACTURER -> Manufacturer)
        label = prop_key
        for prefix in ["DEVICE_", "NPU_", "GPU_", "CPU_", "Intel_"]:
            label = label.replace(prefix, "")
        label = label.title().replace("_", " ")

        try:
            val = core.get_property(real_device, prop_key)
            if val is not None:
                val_str = _format_prop_value(val)
                if val_str and val_str.lower() != "none":
                    info_lines.append(f"  {label:<{_LABEL_WIDTH}}: {val_str}")
        except Exception:  # pylint: disable=broad-except
            pass
    return info_lines


def _get_device_properties(device_alias):
    """
    Query intensive hardware properties from the OpenVINO runtime for diagnostics.
    """
    device_full_name = device_alias
    info_lines = []
    try:
        from openvino import Core  # pylint: disable=import-outside-toplevel, import-error
        core = Core()

        # Resolve alias to physical device ID (e.g. 'GPU' -> 'GPU.0')
        real_device = None
        if device_alias in core.available_devices:
            real_device = device_alias
        else:
            for dev in core.available_devices:
                if dev.startswith(device_alias):
                    real_device = dev
                    break

        if not real_device:
            return None, []

        # Get Descriptive Name
        try:
            device_full_name = core.get_property(
                real_device, "FULL_DEVICE_NAME")
        except Exception:  # pylint: disable=broad-except
            device_full_name = real_device

        # Extract and format supported properties
        info_lines = _extract_hardware_properties(core, real_device)
        info_lines.sort()

    except Exception:  # pylint: disable=broad-except
        pass
    return device_full_name, info_lines


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
    model_status = (
        "Locally Found (Fast)"
        if os.path.exists(config.MODEL_ID)
        else "Hugging Face (Download/Cache)"
    )
    cache_dir = config.OV_CACHE_DIR
    cache_status = (
        "FOUND (Optimized Load)"
        if os.path.exists(cache_dir) and os.listdir(cache_dir)
        else "MISSING (Full Initialization)"
    )
    return model_status, cache_status


def _threads_str():
    """Retrieve current thread allocation settings."""
    return (
        f"ASR={config.ASR_THREADS} | "
        f"Preprocess={config.PREPROCESS_THREADS} | "
        f"FFmpeg={config.FFMPEG_THREADS}"
    )


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
        asr_runtime_val = (
            f"{config.ASR_ENGINE_DEVICE.upper()} "
            f"(Compute: {config.ASR_ENGINE_COMPUTE_TYPE})"
        )

    preprocess_val = (
        f"Vocals={config.ENABLE_VOCAL_SEPARATION} | "
        f"LD-Pre={config.ENABLE_LD_PREPROCESSING}"
    )
    lines = [
        "================================================================",
        f"      {config.APP_NAME} {config.VERSION}",
        "================================================================",
        "  [ENGINE CONFIG]",
        f"  {'Whisper Model ID':<{w}}: {_get_real_model_name()}",
        f"  {'Vocal Separator Model ID':<{w}}: {_get_vocal_separator_model_display()}",
        f"  {'Beam Size':<{w}}: {config.DEFAULT_BEAM_SIZE}",
        f"  {'Threads':<{w}}: {cfg['threads']}",
        f"  {'Preprocess Flags':<{w}}: {preprocess_val}",
        "",
        "  [HARDWARE INFO]",
        f"  {'Pipeline target':<{w}}: {cfg['asr_display']}",
        f"  {'ASR Runtime':<{w}}: {asr_runtime_val}",
        f"  {'Preprocess Device':<{w}}: {cfg['prep_display']}",
        "",
    ]
    if cfg["unique_props"]:
        lines.append("  [DEVICE PROPERTIES]")
        lines.extend(cfg["unique_props"])
        lines.append("")

    lines.extend([
        f"  {'Model Source':<{w}}: {cfg['model_status']}",
        f"  {'Binary Cache Status':<{w}}: {cfg['cache_status']}",
        "================================================================",
    ])
    return lines


def log_banner():
    """Generate and log the high-impact startup banner."""
    logo = _banner_logo()
    model_status, cache_status = _model_and_cache_status()
    threads = _threads_str()

    logger.info("%sWhisper Pro ASR Startup%s", "\033[96m", "\033[0m")
    for logo_line in logo.split("\n"):
        if logo_line.strip():
            logger.info("%s%s%s", "\033[96m", logo_line, "\033[0m")

    # Fetch hardware details
    asr_full, asr_props = _get_device_properties(config.DEVICE)
    prep_full, prep_props = _get_device_properties(config.PREPROCESS_DEVICE)

    asr_display = asr_full or config.ASR_DEVICE_NAME
    prep_display = prep_full or config.PREPROCESS_DEVICE_NAME
    unique_props = _unique_device_props(asr_props, prep_props)

    cfg = {
        "model_status": model_status,
        "cache_status": cache_status,
        "threads": threads,
        "asr_display": asr_display,
        "prep_display": prep_display,
        "unique_props": unique_props,
    }

    for line in _banner_config_lines(cfg):
        logger.info("%s", line)
