"""
Configuration Manager for Whisper Pro ASR

This module handles hardware detection, environment variable parsing, and
model path resolution for both Whisper and UVR/MDX-NET engines.
"""

import importlib
import logging
import os
import shutil
import tempfile

from modules.core.config_helpers import (
    calculate_cpu_parallel_limit,
    detect_hardware,
    get_unit_limit,
    resolve_thread_limits,
)
from modules.core.constants import HALLUCINATION_PHRASES

from . import engine_registry

# Explicitly reference to satisfy unused import check for external consumption
_ = HALLUCINATION_PHRASES

# Set up early logger for configuration phase
logger = logging.getLogger(__name__)

# --- [CORE SERVICE CONFIG] ---
APP_NAME = "Whisper Pro ASR"
VERSION = "1.1.4"
HARDWARE_UNITS = []  # Global registry for accelerator orchestration
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# --- [RESOURCE POOL LIMITS] ---
CPU_CORE_LIMIT = int(os.environ.get("CPU_CORE_LIMIT", 4))


MAX_CUDA = get_unit_limit("MAX_CUDA_UNITS", 1, min_value=0)
MAX_GPU = get_unit_limit("MAX_GPU_UNITS", 1, min_value=0)
MAX_NPU = get_unit_limit("MAX_NPU_UNITS", 1, min_value=0)
MAX_CPU = get_unit_limit("MAX_CPU_UNITS", 1, min_value=1)

# Memory reclamation behavior (unloads models when idle if True)
AGGRESSIVE_OFFLOAD = os.environ.get("AGGRESSIVE_OFFLOAD", "false").lower() == "true"
MODEL_IDLE_TIMEOUT = int(os.environ.get("MODEL_IDLE_TIMEOUT", 300))

# --- [HARDWARE DETECTION & DEVICE MAPPING] ---
# ASR_ENGINE can be: AUTO, FASTER-WHISPER (default), INTEL-WHISPER
ASR_ENGINE_ENV = os.environ.get("ASR_ENGINE", "FASTER-WHISPER").upper()
ASR_DEVICE_ENV = os.environ.get("ASR_DEVICE", "AUTO").upper()
ASR_COMPUTE_ENV = os.environ.get("ASR_COMPUTE_TYPE", "AUTO").upper()

_DETECTED_ENGINE = "FASTER-WHISPER"

# Default to Faster-Whisper (CTranslate2) format
DEFAULT_MODEL = "Systran/faster-whisper-large-v3"

# Check for baked-in OpenVINO model (for Intel Whisper)
OV_MODEL_BAKED = "/app/system_models/whisper-openvino"
OV_MODEL_LEGACY = "/models/whisper-openvino"
OV_MODEL_PATH = OV_MODEL_BAKED if os.path.exists(OV_MODEL_BAKED) else OV_MODEL_LEGACY

# Resolution: Prefer baked-in system models if the default model is selected
ASR_ENV = os.environ.get("ASR_MODEL", DEFAULT_MODEL)
SYS_WHISPER_PATH = "/app/system_models/whisper"

if ASR_ENV == DEFAULT_MODEL and os.path.exists(SYS_WHISPER_PATH) and os.listdir(SYS_WHISPER_PATH):
    MODEL_ID = SYS_WHISPER_PATH
else:
    MODEL_ID = ASR_ENV


# --- [HARDWARE DETECTION] ---
logger.debug("Performing hardware detection...")
_DETECTED_DEVICE, _DETECTED_PREP_DEVICE, _DETECTED_COMPUTE = detect_hardware(MAX_CUDA, MAX_GPU, MAX_NPU, HARDWARE_UNITS)

# --- [DEVICE ASSIGNMENT] ---
if ASR_DEVICE_ENV == "AUTO":
    DEVICE = _DETECTED_DEVICE
else:
    DEVICE = ASR_DEVICE_ENV

ASR_PREPROCESS_DEVICE_ENV = os.environ.get("ASR_PREPROCESS_DEVICE", "AUTO").upper()
if ASR_PREPROCESS_DEVICE_ENV == "AUTO":
    PREPROCESS_DEVICE = _DETECTED_PREP_DEVICE
else:
    PREPROCESS_DEVICE = ASR_PREPROCESS_DEVICE_ENV

# --- [ENGINE SELECTION] ---
# ASR_ENGINE can be: AUTO, FASTER-WHISPER, INTEL-WHISPER, OPENAI-WHISPER, WHISPERX
_resolution_parts = []
if ASR_ENGINE_ENV == "AUTO":
    ASR_ENGINE_SOURCE = "auto"
    ASR_ENGINE, auto_hardware_tier = engine_registry.resolve_auto_engine(HARDWARE_UNITS)
    _resolution_parts.append(f"AUTO -> {ASR_ENGINE} ({auto_hardware_tier})")

    if ASR_DEVICE_ENV == "AUTO":
        DEVICE = engine_registry.resolve_auto_device(HARDWARE_UNITS)

    logger.info(
        "ASR_ENGINE=AUTO resolved to %s using hardware tier %s (order: CUDA > GPU > NPU > CPU)",
        ASR_ENGINE,
        auto_hardware_tier,
    )
else:
    ASR_ENGINE_SOURCE = "explicit"
    ASR_ENGINE = engine_registry.normalize_and_validate_engine(ASR_ENGINE_ENV)
    _resolution_parts.append(f"explicit -> {ASR_ENGINE}")
    if ASR_DEVICE_ENV == "AUTO" and ASR_ENGINE == engine_registry.ENGINE_INTEL_WHISPER:
        DEVICE = engine_registry.resolve_auto_device(HARDWARE_UNITS)

# INTEL-WHISPER requires Intel accelerator hardware for this deployment policy.
# If no Intel GPU/NPU exists, use Faster-Whisper fallback instead of OpenVINO CPU.
if ASR_ENGINE == engine_registry.ENGINE_INTEL_WHISPER:
    has_intel_accelerator = any(unit.get("type") in ["GPU", "NPU"] for unit in HARDWARE_UNITS)
    if not has_intel_accelerator:
        logger.warning("INTEL-WHISPER requested but no Intel GPU/NPU available. Falling back to FASTER-WHISPER.")
        ASR_ENGINE = engine_registry.ENGINE_FASTER_WHISPER
        _resolution_parts.append(f"fallback -> {ASR_ENGINE} (no Intel GPU/NPU)")

ASR_ENGINE_RESOLUTION = " | ".join(_resolution_parts)

# Redirect MODEL_ID if using Intel engine and local OpenVINO model exists
if ASR_ENGINE == "INTEL-WHISPER" and ASR_ENV == DEFAULT_MODEL:
    if os.path.exists(OV_MODEL_PATH):
        MODEL_ID = OV_MODEL_PATH
    else:
        MODEL_ID = "OpenVINO"  # Trigger HF download of OV optimized model

logger.debug("ASR Engine set to: %s", ASR_ENGINE)

# --- [UI & LOGGING DESCRIPTORS] ---
ASR_DEVICE_NAME = "NVIDIA GPU" if DEVICE == "CUDA" else DEVICE
PREPROCESS_DEVICE_NAME = "NVIDIA GPU" if PREPROCESS_DEVICE == "CUDA" else PREPROCESS_DEVICE

# Refine names using hardware properties for the startup banner
if ASR_DEVICE_ENV == "AUTO" and DEVICE in ["NPU", "GPU", "CPU"]:
    try:
        # Re-import locally to avoid undefined global if first detection failed
        _ov = importlib.import_module("openvino")
        core_obj = _ov.Core()
        matching_devs = [d for d in core_obj.available_devices if DEVICE in d]
        if matching_devs:
            dev_id = matching_devs[0]
            ASR_DEVICE_NAME = core_obj.get_property(dev_id, "FULL_DEVICE_NAME")
    except (ImportError, AttributeError, ValueError, TypeError, RuntimeError, OSError):
        pass

if os.environ.get("ASR_PREPROCESS_DEVICE", "AUTO").upper() == "AUTO" and PREPROCESS_DEVICE in ["NPU", "GPU", "CPU"]:
    try:
        _ov = importlib.import_module("openvino")
        core_obj = _ov.Core()
        matching_devs = [d for d in core_obj.available_devices if PREPROCESS_DEVICE in d]
        if matching_devs:
            dev_id = matching_devs[0]
            PREPROCESS_DEVICE_NAME = core_obj.get_property(dev_id, "FULL_DEVICE_NAME")
    except (ImportError, AttributeError, ValueError, TypeError, RuntimeError, OSError):
        pass

# --- [COMPUTE TYPE RESOLUTION] ---
if ASR_COMPUTE_ENV == "AUTO":
    if ASR_DEVICE_ENV == "AUTO":
        COMPUTE_TYPE = _DETECTED_COMPUTE
    else:
        COMPUTE_TYPE = "float16" if DEVICE == "CUDA" else "int8"
else:
    COMPUTE_TYPE = ASR_COMPUTE_ENV.lower()

# Faster-Whisper requires explicitly setting 'cuda' or 'cpu'
# Respect explicit ASR_DEVICE override before falling back to auto-detection.
if DEVICE == "CUDA":
    ASR_ENGINE_DEVICE = "cuda"
elif ASR_DEVICE_ENV == "AUTO" and _DETECTED_DEVICE == "CUDA":
    ASR_ENGINE_DEVICE = "cuda"
else:
    ASR_ENGINE_DEVICE = "cpu"

ASR_ENGINE_COMPUTE_TYPE = COMPUTE_TYPE if DEVICE == "CUDA" else "int8"

# --- [ASR PERFORMANCE PARAMETERS] ---
DEFAULT_BATCH_SIZE = int(os.environ.get("ASR_BATCH_SIZE", 1))
DEFAULT_BEAM_SIZE = int(os.environ.get("ASR_BEAM_SIZE", 5))

# Debug and Logging
DEBUG_MODE = os.environ.get("DEBUG", "false").lower() == "true"
TEXT_LOGS = os.environ.get("TEXT_LOGS", "false").lower() == "true"
AGGRESSIVE_OFFLOAD = os.environ.get("AGGRESSIVE_OFFLOAD", "false").lower() == "true"
VERIFY_RUNTIME = os.environ.get("VERIFY_RUNTIME", "true").lower() == "true"

TELEMETRY_RETENTION_HOURS = int(os.environ.get("TELEMETRY_RETENTION_HOURS", 24))
LOG_RETENTION_DAYS = int(os.environ.get("LOG_RETENTION_DAYS", 7))

# Aliases for API/Test compatibility
ASR_MODEL = MODEL_ID
ASR_DEVICE = DEVICE


def update_env(key, value):
    """Updates an environment variable and re-evaluates dependent config."""
    os.environ[key] = str(value)
    # Note: Full re-evaluation would require a reload of the module or a dedicated
    # refresh function. For now, we update the env so subsequent calls see it.
    logger.info("[Config] Environment updated: %s", key)


INITIAL_STEPS_RATIO = 2.8

# Path for compiled OpenVINO blobs and model downloads
LOCAL_CACHE = "./model_cache"
OV_CACHE_DIR = os.environ.get("OV_CACHE_DIR", LOCAL_CACHE)

# --- [SSD WRITE WEAR OPTIMIZATION] ---
TEMP_DIR = os.environ.get("WHISPER_TEMP_DIR", tempfile.gettempdir())
try:
    os.makedirs(TEMP_DIR, exist_ok=True)
except (PermissionError, OSError):
    TEMP_DIR = tempfile.gettempdir()

# Persistence Directory (Should be mounted to a physical volume for history/logs)
PERSISTENT_DIR = os.environ.get("WHISPER_PERSISTENT_DIR", "/app/data")
try:
    os.makedirs(PERSISTENT_DIR, exist_ok=True)
except (PermissionError, OSError):
    PERSISTENT_DIR = "./test_data"
    try:
        os.makedirs(PERSISTENT_DIR, exist_ok=True)
    except OSError:
        pass

# State and Telemetry Directory (Persistent across restarts)
STATE_DIR = os.environ.get("WHISPER_STATE_DIR", PERSISTENT_DIR)
LOG_DIR = os.environ.get("WHISPER_LOG_DIR", STATE_DIR)
try:
    os.makedirs(STATE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
except (PermissionError, OSError):
    STATE_DIR = "./test_state"
    LOG_DIR = "./test_state"
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
    except OSError:
        pass


def get_custom_mount_points():
    """Discover custom mount points from /proc/mounts to automatically approve volumes."""
    mounts = []
    if not os.path.exists("/proc/mounts"):
        return mounts
    try:
        # Ignore system directories and mounts
        system_roots = {
            "/",
            "/proc",
            "/sys",
            "/dev",
            "/run",
            "/boot",
            "/lib",
            "/lib64",
            "/bin",
            "/sbin",
            "/usr",
            "/var",
            "/etc",
            "/root",
            "/home",
            "/tmp",
            "/sys/firmware",
        }
        with open("/proc/mounts", "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    mp = parts[1]
                    # We want custom mount points that do not belong to system roots
                    # Also skip mounts starting with system paths (like /proc/sys, /sys/fs, /dev/pts, etc.)
                    if mp in system_roots:
                        continue
                    if any(mp.startswith(sr + "/") for sr in system_roots):
                        continue
                    # Skip common Docker internal file mounts
                    if mp.endswith(("/hosts", "/hostname", "/resolv.conf")):
                        continue
                    mounts.append(mp)
    except (FileNotFoundError, PermissionError, OSError, ValueError, IndexError):
        pass
    return mounts


# Approved roots configuration
APPROVED_ROOTS_ENV = os.environ.get("WHISPER_APPROVED_ROOTS", "")
APPROVED_ROOTS = [p.strip() for p in APPROVED_ROOTS_ENV.split(",") if p.strip()]
APPROVED_ROOTS.extend(get_custom_mount_points())

TEMP_DIR_MIN_FREE_BYTES = int(os.environ.get("WHISPER_TEMP_MIN_FREE_MB", 2048)) * 1024 * 1024

PERSISTENT_TEMP_DIR = os.path.join(OV_CACHE_DIR, "temp")
try:
    os.makedirs(PERSISTENT_TEMP_DIR, exist_ok=True)
except PermissionError:
    PERSISTENT_TEMP_DIR = tempfile.gettempdir()


def get_temp_dir(required_bytes=0):
    """Return the best temp directory for transient file I/O.

    Uses a 1.5× headroom multiplier on ``required_bytes`` so that the
    temp filesystem retains breathing room for concurrent operations
    (e.g. a second FFmpeg pass or UVR stems).  For very long movies
    (15 h+) the estimated WAV size alone can approach or exceed the
    tmpfs capacity; the multiplier ensures an early, graceful fallback
    to persistent (SSD) storage instead of an ENOSPC crash.
    """
    # Require 1.5× the estimated file size as headroom for other transient files
    headroom_bytes = int(required_bytes * 1.5) if required_bytes > 0 else 0
    threshold = max(TEMP_DIR_MIN_FREE_BYTES, headroom_bytes)
    try:
        free = shutil.disk_usage(TEMP_DIR).free
        if free < threshold:
            logger.debug(
                "[Config] tmpfs free space (%d MB) below threshold (%d MB) — falling back to persistent temp dir.",
                free // (1024 * 1024),
                threshold // (1024 * 1024),
            )
            return PERSISTENT_TEMP_DIR
    except OSError:
        return PERSISTENT_TEMP_DIR
    return TEMP_DIR


def get_preprocessing_cache_dir():
    """Resolve the preprocessing cache directory dynamically."""
    base = get_temp_dir()
    path = os.path.join(base, "preprocessing")
    try:
        os.makedirs(path, exist_ok=True)
        return path
    except (PermissionError, OSError):
        fallback_base = tempfile.gettempdir()
        path = os.path.join(fallback_base, "preprocessing")
        try:
            os.makedirs(path, exist_ok=True)
        except OSError:
            pass
        return path


PREPROCESSING_CACHE_DIR = get_preprocessing_cache_dir()

# --- [MODEL PATH RESOLUTION] ---
_SYSTEM_MODELS_DIR = "/app/system_models"
DEFAULT_WHISPER = "Systran/faster-whisper-large-v3"
DEFAULT_UVR = "UVR-MDX-NET-Inst_HQ_3.onnx"

logger.debug("Final Whisper Model Path: %s", MODEL_ID)

# 2. Vocal Separation Model (UVR) Resolution
UVR_ENV = os.environ.get("VOCAL_SEPARATION_MODEL", DEFAULT_UVR)
SYS_UVR_DIR = os.path.join(_SYSTEM_MODELS_DIR, "uvr")

if UVR_ENV == DEFAULT_UVR and os.path.exists(SYS_UVR_DIR) and os.listdir(SYS_UVR_DIR):
    logger.debug("Config: Using System UVR Model Directory at %s", SYS_UVR_DIR)
    UVR_MODEL_DIR = SYS_UVR_DIR
else:
    logger.debug("Config: Using User Cache for UVR Models.")
    UVR_MODEL_DIR = os.path.join(OV_CACHE_DIR, "preprocessing_models")

VOCAL_SEPARATION_MODEL = UVR_ENV

# Chunk duration in seconds for UVR stem separation to limit RAM usage / prevent OOM on long files.
# Default is 600 (10 minutes). Set to 0 to disable chunking.
UVR_CHUNK_DURATION = int(os.environ.get("UVR_CHUNK_DURATION", 600))

# Chunk duration in seconds for Intel Whisper transcription to show periodic progress logs.
# Default is 300 (5 minutes).
INTEL_ASR_CHUNK_DURATION = int(os.environ.get("INTEL_ASR_CHUNK_DURATION", 300))

# --- [LANGUAGE PROCESSING & VAD] ---
VAD_MIN_SILENCE_DURATION_MS = int(os.environ.get("VAD_MIN_SILENCE_DURATION_MS", 500))
VAD_SPEECH_PAD_MS = int(os.environ.get("VAD_SPEECH_PAD_MS", 500))

INITIAL_PROMPT = os.environ.get(
    "INITIAL_PROMPT",
    "This video contains speech in multiple languages including "
    "Romanian, English, French, Italian, German, and Spanish.",
)

# --- [PREPROCESSING CONFIGURATION] ---
ENABLE_VOCAL_SEPARATION = os.environ.get("ENABLE_VOCAL_SEPARATION", "false").lower() == "true"

VOCAL_SEPARATION_SEGMENT_DURATION = int(os.environ.get("VOCAL_SEPARATION_SEGMENT_DURATION", 600))

logger.debug("Final Preprocessing Device: %s", PREPROCESS_DEVICE)

DEFAULT_WHISPER_THREADS = int(os.environ.get("ASR_THREADS", 4))
PREPROCESS_THREADS_ENV = int(os.environ.get("ASR_PREPROCESS_THREADS", 4))

# --- [THREAD & PERFORMANCE TUNING] ---


ASR_THREADS, PREPROCESS_THREADS = resolve_thread_limits(
    DEFAULT_WHISPER_THREADS, PREPROCESS_THREADS_ENV, CPU_CORE_LIMIT, MAX_CPU, DEVICE
)

# Industry standard thread limits for shared libraries
os.environ["OMP_NUM_THREADS"] = str(PREPROCESS_THREADS)
os.environ["MKL_NUM_THREADS"] = str(PREPROCESS_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(PREPROCESS_THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(PREPROCESS_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(PREPROCESS_THREADS)
os.environ["ORT_INTRA_OP_NUM_THREADS"] = str(PREPROCESS_THREADS)
os.environ["ORT_INTER_OP_NUM_THREADS"] = "1"

# FFmpeg concurrency for audio preparation
FFMPEG_THREADS = int(os.environ.get("FFMPEG_THREADS", 1))
FFMPEG_HWACCEL = os.environ.get("FFMPEG_HWACCEL", "none")
FFMPEG_FILTER = os.environ.get("FFMPEG_FILTER", "dynaudnorm")


def validate_thread_concurrency():
    """Enforce hardware-aware thread limits to maintain responsiveness."""
    try:
        eff_ffmpeg = FFMPEG_THREADS if FFMPEG_THREADS > 0 else 1
        total_load = PREPROCESS_THREADS + eff_ffmpeg

        if total_load > (CPU_CORE_LIMIT + 2):  # Allow slight over-subscription for I/O
            logger.warning(
                "[Config] OVER-PROVISIONING: PREPROCESS_THREADS (%d) + "
                "FFMPEG_THREADS (%d) = %d, which exceeds logical cores (%d).",
                PREPROCESS_THREADS,
                eff_ffmpeg,
                total_load,
                CPU_CORE_LIMIT,
            )
    except (ValueError, TypeError, AttributeError):
        pass


validate_thread_concurrency()

CPU_PARALLEL_LIMIT = calculate_cpu_parallel_limit(MAX_CPU, CPU_CORE_LIMIT, ASR_THREADS, PREPROCESS_THREADS)


def get_parallel_limit(device):
    """Determine parallel task limit based on physical resource units."""
    if device in ["CUDA", "GPU", "NPU"]:
        try:
            units = [u for u in HARDWARE_UNITS if u.get("type") == device]
            if units:
                return len(units)
        except (AttributeError, TypeError, ValueError):
            pass

        return 1  # Safe default if registry is unavailable or device is absent

    # CPU-bound tasks: Capped by hardware-optimized slot count
    return CPU_PARALLEL_LIMIT


# --- [LANGUAGE DETECTION] ---
# Enable iterative scanning for quiet or long-intro files
SMART_SAMPLING_SEARCH = os.environ.get("SMART_SAMPLING_SEARCH", "false").lower() == "true"
# Enable vocal isolation during language detection (improves identification accuracy)
ENABLE_LD_PREPROCESSING = os.environ.get("ENABLE_LD_PREPROCESSING", "true").lower() == "true"
# Coalesce concurrent identical detect-language requests (same local path) into one leader execution.
ENABLE_LD_REQUEST_COALESCING = os.environ.get("ENABLE_LD_REQUEST_COALESCING", "true").lower() == "true"
# Aggressiveness of VAD during language detection (0.0 to 1.0)
LD_VAD_THRESHOLD = float(os.environ.get("LD_VAD_THRESHOLD", 0.3))
# Minimum confidence threshold to consider a segment's vote in language detection
LD_MIN_CONFIDENCE = float(os.environ.get("LD_MIN_CONFIDENCE", 0.5))


# --- [HALLUCINATION FILTERING] ---
# Known "silence" or "credit" hallucination phrases for removal during post-processing
HALLUCINATION_SILENCE_THRESHOLD = float(os.environ.get("HALLUCINATION_SILENCE_THRESHOLD", 0.85))
HALLUCINATION_REPETITION_THRESHOLD = int(os.environ.get("HALLUCINATION_REPETITION_THRESHOLD", 15))


# --- [SUBTITLE PROMO CARD] ---
SUBTITLE_PROMO_ENABLED = os.environ.get("SUBTITLE_PROMO_ENABLED", "true").lower() == "true"
SUBTITLE_PROMO_TEXT = os.environ.get("SUBTITLE_PROMO_TEXT", "Made with Whisper Pro ASR")
try:
    SUBTITLE_PROMO_DURATION = float(os.environ.get("SUBTITLE_PROMO_DURATION", "3.0"))
except (ValueError, TypeError):
    SUBTITLE_PROMO_DURATION = 3.0
