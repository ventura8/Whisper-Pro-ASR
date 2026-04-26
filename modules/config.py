"""
Configuration Manager for Whisper Pro ASR

This module handles hardware detection, environment variable parsing, and
model path resolution for both Whisper and UVR/MDX-NET engines.
"""
import os
import logging
import shutil
import tempfile
try:
    import ctranslate2
except ImportError:
    ctranslate2 = None
try:
    from openvino import Core
except ImportError:
    Core = None

# Set up early logger for configuration phase
logger = logging.getLogger(__name__)

# --- [CORE SERVICE CONFIG] ---
APP_NAME = "Whisper Pro ASR"
VERSION = "1.0.4"
HARDWARE_UNITS = []  # Global registry for accelerator orchestration

# --- [RESOURCE POOL LIMITS] ---
CPU_CORE_LIMIT = int(os.environ.get("CPU_CORE_LIMIT", 4))


def _get_unit_limit(env_var, default=1):
    """Helper to parse hardware unit limits (supports int, ALL, AUTO)."""
    val = os.environ.get(env_var, str(default)).upper()
    if val in ["ALL", "AUTO"]:
        return 999  # Practically unlimited
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


MAX_CUDA = _get_unit_limit("MAX_CUDA_UNITS", 1)
MAX_GPU = _get_unit_limit("MAX_GPU_UNITS", 1)
MAX_NPU = _get_unit_limit("MAX_NPU_UNITS", 1)
MAX_CPU = _get_unit_limit("MAX_CPU_UNITS", 1)

# pylint: disable=invalid-name

# --- [HARDWARE DETECTION & DEVICE MAPPING] ---
# ASR_ENGINE can be: AUTO (default), FASTER-WHISPER, INTEL-WHISPER
ASR_ENGINE_ENV = os.environ.get("ASR_ENGINE", "AUTO").upper()
ASR_DEVICE_ENV = os.environ.get("ASR_DEVICE", "AUTO").upper()
ASR_COMPUTE_ENV = os.environ.get("ASR_COMPUTE_TYPE", "AUTO").upper()

_DETECTED_ENGINE = "FASTER-WHISPER"
_DETECTED_DEVICE = "CPU"
_DETECTED_PREP_DEVICE = "CPU"
_DETECTED_COMPUTE = "int8"

# Default to Faster-Whisper (CTranslate2) format
DEFAULT_MODEL = "Systran/faster-whisper-large-v3"

# Check for baked-in OpenVINO model (for Intel Whisper)
OV_MODEL_BAKED = "/app/system_models/whisper-openvino"
OV_MODEL_LEGACY = "/models/whisper-openvino"
OV_MODEL_PATH = OV_MODEL_BAKED if os.path.exists(
    OV_MODEL_BAKED) else OV_MODEL_LEGACY

# Resolution: Prefer baked-in system models if the default model is selected
ASR_ENV = os.environ.get("ASR_MODEL", DEFAULT_MODEL)
SYS_WHISPER_PATH = "/app/system_models/whisper"

if (ASR_ENV == DEFAULT_MODEL and
        os.path.exists(SYS_WHISPER_PATH) and
        os.listdir(SYS_WHISPER_PATH)):
    MODEL_ID = SYS_WHISPER_PATH
else:
    MODEL_ID = ASR_ENV

# --- [HARDWARE DETECTION] ---
logger.debug("Performing hardware detection...")
_DETECTED_DEVICE = "CPU"
_DETECTED_PREP_DEVICE = "CPU"
_DETECTED_COMPUTE = "int8"

# 1. NVIDIA Acceleration Check
cuda_count = 0
try:
    if ctranslate2:
        cuda_count = ctranslate2.get_cuda_device_count()
    if cuda_count > 0:
        logger.debug("Auto-detected %d NVIDIA GPU(s).", cuda_count)
        _DETECTED_DEVICE = "CUDA"
        _DETECTED_PREP_DEVICE = "CUDA"
        _DETECTED_COMPUTE = "float16"
        cuda_to_use = min(cuda_count, MAX_CUDA)
        for i in range(cuda_to_use):
            HARDWARE_UNITS.append({"type": "CUDA", "id": f"cuda:{i}", "name": f"NVIDIA GPU {i}"})
except Exception as e:  # pylint: disable=broad-exception-caught
    logger.debug("CUDA detection skipped: %s", e)

# 2. Intel Accelerator Check (OpenVINO)
core = None
try:
    if Core:
        core = Core()
    if core:
        devices = core.available_devices
        logger.debug("OpenVINO Available Devices: %s", devices)

    gpu_count = 0
    npu_count = 0

    for dev in devices:
        if "GPU" in dev:
            if gpu_count >= MAX_GPU:
                continue
            try:
                name = core.get_property(dev, "FULL_DEVICE_NAME")
            except Exception:  # pylint: disable=broad-exception-caught
                name = f"Intel {dev}"
            HARDWARE_UNITS.append({"type": "GPU", "id": dev, "name": name})
            gpu_count += 1
            if _DETECTED_DEVICE == "CPU":
                _DETECTED_PREP_DEVICE = "GPU"
        elif "NPU" in dev:
            if npu_count >= MAX_NPU:
                continue
            try:
                name = core.get_property(dev, "FULL_DEVICE_NAME")
            except Exception:  # pylint: disable=broad-exception-caught
                name = f"Intel {dev}"
            HARDWARE_UNITS.append({"type": "NPU", "id": dev, "name": name})
            npu_count += 1
            if _DETECTED_DEVICE == "CPU":
                _DETECTED_PREP_DEVICE = "NPU"

except Exception as e:  # pylint: disable=broad-exception-caught
    logger.debug("Intel accelerator detection skipped: %s", e)

# Finalize Hardware Units (Only use CPU as a slot if NO accelerators exist)
if not HARDWARE_UNITS:
    logger.info("No accelerators detected. Using Host CPU for all tasks.")
    HARDWARE_UNITS.append({"type": "CPU", "id": "CPU", "name": "Host CPU"})
else:
    logger.info("Accelerators detected. CPU overflow disabled for Vocal Separation.")

logger.info("Hardware Resource Pool Initialized: %s", [u['id'] for u in HARDWARE_UNITS])

logger.debug("Final Hardware Resource Pool: %s", [u['id'] for u in HARDWARE_UNITS])

# --- [DEVICE ASSIGNMENT] ---
if ASR_DEVICE_ENV == "AUTO":
    DEVICE = _DETECTED_DEVICE
else:
    DEVICE = ASR_DEVICE_ENV

ASR_PREPROCESS_DEVICE_ENV = os.environ.get(
    "ASR_PREPROCESS_DEVICE", "AUTO").upper()
if ASR_PREPROCESS_DEVICE_ENV == "AUTO":
    PREPROCESS_DEVICE = _DETECTED_PREP_DEVICE
else:
    PREPROCESS_DEVICE = ASR_PREPROCESS_DEVICE_ENV

# --- [ENGINE SELECTION] ---
# ASR_ENGINE can be: AUTO (default), FASTER-WHISPER, INTEL-WHISPER
ASR_ENGINE = os.environ.get("ASR_ENGINE", "AUTO").upper()
if ASR_ENGINE == "AUTO":
    ASR_ENGINE = "FASTER-WHISPER"

# Redirect MODEL_ID if using Intel engine and local OpenVINO model exists
if ASR_ENGINE == "INTEL-WHISPER" and MODEL_ID == DEFAULT_MODEL:
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
        if Core:
            core_obj = Core()
        matching_devs = [d for d in core_obj.available_devices if DEVICE in d]
        if matching_devs:
            dev_id = matching_devs[0]
            ASR_DEVICE_NAME = core_obj.get_property(dev_id, "FULL_DEVICE_NAME")
    except Exception:  # pylint: disable=broad-exception-caught
        pass

if (os.environ.get("ASR_PREPROCESS_DEVICE", "AUTO").upper() == "AUTO" and
        PREPROCESS_DEVICE in ["NPU", "GPU", "CPU"]):
    try:
        if Core:
            core_obj = Core()
        matching_devs = [d for d in core_obj.available_devices if PREPROCESS_DEVICE in d]
        if matching_devs:
            dev_id = matching_devs[0]
            PREPROCESS_DEVICE_NAME = core_obj.get_property(dev_id, "FULL_DEVICE_NAME")
    except Exception:  # pylint: disable=broad-exception-caught
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
if DEVICE == "CUDA" or _DETECTED_DEVICE == "CUDA":
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
AGGRESSIVE_OFFLOAD = os.environ.get("AGGRESSIVE_OFFLOAD", "true").lower() == "true"

INITIAL_STEPS_RATIO = 2.8

# Path for compiled OpenVINO blobs and model downloads
LOCAL_CACHE = "./model_cache"
OV_CACHE_DIR = os.environ.get("OV_CACHE_DIR", LOCAL_CACHE)

# --- [SSD WRITE WEAR OPTIMIZATION] ---
TEMP_DIR = os.environ.get("WHISPER_TEMP_DIR", tempfile.gettempdir())
os.makedirs(TEMP_DIR, exist_ok=True)

# State and Telemetry Directory (Redirect to RAM-disk/tmpfs to prevent SSD wear)
STATE_DIR = os.environ.get("WHISPER_STATE_DIR", TEMP_DIR)
os.makedirs(STATE_DIR, exist_ok=True)

TEMP_DIR_MIN_FREE_BYTES = int(os.environ.get(
    "WHISPER_TEMP_MIN_FREE_MB", 512)) * 1024 * 1024

PERSISTENT_TEMP_DIR = os.path.join(OV_CACHE_DIR, "temp")
os.makedirs(PERSISTENT_TEMP_DIR, exist_ok=True)


def get_temp_dir(required_bytes=0):
    """Return the best temp directory for transient file I/O."""
    threshold = max(TEMP_DIR_MIN_FREE_BYTES, required_bytes)
    try:
        free = shutil.disk_usage(TEMP_DIR).free
        if free < threshold:
            return PERSISTENT_TEMP_DIR
    except Exception:  # pylint: disable=broad-exception-caught
        return PERSISTENT_TEMP_DIR
    return TEMP_DIR


def get_preprocessing_cache_dir():
    """Resolve the preprocessing cache directory dynamically."""
    base = get_temp_dir()
    path = os.path.join(base, "preprocessing")
    os.makedirs(path, exist_ok=True)
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

# --- [LANGUAGE PROCESSING & VAD] ---
VAD_MIN_SILENCE_DURATION_MS = int(
    os.environ.get("VAD_MIN_SILENCE_DURATION_MS", 500))
VAD_SPEECH_PAD_MS = int(os.environ.get("VAD_SPEECH_PAD_MS", 500))

INITIAL_PROMPT = os.environ.get(
    "INITIAL_PROMPT",
    "This video contains speech in multiple languages including "
    "Romanian, English, French, Italian, German, and Spanish."
)

# --- [PREPROCESSING CONFIGURATION] ---
ENABLE_VOCAL_SEPARATION = os.environ.get(
    "ENABLE_VOCAL_SEPARATION", "false").lower() == "true"

VOCAL_SEPARATION_SEGMENT_DURATION = int(
    os.environ.get("VOCAL_SEPARATION_SEGMENT_DURATION", 600))

logger.debug("Final Preprocessing Device: %s", PREPROCESS_DEVICE)

DEFAULT_WHISPER_THREADS = int(os.environ.get("ASR_THREADS", 4))
PREPROCESS_THREADS_ENV = int(os.environ.get("ASR_PREPROCESS_THREADS", 4))

# --- [THREAD & PERFORMANCE TUNING] ---


def _resolve_thread_limits(requested_asr, requested_prep):
    """Resolve and enforce physical hardware thread limits with priority."""
    # Use CPU_CORE_LIMIT as the effective "core count" for the application logic
    cores = CPU_CORE_LIMIT

    # 1. Handle AUTO scaling
    if MAX_CPU >= 999:
        # If AUTO, we use the requested threads (capped to total budget)
        # The parallel limit will then scale the number of units to fill the remaining budget.
        return min(requested_asr, cores), min(requested_prep, cores)

    # 2. Scale threads per task based on allowed parallel units to fit the global limit
    # Default behavior: If MAX_CPU_UNITS=1, asr_threads can be up to CPU_CORE_LIMIT.
    # If MAX_CPU_UNITS=2, asr_threads is capped at CPU_CORE_LIMIT/2 per task.
    effective_pool = max(1, CPU_CORE_LIMIT // MAX_CPU)

    # Faster-Whisper CPU threads
    asr_threads = min(requested_asr, effective_pool)

    # Threads for ONNX Runtime / OpenVINO (Vocal Separation)
    if DEVICE == "CPU":
        prep_threads = min(requested_prep, effective_pool)
        if asr_threads + prep_threads > effective_pool:
            prep_threads = max(1, effective_pool - asr_threads)
            logger.info("[Config] CPU Priority: Capping Prep threads to %d "
                        "to protect Whisper inference (%d threads)",
                        prep_threads, asr_threads)
    else:
        prep_threads = min(requested_prep, cores)

    if asr_threads < requested_asr:
        logger.info("[Config] Capping ASR_THREADS to %d (Global Limit: %d, Units: %d)",
                    asr_threads, CPU_CORE_LIMIT, MAX_CPU)
    if prep_threads < requested_prep and DEVICE != "CPU":
        logger.info("[Config] Capping ASR_PREPROCESS_THREADS to %d (Hardware limit)", cores)
    return asr_threads, prep_threads


ASR_THREADS, PREPROCESS_THREADS = _resolve_thread_limits(
    DEFAULT_WHISPER_THREADS, PREPROCESS_THREADS_ENV
)

# Industry standard thread limits for shared libraries
os.environ["OMP_NUM_THREADS"] = str(PREPROCESS_THREADS)
os.environ["MKL_NUM_THREADS"] = str(PREPROCESS_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(PREPROCESS_THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(PREPROCESS_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(PREPROCESS_THREADS)
os.environ["ORT_INTRA_OP_NUM_THREADS"] = str(PREPROCESS_THREADS)
os.environ["ORT_INTER_OP_NUM_THREADS"] = str(PREPROCESS_THREADS)

# FFmpeg concurrency for audio preparation
FFMPEG_THREADS = int(os.environ.get("FFMPEG_THREADS", 0))
FFMPEG_HWACCEL = os.environ.get("FFMPEG_HWACCEL", "none")
FFMPEG_FILTER = os.environ.get("FFMPEG_FILTER", "dynaudnorm")


def _validate_thread_concurrency():
    """Enforce hardware-aware thread limits to maintain responsiveness."""
    global FFMPEG_THREADS  # pylint: disable=global-statement
    try:
        logical_cores = CPU_CORE_LIMIT
        if FFMPEG_THREADS == 0 and PREPROCESS_THREADS > 1:
            logger.info(
                "[Config] Parallel prep detected (Threads: %d). "
                "Auto-capping FFMPEG_THREADS to 1 to prevent core thrashing.",
                PREPROCESS_THREADS
            )
            FFMPEG_THREADS = 1

        eff_ffmpeg = FFMPEG_THREADS if FFMPEG_THREADS > 0 else 1
        total_load = PREPROCESS_THREADS * eff_ffmpeg

        if total_load > logical_cores:
            logger.warning(
                "[Config] OVER-PROVISIONING: PREPROCESS_THREADS (%d) * "
                "FFMPEG_THREADS (%d) = %d, which exceeds logical cores (%d).",
                PREPROCESS_THREADS, eff_ffmpeg, total_load, logical_cores
            )
    except Exception:  # pylint: disable=broad-exception-caught
        pass


_validate_thread_concurrency()


def _calculate_cpu_parallel_limit():
    """Calculate how many multi-threaded CPU tasks can run safely."""
    if MAX_CPU < 999:
        return MAX_CPU

    cores = CPU_CORE_LIMIT
    cores_per_task = max(ASR_THREADS, PREPROCESS_THREADS)
    limit = max(1, cores // cores_per_task)
    logger.info("[Resource] Calculated AUTO CPU parallel limit: %d "
                "(Cores: %d, Threads/Task: %d)",
                limit, cores, cores_per_task)
    return limit


CPU_PARALLEL_LIMIT = _calculate_cpu_parallel_limit()


def get_parallel_limit(device):
    """Determine parallel task limit based on physical resource units."""
    if device in ["CUDA", "GPU", "NPU"]:
        try:
            if device == "CUDA":
                if ctranslate2:
                    return max(1, ctranslate2.get_cuda_device_count())

            if device in ["GPU", "NPU"]:
                if Core:
                    core_local = Core()
                # Count distinct physical hardware units of this type
                units = [d for d in core_local.available_devices if device in d]
                return max(1, len(units))
        except Exception:  # pylint: disable=broad-exception-caught
            pass

        return 1  # Safe default if detection fails

    # CPU-bound tasks: Capped by hardware-optimized slot count
    return CPU_PARALLEL_LIMIT


# --- [LANGUAGE DETECTION] ---
# Enable iterative scanning for quiet or long-intro files
SMART_SAMPLING_SEARCH = os.environ.get(
    "SMART_SAMPLING_SEARCH", "false").lower() == "true"
# Enable vocal isolation during language detection (improves identification accuracy)
ENABLE_LD_PREPROCESSING = os.environ.get(
    "ENABLE_LD_PREPROCESSING", "true").lower() == "true"
# Aggressiveness of VAD during language detection (0.0 to 1.0)
LD_VAD_THRESHOLD = float(os.environ.get("LD_VAD_THRESHOLD", 0.3))

# --- [HALLUCINATION FILTERING] ---
# Known "silence" or "credit" hallucination phrases for removal during post-processing
HALLUCINATION_SILENCE_THRESHOLD = float(os.environ.get("HALLUCINATION_SILENCE_THRESHOLD", 0.85))
HALLUCINATION_REPETITION_THRESHOLD = int(os.environ.get("HALLUCINATION_REPETITION_THRESHOLD", 15))

HALLUCINATION_PHRASES = [
    # Romanian
    "nu uitați să dați like", "nu uitati sa dati like",
    "să lăsați un comentariu", "sa lasati un comentariu",
    "să distribuiți", "sa distribuiti",
    "abonați-vă la canal", "abonati-va la canal",
    "nu uitați să vă abonați", "nu uitati sa va abonati",
    "pentru a nu rata videoclipurile noastre",
    "nu uitați să dați like, să lăsați un comentariu și "
    "să distribuiți acest material video pe alte rețele sociale",
    "nu uitati sa dati like, sa lasati un comentariu si "
    "sa distribuiti acest material video pe alte retele sociale",
    "nu uitați să vă abonați la canal, să vă mulțumim și la rețeta următoare",
    "abonati-va la canal, sa va multumim si la reteta urmatoare",
    "vă mulțumim pentru vizionare", "va multumim pentru vizionare",
    "nu uitați să apăsați butonul de like",
    # English
    "thank you for watching", "thanks for watching",
    "subscribe to my channel", "please subscribe",
    "like and subscribe", "hit the like button",
    "leave a comment", "share this video",
    "see you in the next", "bye bye",
    # French
    "merci d'avoir regardé", "n'oubliez pas de vous abonner",
    "laissez un commentaire", "à bientôt",
    # German
    "danke fürs zuschauen", "vergisst nicht zu abonnieren",
    # Spanish
    "gracias por ver", "no olvides suscribirte",
    # Italian
    "grazie per aver guardato", "non dimenticare di iscriverti",
]
