"""
Configuration Manager for Whisper Pro ASR

This module handles hardware detection, environment variable parsing, and 
model path resolution for both Whisper and UVR/MDX-NET engines.
"""
import os
import logging

# Set up early logger for configuration phase
logger = logging.getLogger(__name__)

# --- [CORE SERVICE CONFIG] ---
APP_NAME = "Whisper Pro ASR"
VERSION = "1.0.0"

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
try:
    # 1. NVIDIA Acceleration Check
    import ctranslate2  # pylint: disable=import-error
    if ctranslate2.get_cuda_device_count() > 0:
        logger.debug("Auto-detected %d NVIDIA GPU(s).",
                     ctranslate2.get_cuda_device_count())
        _DETECTED_DEVICE = "CUDA"
        _DETECTED_PREP_DEVICE = "CUDA"
        _DETECTED_COMPUTE = "float16"
    else:
        # 2. Intel Accelerator Check (OpenVINO)
        from openvino import Core  # pylint: disable=import-error
        core = Core()
        devices = core.available_devices
        logger.debug("OpenVINO Available Devices: %s", devices)

        has_npu = any("NPU" in d for d in devices)
        has_gpu = any("GPU" in d for d in devices)

        if has_gpu:
            GPU_DEV = [d for d in devices if 'GPU' in d][0]
            try:
                GPU_FULL_NAME = core.get_property(GPU_DEV, "FULL_DEVICE_NAME")
            except Exception:  # pylint: disable=broad-exception-caught
                GPU_FULL_NAME = "Intel GPU"
            logger.debug("Auto-detected %s (%s).", GPU_FULL_NAME, GPU_DEV)
            # Preference: We use CPU for the ASR Engine (Whisper) to ensure
            # full support for Beam Search (Quality), which is often limited on iGPU/NPU.
            # We still use GPU for Preprocessing (Vocal Separation).
            _DETECTED_DEVICE = "CPU"
            _DETECTED_PREP_DEVICE = "GPU"
        elif has_npu:
            NPU_DEV = [d for d in devices if 'NPU' in d][0]
            try:
                NPU_FULL_NAME = core.get_property(NPU_DEV, "FULL_DEVICE_NAME")
            except Exception:  # pylint: disable=broad-exception-caught
                NPU_FULL_NAME = "Intel NPU"
            logger.debug("Auto-detected %s (%s).", NPU_FULL_NAME, NPU_DEV)
            _DETECTED_DEVICE = "CPU"
            _DETECTED_PREP_DEVICE = "NPU"

        # if has_npu or has_gpu:
        #     _DETECTED_ENGINE = "INTEL-WHISPER"

        _DETECTED_COMPUTE = "int8"

except Exception as e:  # pylint: disable=broad-exception-caught
    logger.debug("Hardware detection notice: %s. Defaulting to CPU.", e)

# --- [DEVICE ASSIGNMENT] ---
if ASR_DEVICE_ENV == "AUTO":
    DEVICE = _DETECTED_DEVICE
else:
    DEVICE = ASR_DEVICE_ENV

ASR_PREPROCESS_DEVICE_ENV = os.environ.get(
    "ASR_PREPROCESS_DEVICE", "AUTO").upper()
if ASR_PREPROCESS_DEVICE_ENV == "AUTO":
    # If ASR_DEVICE was manual (e.g. CUDA), we still prefer the same for prep if available
    # but if ASR_DEVICE is CPU, we might still want to use a detected GPU for prep.
    # The _DETECTED_PREP_DEVICE already has the best priority logic.
    PREPROCESS_DEVICE = _DETECTED_PREP_DEVICE
else:
    PREPROCESS_DEVICE = ASR_PREPROCESS_DEVICE_ENV

# --- [ENGINE SELECTION] ---
if ASR_ENGINE_ENV == "AUTO":
    # User override: If ASR_DEVICE is explicitly CUDA or CPU, default to FASTER-WHISPER
    if ASR_DEVICE_ENV in ["CUDA", "CPU"]:
        ASR_ENGINE = "FASTER-WHISPER"
    else:
        ASR_ENGINE = _DETECTED_ENGINE
else:
    ASR_ENGINE = ASR_ENGINE_ENV

# Redirect MODEL_ID to OpenVINO version if Intel engine is used
if ASR_ENGINE == "INTEL-WHISPER":
    # If no explicit path provider, prefer our baked-in OpenVINO model.
    user_model_env = os.environ.get("ASR_MODEL")
    if user_model_env is None or user_model_env == DEFAULT_MODEL:
        if os.path.exists(OV_MODEL_PATH):
            MODEL_ID = OV_MODEL_PATH
            logger.debug("Redirected MODEL_ID to OpenVINO path: %s", MODEL_ID)
        else:
            # Fallback to official OpenVINO HF ID for download.
            MODEL_ID = "OpenVINO/whisper-large-v3-fp16-ov"
            logger.debug(
                "No local OpenVINO model found. Falling back to HF ID: %s", MODEL_ID)

logger.debug("ASR Engine set to: %s", ASR_ENGINE)

# --- [UI & LOGGING DESCRIPTORS] ---
ASR_DEVICE_NAME = "NVIDIA GPU" if DEVICE == "CUDA" else DEVICE
PREPROCESS_DEVICE_NAME = "NVIDIA GPU" if PREPROCESS_DEVICE == "CUDA" else PREPROCESS_DEVICE

# Refine names using hardware properties for the startup banner
if ASR_DEVICE_ENV == "AUTO" and DEVICE in ["NPU", "GPU", "CPU"]:
    try:
        from openvino import Core  # pylint: disable=import-error
        core = Core()
        dev_id = [d for d in core.available_devices if DEVICE in d][0]
        ASR_DEVICE_NAME = core.get_property(dev_id, "FULL_DEVICE_NAME")
    except Exception:  # pylint: disable=broad-exception-caught
        pass

if (os.environ.get("ASR_PREPROCESS_DEVICE", "AUTO").upper() == "AUTO" and
        PREPROCESS_DEVICE in ["NPU", "GPU", "CPU"]):
    try:
        from openvino import Core  # pylint: disable=import-error
        core = Core()
        dev_id = [d for d in core.available_devices if PREPROCESS_DEVICE in d][0]
        PREPROCESS_DEVICE_NAME = core.get_property(dev_id, "FULL_DEVICE_NAME")
    except Exception:  # pylint: disable=broad-exception-caught
        pass

# --- [COMPUTE TYPE RESOLUTION] ---
if ASR_COMPUTE_ENV == "AUTO":
    if ASR_DEVICE_ENV == "AUTO":
        COMPUTE_TYPE = _DETECTED_COMPUTE
    else:
        # Fallback defaults for forced devices
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
# ASR_BATCH_SIZE is used for parallel chunk inference
DEFAULT_BATCH_SIZE = int(os.environ.get("ASR_BATCH_SIZE", 1))
# ASR_BEAM_SIZE improves accuracy but adds latency and VRAM/NPU memory pressure
DEFAULT_BEAM_SIZE = int(os.environ.get("ASR_BEAM_SIZE", 5))

# Debug and Logging
DEBUG_MODE = os.environ.get("DEBUG", "false").lower() == "true"
TEXT_LOGS = os.environ.get("TEXT_LOGS", "false").lower() == "true"

# Calibration stats for ETA calculation
INITIAL_STEPS_RATIO = 2.8

# Path for compiled OpenVINO blobs and model downloads
LOCAL_CACHE = "./model_cache"
OV_CACHE_DIR = os.environ.get("OV_CACHE_DIR", LOCAL_CACHE)
PREPROCESSING_CACHE_DIR = os.path.join(OV_CACHE_DIR, "preprocessing")

# --- [MODEL PATH RESOLUTION] ---
# "Batteries Included" logic: If use has not changed the default model,
# we prefer the baked-in models in /app/system_models to avoid downloads.
_SYSTEM_MODELS_DIR = "/app/system_models"
DEFAULT_WHISPER = "Systran/faster-whisper-large-v3"
DEFAULT_UVR = "UVR-MDX-NET-Inst_HQ_3.onnx"

# 1. Whisper Model Resolution - Successfully finalized in hardware detection phase.
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
# VAD (Voice Activity Detection) parameters
VAD_MIN_SILENCE_DURATION_MS = int(
    os.environ.get("VAD_MIN_SILENCE_DURATION_MS", 500))
VAD_SPEECH_PAD_MS = int(os.environ.get("VAD_SPEECH_PAD_MS", 500))

# Contextual prompt injected into the decoder
INITIAL_PROMPT = os.environ.get(
    "INITIAL_PROMPT", "Transcribe the following audio file.")

# --- [PREPROCESSING CONFIGURATION] ---
# Vocal Separation (utilizing the UVR/MDX-Net architecture)
ENABLE_VOCAL_SEPARATION = os.environ.get(
    "ENABLE_VOCAL_SEPARATION", "false").lower() == "true"

# Preprocessing Device resolution - Finalized in hardware detection phase.
logger.debug("Final Preprocessing Device: %s", PREPROCESS_DEVICE)

# --- [THREAD & PERFORMANCE TUNING] ---
# Faster-Whisper CPU threads (used for transcription when using beam search)
ASR_THREADS = int(os.environ.get("ASR_THREADS", 4))

# Threads for ONNX Runtime / OpenVINO (Vocal Separation)
PREPROCESS_THREADS = int(os.environ.get("ASR_PREPROCESS_THREADS", 4))

# Industry standard thread limits for shared libraries
os.environ["OMP_NUM_THREADS"] = str(PREPROCESS_THREADS)
os.environ["MKL_NUM_THREADS"] = str(PREPROCESS_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(PREPROCESS_THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(PREPROCESS_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(PREPROCESS_THREADS)
os.environ["ORT_INTRA_OP_NUM_THREADS"] = str(PREPROCESS_THREADS)
os.environ["ORT_INTER_OP_NUM_THREADS"] = str(PREPROCESS_THREADS)

# FFmpeg concurrency for audio preparation (0 = auto-detect)
FFMPEG_THREADS = int(os.environ.get("FFMPEG_THREADS", 0))

# --- [LANGUAGE DETECTION] ---
# Enable iterative scanning for quiet or long-intro files
SMART_SAMPLING_SEARCH = os.environ.get(
    "SMART_SAMPLING_SEARCH", "false").lower() == "true"
# Enable vocal isolation during language detection (improves identification accuracy)
ENABLE_LD_PREPROCESSING = os.environ.get(
    "ENABLE_LD_PREPROCESSING", "false").lower() == "true"

# --- [HALLUCINATION FILTERING] ---
# Known "silence" or "credit" hallucination phrases for removal during post-processing
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
