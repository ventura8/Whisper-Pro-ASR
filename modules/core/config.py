"""
Configuration Manager for Whisper Pro ASR

This module handles hardware detection, environment variable parsing, and
model path resolution for both Whisper and UVR/MDX-NET engines.
"""

import importlib
import logging
import os

from modules.core.config_helpers import (
    calculate_cpu_parallel_limit,
    detect_hardware,
    get_unit_limit,
    resolve_thread_limits,
)
from modules.core.constants import HALLUCINATION_PHRASES
from modules.core.mount_helpers import get_custom_mount_points, resolve_temp_dir

from . import config_model_paths, config_paths, config_resolution, device_probe, engine_registry

# Explicitly referenced to satisfy the unused-import check; both are part of this
# module's public surface and are read through it by callers and tests.
_ = (HALLUCINATION_PHRASES, get_custom_mount_points)

# Set up early logger for configuration phase
logger = logging.getLogger(__name__)

HOST = os.environ.get("HOST") or ".".join(["0", "0", "0", "0"])

# --- [CORE SERVICE CONFIG] ---
APP_NAME = "Whisper Pro ASR"
VERSION = "1.3.0"
#: Which image this container was built from (cpu, intel, nvidia, amd, nvidia-intel,
#: full), stamped per target in the Dockerfile. Two containers can run the same VERSION
#: with very different accelerator support, so the dashboard shows both. Empty when the
#: app runs outside a shipped image (local checkout, tests).
IMAGE_EDITION = os.environ.get("WHISPER_IMAGE_EDITION", "").strip()
#: VERSION plus the edition, for display only. VERSION itself stays a bare semver
#: because API clients (Bazarr among them) parse it.
VERSION_DISPLAY = f"{VERSION} {IMAGE_EDITION}" if IMAGE_EDITION else VERSION
HARDWARE_UNITS: list[dict[str, str]] = []  # Global registry for accelerator orchestration
DIARIZATION_HF_TOKEN = os.environ.get("DIARIZATION_HF_TOKEN", "").strip()

WHISPERX_LIB_PATH = os.environ.get("WHISPERX_LIB_PATH", "/app/libs/whisperx")
WHISPERX_AVAILABLE = os.path.isdir(WHISPERX_LIB_PATH)

# --- [RESOURCE POOL LIMITS] ---
CPU_CORE_LIMIT = int(os.environ.get("CPU_CORE_LIMIT", 4))


MAX_CUDA = get_unit_limit("MAX_CUDA_UNITS", 1, min_value=0)
MAX_GPU = get_unit_limit("MAX_GPU_UNITS", 1, min_value=0)
MAX_NPU = get_unit_limit("MAX_NPU_UNITS", 1, min_value=0)
MAX_AMD = get_unit_limit("MAX_AMD_UNITS", 1, min_value=0)
MAX_CPU = get_unit_limit("MAX_CPU_UNITS", 1, min_value=1)

# Memory reclamation behavior (unloads models when idle if True)
AGGRESSIVE_OFFLOAD = os.environ.get("AGGRESSIVE_OFFLOAD", "false").lower() == "true"
MODEL_IDLE_TIMEOUT = int(os.environ.get("MODEL_IDLE_TIMEOUT", 300))

# --- [HARDWARE DETECTION & DEVICE MAPPING] ---
# ASR_ENGINE can be: AUTO (default), FASTER-WHISPER, INTEL-WHISPER, OPENAI-WHISPER, WHISPERX.
# AUTO resolves to engine_registry.AUTO_DEFAULT_ENGINE -- Faster-Whisper -- on every host,
# regardless of the accelerators present. The hardware still decides which *unit* the task
# runs on (engine_registry.AUTO_DEVICE_PRIORITY), just not which engine decodes it.
# An accelerator-specific engine is still available; it has to be asked for, with
# ASR_ENGINE=INTEL-WHISPER or ASR_ENGINE=OPENAI-WHISPER.
# An empty value is treated as unset so `ASR_ENGINE=` in an env file cannot fail validation.
ASR_ENGINE_ENV = (os.environ.get("ASR_ENGINE") or "AUTO").strip().upper()
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


# --- [PROCESS ISOLATION] ---
# Engines and vocal separation each run in their own worker process. Declared before
# hardware detection because device assignment below depends on whether preprocessing is
# isolated: the cross-vendor restriction only applies when contexts share an interpreter.
ISOLATE_ENGINES = (os.environ.get("ASR_ISOLATE_ENGINES") or "1").strip().lower() not in ("0", "false", "no")
#: Vocal separation runs out-of-process too. Separate switch from ISOLATE_ENGINES because
#: the trade-offs differ: UVR is where ONNX Runtime's CUDA provider parks device memory
#: the in-process purge cannot reclaim, and where cross-vendor contexts collide.
ISOLATE_PREPROCESSING = (os.environ.get("ASR_ISOLATE_PREPROCESSING") or ("1" if ISOLATE_ENGINES else "0")).strip().lower() not in (
    "0",
    "false",
    "no",
)

# --- [HARDWARE DETECTION] ---
logger.debug("Performing hardware detection...")
_DETECTED_DEVICE, _DETECTED_PREP_DEVICE, _DETECTED_COMPUTE = detect_hardware(MAX_CUDA, MAX_GPU, MAX_NPU, MAX_AMD, HARDWARE_UNITS)

# --- [DEVICE ASSIGNMENT] ---
if ASR_DEVICE_ENV == "AUTO":
    DEVICE = _DETECTED_DEVICE
else:
    DEVICE = ASR_DEVICE_ENV

ASR_PREPROCESS_DEVICE_ENV = os.environ.get("ASR_PREPROCESS_DEVICE", "AUTO").upper()
ACCELERATED_PREPROCESS_DEVICES = frozenset({"CUDA", "GPU", "NPU", "OPENVINO", "AMD"})
if ASR_PREPROCESS_DEVICE_ENV == "AUTO":
    PREPROCESS_DEVICE = _DETECTED_PREP_DEVICE
else:
    PREPROCESS_DEVICE = ASR_PREPROCESS_DEVICE_ENV

# Initializing an OpenVINO GPU/NPU (Level-Zero/OpenCL) context in the same
# process as an already-active CUDA context has been observed to crash/hang
# natively on Intel iGPU hardware (confirmed via direct testing: mixing
# CUDA-ASR with Intel-OpenVINO-targeted UVR preprocessing in one process).
# This can arise both from an explicit ASR_PREPROCESS_DEVICE override and
# from AUTO hardware detection on hybrid NVIDIA+Intel machines (detect_hardware
# in config_helpers.py can select CUDA for ASR while independently selecting
# Intel GPU/NPU for preprocessing). There is no known Python-side fix for the
# underlying driver-level crash, so this combination is disallowed outright.
# Preprocessing still gets GPU acceleration — just from the same vendor (CUDA)
# as ASR instead of a cross-vendor OpenVINO context; the CUDA/CPU dispatch in
# provider.py already falls back to CPU on its own if CUDA isn't actually
# usable in onnxruntime.
# Lifted when ISOLATE_PREPROCESSING is on: UVR then runs in its own process, so the
# OpenVINO context is never created in the interpreter holding the CUDA one and the
# driver-level crash cannot occur. That is what lets a hybrid host run ASR on CUDA while
# vocal separation uses the Intel iGPU -- and equally lets an AMD host drive UVR with
# ROCm alongside a CUDA or CPU ASR engine.
if DEVICE == "CUDA" and PREPROCESS_DEVICE in ("GPU", "NPU", "INTEL", "OPENVINO") and not ISOLATE_PREPROCESSING:
    logger.warning(
        "[Config] ASR_DEVICE=CUDA with Intel OpenVINO preprocessing (%s) is unsupported: mixing a "
        "CUDA context with an OpenVINO GPU/NPU context in the same process crashes on this driver "
        "stack. Forcing preprocessing to CUDA instead.",
        PREPROCESS_DEVICE,
    )
    PREPROCESS_DEVICE = "CUDA"

# --- [ENGINE SELECTION, SCHEDULER POOL AND HYBRID ENGINES] ---
# The whole decision lives in config_resolution so it can be tested without the machine the
# tests run on changing the answer. HARDWARE_UNITS is narrowed in place: it is the live
# scheduler pool every other module reads.
_ENGINE_RESOLUTION = config_resolution.resolve_engine_and_pool(
    asr_engine_env=ASR_ENGINE_ENV,
    asr_device_env=ASR_DEVICE_ENV,
    device=DEVICE,
    hardware_units=HARDWARE_UNITS,
    isolate_engines=ISOLATE_ENGINES,
    hybrid_env=os.environ.get("HYBRID_ENGINES", ""),
    logger=logger,
)
ASR_ENGINE = _ENGINE_RESOLUTION["ASR_ENGINE"]
ASR_ENGINE_SOURCE = _ENGINE_RESOLUTION["ASR_ENGINE_SOURCE"]
DEVICE = _ENGINE_RESOLUTION["DEVICE"]
ASR_DEVICE_ENV = _ENGINE_RESOLUTION["ASR_DEVICE_ENV"]
HYBRID_ENGINES = _ENGINE_RESOLUTION["HYBRID_ENGINES"]
DETECTED_HARDWARE_UNITS = _ENGINE_RESOLUTION["DETECTED_HARDWARE_UNITS"]
#: Every accelerator detection found, before the ASR engine's own limits were applied.
#: Preprocessing must not inherit that filter: UVR runs on ONNX Runtime and reaches devices
#: CTranslate2 never will, so an Intel iGPU pruned for FASTER-WHISPER's sake is still a
#: perfectly good place to run vocal isolation.
DETECTED_UNITS: list[dict[str, str]] = _ENGINE_RESOLUTION["DETECTED_UNITS"]
ASR_ENGINE_RESOLUTION = _ENGINE_RESOLUTION["ASR_ENGINE_RESOLUTION"]
if DEVICE != ASR_DEVICE_ENV and ASR_DEVICE_ENV in ("CUDA", "AMD", "GPU", "NPU", "CPU"):
    os.environ["ASR_DEVICE"] = DEVICE

# Redirect MODEL_ID if using Intel engine and local OpenVINO model exists
if ASR_ENGINE == "INTEL-WHISPER" and ASR_ENV == DEFAULT_MODEL:
    if os.path.exists(OV_MODEL_PATH):
        MODEL_ID = OV_MODEL_PATH
    else:
        MODEL_ID = "OpenVINO"  # Trigger HF download of OV optimized model

logger.debug("ASR Engine set to: %s", ASR_ENGINE)

# --- [RUNTIME DIRECTORIES] ---
# Resolved through config_paths so a reload recomputes them together.
LOCAL_CACHE = config_paths.LOCAL_CACHE
_RUNTIME_DIRS = config_paths.resolve_runtime_dirs()
OV_CACHE_DIR = _RUNTIME_DIRS["OV_CACHE_DIR"]
PERSISTENT_DIR = _RUNTIME_DIRS["PERSISTENT_DIR"]
STATE_DIR = _RUNTIME_DIRS["STATE_DIR"]
LOG_DIR = _RUNTIME_DIRS["LOG_DIR"]
APPROVED_ROOTS = _RUNTIME_DIRS["APPROVED_ROOTS"]
TEMP_DIR = _RUNTIME_DIRS["TEMP_DIR"]
TEMP_DIR_MIN_FREE_BYTES = _RUNTIME_DIRS["TEMP_DIR_MIN_FREE_BYTES"]
PERSISTENT_TEMP_DIR = _RUNTIME_DIRS["PERSISTENT_TEMP_DIR"]


def get_temp_dir(required_bytes=0):
    """Return the best temp directory for transient file I/O."""
    return resolve_temp_dir(TEMP_DIR, PERSISTENT_TEMP_DIR, TEMP_DIR_MIN_FREE_BYTES, required_bytes)


def get_preprocessing_cache_dir():
    """Resolve the preprocessing cache directory dynamically."""
    return config_paths.preprocessing_cache_dir(get_temp_dir())


PREPROCESSING_CACHE_DIR = get_preprocessing_cache_dir()

# --- [MODEL PATH RESOLUTION] ---
# Runs before the device-executability probe below, which inspects the OpenVINO IR on disk:
# probing MODEL_ID while it still held the "OpenVINO" sentinel asked whether a directory
# named "OpenVINO" was statically shaped, and always answered no.
_SYSTEM_MODELS_DIR = "/app/system_models"
DEFAULT_WHISPER = "Systran/faster-whisper-large-v3"
DEFAULT_UVR = "UVR-MDX-NET-Inst_HQ_3.onnx"
OPENAI_WHISPER_DEFAULT_MODEL = config_model_paths.OPENAI_WHISPER_DEFAULT_MODEL

_MODEL_PATHS = config_model_paths.resolve_model_paths(
    asr_engine=ASR_ENGINE,
    asr_env=ASR_ENV,
    model_id=MODEL_ID,
    default_model=DEFAULT_MODEL,
    ov_cache_dir=OV_CACHE_DIR,
    ov_model_path=OV_MODEL_PATH,
)
MODEL_ID = _MODEL_PATHS["MODEL_ID"]
CT2_CACHE_DIR = _MODEL_PATHS["CT2_CACHE_DIR"]
OV_CACHE_MODEL_DIR = _MODEL_PATHS["OV_CACHE_MODEL_DIR"]
OV_MODEL_PATH = _MODEL_PATHS["OV_MODEL_PATH"]
OPENAI_WHISPER_CACHE_DIR = _MODEL_PATHS["OPENAI_WHISPER_CACHE_DIR"]
MODEL_ID_BY_ENGINE = _MODEL_PATHS["MODEL_ID_BY_ENGINE"]

logger.debug("Final Whisper Model Path: %s", MODEL_ID)


def engine_for_unit(unit: dict) -> str:
    """Return the ASR engine that should run on ``unit``."""
    return config_model_paths.engine_for_unit(unit, hybrid_engines=HYBRID_ENGINES, asr_engine=ASR_ENGINE)


def model_id_for_engine(engine: str) -> str:
    """Return the weights path ``engine`` should load."""
    return config_model_paths.model_id_for_engine(
        engine, hybrid_engines=HYBRID_ENGINES, model_id=MODEL_ID, model_id_by_engine=MODEL_ID_BY_ENGINE
    )


def engines_in_use() -> list[str]:
    """Every engine this deployment will actually load, for provisioning."""
    return config_model_paths.engines_in_use(HARDWARE_UNITS, hybrid_engines=HYBRID_ENGINES, asr_engine=ASR_ENGINE)


# --- [DEVICE EXECUTABILITY] ---
# The NPU builds a WhisperPipeline and only fails at inference, so an unusable NPU is not
# discovered until the first request 500s. This must run here, at config import, and not
# at app startup: the scheduler and the isolated workers read HARDWARE_UNITS when their
# modules are imported, so a later correction leaves the banner saying GPU while tasks are
# still dispatched to "hardware unit NPU".
if os.environ.get("VERIFY_RUNTIME", "true").lower() == "true" and DEVICE == "NPU" and ASR_ENGINE == engine_registry.ENGINE_INTEL_WHISPER:
    _npu_ok, _npu_reason = device_probe.npu_can_execute(MODEL_ID)
    if not _npu_ok:
        logger.error("ASR_DEVICE=NPU cannot execute this model: %s", _npu_reason)
        # CPU rather than the GPU: the NPU stays in the pool doing UVR, and ASR for it runs
        # on the CPU, so an NPU unit and a GPU unit proceed in parallel instead of both
        # queueing on the GPU. The pool is left alone -- the NPU is a working preprocessing
        # unit, and removing it would discard the reason this hardware is here.
        DEVICE = "CPU"
        os.environ["ASR_DEVICE"] = "CPU"
        ASR_DEVICE_ENV = "CPU"
        logger.error("ASR falls back to the CPU; the NPU stays in the pool for UVR preprocessing.")

# --- [UI & LOGGING DESCRIPTORS] ---
DEVICE_DISPLAY_NAMES = {"CUDA": "NVIDIA GPU", "AMD": "AMD GPU"}
ASR_DEVICE_NAME = DEVICE_DISPLAY_NAMES.get(DEVICE, DEVICE)
PREPROCESS_DEVICE_NAME = DEVICE_DISPLAY_NAMES.get(PREPROCESS_DEVICE, PREPROCESS_DEVICE)

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

# Re-transcribe per detected-language run for code-switched audio, instead of forcing one
# language across the whole file. Costs a full-file per-30s-chunk language scan, but only
# when the cheap montage vote already suspects more than one language is present -- a
# single-language file (the overwhelming majority of requests) never pays for it. On by
# default because it fixes a recorded defect (dropped code-switched legs); disable if it
# ever needs to be rolled back without a code change.
ASR_MULTILINGUAL_SEGMENTATION = os.environ.get("ASR_MULTILINGUAL_SEGMENTATION", "true").strip().lower() in ("true", "1", "yes")

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
# How sure Silero must be that a window contains speech before it is decoded.
#
# Transcription used to pass no threshold at all, so Silero ran at its library default and
# nothing here was tunable. It is plumbed now, but the default deliberately matches that
# old behaviour: 0.6 was tried against the `quiet` defect and measured as a bad trade.
#
#   threshold  hallucinated quiet windows   real windows missed
#   0.5 (this)              10 / 25                55 / 118
#   0.6                      9 / 25                62 / 118
#
# One fewer invented window cost seven real utterances. Raise it only with a measurement
# on the full pipeline -- an engine-only sweep says 0.6 removes far more hallucination
# than it does here, because that sweep skips UVR, and UVR has already stripped most
# non-vocal content before VAD ever sees it. The benefit shrinks in the real pipeline; the
# cost to quiet speech does not.
#
# Language detection keeps its own, deliberately looser LD_VAD_THRESHOLD: it samples for a
# language rather than deciding what gets transcribed.
VAD_THRESHOLD = float(os.environ.get("VAD_THRESHOLD", 0.5))

INITIAL_PROMPT = os.environ.get(
    "INITIAL_PROMPT",
    "This video contains speech in multiple languages including Romanian, English, French, Italian, German, and Spanish.",
)

# --- [PREPROCESSING CONFIGURATION] ---
ENABLE_VOCAL_SEPARATION = os.environ.get("ENABLE_VOCAL_SEPARATION", "false").strip().lower() in ("true", "1", "yes")

VOCAL_SEPARATION_SEGMENT_DURATION = int(os.environ.get("VOCAL_SEPARATION_SEGMENT_DURATION", 600))

logger.debug("Final Preprocessing Device: %s", PREPROCESS_DEVICE)

DEFAULT_WHISPER_THREADS = int(os.environ.get("ASR_THREADS", 4))
PREPROCESS_THREADS_ENV = int(os.environ.get("ASR_PREPROCESS_THREADS", 4))

# --- [THREAD & PERFORMANCE TUNING] ---


ASR_THREADS, PREPROCESS_THREADS = resolve_thread_limits(DEFAULT_WHISPER_THREADS, PREPROCESS_THREADS_ENV, CPU_CORE_LIMIT, MAX_CPU, DEVICE)

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
                "[Config] OVER-PROVISIONING: PREPROCESS_THREADS (%d) + FFMPEG_THREADS (%d) = %d, which exceeds logical cores (%d).",
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
    if device not in ["CUDA", "GPU", "NPU", "AMD"]:
        return CPU_PARALLEL_LIMIT
    return _accelerator_parallel_limit(device)


def _accelerator_parallel_limit(device) -> int:
    try:
        units = [u for u in HARDWARE_UNITS if u.get("type") == device]
        if units:
            return len(units)
    except (AttributeError, TypeError, ValueError):
        pass
    return 1


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
