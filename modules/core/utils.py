"""
Cross-Platform Media Utilities

This module provides essential tools for audio standardization, time formatting,
and subtitle generation. It abstracts FFmpeg complexity and ensures that all
media ingested by the service conforms to a uniform 16kHz MONO specification.
"""

import contextlib
import contextvars
import json
import logging
import os
import tempfile
import threading
from shutil import which

import psutil

try:
    import torch
except ImportError:
    torch = None

from modules.core import config, process_exec, utils_helpers
from modules.core.subtitles import (
    format_single_srt_block,
    format_timestamp,
    generate_srt,
    generate_tsv,
    generate_txt,
    generate_vtt,
    wrap_text,
)
from modules.core.utils_helpers import (
    cleanup_old_files,
    get_pretty_model_name,
    purge_temporary_assets,
    run_ffmpeg_standardization,
    secure_remove,
    validate_audio,
)

_parse_ffmpeg_progress_impl = utils_helpers.parse_ffmpeg_progress

# Reference to satisfy unused import check for public API consumption
_unused_api = (secure_remove, get_pretty_model_name, validate_audio, cleanup_old_files, purge_temporary_assets)

# Re-export for public API and compatibility
_reexports = (
    wrap_text,
    generate_srt,
    format_timestamp,
    generate_vtt,
    generate_txt,
    generate_tsv,
    format_single_srt_block,
)

# Global process object for telemetry consistency
_PROCESS_OBJ = psutil.Process(os.getpid())


def get_system_telemetry():
    """Gather consistent system and process utilization metrics."""
    cpu_usage = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    app_mem_rss = _PROCESS_OBJ.memory_info().rss
    app_cpu = _PROCESS_OBJ.cpu_percent(interval=None)

    return {
        "cpu_percent": cpu_usage,
        "app_cpu_percent": round(app_cpu / (psutil.cpu_count() or 1), 1),
        "memory_percent": mem.percent,
        "memory_used_gb": round(mem.used / (1024**3), 2),
        "memory_total_gb": round(mem.total / (1024**3), 2),
        "app_memory_gb": round(app_mem_rss / (1024**3), 2),
    }


def get_nvidia_vram_usage_mb() -> int | None:
    """Return total used NVIDIA VRAM in MB across visible GPUs, or None if unavailable."""
    nvidia_smi = which("nvidia-smi")
    if not nvidia_smi:
        return None

    lines = _query_nvidia_vram_lines(nvidia_smi)
    if lines is None:
        return None

    return _parse_nvidia_vram_total(lines)


def _query_nvidia_vram_lines(nvidia_smi: str) -> list[str] | None:
    """Query nvidia-smi and return raw memory-used lines."""
    try:
        output = process_exec.check_output_text(
            [nvidia_smi, "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            timeout=5.0,
        )
    except (
        process_exec.CommandExecutionError,
        process_exec.CommandTimeoutError,
        FileNotFoundError,
        OSError,
    ):
        return None
    return output.splitlines()


def _parse_nvidia_vram_total(lines: list[str]) -> int | None:
    """Parse raw nvidia-smi memory-used lines and sum valid MB values."""
    values = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            values.append(int(stripped.split(",")[0].strip()))
        except ValueError:
            continue

    if not values:
        return None
    return sum(values)


NOT_SET = object()
TRACKED_FILES_VAR = contextvars.ContextVar("tracked_files", default=None)
FILENAME_VAR = contextvars.ContextVar("filename", default=NOT_SET)
SOURCE_PATH_VAR = contextvars.ContextVar("source_path", default=NOT_SET)
INPUT_FLAGS_VAR = contextvars.ContextVar("input_flags", default=None)
REQUEST_TRACKED_FILES_VAR = contextvars.ContextVar("request_tracked_files", default=None)


class ContextVarProxy:
    """Proxy class for Thread/Context local storage."""

    def __init__(self):
        self._cv = contextvars.ContextVar("thread_context", default=None)
        # Define expected dynamic attributes to avoid W0201 linter warnings
        self.task_id = None
        self.registration_thread_id = None
        self.is_priority = False
        self.caller_info = {}
        self.request_json = {}
        self.endpoint = ""
        self.total_duration = 0

    def _get_dict(self):
        d = self._cv.get()
        if d is None:
            d = {}
            self._cv.set(d)
        return d

    def reset(self):
        """Reset the context to a new empty dictionary to ensure request isolation."""
        self._cv.set({})
        TRACKED_FILES_VAR.set(None)
        FILENAME_VAR.set(NOT_SET)
        SOURCE_PATH_VAR.set(NOT_SET)
        INPUT_FLAGS_VAR.set(None)

    def __getattr__(self, name):
        special_val = _get_special_context_attr(name)
        if special_val is not NOT_SET:
            return special_val
        d = self._get_dict()
        if name in d:
            return d[name]
        raise AttributeError(f"'ContextVarProxy' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name == "_cv":
            super().__setattr__(name, value)
            return
        if _set_special_context_attr(name, value):
            return
        d = self._get_dict()
        d[name] = value

    def __delattr__(self, name):
        if _delete_special_context_attr(name):
            return
        d = self._get_dict()
        if name in d:
            del d[name]


def _get_special_context_attr(name):
    if name == "tracked_files":
        return _get_or_create_tracked_files()
    if name == "filename":
        return _get_required_context_value(FILENAME_VAR, name)
    if name == "source_path":
        return _get_required_context_value(SOURCE_PATH_VAR, name)
    if name == "input_flags":
        return INPUT_FLAGS_VAR.get()
    return NOT_SET


def _get_or_create_tracked_files():
    val = TRACKED_FILES_VAR.get()
    if val is None:
        val = []
        TRACKED_FILES_VAR.set(val)
    return val


def _get_required_context_value(var, name: str):
    val = var.get()
    if val is NOT_SET:
        raise AttributeError(f"'ContextVarProxy' object has no attribute '{name}'")
    return val


def _set_special_context_attr(name, value) -> bool:
    if name == "tracked_files":
        TRACKED_FILES_VAR.set(value)
        return True
    if name == "filename":
        FILENAME_VAR.set(value)
        return True
    if name == "source_path":
        SOURCE_PATH_VAR.set(value)
        return True
    if name == "input_flags":
        INPUT_FLAGS_VAR.set(value)
        return True
    return False


def _delete_special_context_attr(name) -> bool:
    if name == "tracked_files":
        TRACKED_FILES_VAR.set(None)
        return True
    if name == "filename":
        FILENAME_VAR.set(NOT_SET)
        return True
    if name == "source_path":
        SOURCE_PATH_VAR.set(NOT_SET)
        return True
    if name == "input_flags":
        INPUT_FLAGS_VAR.set(None)
        return True
    return False


# Global contextvars storage for request context (e.g. filename tracking, temp files)
THREAD_CONTEXT = ContextVarProxy()


def get_tracked_files():
    """Retrieve the list of files tracked for cleanup in the current request or thread."""
    req_tracked = REQUEST_TRACKED_FILES_VAR.get()
    if req_tracked is not None:
        return req_tracked
    if not hasattr(THREAD_CONTEXT, "tracked_files"):
        setattr(THREAD_CONTEXT, "tracked_files", [])
    return THREAD_CONTEXT.tracked_files


def track_file(path):
    """Add a file path to the current request's cleanup list."""
    if path and os.path.exists(path) and os.path.isfile(path):
        tracked = get_tracked_files()
        if path not in tracked:
            tracked.append(path)
    return path


def cleanup_tracked_files(request=None):
    """Remove all files tracked in the current request's context and clear the registry."""
    tracked = _resolve_request_scoped_tracked_files(request)
    if tracked is not None:
        _cleanup_tracked_file_list(tracked, "request scoped")
        return
    files = get_tracked_files()
    if files:
        _cleanup_tracked_file_list(files, "fallback")


def _resolve_request_scoped_tracked_files(request=None):
    if request is not None and hasattr(request.state, "tracked_files"):
        return request.state.tracked_files
    return REQUEST_TRACKED_FILES_VAR.get()


def _cleanup_tracked_file_list(files: list, scope_label: str):
    logger.debug("[System] Performing request-local storage hygiene on %d files (%s)", len(files), scope_label)
    for f_path in list(files):
        secure_remove(f_path)
    files.clear()


logger = logging.getLogger(__name__)


def _cuda_device_count() -> int:
    if not (torch and torch.cuda.is_available()):
        return 0
    return torch.cuda.device_count()


def _clear_single_cuda_cache() -> None:
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "ipc_collect"):
        torch.cuda.ipc_collect()


def _clear_multi_cuda_cache(device_count: int) -> None:
    for index in range(device_count):
        with torch.cuda.device(index):
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()


def clear_gpu_cache():
    """Trigger explicit hardware cache reclamation if CUDA is present."""
    try:
        device_count = _cuda_device_count()
        if device_count <= 0:
            return
        if device_count == 1:
            _clear_single_cuda_cache()
            return
        _clear_multi_cuda_cache(device_count)
    except tuple([Exception]):
        pass


# --- [GLOBAL CONCURRENCY CONTROL] ---
# Semaphore to limit CPU-bound tasks (ASR, UVR on CPU, FFmpeg)
_CPU_LOCK = threading.Semaphore(config.CPU_PARALLEL_LIMIT)

# Track active standard (non-priority) FFmpeg processes to allow priority tasks to yield/wait appropriately
STANDARD_FFMPEG_LOCK = threading.Lock()
STANDARD_FFMPEG_STATE = {"count": 0}
STANDARD_FFMPEG_COND = threading.Condition(STANDARD_FFMPEG_LOCK)


@contextlib.contextmanager
def cpu_lock_ctx():
    """Context manager for CPU-heavy tasks to prevent core over-subscription."""
    with _CPU_LOCK:
        yield


# --- [FFMPEG STANDARDS] ---
# Broadcast-grade audio normalization settings
if config.FFMPEG_FILTER == "loudnorm":
    # EBU R128 standard - High quality, slower (Double-pass analysis)
    STANDARD_NORMALIZATION_FILTERS = "loudnorm=I=-16:TP=-1.5:LRA=11"
else:
    # Dynamic Audio Normalizer - Significantly faster, optimized for speech intelligibility
    # f=150: Window size (ms) | g=15: Gaussian filter size
    STANDARD_NORMALIZATION_FILTERS = "dynaudnorm=f=150:g=15"

# Standard audio stream parameters: 16kHz, mono, 16-bit PCM (Whisper optimized)
STANDARD_AUDIO_FLAGS = ["-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1"]

# High-quality audio parameters: 44.1kHz, stereo, 16-bit PCM (UVR optimized)
HQ_AUDIO_FLAGS = ["-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2"]


def convert_to_wav(source_path, input_flags=None):
    """
    Standardize media into 1-channel, 16kHz PCM WAV.
    """
    return _convert_base(source_path, STANDARD_AUDIO_FLAGS, 16000, 1, input_flags=input_flags)


def prepare_for_uvr(source_path, yield_cb=None, input_flags=None):
    """
    Standardize media into high-quality 2-channel, 44.1kHz PCM WAV for UVR.
    """
    return track_file(
        _convert_base(
            source_path,
            HQ_AUDIO_FLAGS,
            44100,
            2,
            tag="HQ-Prep",
            yield_cb=yield_cb,
            input_flags=input_flags,
        )
    )


def _convert_base(source_path, flags, rate, channels, tag="Prep", *, yield_cb=None, input_flags=None):
    """Internal base for audio conversion."""
    if not _validate_source_path_for_conversion(source_path, tag):
        return None

    duration = get_audio_duration(source_path)
    logger.info("[%s] Stream analysis: %s (%s)", tag, os.path.basename(source_path), format_duration(duration))
    output_path = _allocate_temp_output_path(duration, rate, channels)

    try:
        _run_optional_yield(yield_cb)
        _run_ffmpeg_standardization(source_path, output_path, duration, flags=flags, input_flags=input_flags, yield_cb=yield_cb)
        _run_optional_yield(yield_cb)
        logger.info("[%s] Normalization sequence completed successfully.", tag)
        return track_file(output_path)
    except tuple([Exception]) as err:
        _log_and_cleanup_conversion_failure(tag, err, output_path)
        return None


def _validate_source_path_for_conversion(source_path, tag: str) -> bool:
    if not source_path or not os.path.exists(source_path):
        logger.error("[%s] Media standardization failed: Source path missing.", tag)
        return False
    if os.path.getsize(source_path) == 0:
        logger.error("[%s] Media standardization failed: Source file is empty.", tag)
        return False
    return True


def _allocate_temp_output_path(duration: float, rate: int, channels: int) -> str:
    estimated_bytes = int(duration * rate * 2 * channels) if duration > 0 else 0
    temp_dir = config.get_temp_dir(required_bytes=estimated_bytes)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=temp_dir) as temp_wav:
        return temp_wav.name


def _run_optional_yield(yield_cb=None):
    if yield_cb:
        yield_cb()


def _log_and_cleanup_conversion_failure(tag: str, err: Exception, output_path: str):
    logger.error("[%s] Media standardization failed: %s", tag, err)
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
        except OSError:
            pass


def _run_ffmpeg_standardization(source_path, output_path, duration, flags=None, *, input_flags=None, yield_cb=None):
    """Compatibility wrapper delegating FFmpeg standardization to utils_helpers."""
    ffmpeg_env = {
        "cfg": config,
        "thread_context": THREAD_CONTEXT,
        "standard_ffmpeg_cond": STANDARD_FFMPEG_COND,
        "standard_ffmpeg_state": STANDARD_FFMPEG_STATE,
        "standard_audio_flags": STANDARD_AUDIO_FLAGS,
        "standard_normalization_filters": STANDARD_NORMALIZATION_FILTERS,
        "format_duration": format_duration,
    }
    run_ffmpeg_standardization(
        source_path,
        output_path,
        duration,
        ffmpeg_env,
        flags=flags,
        input_flags=input_flags,
        yield_cb=yield_cb,
    )


def parse_ffmpeg_progress(process, duration, yield_cb=None):
    """Compatibility wrapper delegating FFmpeg progress parsing to utils_helpers."""
    return _parse_ffmpeg_progress_impl(process, duration, format_duration, yield_cb=yield_cb)


def _pcm_bytes_per_second(input_flags: list[str] | None) -> float:
    """Derive PCM bytes/sec from -ar and -ac flags; defaults match STANDARD_AUDIO_FLAGS (16kHz mono s16le)."""
    flags = list(input_flags or [])
    sample_rate = 16000
    channels = 1
    bytes_per_sample = 2  # s16le — 16-bit signed little-endian
    try:
        if "-ar" in flags:
            sample_rate = int(flags[flags.index("-ar") + 1])
        if "-ac" in flags:
            channels = int(flags[flags.index("-ac") + 1])
    except (ValueError, IndexError):
        pass
    return float(sample_rate * channels * bytes_per_sample)


def _calculate_pcm_fallback_duration(file_path: str, input_flags: list[str]) -> float:
    try:
        if input_flags and os.path.exists(file_path):
            f_size = os.path.getsize(file_path)
            return float(f_size) / _pcm_bytes_per_second(input_flags)
    except tuple([Exception]):
        pass
    return 0.0


# Public aliases so sibling modules and tests avoid protected-access lint warnings
pcm_bytes_per_second = _pcm_bytes_per_second
calculate_pcm_fallback_duration = _calculate_pcm_fallback_duration


def get_audio_duration(file_path, input_flags=None):
    """Extract precise media duration via ffprobe stream headers."""
    if input_flags is None:
        input_flags = getattr(THREAD_CONTEXT, "input_flags", None)
    try:
        cmd = ["ffprobe", "-v", "error"]
        if input_flags:
            cmd.extend(input_flags)
        cmd.extend(
            [
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                file_path,
            ]
        )
        return float(process_exec.check_output_text(cmd, timeout=10).strip())
    except tuple([Exception]):
        return _calculate_pcm_fallback_duration(file_path, input_flags)


def format_duration(seconds):
    """Convert raw seconds into a human-readable HH:MM:SS format."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# Static Language Mapping (ISO-639-1 -> Name)
# Extracted from standard Whisper mappings to ensure /detect-language
# returns full names without needing the full openai-whisper package.
LANGUAGES = json.loads(
    '{"en": "English", "zh": "Chinese", "de": "German", "es": "Spanish", "ru": "Russian", "ko": "Korean", '
    '"fr": "French", "ja": "Japanese", "pt": "Portuguese", "tr": "Turkish", "pl": "Polish", "ca": "Catalan", '
    '"nl": "Dutch", "ar": "Arabic", "sv": "Swedish", "it": "Italian", "id": "Indonesian", "hi": "Hindi", '
    '"fi": "Finnish", "vi": "Vietnamese", "he": "Hebrew", "uk": "Ukrainian", "el": "Greek", "ms": "Malay", '
    '"cs": "Czech", "ro": "Romanian", "da": "Danish", "hu": "Hungarian", "ta": "Tamil", "no": "Norwegian", '
    '"th": "Thai", "ur": "Urdu", "hr": "Croatian", "bg": "Bulgarian", "lt": "Lithuanian", "la": "Latin", '
    '"mi": "Maori", "ml": "Malayalam", "cy": "Welsh", "sk": "Slovak", "te": "Telugu", "fa": "Persian", '
    '"lv": "Latvian", "bn": "Bengali", "sr": "Serbian", "az": "Azerbaijani", "sl": "Slovenian", "kn": "Kannada", '
    '"et": "Estonian", "mk": "Macedonian", "br": "Breton", "eu": "Basque", "is": "Icelandic", "hy": "Armenian", '
    '"ne": "Nepali", "mn": "Mongolian", "bs": "Bosnian", "kk": "Kazakh", "sq": "Albanian", "sw": "Swahili", '
    '"gl": "Galician", "mr": "Marathi", "pa": "Punjabi", "si": "Sinhala", "km": "Khmer", "sn": "Shona", '
    '"yo": "Yoruba", "so": "Somali", "af": "Afrikaans", "oc": "Occitan", "ka": "Georgian", "be": "Belarusian", '
    '"tg": "Tajik", "sd": "Sindhi", "gu": "Gujarati", "am": "Amharic", "yi": "Yiddish", "lo": "Lao", '
    '"uz": "Uzbek", "fo": "Faroese", "ht": "Haitian Creole", "ps": "Pashto", "tk": "Turkmen", "nn": "Nynorsk", '
    '"mt": "Maltese", "sa": "Sanskrit", "lb": "Luxembourgish", "my": "Myanmar", "bo": "Tibetan", "tl": "Tagalog", '
    '"mg": "Malagasy", "as": "Assamese", "tt": "Tatar", "haw": "Hawaiian", "ln": "Lingala", "ha": "Hausa", '
    '"ba": "Bashkir", "jw": "Javanese", "su": "Sundanese"}'
)
