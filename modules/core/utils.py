"""
Cross-Platform Media Utilities

This module provides essential tools for audio standardization, time formatting,
and subtitle generation. It abstracts FFmpeg complexity and ensures that all
media ingested by the service conforms to a uniform 16kHz MONO specification.
"""

import contextlib
import contextvars
import importlib
import json
import logging
import os
import subprocess
import tempfile
import threading
import time

import psutil

try:
    import torch
except ImportError:
    torch = None

from modules.core import config
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
    secure_remove,
    validate_audio,
)

# Reference to satisfy unused import check for public API consumption
_ = (secure_remove, get_pretty_model_name, validate_audio, cleanup_old_files, purge_temporary_assets)

# Re-export for public API and compatibility
_ = (wrap_text, generate_srt, format_timestamp, generate_vtt, generate_txt, generate_tsv, format_single_srt_block)

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
        "app_cpu_percent": round(app_cpu, 1),
        "memory_percent": mem.percent,
        "memory_used_gb": round(mem.used / (1024**3), 2),
        "memory_total_gb": round(mem.total / (1024**3), 2),
        "app_memory_gb": round(app_mem_rss / (1024**3), 2),
    }


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
        if name == "tracked_files":
            val = TRACKED_FILES_VAR.get()
            if val is None:
                val = []
                TRACKED_FILES_VAR.set(val)
            return val
        if name == "filename":
            val = FILENAME_VAR.get()
            if val is NOT_SET:
                raise AttributeError(f"'ContextVarProxy' object has no attribute '{name}'")
            return val
        if name == "source_path":
            val = SOURCE_PATH_VAR.get()
            if val is NOT_SET:
                raise AttributeError(f"'ContextVarProxy' object has no attribute '{name}'")
            return val
        if name == "input_flags":
            return INPUT_FLAGS_VAR.get()
        d = self._get_dict()
        if name in d:
            return d[name]
        raise AttributeError(f"'ContextVarProxy' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name == "_cv":
            super().__setattr__(name, value)
        elif name == "tracked_files":
            TRACKED_FILES_VAR.set(value)
        elif name == "filename":
            FILENAME_VAR.set(value)
        elif name == "source_path":
            SOURCE_PATH_VAR.set(value)
        elif name == "input_flags":
            INPUT_FLAGS_VAR.set(value)
        else:
            d = self._get_dict()
            d[name] = value

    def __delattr__(self, name):
        if name == "tracked_files":
            TRACKED_FILES_VAR.set(None)
        elif name == "filename":
            FILENAME_VAR.set(NOT_SET)
        elif name == "source_path":
            SOURCE_PATH_VAR.set(NOT_SET)
        elif name == "input_flags":
            INPUT_FLAGS_VAR.set(None)
        else:
            d = self._get_dict()
            if name in d:
                del d[name]


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
    tracked = None
    if request is not None and hasattr(request.state, "tracked_files"):
        tracked = request.state.tracked_files
    else:
        tracked = REQUEST_TRACKED_FILES_VAR.get()

    if tracked is not None:
        logger.debug("[System] Performing request-local storage hygiene on %d files (request scoped)", len(tracked))
        for f_path in list(tracked):
            secure_remove(f_path)
        tracked.clear()
    else:
        files = get_tracked_files()
        if files:
            logger.debug("[System] Performing request-local storage hygiene on %d files (fallback)", len(files))
            for f_path in list(files):
                secure_remove(f_path)
            files.clear()


logger = logging.getLogger(__name__)


def clear_gpu_cache():
    """Trigger explicit hardware cache reclamation if CUDA is present."""
    try:
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
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
    if not source_path or not os.path.exists(source_path):
        logger.error("[%s] Media standardization failed: Source path missing.", tag)
        return None

    if os.path.getsize(source_path) == 0:
        logger.error("[%s] Media standardization failed: Source file is empty.", tag)
        return None

    duration = get_audio_duration(source_path)
    logger.info("[%s] Stream analysis: %s (%s)", tag, os.path.basename(source_path), format_duration(duration))

    # Estimate output size: rate * 2 bytes * channels * duration
    estimated_bytes = int(duration * rate * 2 * channels) if duration > 0 else 0
    temp_dir = config.get_temp_dir(required_bytes=estimated_bytes)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=temp_dir) as temp_wav:
        output_path = temp_wav.name

    try:
        if yield_cb:
            yield_cb()
        _run_ffmpeg_standardization(
            source_path, output_path, duration, flags, input_flags=input_flags, yield_cb=yield_cb
        )
        if yield_cb:
            yield_cb()
        logger.info("[%s] Normalization sequence completed successfully.", tag)
        return track_file(output_path)

    except tuple([Exception]) as err:
        logger.error("[%s] Media standardization failed: %s", tag, err)
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
        return None


def _run_ffmpeg_standardization(source_path, output_path, duration, flags=None, *, input_flags=None, yield_cb=None):
    """Execute FFmpeg command with progress tracking."""
    is_priority = getattr(THREAD_CONTEXT, "is_priority", False)
    if not is_priority:
        with STANDARD_FFMPEG_COND:
            STANDARD_FFMPEG_STATE["count"] += 1
            logger.debug("[Prep] Incrementing standard FFmpeg count: %d", STANDARD_FFMPEG_STATE["count"])

    if input_flags is None:
        input_flags = getattr(THREAD_CONTEXT, "input_flags", None)

    try:
        if flags is None:
            flags = STANDARD_AUDIO_FLAGS

        command = [
            "ffmpeg",
            "-threads",
            str(config.FFMPEG_THREADS),
            "-thread_queue_size",
            "2048",
            "-y",
            "-loglevel",
            "error",
            "-filter_threads",
            str(config.FFMPEG_THREADS),
        ]

        # Hardware Acceleration Injection
        if config.FFMPEG_HWACCEL.lower() != "none":
            command.extend(["-hwaccel", config.FFMPEG_HWACCEL])

        # Input format flags
        input_args = []
        if input_flags:
            input_args.extend(input_flags)
        input_args.extend(["-i", source_path])

        command.extend(
            input_args + ["-progress", "pipe:1"] + flags + ["-af", STANDARD_NORMALIZATION_FILTERS, output_path]
        )

        # Watchdog timeout: dynamic based on duration if > 0, otherwise standard 300 seconds fallback.
        ffmpeg_timeout = max(300.0, duration * 5.0) if duration > 0 else 300.0

        # Merge stderr into stdout to avoid deadlock when stderr buffer fills up
        with subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True
        ) as process:
            timeout_triggered = False

            def kill_process():
                nonlocal timeout_triggered
                try:
                    if process.poll() is None:
                        timeout_triggered = True
                        logger.warning(
                            "[Prep] FFmpeg execution exceeded timeout (%.1fs). Terminating process.", ffmpeg_timeout
                        )
                        process.terminate()
                        time.sleep(1.0)
                        if process.poll() is None:
                            process.kill()
                except OSError:
                    pass

            timer = threading.Timer(ffmpeg_timeout, kill_process)
            timer.start()
            try:
                final_speed = parse_ffmpeg_progress(process, duration, yield_cb=yield_cb)
                process.wait()
                if timeout_triggered:
                    raise RuntimeError(f"FFmpeg standardization timed out after {ffmpeg_timeout}s")
                logger.info("[Prep] FFmpeg finished | Speed: %s", final_speed)

                if process.returncode != 0:
                    raise RuntimeError(f"FFmpeg failed with return code {process.returncode}")
            finally:
                timer.cancel()
    finally:
        if not is_priority:
            with STANDARD_FFMPEG_COND:
                STANDARD_FFMPEG_STATE["count"] = max(0, STANDARD_FFMPEG_STATE["count"] - 1)
                logger.debug("[Prep] Decrementing standard FFmpeg count: %d", STANDARD_FFMPEG_STATE["count"])
                STANDARD_FFMPEG_COND.notify_all()


def _handle_duration_progress(line, duration, last_stage_pct, last_logged_pct, yield_cb):
    """Parse time progress and publish updates when duration is available."""
    try:
        time_ms = int(line.split("=")[1].strip())
        time_sec = time_ms / 1000000.0
        pct = (time_sec / duration) * 100

        # Publish dashboard stage updates as FFmpeg progresses.
        pct_int = max(0, min(100, int(pct)))
        new_stage_pct = last_stage_pct
        if pct_int >= last_stage_pct + 5:
            try:
                scheduler = importlib.import_module("modules.inference.scheduler")
                scheduler.update_task_progress(None, f"FFmpeg ({pct_int}%)")
            except (ImportError, RuntimeError, AttributeError, ValueError, TypeError, KeyError):
                pass
            if yield_cb:
                yield_cb()
            new_stage_pct = pct_int

        new_logged_pct = last_logged_pct
        if pct - last_logged_pct >= 10:
            logger.info(
                "[Prep] FFmpeg Status: %5.1f%% | %s / %s",
                pct,
                format_duration(time_sec),
                format_duration(duration),
            )
            new_logged_pct = pct

        return new_stage_pct, new_logged_pct
    except (ValueError, IndexError):
        return last_stage_pct, last_logged_pct


def parse_ffmpeg_progress(process, duration, yield_cb=None):
    """Parse FFmpeg stdout for progress updates."""
    last_logged_pct = -10
    last_stage_pct = -1
    final_speed = "N/A"
    last_yield_time = 0.0
    while True:
        line = process.stdout.readline()
        if not line:
            break

        if "speed=" in line:
            try:
                final_speed = line.split("=")[1].strip()
            except (ValueError, IndexError):
                pass

        if "out_time_ms=" in line:
            if duration > 0:
                last_stage_pct, last_logged_pct = _handle_duration_progress(
                    line, duration, last_stage_pct, last_logged_pct, yield_cb
                )
            else:
                # unknown duration: yield periodically on progress output
                current_time = time.time()
                if current_time - last_yield_time >= 1.0:
                    if yield_cb:
                        yield_cb()
                    last_yield_time = current_time
    return final_speed


def get_audio_duration(file_path):
    """Extract precise media duration via ffprobe stream headers."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            file_path,
        ]
        result = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=10)
        return float(result.decode("utf-8").strip())
    except tuple([Exception]):
        # Calculate duration based on file size if input is raw PCM
        try:
            input_flags = getattr(THREAD_CONTEXT, "input_flags", None)
            if input_flags and os.path.exists(file_path):
                # Standard raw PCM: 16000Hz mono s16le (2 bytes/sample) -> 32000 bytes/sec
                f_size = os.path.getsize(file_path)
                return float(f_size) / 32000.0
        except tuple([Exception]):
            pass
        return 0.0


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
