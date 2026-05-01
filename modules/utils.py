"""
Cross-Platform Media Utilities

This module provides essential tools for audio standardization, time formatting,
and subtitle generation. It abstracts FFmpeg complexity and ensures that all
media ingested by the service conforms to a uniform 16kHz MONO specification.
"""
# pylint: disable=broad-exception-caught, cyclic-import
import os
import tempfile
import subprocess
import logging
import threading
import contextlib
import shutil
import time
import psutil
try:
    import torch
except ImportError:
    torch = None

from modules import config

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
        "app_memory_gb": round(app_mem_rss / (1024**3), 2)
    }


# Global thread-local storage for request context (e.g. filename tracking, temp files)
THREAD_CONTEXT = threading.local()


def get_tracked_files():
    """Retrieve the list of files tracked for cleanup in the current thread."""
    if not hasattr(THREAD_CONTEXT, 'tracked_files'):
        THREAD_CONTEXT.tracked_files = []
    return THREAD_CONTEXT.tracked_files


def track_file(path):
    """Add a file path to the current thread's cleanup list."""
    if path and os.path.exists(path) and os.path.isfile(path):
        tracked = get_tracked_files()
        if path not in tracked:
            tracked.append(path)
    return path


def cleanup_tracked_files():
    """Remove all files tracked in the current thread and clear the registry."""
    files = get_tracked_files()
    if not files:
        return

    logger.debug("[System] Performing request-local storage hygiene on %d files", len(files))
    for f_path in list(files):
        secure_remove(f_path)
    files.clear()


logger = logging.getLogger(__name__)


def clear_gpu_cache():
    """Trigger explicit hardware cache reclamation if CUDA is present."""
    try:
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# --- [GLOBAL CONCURRENCY CONTROL] ---
# Semaphore to limit CPU-bound tasks (ASR, UVR on CPU, FFmpeg)
_CPU_LOCK = threading.Semaphore(config.CPU_PARALLEL_LIMIT)


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
STANDARD_AUDIO_FLAGS = [
    "-vn",
    "-acodec", "pcm_s16le",
    "-ar", "16000",
    "-ac", "1"
]

# High-quality audio parameters: 44.1kHz, stereo, 16-bit PCM (UVR optimized)
HQ_AUDIO_FLAGS = [
    "-vn",
    "-acodec", "pcm_s16le",
    "-ar", "44100",
    "-ac", "2"
]


def convert_to_wav(source_path):
    """
    Standardize media into 1-channel, 16kHz PCM WAV.
    """
    return _convert_base(source_path, STANDARD_AUDIO_FLAGS, 16000, 1)


def prepare_for_uvr(source_path):
    """
    Standardize media into high-quality 2-channel, 44.1kHz PCM WAV for UVR.
    """
    # Optimization: Skip if already a WAV/FLAC to avoid double-handling
    if source_path and source_path.lower().endswith(('.wav', '.flac')):
        return source_path

    return track_file(_convert_base(source_path, HQ_AUDIO_FLAGS, 44100, 2, tag="HQ-Prep"))


def _convert_base(source_path, flags, rate, channels, tag="Prep"):
    """Internal base for audio conversion."""
    if not source_path or not os.path.exists(source_path):
        logger.error("[%s] Media standardization failed: Source path missing.", tag)
        return None

    if os.path.getsize(source_path) == 0:
        logger.error("[%s] Media standardization failed: Source file is empty.", tag)
        return None

    duration = get_audio_duration(source_path)
    logger.info("[%s] Stream analysis: %s (%s)",
                tag, os.path.basename(source_path), format_duration(duration))

    # Estimate output size: rate * 2 bytes * channels * duration
    estimated_bytes = int(duration * rate * 2 * channels) if duration > 0 else 0
    temp_dir = config.get_temp_dir(required_bytes=estimated_bytes)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False,
                                     dir=temp_dir) as temp_wav:
        output_path = temp_wav.name

    try:
        _run_ffmpeg_standardization(source_path, output_path, duration, flags)
        logger.info("[%s] Normalization sequence completed successfully.", tag)
        return track_file(output_path)

    except Exception as err:  # pylint: disable=broad-exception-caught
        logger.error("[%s] Media standardization failed: %s", tag, err)
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except Exception:  # pylint: disable=broad-exception-caught
                pass
        return None


def _run_ffmpeg_standardization(source_path, output_path, duration, flags=None):
    """Execute FFmpeg command with progress tracking."""
    if flags is None:
        flags = STANDARD_AUDIO_FLAGS

    command = [
        "ffmpeg",
        "-threads", str(config.FFMPEG_THREADS),
        "-thread_queue_size", "2048",
        "-y",
        "-loglevel", "error",
        "-filter_threads", str(config.FFMPEG_THREADS),
    ]

    # Hardware Acceleration Injection
    if config.FFMPEG_HWACCEL.lower() != "none":
        command.extend(["-hwaccel", config.FFMPEG_HWACCEL])

    command.extend([
        "-i", source_path,
        "-progress", "pipe:1"
    ] + flags + ["-af", STANDARD_NORMALIZATION_FILTERS, output_path])

    # Merge stderr into stdout to avoid deadlock when stderr buffer fills up
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True
    ) as process:
        _parse_ffmpeg_progress(process, duration)
        process.wait()

        if process.returncode != 0:
            raise RuntimeError(f"FFmpeg failed with return code {process.returncode}")


def _parse_ffmpeg_progress(process, duration):
    """Parse FFmpeg stdout for progress updates."""
    last_logged_pct = -10
    while True:
        line = process.stdout.readline()
        if not line:
            break

        if "out_time_ms=" in line and duration > 0:
            try:
                time_ms = int(line.split("=")[1].strip())
                time_sec = time_ms / 1000000.0
                pct = (time_sec / duration) * 100
                if pct - last_logged_pct >= 10:
                    logger.info("[Prep] FFmpeg Status: %5.1f%% | %s / %s",
                                pct, format_duration(time_sec), format_duration(duration))
                    last_logged_pct = pct
            except (ValueError, IndexError):
                pass


def get_audio_duration(file_path):
    """Extract precise media duration via ffprobe stream headers."""
    try:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", file_path
        ]
        result = subprocess.check_output(cmd, timeout=10)
        return float(result.decode("utf-8").strip())
    except Exception:
        return 0.0


def format_duration(seconds):
    """Convert raw seconds into a human-readable HH:MM:SS format."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def generate_srt(result):
    """
    Compose industrial-standard SubRip (SRT) content from segment metadata.

    Handles time formatting, sequence indexing, and empty signal fallbacks.
    """
    if not result:
        return "[No dialogue detected]"

    # Fallback for plain-text response format (legacy)
    if "segments" not in result:
        text = result.get("text", "").strip()
        if text:
            return f"1\n00:00:00,000 --> 00:00:05,000\n{text}\n"
        return "[No dialogue detected]"

    srt_lines = []
    for idx, segment in enumerate(result.get("segments", []), start=1):
        try:
            # Try 'timestamp' tuple first, then individual 'start'/'end' keys
            if "timestamp" in segment:
                start_ts, end_ts = segment["timestamp"]
            else:
                start_ts = segment.get("start", 0.0)
                end_ts = segment.get("end", 5.0)

            if start_ts is None:
                start_ts = 0.0
            if end_ts is None:
                end_ts = 0.0
        except (ValueError, TypeError, KeyError):
            start_ts, end_ts = 0.0, 5.0

        start_fmt = format_timestamp(start_ts)
        end_fmt = format_timestamp(end_ts)
        text = segment.get("text", "").strip()

        # Format SRT block: Index -> Timestamps -> Content
        srt_lines.append(f"{idx}\n{start_fmt} --> {end_fmt}\n{text}\n")

    if not srt_lines:
        return "[No dialogue detected]"

    return "\n".join(srt_lines)


def format_timestamp(seconds):
    """Generate the millisecond-precision timestamp required for SRT specifications."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_vtt(result):
    """
    Generate WebVTT content for web-native subtitles.
    """
    if not result:
        return "WEBVTT\n\n[No dialogue detected]"

    segments = result.get("segments", [])
    if not segments:
        # Fallback for text-only result
        text = result.get("text", "").strip()
        if text:
            return f"WEBVTT\n\n00:00:00.000 --> 00:00:05.000\n{text}\n"
        return "WEBVTT\n\n[No dialogue detected]"

    vtt_lines = ["WEBVTT", ""]
    for idx, segment in enumerate(segments, start=1):
        try:
            if "timestamp" in segment:
                start_ts, end_ts = segment["timestamp"]
            else:
                start_ts = segment.get("start", 0.0)
                end_ts = segment.get("end", 5.0)

            if start_ts is None:
                start_ts = 0.0
            if end_ts is None:
                end_ts = 0.0
        except (ValueError, TypeError, KeyError):
            start_ts, end_ts = 0.0, 5.0

        # VTT uses dot for milliseconds
        start_fmt = format_timestamp(start_ts).replace(',', '.')
        end_fmt = format_timestamp(end_ts).replace(',', '.')
        text = segment.get("text", "").strip()

        vtt_lines.append(f"{idx}\n{start_fmt} --> {end_fmt}\n{text}\n")

    return "\n".join(vtt_lines)


def generate_txt(result):
    """
    Generate plain text transcript.
    """
    if not result:
        return ""
    return result.get("text", "").strip()


def generate_tsv(result):
    """
    Generate Tab-Separated Values (TSV) format.
    """
    if not result:
        return "start\tend\ttext"

    tsv_lines = ["start\tend\ttext"]
    for segment in result.get("segments", []):
        try:
            if "timestamp" in segment:
                start_ts, end_ts = segment["timestamp"]
            else:
                start_ts = segment.get("start", 0.0)
                end_ts = segment.get("end", 0.0)

            if start_ts is None:
                start_ts = 0.0
            if end_ts is None:
                end_ts = 0.0

            start_ms = int(start_ts * 1000)
            end_ms = int(end_ts * 1000)
            text = segment.get("text", "").strip().replace(
                "\t", " ").replace("\n", " ")

            tsv_lines.append(f"{start_ms}\t{end_ms}\t{text}")
        except Exception:
            continue

    return "\n".join(tsv_lines)


# Static Language Mapping (ISO-639-1 -> Name)
# Extracted from standard Whisper mappings to ensure /detect-language
# returns full names without needing the full openai-whisper package.
LANGUAGES = {
    "en": "English", "zh": "Chinese", "de": "German", "es": "Spanish", "ru": "Russian",
    "ko": "Korean", "fr": "French", "ja": "Japanese", "pt": "Portuguese", "tr": "Turkish",
    "pl": "Polish", "ca": "Catalan", "nl": "Dutch", "ar": "Arabic", "sv": "Swedish",
    "it": "Italian", "id": "Indonesian", "hi": "Hindi", "fi": "Finnish", "vi": "Vietnamese",
    "he": "Hebrew", "uk": "Ukrainian", "el": "Greek", "ms": "Malay", "cs": "Czech",
    "ro": "Romanian", "da": "Danish", "hu": "Hungarian", "ta": "Tamil", "no": "Norwegian",
    "th": "Thai", "ur": "Urdu", "hr": "Croatian", "bg": "Bulgarian", "lt": "Lithuanian",
    "la": "Latin", "mi": "Maori", "ml": "Malayalam", "cy": "Welsh", "sk": "Slovak",
    "te": "Telugu", "fa": "Persian", "lv": "Latvian", "bn": "Bengali", "sr": "Serbian",
    "az": "Azerbaijani", "sl": "Slovenian", "kn": "Kannada", "et": "Estonian",
    "mk": "Macedonian", "br": "Breton", "eu": "Basque", "is": "Icelandic", "hy": "Armenian",
    "ne": "Nepali", "mn": "Mongolian", "bs": "Bosnian", "kk": "Kazakh", "sq": "Albanian",
    "sw": "Swahili", "gl": "Galician", "mr": "Marathi", "pa": "Punjabi", "si": "Sinhala",
    "km": "Khmer", "sn": "Shona", "yo": "Yoruba", "so": "Somali", "af": "Afrikaans",
    "oc": "Occitan", "ka": "Georgian", "be": "Belarusian", "tg": "Tajik", "sd": "Sindhi",
    "gu": "Gujarati", "am": "Amharic", "yi": "Yiddish", "lo": "Lao", "uz": "Uzbek",
    "fo": "Faroese", "ht": "Haitian Creole", "ps": "Pashto", "tk": "Turkmen", "nn": "Nynorsk",
    "mt": "Maltese", "sa": "Sanskrit", "lb": "Luxembourgish", "my": "Myanmar", "bo": "Tibetan",
    "tl": "Tagalog", "mg": "Malagasy", "as": "Assamese", "tt": "Tatar", "haw": "Hawaiian",
    "ln": "Lingala", "ha": "Hausa", "ba": "Bashkir", "jw": "Javanese", "su": "Sundanese"
}


def secure_remove(file_path):
    """Safely remove a file if it exists, ignoring errors."""
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception:
            pass


def get_pretty_model_name(model_path):
    """Convert technical model identifiers to human-readable names."""
    if not model_path:
        return "Unknown Engine"

    name = str(model_path).rsplit('/', maxsplit=1)[-1]

    # Common mappings (Sorted by specificity to avoid partial matches)
    mappings = {
        "faster-whisper-large-v3": "Whisper Large v3",
        "faster-whisper-medium": "Whisper Medium",
        "faster-whisper-small": "Whisper Small",
        "faster-whisper-base": "Whisper Base",
        "faster-whisper-tiny": "Whisper Tiny",
        "distil-large-v3": "Distil Large v3",
        "distil-medium": "Distil Medium",
        "distil-small": "Distil Small",
        "whisper-tiny": "Whisper Tiny",
        "whisper-base": "Whisper Base",
        "whisper-small": "Whisper Small",
        "whisper-medium": "Whisper Medium",
        "whisper-large": "Whisper Large",
        "UVR-MDX-NET-Inst_HQ_3.onnx": "Vocal Isolation HQ",
        "UVR-MDX-NET-Voc_FT.onnx": "Vocal Isolation FT",
        "silero_vad.onnx": "Silero VAD",
        "whisper": "Whisper Engine"
    }

    for key, pretty in mappings.items():
        if key in name:
            return pretty

    # Cleanup path and return (handle both dashes and underscores)
    return name.replace('.onnx', '').replace('_', ' ').replace('-', ' ').title()


def validate_audio(file_path):
    """Checks if the audio file is valid (exists and non-empty)."""
    if not file_path or not os.path.exists(file_path):
        return False
    if os.path.getsize(file_path) == 0:
        return False
    return True


def cleanup_old_files(directory, days=7):
    """Deletes files older than specified days in a directory."""
    if not os.path.exists(directory):
        return

    now = time.time()
    cutoff = now - (days * 86400)

    for root, _, files in os.walk(directory):
        for name in files:
            file_path = os.path.join(root, name)
            try:
                if os.path.getmtime(file_path) < cutoff:
                    os.remove(file_path)
                    logger.debug("[System] Pruned old file: %s", name)
            except Exception as e:
                logger.warning("[System] Failed to prune %s: %s", name, e)


def purge_temporary_assets():
    """Purge orphaned transcription files from the temp directory."""
    temp_dir = os.getenv("WHISPER_TEMP_DIR", tempfile.gettempdir())
    if os.path.exists(temp_dir):
        try:
            for f in os.listdir(temp_dir):
                fpath = os.path.join(temp_dir, f)
                if os.path.isfile(fpath) and f.endswith(('.wav', '.mp3', '.tmp', '.json')):
                    os.remove(fpath)
                elif os.path.isdir(fpath) and (f.startswith('tm') or f in ['preprocessing']):
                    shutil.rmtree(fpath)
            logger.info("[System] Purged temporary asset cache")
        except (OSError, IOError) as exc:
            logger.error("[System] Cleanup failed: %s", exc)
