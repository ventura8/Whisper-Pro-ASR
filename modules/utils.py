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

from . import config

logger = logging.getLogger(__name__)

# --- [FFMPEG STANDARDS] ---
# Broadcast-grade audio normalization settings
STANDARD_NORMALIZATION_FILTERS = "loudnorm=I=-16:TP=-1.5:LRA=11"

# Standard audio stream parameters: 16kHz, mono, 16-bit PCM
STANDARD_AUDIO_FLAGS = [
    "-vn",
    "-acodec", "pcm_s16le",
    "-ar", "16000",
    "-ac", "1"
]


def convert_to_wav(source_path):
    """
    Standardize media into 1-channel, 16kHz PCM WAV.
    """
    if not source_path or not os.path.exists(source_path):
        logger.error("[Prep] Media standardization failed: Source path missing.")
        return None

    if os.path.getsize(source_path) == 0:
        logger.error("[Prep] Media standardization failed: Source file is empty.")
        return None

    duration = get_audio_duration(source_path)
    logger.info("[Prep] Stream analysis: %s (%s)",
                os.path.basename(source_path), format_duration(duration))

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        output_path = temp_wav.name

    try:
        _run_ffmpeg_standardization(source_path, output_path, duration)
        logger.info("[Prep] Normalization sequence completed successfully.")
        return output_path

    except Exception as err: # pylint: disable=broad-exception-caught
        logger.error("[Prep] Media standardization failed: %s", err)
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except Exception: # pylint: disable=broad-exception-caught
                pass
        return None


def _run_ffmpeg_standardization(source_path, output_path, duration):
    """Execute FFmpeg command with progress tracking."""
    command = [
        "ffmpeg", "-threads", str(config.FFMPEG_THREADS),
        "-thread_queue_size", "2048",
        "-y",
        "-filter_threads", str(config.FFMPEG_THREADS),
        "-i", source_path,
        "-progress", "pipe:1"
    ] + STANDARD_AUDIO_FLAGS + ["-af", STANDARD_NORMALIZATION_FILTERS, output_path]

    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    ) as process:
        _parse_ffmpeg_progress(process, duration)
        _, stderr = process.communicate()

        if process.returncode != 0:
            err_lines = stderr.splitlines()
            err_snippet = "\n".join(err_lines[-10:]) if err_lines else "No error details"
            raise RuntimeError(f"FFmpeg failed (code {process.returncode}). Details: {err_snippet}")


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
    if "chunks" not in result:
        text = result.get("text", "").strip()
        if text:
            return f"1\n00:00:00,000 --> 00:00:05,000\n{text}\n"
        return "[No dialogue detected]"

    srt_lines = []
    for idx, chunk in enumerate(result.get("chunks", []), start=1):
        try:
            # Extract timestamp tuple
            start_ts, end_ts = chunk.get("timestamp", (0.0, 5.0))
            if start_ts is None or end_ts is None:
                start_ts, end_ts = 0.0, 5.0
        except (ValueError, TypeError):
            start_ts, end_ts = 0.0, 5.0

        start_fmt = format_timestamp(start_ts)
        end_fmt = format_timestamp(end_ts)
        text = chunk.get("text", "").strip()

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

    chunks = result.get("chunks", [])
    if not chunks:
        # Fallback for text-only result
        text = result.get("text", "").strip()
        if text:
            return f"WEBVTT\n\n00:00:00.000 --> 00:00:05.000\n{text}\n"
        return "WEBVTT\n\n[No dialogue detected]"

    vtt_lines = ["WEBVTT", ""]
    for idx, chunk in enumerate(chunks, start=1):
        try:
            start_ts, end_ts = chunk.get("timestamp", (0.0, 5.0))
            if start_ts is None or end_ts is None:
                start_ts, end_ts = 0.0, 5.0
        except (ValueError, TypeError):
            start_ts, end_ts = 0.0, 5.0

        # VTT uses dot for milliseconds
        start_fmt = format_timestamp(start_ts).replace(',', '.')
        end_fmt = format_timestamp(end_ts).replace(',', '.')
        text = chunk.get("text", "").strip()

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
    for chunk in result.get("chunks", []):
        try:
            start_ts, end_ts = chunk.get("timestamp", (0.0, 0.0))
            if start_ts is None:
                start_ts = 0.0
            if end_ts is None:
                end_ts = 0.0

            start_ms = int(start_ts * 1000)
            end_ms = int(end_ts * 1000)
            text = chunk.get("text", "").strip().replace(
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
