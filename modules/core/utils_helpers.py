"""
Utility Helper Functions for Whisper Pro ASR
"""

import logging
import os
import shutil
import tempfile
import time

logger = logging.getLogger(__name__)


def secure_remove(file_path):
    """Safely remove a file if it exists, ignoring errors."""
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except tuple([Exception]):
            pass


def get_pretty_model_name(model_path):
    """Convert technical model identifiers to human-readable names."""
    if not model_path:
        return "Unknown Engine"

    name = str(model_path).rsplit("/", maxsplit=1)[-1]

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
        "whisper": "Whisper Engine",
    }

    for key, pretty in mappings.items():
        if key in name:
            return pretty

    # Cleanup path and return (handle both dashes and underscores)
    return name.replace(".onnx", "").replace("_", " ").replace("-", " ").title()


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
            except tuple([Exception]) as e:
                logger.warning("[System] Failed to prune %s: %s", name, e)


def purge_temporary_assets():
    """Purge orphaned transcription files from the temp directory."""
    temp_dir = os.getenv("WHISPER_TEMP_DIR")
    if not temp_dir:
        temp_dir = os.path.join(tempfile.gettempdir(), "whisper")

    if os.path.exists(temp_dir):
        try:
            for f in os.listdir(temp_dir):
                fpath = os.path.join(temp_dir, f)
                if os.path.isfile(fpath) and (
                    f.startswith(("tmp_", "upload_", "whisper_", "processed_"))
                    or f.endswith((".wav", ".mp3", ".tmp", ".json"))
                ):
                    os.remove(fpath)
                elif os.path.isdir(fpath) and (f.startswith("whisper_") or f in ["preprocessing"]):
                    shutil.rmtree(fpath)
            logger.info("[System] Purged temporary asset cache")
        except tuple([Exception]) as exc:
            logger.error("[System] Cleanup failed: %s", exc)
