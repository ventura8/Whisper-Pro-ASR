import os
import sys
import types
import torch
import shutil
import logging
import json
import subprocess
from pathlib import Path

# Shim for legacy libraries that expect torchaudio.backend (removed in 2.1+)
# Must be applied BEFORE importing libraries that might depend on it (like demucs)
import torchaudio
if not hasattr(torchaudio, "backend"):
    # Create or get backend
    try:
        import torchaudio._backend as _backend
        backend_obj = _backend
    except ImportError:
        backend_obj = types.ModuleType("torchaudio.backend")

    # Ensure it's a package and in sys.modules
    if not hasattr(backend_obj, "__path__"):
        backend_obj.__path__ = []

    sys.modules["torchaudio.backend"] = backend_obj
    torchaudio.backend = backend_obj

    # Mock 'common' submodule which is often requested for AudioBackend types
    if "torchaudio.backend.common" not in sys.modules:
        mock_common = types.ModuleType("torchaudio.backend.common")
        # Add common placeholder if a library checks for it
        mock_common.AudioBackend = object
        # Add AudioMetaData as it's often used (e.g., in DeepFilterNet)
        mock_common.AudioMetaData = getattr(torchaudio, "AudioMetaData", object)
        sys.modules["torchaudio.backend.common"] = mock_common
        backend_obj.common = mock_common

import urllib.request

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("model_exporter")

MODELS_DIR = Path("/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def export_whisper(model_name):
    """Export Whisper model using optimum-cli."""
    logger.info(f"Exporting Whisper model: {model_name}...")
    out_dir = MODELS_DIR / "whisper-openvino"

    cmd = [
        "optimum-cli", "export", "openvino",
        "--model", model_name,
        "--task", "automatic-speech-recognition",
        "--weight-format", "int8",
        "--sym",
        str(out_dir)
    ]

    try:
        subprocess.run(cmd, check=True)
        logger.info("Whisper export successful.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Whisper export failed: {e}")
        sys.exit(1)


def warmup_uvr():
    """Download UVR/MDX-NET model and initialize separator to cache onnx files."""
    logger.info("Pre-caching UVR/MDX-NET model (Vocal Separation)...")
    from audio_separator.separator import Separator

    # We use a temp directory to trigger the download, which audio-separator caches in its own internal structure
    # However, we want to specify a path that matches our runtime config /models/preprocessing
    model_dir = Path("/models/preprocessing")
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        separator = Separator(
            model_file_dir=str(model_dir),
            output_dir="/tmp",
            output_format="WAV"
        )

        # This will download the .ckpt and export to .onnx if needed
        model_name = os.environ.get("VOCAL_SEPARATION_MODEL", "UVR-MDX-NET-Inst_HQ_3.onnx")
        logger.info(f"Downloading/Loading {model_name}...")
        separator.load_model(model_name)
        logger.info("UVR/MDX-NET pre-cache successful.")
    except Exception as e:
        logger.error(f"UVR/MDX-NET pre-cache failed: {e}")
        # Not exiting as the build can still proceed and download at runtime
        pass


if __name__ == "__main__":
    whisper_model = os.environ.get("WHISPER_MODEL_NAME", "openai/whisper-large-v3")

    # Run exports
    export_whisper(whisper_model)
    warmup_uvr()

    logger.info("All model exports and warmups complete.")
