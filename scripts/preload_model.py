import argparse
import hashlib
import logging
import os
import shutil
import subprocess
import sys

import torch
from audio_separator.separator import Separator
from faster_whisper import download_model

# Set up logging to stdout
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

# Configuration
WHISPER_ID = "Systran/faster-whisper-large-v3"
OV_SOURCE_ID = "openai/whisper-large-v3"
UVR_MODEL = "UVR-MDX-NET-Inst_HQ_3.onnx"
UVR_MODEL_SHA256 = "317554b07fe1ea5279a77f2b1520a41ea4b93432560c4ffd08792c30fddf9adc"
SILERO_VAD_MODEL_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"

# The official OpenVINO pre-converted model for GenAI 2025.4
OV_MODEL_ID = "OpenVINO/whisper-large-v3-fp16-ov"

SYSTEM_DIR = "/app/system_models"
WHISPER_DIR = os.path.join(SYSTEM_DIR, "whisper")
OV_WHISPER_DIR = os.path.join(SYSTEM_DIR, "whisper-openvino")
UVR_DIR = os.path.join(SYSTEM_DIR, "uvr")
VAD_DIR = os.path.join(SYSTEM_DIR, "vad")

CACHE_DIR = None
SKIP_INTEL_WHISPER = False


def _cache_path(cache_name):
    return os.path.join(CACHE_DIR, cache_name) if CACHE_DIR else None


def _replace_directory(source_dir, target_dir):
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)


def _cache_directory(source_dir, cache_name):
    if not CACHE_DIR:
        return
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_dir = _cache_path(cache_name)
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    shutil.copytree(source_dir, cache_dir)


def _restore_directory_from_cache(cache_name, target_dir, description, validator=None):
    if not CACHE_DIR:
        return False
    cache_dir = _cache_path(cache_name)
    if not os.path.exists(cache_dir):
        return False
    if validator and not validator(cache_dir):
        return False

    logger.info("Restoring %s from cache...", description)
    _replace_directory(cache_dir, target_dir)
    return True


def _run_subprocess_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for line in process.stdout:
            logger.info(line.rstrip("\n"))
    finally:
        process.wait()
    return process.returncode == 0


def _download_openvino_source():
    logger.info("Downloading official OpenAI Whisper weights for Intel conversion: %s", OV_SOURCE_ID)
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=OV_SOURCE_ID,
            local_dir=OV_WHISPER_DIR,
            local_dir_use_symlinks=False,
            max_workers=4,
        )
        logger.info("Whisper (OpenVINO) source weights ready in %s", OV_WHISPER_DIR)
        _cache_directory(OV_WHISPER_DIR, "whisper-openvino")
    except Exception as exc:
        logger.error("Failed to download OpenVINO Whisper model: %s", exc)


def _ov_has_required_files(directory):
    files = set(os.listdir(directory))
    critical_patterns = {
        "openvino_encoder_model.xml",
        "openvino_encoder_model.bin",
        "openvino_decoder_model.xml",
        "openvino_decoder_model.bin",
    }
    missing = sorted(critical_patterns.difference(files))
    if missing:
        logger.warning("Model directory %s is missing critical files: %s", directory, missing)
        return False
    return True


def _ov_bins_are_large_enough(directory):
    for filename in ("openvino_encoder_model.bin", "openvino_decoder_model.bin"):
        file_path = os.path.join(directory, filename)
        if not os.path.exists(file_path):
            return False
        size_bytes = os.path.getsize(file_path)
        if size_bytes < 50 * 1024 * 1024:
            logger.error("File %s is too small (%d bytes). Corrupted or empty.", filename, size_bytes)
            return False
    return True


def _download_ct2_whisper():
    logger.info("Downloading Whisper Model (CT2): %s to %s...", WHISPER_ID, WHISPER_DIR)
    try:
        download_model(WHISPER_ID, output_dir=WHISPER_DIR)
        logger.info("Whisper (CT2) downloaded successfully.")
        _cache_directory(WHISPER_DIR, "whisper")
        return True
    except Exception as exc:
        logger.error("Failed to download Whisper model: %s", exc)
        return False


def _ensure_ct2_whisper():
    if os.path.exists(os.path.join(WHISPER_DIR, "model.bin")):
        logger.info("Faster-Whisper model already exists in %s. Skipping.", WHISPER_DIR)
        return

    if _restore_directory_from_cache("whisper", WHISPER_DIR, "Whisper (CT2)"):
        return

    if not _download_ct2_whisper():
        sys.exit(1)


def _export_openvino_whisper():
    if not shutil.which("optimum-cli"):
        logger.info("optimum-cli not found. Skipping build-time conversion.")
        return False

    logger.info("Exporting Whisper Model to OpenVINO using optimum-cli...")
    try:
        cmd = [
            "optimum-cli",
            "export",
            "openvino",
            "--model",
            "openai/whisper-large-v3",
            "--task",
            "automatic-speech-recognition",
            "--weight-format",
            "fp16",
            OV_WHISPER_DIR,
        ]
        logger.info("Running: %s", " ".join(cmd))
        if _run_subprocess_command(cmd) and verify_ov_model(OV_WHISPER_DIR):
            logger.info("Whisper (OpenVINO) exported successfully.")
            _cache_directory(OV_WHISPER_DIR, "whisper-openvino")
            return True
        logger.warning("Optimum export failed or produced invalid model files.")
    except Exception as exc:
        logger.warning("Exception during optimum export: %s", exc)
    return False


def _download_uvr_direct(target_file):
    import tempfile

    import requests

    direct_url = f"https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/{UVR_MODEL}"
    logger.info("Attempting direct UVR model download from %s...", direct_url)
    with requests.get(direct_url, stream=True, timeout=120) as resp:
        resp.raise_for_status()

        target_dir = os.path.dirname(os.path.abspath(target_file))
        os.makedirs(target_dir, exist_ok=True)
        temp_fd, temp_path = tempfile.mkstemp(dir=target_dir, prefix="uvr_dl_")
        sha256 = hashlib.sha256()
        try:
            with os.fdopen(temp_fd, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    if not chunk:
                        continue
                    f.write(chunk)
                    sha256.update(chunk)
            digest = sha256.hexdigest()
            if digest != UVR_MODEL_SHA256:
                raise RuntimeError(f"UVR model checksum mismatch: expected {UVR_MODEL_SHA256}, got {digest}")
            os.replace(temp_path, target_file)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


def _download_silero_vad_direct(target_file):
    import tempfile

    import requests

    vad_url = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
    logger.info("Attempting direct Silero VAD model download from %s...", vad_url)
    with requests.get(vad_url, stream=True, timeout=30) as resp:
        resp.raise_for_status()

        target_dir = os.path.dirname(os.path.abspath(target_file))
        os.makedirs(target_dir, exist_ok=True)

        temp_fd, temp_path = tempfile.mkstemp(dir=target_dir, prefix="vad_dl_")
        sha256 = hashlib.sha256()
        try:
            with os.fdopen(temp_fd, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    f.write(chunk)
                    sha256.update(chunk)

            digest = sha256.hexdigest()
            if digest != SILERO_VAD_MODEL_SHA256:
                raise RuntimeError(f"Silero VAD checksum mismatch: expected {SILERO_VAD_MODEL_SHA256}, got {digest}")
            os.replace(temp_path, target_file)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


def _ensure_uvr_model():
    target_file = os.path.join(UVR_DIR, UVR_MODEL)
    if os.path.exists(target_file):
        logger.info("UVR model already exists in %s. Skipping.", UVR_DIR)
        return

    if _restore_directory_from_cache("uvr", UVR_DIR, "UVR Model"):
        return

    os.makedirs(UVR_DIR, exist_ok=True)
    logger.info("Downloading UVR Model: %s to %s...", UVR_MODEL, UVR_DIR)
    try:
        sep = Separator(model_file_dir=UVR_DIR, output_dir="/tmp")
        sep.load_model(UVR_MODEL)
        logger.info("UVR Model downloaded successfully.")
        _cache_directory(UVR_DIR, "uvr")
    except Exception as exc:
        logger.warning("Separator load_model failed (%s), attempting direct download...", exc)
        try:
            _download_uvr_direct(target_file)
            logger.info("Direct UVR Model download succeeded.")
            _cache_directory(UVR_DIR, "uvr")
        except Exception as direct_exc:
            logger.error("Failed to download UVR model: %s", direct_exc)
            sys.exit(1)


def _ensure_vad_model():
    target_file = os.path.join(VAD_DIR, "silero_vad.onnx")
    if os.path.exists(target_file):
        logger.info("VAD model already exists in %s. Skipping.", VAD_DIR)
        return

    if _restore_directory_from_cache("vad", VAD_DIR, "VAD Model"):
        return

    logger.info("Downloading Silero VAD ONNX model to %s (sha256-verified)...", VAD_DIR)
    try:
        _download_silero_vad_direct(target_file)
        logger.info("Silero VAD ONNX downloaded successfully: %s", target_file)
        _cache_directory(VAD_DIR, "vad")
    except Exception as exc:
        logger.warning("Silero VAD ONNX download skipped (%s). Runtime will fallback or download when needed.", exc)


def verify_ov_model(directory):
    """Verify that the directory contains a valid OpenVINO GenAI Whisper model."""
    if not os.path.exists(directory):
        return False

    if not _ov_has_required_files(directory):
        return False

    return _ov_bins_are_large_enough(directory)


def preload_whisper():
    # 1. CTranslate2 (Faster-Whisper)
    logger.info("--- [1/4] Preparing Faster-Whisper Model ---")
    _ensure_ct2_whisper()

    # 2. OpenVINO (Intel-Whisper)
    logger.info("--- [2/4] Preparing OpenVINO Whisper Model ---")

    if SKIP_INTEL_WHISPER:
        logger.info("Intel Whisper preloading is disabled via flag. Skipping.")
        return

    if verify_ov_model(OV_WHISPER_DIR):
        logger.info("OpenVINO Whisper model already exists and is valid. Skipping.")
        return

    if _restore_directory_from_cache("whisper-openvino", OV_WHISPER_DIR, "Whisper (OpenVINO)", validator=verify_ov_model):
        return

    if _export_openvino_whisper():
        return

    _download_openvino_source()


def preload_uvr():
    logger.info("--- [3/4] Preparing UVR Model ---")
    _ensure_uvr_model()


def preload_vad():
    logger.info("--- [4/4] Preparing VAD Model (C++ ONNX) ---")
    _ensure_vad_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=str, help="Persistent cache directory between builds")
    parser.add_argument("--skip-intel-whisper", action="store_true", help="Skip preloading Intel Whisper models")
    args = parser.parse_args()

    # Automatically use /root/.cache/model_downloads if no cache-dir is provided but we are in Docker
    CACHE_DIR = args.cache_dir
    if not CACHE_DIR and os.path.exists("/root/.cache"):
        CACHE_DIR = "/root/.cache/model_downloads"

    SKIP_INTEL_WHISPER = args.skip_intel_whisper

    preload_whisper()
    preload_uvr()
    preload_vad()
