import argparse
import hashlib
import logging
import os
import shutil
import subprocess
import sys

import torch
from modules.core import config, model_integrity, model_provisioning

# Set up logging to stdout
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

# Configuration
WHISPER_ID = model_provisioning.WHISPER_ID
OV_SOURCE_ID = "openai/whisper-large-v3"
UVR_MODEL = model_provisioning.UVR_MODEL
UVR_MODEL_SHA256 = model_provisioning.UVR_MODEL_SHA256
SILERO_VAD_MODEL_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"

# The official OpenVINO pre-converted model for GenAI 2025.4
OV_MODEL_ID = model_provisioning.OV_MODEL_ID

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


def _download_openvino_genai():
    """Download the official pre-converted OpenVINO IR (fp16).

    optimum-cli is not installed in the runtime image, so the source-weight
    conversion path never runs. This fetches the ready-made IR instead, which
    is both usable by the Intel engine and ~22GB smaller than the raw weights.
    """
    logger.info("Downloading pre-converted OpenVINO Whisper model: %s", OV_MODEL_ID)
    try:
        from huggingface_hub import snapshot_download

        # No local_dir_use_symlinks: huggingface_hub 1.x removed the parameter and takes
        # no **kwargs, so passing it raises TypeError before anything is fetched.
        snapshot_download(
            repo_id=OV_MODEL_ID,
            local_dir=OV_WHISPER_DIR,
            max_workers=4,
        )
        if not verify_ov_model(OV_WHISPER_DIR):
            logger.error("Downloaded OpenVINO model in %s failed validation.", OV_WHISPER_DIR)
            return False
        logger.info("Whisper (OpenVINO) IR ready in %s", OV_WHISPER_DIR)
        _cache_directory(OV_WHISPER_DIR, "whisper-openvino")
        return True
    except Exception as exc:
        logger.error("Failed to download pre-converted OpenVINO Whisper model: %s", exc)
        return False


# The raw-source fallback was removed deliberately. It downloaded ~22GB of unconverted
# OpenAI weights into the directory the Intel engine reads, logged "source weights ready",
# and returned success -- leaving a directory that ov_genai.WhisperPipeline cannot load.
# Conversion needs optimum-cli, which is not installed in the runtime image, so there was
# never a path from those weights to a usable IR. Failing here is the honest outcome:
# preload runs at build/provision time, where a clear error is cheap and a silently broken
# Intel engine is not.


def _download_ct2_whisper():
    success = model_provisioning.ensure_ct2_whisper(WHISPER_DIR, WHISPER_ID)
    if success:
        _cache_directory(WHISPER_DIR, "whisper")
    return success


def _ensure_ct2_whisper():
    if model_integrity.verify_ct2_model_dir(WHISPER_DIR):
        logger.info("Faster-Whisper model already exists and is valid in %s. Skipping.", WHISPER_DIR)
        return

    if _restore_directory_from_cache(
        "whisper",
        WHISPER_DIR,
        "Whisper (CT2)",
        validator=model_integrity.verify_ct2_model_dir,
    ):
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

    def _validate_uvr(p):
        return model_integrity.verify_onnx_model_file(p, min_bytes=10 * 1024 * 1024, expected_sha256=UVR_MODEL_SHA256)

    if _validate_uvr(target_file):
        logger.info("UVR model already exists and is valid in %s. Skipping.", UVR_DIR)
        return

    if _restore_directory_from_cache(
        "uvr",
        UVR_DIR,
        "UVR Model",
        validator=lambda d: _validate_uvr(os.path.join(d, UVR_MODEL)),
    ):
        return

    # The separator-then-direct-download fallback, the checksum validation and the bounded
    # retry all live in model_provisioning.ensure_uvr_model, which the runtime path already
    # uses. This module reimplemented the same three things, so a change to the download
    # policy had to be made twice and the preloader silently kept the older behaviour when
    # it was not. Only the preloader-specific step -- seeding the build cache -- stays here.
    if not model_provisioning.ensure_uvr_model(UVR_DIR, CACHE_DIR or config.PERSISTENT_TEMP_DIR):
        sys.exit(1)

    _cache_directory(UVR_DIR, "uvr")


def _ensure_vad_model():
    target_file = os.path.join(VAD_DIR, "silero_vad.onnx")

    def _validate_vad(p):
        return model_integrity.verify_onnx_model_file(p, min_bytes=500 * 1024, expected_sha256=SILERO_VAD_MODEL_SHA256)

    if _validate_vad(target_file):
        logger.info("VAD model already exists and is valid in %s. Skipping.", VAD_DIR)
        return

    if _restore_directory_from_cache(
        "vad",
        VAD_DIR,
        "VAD Model",
        validator=lambda d: _validate_vad(os.path.join(d, "silero_vad.onnx")),
    ):
        return

    def _do_vad_download():
        _download_silero_vad_direct(target_file)

    success = model_integrity.download_with_integrity_retry(
        download_fn=_do_vad_download,
        validator_fn=_validate_vad,
        target_path=target_file,
        max_retries=2,
        description="Silero VAD ONNX model",
    )
    if not success:
        sys.exit(1)
    _cache_directory(VAD_DIR, "vad")


def verify_ov_model(directory):
    """Verify that the directory contains a valid OpenVINO GenAI Whisper model."""
    return model_integrity.verify_openvino_model_dir(directory)


def _openvino_model_already_available():
    """Return True when the OpenVINO IR is present, or restorable from the build cache."""
    if verify_ov_model(OV_WHISPER_DIR):
        logger.info("OpenVINO Whisper model already exists and is valid. Skipping.")
        return True
    return _restore_directory_from_cache("whisper-openvino", OV_WHISPER_DIR, "Whisper (OpenVINO)", validator=verify_ov_model)


def preload_whisper():
    # 1. CTranslate2 (Faster-Whisper)
    logger.info("--- [1/4] Preparing Faster-Whisper Model ---")
    _ensure_ct2_whisper()

    # 2. OpenVINO (Intel-Whisper)
    logger.info("--- [2/4] Preparing OpenVINO Whisper Model ---")

    if SKIP_INTEL_WHISPER:
        logger.info("Intel Whisper preloading is disabled via flag. Skipping.")
        return

    if _openvino_model_already_available():
        return

    if _export_openvino_whisper():
        return

    if _download_openvino_genai():
        return

    logger.error(
        "Could not obtain a usable OpenVINO IR for the Intel engine: neither the optimum "
        "export nor the pre-converted download (%s) produced one.",
        OV_MODEL_ID,
    )
    logger.error("Re-run with --skip-intel-whisper to provision the other models without it.")
    sys.exit(1)


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
