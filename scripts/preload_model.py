import os
import logging
import sys
import torch
import shutil
import argparse
import subprocess
from faster_whisper import download_model
from audio_separator.separator import Separator

# Set up logging to stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

# Configuration
WHISPER_ID = "Systran/faster-whisper-large-v3"
OV_SOURCE_ID = "openai/whisper-large-v3"
UVR_MODEL = "UVR-MDX-NET-Inst_HQ_3.onnx"

# The official OpenVINO pre-converted model for GenAI 2025.4
OV_MODEL_ID = "OpenVINO/whisper-large-v3-fp16-ov"

SYSTEM_DIR = "/app/system_models"
WHISPER_DIR = os.path.join(SYSTEM_DIR, "whisper")
OV_WHISPER_DIR = os.path.join(SYSTEM_DIR, "whisper-openvino")
UVR_DIR = os.path.join(SYSTEM_DIR, "uvr")
VAD_DIR = os.path.join(SYSTEM_DIR, "vad")

CACHE_DIR = None
SKIP_INTEL_WHISPER = False

def verify_ov_model(directory):
    """Verify that the directory contains a valid OpenVINO GenAI Whisper model."""
    if not os.path.exists(directory):
        return False
    
    # Critical files for OpenVINO GenAI WhisperPipeline
    critical_patterns = [
        "openvino_encoder_model.xml",
        "openvino_encoder_model.bin",
        "openvino_decoder_model.xml",
        "openvino_decoder_model.bin"
    ]
    
    files = os.listdir(directory)
    missing = [p for p in critical_patterns if p not in files]
    
    if missing:
        logger.warning("Model directory %s is missing critical files: %s", directory, missing)
        return False
    
    # Check for empty bin files (the "Empty weights data" error)
    # The bin file for large-v3 should be > 100MB even if quantized, and ~1.5GB if FP16.
    for f in ["openvino_encoder_model.bin", "openvino_decoder_model.bin"]:
        fpath = os.path.join(directory, f)
        if os.path.exists(fpath):
            fsize = os.path.getsize(fpath)
            if fsize < 50 * 1024 * 1024: # Less than 50MB is definitely wrong for Large V3
                logger.error("File %s is too small (%d bytes). Corrupted or empty.", f, fsize)
                return False
        else:
            return False
                
    return True

def preload_whisper():
    # 1. CTranslate2 (Faster-Whisper)
    logger.info("--- [1/4] Preparing Faster-Whisper Model ---")
    
    # Idempotency check: If already present and looks okay, skip
    if os.path.exists(os.path.join(WHISPER_DIR, "model.bin")):
        logger.info("Faster-Whisper model already exists in %s. Skipping.", WHISPER_DIR)
    else:
        # Check cache first
        if CACHE_DIR and os.path.exists(os.path.join(CACHE_DIR, "whisper")):
            logger.info("Restoring Whisper (CT2) from cache...")
            if os.path.exists(WHISPER_DIR): shutil.rmtree(WHISPER_DIR)
            shutil.copytree(os.path.join(CACHE_DIR, "whisper"), WHISPER_DIR)
        else:
            logger.info("Downloading Whisper Model (CT2): %s to %s...", WHISPER_ID, WHISPER_DIR)
            try:
                download_model(WHISPER_ID, output_dir=WHISPER_DIR)
                logger.info("Whisper (CT2) downloaded successfully.")
                if CACHE_DIR:
                    os.makedirs(CACHE_DIR, exist_ok=True)
                    cache_path = os.path.join(CACHE_DIR, "whisper")
                    if os.path.exists(cache_path): shutil.rmtree(cache_path)
                    shutil.copytree(WHISPER_DIR, cache_path)
            except Exception as e:
                logger.error("Failed to download Whisper model: %s", e)
                sys.exit(1)

    # 2. OpenVINO (Intel-Whisper)
    logger.info("--- [2/4] Preparing OpenVINO Whisper Model ---")
    
    if SKIP_INTEL_WHISPER:
        logger.info("Intel Whisper preloading is disabled via flag. Skipping.")
        return

    # Idempotency check
    if verify_ov_model(OV_WHISPER_DIR):
        logger.info("OpenVINO Whisper model already exists and is valid. Skipping.")
        return

    # Check cache first
    if CACHE_DIR and os.path.exists(os.path.join(CACHE_DIR, "whisper-openvino")):
        if verify_ov_model(os.path.join(CACHE_DIR, "whisper-openvino")):
            logger.info("Restoring Whisper (OpenVINO) from cache...")
            if os.path.exists(OV_WHISPER_DIR): shutil.rmtree(OV_WHISPER_DIR)
            shutil.copytree(os.path.join(CACHE_DIR, "whisper-openvino"), OV_WHISPER_DIR)
            return

    # Attempt export if optimum-cli is available
    if shutil.which("optimum-cli"):
        logger.info("Exporting Whisper Model to OpenVINO using optimum-cli...")
        try:
            source_id = "openai/whisper-large-v3"
            cmd = [
                "optimum-cli", "export", "openvino",
                "--model", source_id,
                "--task", "automatic-speech-recognition",
                "--weight-format", "fp16", # FP16 is safer for build
                OV_WHISPER_DIR
            ]
            
            logger.info("Running: %s", " ".join(cmd))
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                print(line, end="")
            process.wait()
            
            if process.returncode == 0 and verify_ov_model(OV_WHISPER_DIR):
                logger.info("Whisper (OpenVINO) exported successfully.")
                if CACHE_DIR:
                    os.makedirs(CACHE_DIR, exist_ok=True)
                    cache_path = os.path.join(CACHE_DIR, "whisper-openvino")
                    if os.path.exists(cache_path): shutil.rmtree(cache_path)
                    shutil.copytree(OV_WHISPER_DIR, cache_path)
                return
            else:
                logger.warning("Optimum export failed or produced invalid model files.")
        except Exception as e:
            logger.warning("Exception during optimum export: %s", e)
    else:
        logger.info("optimum-cli not found. Skipping build-time conversion.")

    # Approach: Download the original OpenAI weights for Intel conversion
    logger.info("Downloading official OpenAI Whisper weights for Intel conversion: %s", OV_SOURCE_ID)
    try:
        from huggingface_hub import snapshot_download
        
        snapshot_download(
            repo_id=OV_SOURCE_ID,
            local_dir=OV_WHISPER_DIR,
            local_dir_use_symlinks=False,
            max_workers=4
        )
        
        logger.info("Whisper (OpenVINO) source weights ready in %s", OV_WHISPER_DIR)
        if CACHE_DIR:
            os.makedirs(CACHE_DIR, exist_ok=True)
            cache_path = os.path.join(CACHE_DIR, "whisper-openvino")
            if os.path.exists(cache_path): shutil.rmtree(cache_path)
            shutil.copytree(OV_WHISPER_DIR, cache_path)
            
    except Exception as e:
        logger.error("Failed to download OpenVINO Whisper model: %s", e)

def preload_uvr():
    logger.info("--- [3/4] Preparing UVR Model ---")
    
    # Idempotency check
    if os.path.exists(os.path.join(UVR_DIR, UVR_MODEL)):
        logger.info("UVR model already exists in %s. Skipping.", UVR_DIR)
        return

    if CACHE_DIR and os.path.exists(os.path.join(CACHE_DIR, "uvr")):
        logger.info("Restoring UVR Model from cache...")
        if os.path.exists(UVR_DIR): shutil.rmtree(UVR_DIR)
        shutil.copytree(os.path.join(CACHE_DIR, "uvr"), UVR_DIR)
        return

    logger.info("Downloading UVR Model: %s to %s...", UVR_MODEL, UVR_DIR)
    try:
        sep = Separator(model_file_dir=UVR_DIR, output_dir="/tmp")
        sep.load_model(UVR_MODEL)
        logger.info("UVR Model downloaded successfully.")

        if CACHE_DIR:
            os.makedirs(CACHE_DIR, exist_ok=True)
            cache_path = os.path.join(CACHE_DIR, "uvr")
            if os.path.exists(cache_path): shutil.rmtree(cache_path)
            shutil.copytree(UVR_DIR, cache_path)
    except Exception as e:
        logger.error("Failed to download UVR model: %s", e)
        sys.exit(1)

def preload_vad():
    logger.info("--- [4/4] Preparing VAD Model (C++ ONNX) ---")
    
    # Idempotency check
    if os.path.exists(os.path.join(VAD_DIR, "silero_vad.onnx")):
        logger.info("VAD model already exists in %s. Skipping.", VAD_DIR)
        return

    if CACHE_DIR and os.path.exists(os.path.join(CACHE_DIR, "vad")):
        logger.info("Restoring VAD Model from cache...")
        if os.path.exists(VAD_DIR): shutil.rmtree(VAD_DIR)
        shutil.copytree(os.path.join(CACHE_DIR, "vad"), VAD_DIR)
        return

    logger.info("Downloading Silero VAD ONNX model to %s...", VAD_DIR)
    try:
        os.makedirs(VAD_DIR, exist_ok=True)
        import requests
        # Direct URL to the official Silero VAD ONNX model
        vad_url = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
        target_path = os.path.join(VAD_DIR, "silero_vad.onnx")
        
        response = requests.get(vad_url, stream=True, timeout=30)
        response.raise_for_status()
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info("Silero VAD ONNX downloaded successfully: %s", target_path)
        
        if CACHE_DIR:
            os.makedirs(CACHE_DIR, exist_ok=True)
            cache_path = os.path.join(CACHE_DIR, "vad")
            if os.path.exists(cache_path): shutil.rmtree(cache_path)
            shutil.copytree(VAD_DIR, cache_path)
    except Exception as e:
        logger.error("Failed to download Silero VAD ONNX: %s", e)
        sys.exit(1)

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

