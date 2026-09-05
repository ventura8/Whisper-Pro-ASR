"""
Model Download and Storage Integrity Verification Utilities.

Provides checksum verification, model structural sanity checks, corruption
detection, and automatic deletion + redownload helpers.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Minimum expected file sizes to detect empty/truncated model files (in bytes)
MIN_CT2_BIN_BYTES = 10 * 1024 * 1024  # 10 MB
MIN_OPENVINO_BIN_BYTES = 50 * 1024 * 1024  # 50 MB
MIN_ONNX_MODEL_BYTES = 1 * 1024 * 1024  # 1 MB

UVR_MDX_HQ3_SHA256 = "317554b07fe1ea5279a77f2b1520a41ea4b93432560c4ffd08792c30fddf9adc"


def compute_file_sha256(file_path: str | Path, chunk_size: int = 65536) -> str:
    """Compute SHA-256 hex digest of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_file_sha256(file_path: str | Path, expected_sha256: str) -> bool:
    """Verify that a file exists, is non-empty, and matches the expected SHA-256 hash."""
    path = Path(file_path)
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        digest = compute_file_sha256(path)
        return digest.lower() == expected_sha256.lower()
    except (OSError, PermissionError) as exc:
        logger.warning("[Integrity] Failed to compute hash for %s: %s", path, exc)
        return False


def verify_file_min_size(file_path: str | Path, min_bytes: int = 1) -> bool:
    """Check if file exists and exceeds minimum size in bytes."""
    path = Path(file_path)
    if not path.is_file():
        return False
    try:
        return path.stat().st_size >= min_bytes
    except OSError:
        return False


def _has_ct2_binary(path: Path, min_bin_bytes: int) -> bool:
    return verify_file_min_size(path / "model.bin", min_bin_bytes) or verify_file_min_size(path / "model.safetensors", min_bin_bytes)


def _has_ct2_configs(path: Path) -> bool:

    for name in ("config.json", "preprocessor_config.json", "tokenizer.json"):
        if not verify_file_min_size(path / name, 1):
            logger.warning("[Integrity] CT2 model directory %s is missing %s.", path, name)
            return False
    return True


def verify_ct2_model_dir(dir_path: str | Path, min_bin_bytes: int = MIN_CT2_BIN_BYTES) -> bool:
    """
    Verify structural and file-size integrity of a CTranslate2 (Faster-Whisper) model directory.
    Requires model.bin (or model.safetensors) and config.json / vocabulary files.
    """
    path = Path(dir_path)
    if not path.is_dir():
        return False
    if not _has_ct2_binary(path, min_bin_bytes):
        logger.warning("[Integrity] CT2 model directory %s is missing valid weights binary.", path)
        return False
    return _has_ct2_configs(path)


def _ov_xml_bin_pair_valid(path: Path, prefix: str, min_bin_bytes: int) -> bool:
    xml_file = path / f"{prefix}.xml"
    bin_file = path / f"{prefix}.bin"
    if not xml_file.is_file() or xml_file.stat().st_size == 0:
        logger.warning("[Integrity] OpenVINO model missing XML file: %s", xml_file)
        return False
    if not verify_file_min_size(bin_file, min_bin_bytes):
        logger.warning("[Integrity] OpenVINO model missing/truncated BIN file: %s", bin_file)
        return False
    return True


def _ov_auxiliary_assets_valid(path: Path) -> bool:
    aux_pairs = ["openvino_tokenizer", "openvino_detokenizer"]
    for prefix in aux_pairs:
        if not (path / f"{prefix}.xml").is_file() or not verify_file_min_size(path / f"{prefix}.bin", 1):
            logger.warning("[Integrity] OpenVINO model missing tokenizer asset: %s", prefix)
            return False
    if not verify_file_min_size(path / "generation_config.json", 1):
        logger.warning("[Integrity] OpenVINO model missing generation_config.json: %s", path)
        return False
    return True


def verify_openvino_model_dir(dir_path: str | Path, min_bin_bytes: int = MIN_OPENVINO_BIN_BYTES) -> bool:
    """
    Verify structural and file-size integrity of an OpenVINO GenAI Whisper model directory.
    Requires encoder and decoder XML/BIN pairs, tokenizer/detokenizer assets, and generation_config.json.
    """
    path = Path(dir_path)
    if not path.is_dir():
        return False

    encoder_valid = _ov_xml_bin_pair_valid(path, "openvino_encoder_model", min_bin_bytes)
    decoder_valid = _ov_xml_bin_pair_valid(path, "openvino_decoder_model", min_bin_bytes)
    aux_valid = _ov_auxiliary_assets_valid(path)

    return encoder_valid and decoder_valid and aux_valid


def verify_onnx_model_file(
    file_path: str | Path,
    min_bytes: int = MIN_ONNX_MODEL_BYTES,
    expected_sha256: str | None = None,
) -> bool:
    """Verify an ONNX model file exists, meets minimum size, and optionally matches checksum."""
    path = Path(file_path)

    if not verify_file_min_size(path, min_bytes):
        logger.warning("[Integrity] ONNX file missing or too small (%s)", path)
        return False

    if expected_sha256:
        return verify_file_sha256(path, expected_sha256)

    return True


def purge_corrupted_path(target_path: str | Path, description: str = "asset") -> bool:
    """Safely and recursively remove a corrupted file or directory."""
    path = Path(target_path)
    if not path.exists():
        return True

    logger.warning("[Integrity] Purging corrupted %s at %s...", description, path)
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        logger.info("[Integrity] Successfully removed corrupted %s: %s", description, path)
        return True
    except (OSError, PermissionError) as exc:
        logger.error("[Integrity] Failed to remove corrupted %s at %s: %s", description, path, exc)
        return False


def _execute_download_attempt(
    download_fn: Callable[[], Any],
    validator_fn: Callable[[str | Path], bool],
    target_path: str | Path,
    description: str,
    *,
    attempt: int,
    max_retries: int,
) -> bool:
    logger.info("[Integrity] Attempt %d/%d: downloading/loading %s...", attempt, max_retries, description)
    try:
        download_fn()
        if validator_fn(target_path):
            logger.info("[Integrity] Validation succeeded for %s on attempt %d.", description, attempt)
            return True
        logger.warning("[Integrity] Validation failed for %s after attempt %d.", description, attempt)
    except (RuntimeError, ValueError, OSError, ImportError, TypeError, KeyError) as exc:
        logger.warning("[Integrity] Download error for %s on attempt %d: %s", description, attempt, exc)

    purge_corrupted_path(target_path, description)
    return False


def download_with_integrity_retry(
    download_fn: Callable[[], Any],
    validator_fn: Callable[[str | Path], bool],
    target_path: str | Path,
    max_retries: int = 2,
    description: str = "model",
) -> bool:
    """
    Execute a download function with automatic validation, corruption purge, and retry.

    Note: `max_retries` represents the total number of download and validation attempts
    (e.g., `max_retries=2` executes up to two total attempts).
    """

    path = Path(target_path)

    # Initial check: if already present and valid, skip download
    if path.exists():
        if validator_fn(path):
            logger.info("[Integrity] %s already exists and passed integrity verification.", description)
            return True
        logger.warning("[Integrity] Existing %s failed integrity check; purging and redownloading.", description)
        purge_corrupted_path(path, description)

    for attempt in range(1, max_retries + 1):
        if _execute_download_attempt(download_fn, validator_fn, target_path, description, attempt=attempt, max_retries=max_retries):
            return True

    logger.error("[Integrity] All %d download/validation attempts failed for %s.", max_retries, description)
    return False
