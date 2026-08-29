"""
Unit tests for model download and integrity verification module (model_integrity.py).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest import mock

from modules.core import model_integrity


def test_compute_file_sha256(tmp_path: Path):
    """Test compute_file_sha256 matches hashlib output."""
    sample = tmp_path / "sample.bin"
    sample.write_bytes(b"whisper-pro-test-content")
    expected = hashlib.sha256(b"whisper-pro-test-content").hexdigest()
    assert model_integrity.compute_file_sha256(sample) == expected


def test_verify_file_sha256_success(tmp_path: Path):
    """Test successful sha256 checksum verification."""
    sample = tmp_path / "valid.bin"
    sample.write_bytes(b"valid-model-bytes")
    digest = hashlib.sha256(b"valid-model-bytes").hexdigest()
    assert model_integrity.verify_file_sha256(sample, digest) is True
    assert model_integrity.verify_file_sha256(sample, digest.upper()) is True


def test_verify_file_sha256_mismatch_and_missing(tmp_path: Path):
    """Test mismatched sha256 or missing files return False."""
    sample = tmp_path / "corrupt.bin"
    sample.write_bytes(b"corrupt-bytes")
    assert model_integrity.verify_file_sha256(sample, "0" * 64) is False
    assert model_integrity.verify_file_sha256(tmp_path / "non_existent.bin", "0" * 64) is False

    empty = tmp_path / "empty.bin"
    empty.touch()
    assert model_integrity.verify_file_sha256(empty, hashlib.sha256(b"").hexdigest()) is False


def test_verify_ct2_model_dir(tmp_path: Path):
    """Test CTranslate2 model directory structure verification."""
    model_dir = tmp_path / "ct2_model"
    model_dir.mkdir()

    # Empty dir fails
    assert model_integrity.verify_ct2_model_dir(model_dir) is False

    # Add model.bin under size limit
    bin_file = model_dir / "model.bin"
    bin_file.write_bytes(b"small")
    assert model_integrity.verify_ct2_model_dir(model_dir, min_bin_bytes=100) is False

    # Add large enough model.bin but missing config.json
    bin_file.write_bytes(b"x" * 200)
    assert model_integrity.verify_ct2_model_dir(model_dir, min_bin_bytes=100) is False

    # Add valid config.json but missing preprocessor_config.json
    (model_dir / "config.json").write_text('{"model_type": "whisper"}')
    assert model_integrity.verify_ct2_model_dir(model_dir, min_bin_bytes=100) is False

    # Add preprocessor_config.json but missing tokenizer.json
    (model_dir / "preprocessor_config.json").write_text('{"feature_extractor_type": "WhisperFeatureExtractor"}')
    assert model_integrity.verify_ct2_model_dir(model_dir, min_bin_bytes=100) is False

    # Add tokenizer.json
    (model_dir / "tokenizer.json").write_text('{"version": "1.0"}')
    assert model_integrity.verify_ct2_model_dir(model_dir, min_bin_bytes=100) is True

    # Safetensors variant
    bin_file.unlink()
    (model_dir / "model.safetensors").write_bytes(b"x" * 200)
    assert model_integrity.verify_ct2_model_dir(model_dir, min_bin_bytes=100) is True


def test_verify_openvino_model_dir(tmp_path: Path):
    """Test OpenVINO model directory structure and size checks."""
    ov_dir = tmp_path / "ov_model"
    ov_dir.mkdir()

    assert model_integrity.verify_openvino_model_dir(ov_dir) is False

    # Create encoder files
    (ov_dir / "openvino_encoder_model.xml").write_text("<xml/>")
    (ov_dir / "openvino_encoder_model.bin").write_bytes(b"x" * 100)

    # Missing decoder files
    assert model_integrity.verify_openvino_model_dir(ov_dir, min_bin_bytes=50) is False

    # Create decoder files
    (ov_dir / "openvino_decoder_model.xml").write_text("<xml/>")
    (ov_dir / "openvino_decoder_model.bin").write_bytes(b"x" * 100)

    # Missing tokenizer assets
    assert model_integrity.verify_openvino_model_dir(ov_dir, min_bin_bytes=50) is False

    (ov_dir / "openvino_tokenizer.xml").write_text("<xml/>")
    (ov_dir / "openvino_tokenizer.bin").write_bytes(b"x" * 10)
    (ov_dir / "openvino_detokenizer.xml").write_text("<xml/>")
    (ov_dir / "openvino_detokenizer.bin").write_bytes(b"x" * 10)
    (ov_dir / "generation_config.json").write_text('{"max_length": 448}')

    assert model_integrity.verify_openvino_model_dir(ov_dir, min_bin_bytes=50) is True
    # If bin too small
    assert model_integrity.verify_openvino_model_dir(ov_dir, min_bin_bytes=200) is False


def test_verify_onnx_model_file(tmp_path: Path):
    """Test ONNX model file size and hash validation."""
    onnx_file = tmp_path / "model.onnx"
    onnx_file.write_bytes(b"onnx-data-12345")
    expected_hash = hashlib.sha256(b"onnx-data-12345").hexdigest()

    assert model_integrity.verify_onnx_model_file(onnx_file, min_bytes=5) is True
    assert model_integrity.verify_onnx_model_file(onnx_file, min_bytes=5, expected_sha256=expected_hash) is True
    assert model_integrity.verify_onnx_model_file(onnx_file, min_bytes=50) is False
    assert model_integrity.verify_onnx_model_file(onnx_file, min_bytes=5, expected_sha256="wrong") is False


def test_purge_corrupted_path(tmp_path: Path):
    """Test purge_corrupted_path safely deletes files and directories."""
    file_path = tmp_path / "corrupt_file.bin"
    file_path.write_bytes(b"bad")
    assert model_integrity.purge_corrupted_path(file_path) is True
    assert not file_path.exists()

    dir_path = tmp_path / "corrupt_dir"
    dir_path.mkdir()
    (dir_path / "sub.bin").write_bytes(b"bad")
    assert model_integrity.purge_corrupted_path(dir_path) is True
    assert not dir_path.exists()

    # Non-existent path returns True safely
    assert model_integrity.purge_corrupted_path(tmp_path / "missing") is True


def test_download_with_integrity_retry_first_success(tmp_path: Path):
    """Test download succeeds on first attempt when valid."""
    target = tmp_path / "downloaded.bin"
    calls = []

    def download():
        calls.append(1)
        target.write_bytes(b"good")

    def validator(path_to_check: Path) -> bool:
        return path_to_check.exists() and path_to_check.read_bytes() == b"good"

    success = model_integrity.download_with_integrity_retry(download, validator, target, max_retries=2)
    assert success is True
    assert len(calls) == 1


def test_download_with_integrity_retry_recovers_after_corruption(tmp_path: Path):
    """Test download recovers after initial corrupt attempt."""
    target = tmp_path / "retry_download.bin"
    attempts = 0

    def download():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            target.write_bytes(b"corrupt")
        else:
            target.write_bytes(b"clean")

    def validator(path_to_check: Path) -> bool:
        return path_to_check.exists() and path_to_check.read_bytes() == b"clean"

    success = model_integrity.download_with_integrity_retry(download, validator, target, max_retries=3)
    assert success is True
    assert attempts == 2
    assert target.read_bytes() == b"clean"


def test_download_with_integrity_retry_exhausts_and_fails(tmp_path: Path):
    """Test download fails cleanly and purges when all attempts are corrupted."""
    target = tmp_path / "failing_download.bin"
    attempts = 0

    def download():
        nonlocal attempts
        attempts += 1
        target.write_bytes(b"always-bad")

    def validator(_unused_path: Path) -> bool:
        return False

    success = model_integrity.download_with_integrity_retry(download, validator, target, max_retries=2)
    assert success is False
    assert attempts == 2
    assert not target.exists()  # Purged after failed attempt


def test_download_with_integrity_retry_existing_valid_file(tmp_path: Path):
    """Test existing valid model file immediately skips download."""
    target = tmp_path / "existing_valid.bin"
    target.write_bytes(b"valid")
    called = False

    def download():
        nonlocal called
        called = True

    def validator(p: Path) -> bool:
        return p.exists() and p.read_bytes() == b"valid"

    success = model_integrity.download_with_integrity_retry(download, validator, target)
    assert success is True
    assert not called


def test_download_with_integrity_retry_handles_download_exception(tmp_path: Path):
    """Test download function exceptions are caught, logged, and retried."""
    target = tmp_path / "throwing_download.bin"
    attempts = 0

    def download():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("Connection dropped")
        target.write_bytes(b"recovered")

    def validator(p: Path) -> bool:
        return p.exists() and p.read_bytes() == b"recovered"

    success = model_integrity.download_with_integrity_retry(download, validator, target, max_retries=2)
    assert success is True
    assert attempts == 2


def test_verify_file_sha256_unreadable_file(tmp_path: Path):
    """Test unreadable or permission error during sha256 calculation returns False."""
    sample = tmp_path / "unreadable.bin"
    sample.write_bytes(b"test")
    # The file's OWN digest, not an unrelated zero hash: against "0" * 64 the call returns
    # False on a plain checksum mismatch too, so the test passed whether or not the
    # PermissionError was handled -- it could not distinguish the behaviour it names.
    real_digest = hashlib.sha256(b"test").hexdigest()
    assert model_integrity.verify_file_sha256(sample, real_digest) is True
    with mock.patch("builtins.open", side_effect=PermissionError("Permission denied")):
        assert model_integrity.verify_file_sha256(sample, real_digest) is False


def test_vad_preload_download_failure_handling(tmp_path: Path):
    """Test that failed VAD download returns False and can trigger clean failure."""
    target_file = tmp_path / "silero_vad.onnx"

    def failing_download():
        target_file.write_bytes(b"corrupt")

    def validator(p: Path) -> bool:
        return model_integrity.verify_onnx_model_file(
            p, min_bytes=500 * 1024, expected_sha256="1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
        )

    res = model_integrity.download_with_integrity_retry(
        download_fn=failing_download,
        validator_fn=validator,
        target_path=target_file,
        max_retries=2,
        description="Silero VAD ONNX model",
    )
    assert res is False
    assert not target_file.exists()
