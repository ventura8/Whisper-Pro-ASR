"""Regression tests: an OpenVINO model directory must survive a Faster-Whisper load failure.

On a hybrid NVIDIA+Intel host, an explicit INTEL-WHISPER task could be scheduled onto the
CUDA unit. The factory then degraded to FasterWhisperEngine and handed CTranslate2 the
OpenVINO IR directory, which failed to load, was misread as a corrupt CT2 model, and was
deleted -- taking the weights the Intel unit needed with it.
"""

import os
from unittest import mock

from modules.core import model_integrity
from modules.inference.engines.faster_whisper_engine import FasterWhisperEngine

OV_FILES = {
    "openvino_encoder_model.xml": 4096,
    "openvino_encoder_model.bin": model_integrity.MIN_OPENVINO_BIN_BYTES + 1,
    "openvino_decoder_model.xml": 4096,
    "openvino_decoder_model.bin": model_integrity.MIN_OPENVINO_BIN_BYTES + 1,
    "openvino_tokenizer.xml": 1024,
    "openvino_tokenizer.bin": 1024,
    "openvino_detokenizer.xml": 1024,
    "openvino_detokenizer.bin": 1024,
    "generation_config.json": 128,
}


def _make_ov_model_dir(tmp_path):
    """Create a directory the OpenVINO verifier accepts, without materialising the bytes.

    The verifier reads ``st_size``, so the two ~50MB .bin files are created sparse:
    allocating them for real wrote 100MB per test and made this module the slowest in the
    suite for no added coverage.
    """
    model_dir = tmp_path / "whisper-openvino"
    model_dir.mkdir()
    for name, size in OV_FILES.items():
        path = model_dir / name
        path.touch()
        os.truncate(path, size)
    return str(model_dir)


def test_ov_dir_is_recognised_as_valid(tmp_path):
    """Guard the fixture itself: the directory must pass OpenVINO verification."""
    assert model_integrity.verify_openvino_model_dir(_make_ov_model_dir(tmp_path)) is True


def test_purge_local_dir_spares_openvino_model(tmp_path):
    model_dir = _make_ov_model_dir(tmp_path)
    engine = FasterWhisperEngine.__new__(FasterWhisperEngine)

    engine._purge_corrupted_local_dir(model_dir, RuntimeError("not a CT2 model"))

    assert os.path.isdir(model_dir)
    assert os.path.exists(os.path.join(model_dir, "openvino_encoder_model.xml"))


def test_purge_local_dir_still_removes_corrupt_ct2_dir(tmp_path):
    """The protection must not disarm the purge for a genuinely corrupt CT2 directory."""
    corrupt = tmp_path / "whisper"
    corrupt.mkdir()
    (corrupt / "config.json").write_text("{}")

    engine = FasterWhisperEngine.__new__(FasterWhisperEngine)
    engine._purge_corrupted_local_dir(str(corrupt), RuntimeError("boom"))

    assert not os.path.exists(str(corrupt))


def test_cached_snapshot_retry_spares_openvino_model(tmp_path):
    model_dir = _make_ov_model_dir(tmp_path)
    engine = FasterWhisperEngine.__new__(FasterWhisperEngine)

    with mock.patch.object(FasterWhisperEngine, "_resolve_hf_snapshot_dir", return_value=model_dir):
        result = engine._try_retry_cached_model(
            model_dir,
            RuntimeError("not a CT2 model"),
            device="cpu",
            device_index=0,
            compute_type="int8",
            cpu_threads=1,
            download_root=str(tmp_path),
        )

    assert result is False
    assert os.path.isdir(model_dir)
