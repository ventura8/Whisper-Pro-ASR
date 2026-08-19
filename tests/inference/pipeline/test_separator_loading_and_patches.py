"""Separator loading and audio-separator ONNX patching tests.

This module hosts the subset of `test_preprocessing.py` that covers:
- separator default model loading
- audio-separator ONNX checks + safe_separate validation

Keeping these focused helps reduce per-file size.
"""

from __future__ import annotations

from unittest import mock

import pytest

from modules.inference.pipeline.preprocessing import PreprocessingManager
from modules.inference.pipeline.preprocessing import helpers as preprocessing_helpers


@pytest.fixture
def prep_manager():
    """Provide a clean PreprocessingManager instance for isolated unit tests."""
    unit = {"id": "CPU", "type": "CPU", "name": "CPU"}
    return PreprocessingManager(assigned_unit=unit)


class TestLoadSeparatorModelDefault:
    """Tests for PreprocessingManager._load_separator_model_default."""

    def test_accelerator_fallback_to_cpu_success(self, prep_manager):
        separator = mock.MagicMock()
        separator.onnx_execution_provider = [
            "ROCMExecutionProvider",
            "CPUExecutionProvider",
        ]
        call_count = 0

        def fake_load_model(model_name):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("AMD driver failure")

        separator.load_model.side_effect = fake_load_model
        prep_manager._load_separator_model_default(separator)
        assert separator.onnx_execution_provider == ["CPUExecutionProvider"]
        assert call_count == 2

    def test_cpu_load_failure_raises(self, prep_manager):
        separator = mock.MagicMock()
        separator.onnx_execution_provider = ["CPUExecutionProvider"]
        separator.load_model.side_effect = RuntimeError("CPU load failure")
        with pytest.raises(RuntimeError, match="CPU load failure"):
            prep_manager._load_separator_model_default(separator)
        assert prep_manager.separator is None


class TestAudioSeparatorPatches:
    """Tests for _patch_audio_separator_onnx_check wrappers."""

    def test_safe_download_model_files_existing_and_delegation(self, tmp_path):
        fake_cls = type("FakeSeparator", (), {})
        model_file = tmp_path / "UVR-MDX-NET-Inst_HQ_3.onnx"
        model_file.write_text("model_bytes")

        with mock.patch("importlib.import_module") as mock_import:
            mock_import.return_value = mock.MagicMock(Separator=fake_cls)
            preprocessing_helpers._patch_audio_separator_onnx_check()

        instance = fake_cls()
        instance.model_file_dir = str(tmp_path)

        # Test existing file branch
        res = instance.download_model_files("UVR-MDX-NET-Inst_HQ_3.onnx")
        assert res[0] == "UVR-MDX-NET-Inst_HQ_3.onnx"
        assert res[3] == str(model_file)

        # Test absent file branch with no original download
        with pytest.raises(FileNotFoundError, match="Model file absent.onnx not found"):
            instance.download_model_files("absent.onnx")

    def test_safe_separate_validation(self):
        fake_cls = type("FakeSeparator", (), {})
        fake_cls.separate = mock.MagicMock(return_value=["vocals.wav"])
        with mock.patch("importlib.import_module") as mock_import:
            mock_import.return_value = mock.MagicMock(Separator=fake_cls)
            preprocessing_helpers._patch_audio_separator_onnx_check()

        instance = fake_cls()
        assert instance.separate("/fake/path.wav") == ["vocals.wav"]

        # When orig_separate returns empty list, safe_separate raises RuntimeError
        fake_cls.is_patched = False
        fake_cls.download_model_files = mock.MagicMock()
        fake_cls.separate = mock.MagicMock(return_value=[])
        with mock.patch("importlib.import_module") as mock_import:
            mock_import.return_value = mock.MagicMock(Separator=fake_cls)
            preprocessing_helpers._patch_audio_separator_onnx_check()

        inst2 = fake_cls()
        with pytest.raises(RuntimeError, match="failed to process file"):
            inst2.separate("/fake/audio.wav")

    def test_hash_loader_delegates_and_defaults_only_for_hq3(self):
        """Hash metadata fallback applies only to the default UVR HQ_3 model."""
        from modules.inference.pipeline.preprocessing import helpers as preprocessing_helpers

        orig = mock.Mock(return_value={"mdx_dim_f_set": 99})
        assert preprocessing_helpers._safe_load_model_data_using_hash(orig, "other-model")["mdx_dim_f_set"] == 99

        orig_fail = mock.Mock(side_effect=ValueError("offline"))
        defaulted = preprocessing_helpers._safe_load_model_data_using_hash(orig_fail, "UVR-MDX-NET-Inst_HQ_3")
        assert defaulted["mdx_n_fft_scale_set"] == 6144

        with pytest.raises(ValueError, match="offline"):
            preprocessing_helpers._safe_load_model_data_using_hash(orig_fail, "other-model.onnx")
