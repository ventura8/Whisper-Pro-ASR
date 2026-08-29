"""Tests for modules/config.py"""

import importlib
import os
import tempfile
from typing import Any
from unittest import mock

import pytest

import modules.core.config as config_module
from tests.config_reload_helpers import npu_cannot_execute

pytestmark = pytest.mark.usefixtures("restore_config_after_reload")


class TestConfig:
    """Test suite for config module."""

    def test_default_model_value(self):
        """Test that DEFAULT_MODEL is set correctly."""
        assert config_module.DEFAULT_WHISPER in {
            "Systran/faster-whisper-large-v3",
            "openai/whisper-large-v3",
        }

    def test_model_id_from_env(self):
        """Test MODEL_ID reads from environment."""
        with mock.patch.dict(os.environ, {"ASR_MODEL": "/custom/model/path"}):
            importlib.reload(config_module)

            assert config_module.MODEL_ID == "/custom/model/path"

    def test_default_batch_size(self):
        """Test DEFAULT_BATCH_SIZE defaults to 1."""
        with mock.patch.dict(os.environ, {}, clear=True):
            importlib.reload(config_module)

            assert config_module.DEFAULT_BATCH_SIZE == 1

    def test_batch_size_from_env(self):
        """Test ASR_BATCH_SIZE from environment."""
        with mock.patch.dict(os.environ, {"ASR_BATCH_SIZE": "4"}):
            importlib.reload(config_module)

            assert config_module.DEFAULT_BATCH_SIZE == 4

    def test_default_beam_size(self):
        """Test DEFAULT_BEAM_SIZE defaults to 5."""
        with mock.patch.dict(os.environ, {}, clear=True):
            importlib.reload(config_module)

            assert config_module.DEFAULT_BEAM_SIZE == 5

    def test_beam_size_from_env(self):
        """Test ASR_BEAM_SIZE from environment."""
        with mock.patch.dict(os.environ, {"ASR_BEAM_SIZE": "1"}):
            importlib.reload(config_module)

            assert config_module.DEFAULT_BEAM_SIZE == 1

    def test_debug_mode_false_default(self):
        """Test DEBUG_MODE defaults to False."""
        with mock.patch.dict(os.environ, {}, clear=True):
            importlib.reload(config_module)

            assert config_module.DEBUG_MODE is False

    def test_debug_mode_true(self):
        """Test DEBUG_MODE is True when DEBUG=true."""
        with mock.patch.dict(os.environ, {"DEBUG": "true"}):
            importlib.reload(config_module)

            assert config_module.DEBUG_MODE is True

    def test_debug_mode_case_insensitive(self):
        """Test DEBUG_MODE handles case variations."""
        with mock.patch.dict(os.environ, {"DEBUG": "TRUE"}):
            importlib.reload(config_module)

            assert config_module.DEBUG_MODE is True

    def test_initial_steps_ratio(self):
        """Test INITIAL_STEPS_RATIO constant."""
        assert config_module.INITIAL_STEPS_RATIO == 2.8


class TestConfigEnv:
    """Config from environment and defaults."""

    def test_ov_cache_dir_default(self):
        """Test OV_CACHE_DIR defaults to './model_cache'."""
        with mock.patch.dict(os.environ, {}, clear=True):
            importlib.reload(config_module)

            assert config_module.OV_CACHE_DIR == "./model_cache"

    def test_ov_cache_dir_from_env(self):
        """Test OV_CACHE_DIR can be set via env."""
        custom_cache = os.path.join(tempfile.gettempdir(), "custom_cache")
        with mock.patch.dict(os.environ, {"OV_CACHE_DIR": custom_cache}):
            importlib.reload(config_module)

            assert config_module.OV_CACHE_DIR == custom_cache

    def test_app_constants(self):
        """Test app name and version constants."""
        assert "Whisper" in config_module.APP_NAME
        assert config_module.VERSION == "1.3.0"

    def test_device_constant_exists(self):
        """Test DEVICE constant exists."""
        assert hasattr(config_module, "DEVICE")
        assert config_module.DEVICE in ["CPU", "CUDA", "GPU", "NPU", "AMD"]

    def test_asr_threads_default(self):
        """Test ASR_THREADS defaults to 4."""
        with mock.patch.dict(os.environ, {"CPU_CORE_LIMIT": "64"}, clear=True):
            importlib.reload(config_module)

            assert config_module.ASR_THREADS == 4

    def test_asr_threads_from_env(self):
        """Test ASR_THREADS can be set via env."""
        with mock.patch.dict(os.environ, {"ASR_THREADS": "8", "CPU_CORE_LIMIT": "64"}):
            importlib.reload(config_module)

            assert config_module.ASR_THREADS == 8

    def test_ffmpeg_threads_default(self):
        """Test FFMPEG_THREADS defaults to 1 when parallel prep is active."""
        with mock.patch.dict(os.environ, {"CPU_CORE_LIMIT": "64"}, clear=True):
            importlib.reload(config_module)

            # Parallel mode forces FFmpeg to 1 by default to prevent over-provisioning
            assert config_module.FFMPEG_THREADS == 1

    def test_ffmpeg_threads_manual_zero(self):
        """Test FFMPEG_THREADS stays 0 if explicitly set and PREPROCESS_THREADS=1."""
        with mock.patch.dict(os.environ, {"FFMPEG_THREADS": "0", "ASR_PREPROCESS_THREADS": "1"}):
            importlib.reload(config_module)

            assert config_module.FFMPEG_THREADS == 0

    def test_thread_capping_to_cores(self):
        """Test that threads are capped to logical core count with priority."""
        env = {"ASR_THREADS": "64", "ASR_PREPROCESS_THREADS": "64", "CPU_CORE_LIMIT": "8"}
        with mock.patch.dict(os.environ, env):
            # Force CPU mode for test priority check
            with mock.patch("modules.core.config.DEVICE", "CPU"):
                importlib.reload(config_module)
                assert config_module.ASR_THREADS == 8
                # Prep is now allowed to use the full pool sequentially
                assert config_module.PREPROCESS_THREADS == 8

    def test_enable_vocal_separation_default(self):
        """Test ENABLE_VOCAL_SEPARATION defaults to False."""
        with mock.patch.dict(os.environ, {}, clear=True):
            importlib.reload(config_module)

            assert config_module.ENABLE_VOCAL_SEPARATION is False

    def test_enable_vocal_separation_false(self):
        """Test ENABLE_VOCAL_SEPARATION can be set to False."""
        with mock.patch.dict(os.environ, {"ENABLE_VOCAL_SEPARATION": "false"}):
            importlib.reload(config_module)

            assert config_module.ENABLE_VOCAL_SEPARATION is False

    def test_hallucination_phrases_exist(self):
        """Test HALLUCINATION_PHRASES list exists and is populated."""
        assert hasattr(config_module, "HALLUCINATION_PHRASES")
        assert isinstance(config_module.HALLUCINATION_PHRASES, list)
        assert len(config_module.HALLUCINATION_PHRASES) > 0

    def test_compute_type_exists(self):
        """Test COMPUTE_TYPE exists."""
        assert hasattr(config_module, "COMPUTE_TYPE")


class TestConfigHardware:
    """Config hardware detection and device overrides."""

    def test_hardware_detection_logic_cuda(self):
        """Test CUDA detection path."""
        with mock.patch.dict(os.environ, {"ASR_DEVICE": "AUTO"}):
            with mock.patch("ctranslate2.get_cuda_device_count", return_value=1):
                importlib.reload(config_module)
                assert config_module.DEVICE == "CUDA"

    def test_hardware_detection_logic_npu(self):
        """Test NPU detection path."""
        with mock.patch.dict(os.environ, {"ASR_DEVICE": "AUTO"}):
            with mock.patch("ctranslate2.get_cuda_device_count", return_value=0):
                mock_core = mock.MagicMock()
                mock_core.available_devices = ["NPU", "CPU"]
                mock_core.get_property.return_value = "Intel(R) AI Boost"
                # Pinned rather than left to the probe: config now hands it the resolved IR
                # directory, so the answer would otherwise depend on whether the machine
                # running the tests happens to have weights in ./model_cache.
                with mock.patch("openvino.Core", return_value=mock_core), npu_cannot_execute():
                    importlib.reload(config_module)
                    # The NPU is detected and kept as a unit, but ASR does not run on it:
                    # the default engine is CTranslate2, which has no OpenVINO backend, so
                    # the device claim is corrected to CPU while preprocessing -- which the
                    # NPU does run -- stays on it.
                    assert config_module.DEVICE == "CPU"
                    assert config_module.PREPROCESS_DEVICE == "NPU"
                    assert any(u.get("id") == "NPU" for u in config_module.HARDWARE_UNITS if u.get("type") == "NPU")

    def test_hardware_detection_logic_gpu_amd(self):
        """Test AMD/Intel GPU detection path (OpenVINO GPU)."""
        with mock.patch.dict(os.environ, {"ASR_DEVICE": "AUTO"}):
            with mock.patch("ctranslate2.get_cuda_device_count", return_value=0):
                mock_core = mock.MagicMock()
                mock_core.available_devices = ["GPU", "CPU"]
                mock_core.get_property.return_value = "Intel(R) Arc(TM) Graphics"
                with mock.patch("openvino.Core", return_value=mock_core):
                    importlib.reload(config_module)
                    # The iGPU is detected and drives preprocessing, but not ASR: the
                    # default engine is CTranslate2, which has no OpenVINO backend, so the
                    # ASR device claim is CPU rather than a device it cannot address.
                    assert config_module.DEVICE == "CPU"
                    assert config_module.PREPROCESS_DEVICE == "GPU"

    def test_hardware_resource_pooling(self):
        """Test that HARDWARE_UNITS pool is correctly populated."""
        with mock.patch.dict(os.environ, {"ASR_DEVICE": "AUTO"}):
            with mock.patch("ctranslate2.get_cuda_device_count", return_value=0):
                mock_core = mock.MagicMock()
                mock_core.available_devices = ["GPU.0", "NPU.0", "CPU"]
                mock_core.get_property.return_value = "Intel Accelerator"
                with mock.patch("openvino.Core", return_value=mock_core):
                    importlib.reload(config_module)
                    # Should have 2 units (GPU and NPU)
                    assert len(config_module.HARDWARE_UNITS) == 2
                    assert any(u["type"] == "GPU" for u in config_module.HARDWARE_UNITS)
                    assert any(u["type"] == "NPU" for u in config_module.HARDWARE_UNITS)

    def test_hardware_unit_limits(self):
        """Test that MAX_*_UNITS correctly limits the pool."""
        with mock.patch.dict(os.environ, {"MAX_GPU_UNITS": "1", "MAX_NPU_UNITS": "0"}):
            with mock.patch("ctranslate2.get_cuda_device_count", return_value=0):
                mock_core = mock.MagicMock()
                mock_core.available_devices = ["GPU.0", "GPU.1", "NPU.0", "CPU"]
                mock_core.get_property.return_value = "Intel Accelerator"
                with mock.patch("openvino.Core", return_value=mock_core):
                    importlib.reload(config_module)
                    # Should only have 1 GPU and 0 NPUs
                    assert len([u for u in config_module.HARDWARE_UNITS if u["type"] == "GPU"]) == 1
                    assert len([u for u in config_module.HARDWARE_UNITS if u["type"] == "NPU"]) == 0

    def test_max_cpu_units_logic(self):
        """Test MAX_CPU_UNITS parsing and impact on CPU_PARALLEL_LIMIT."""
        with mock.patch.dict(os.environ, {"MAX_CPU_UNITS": "2"}):
            importlib.reload(config_module)
            assert config_module.MAX_CPU == 2
            assert config_module.CPU_PARALLEL_LIMIT == 2

    def test_cpu_parallel_limit_auto_scaling(self):
        """Test auto-scaling of CPU_PARALLEL_LIMIT when MAX_CPU is AUTO."""
        with mock.patch.dict(
            os.environ,
            {"MAX_CPU_UNITS": "AUTO", "ASR_THREADS": "2", "ASR_PREPROCESS_THREADS": "2", "CPU_CORE_LIMIT": "8"},
        ):
            importlib.reload(config_module)
            # cores // max(threads) = 8 // 2 = 4
            assert config_module.CPU_PARALLEL_LIMIT == 4

    def test_cpu_parallel_limit_auto_zero_threads_safe(self):
        """MAX_CPU=AUTO with zero thread envs should not crash and should compute deterministically."""
        with mock.patch.dict(
            os.environ,
            {"MAX_CPU_UNITS": "AUTO", "ASR_THREADS": "0", "ASR_PREPROCESS_THREADS": "0", "CPU_CORE_LIMIT": "8"},
        ):
            importlib.reload(config_module)
            assert config_module.CPU_PARALLEL_LIMIT == 8

    def test_hardware_property_exception(self):
        """Test that hardware names fall back on property exception."""
        with mock.patch.dict(os.environ, {"ASR_DEVICE": "AUTO"}):
            with mock.patch("ctranslate2.get_cuda_device_count", return_value=0):
                mock_core = mock.MagicMock()
                mock_core.available_devices = ["GPU"]
                mock_core.get_property.side_effect = RuntimeError("Property fail")
                with mock.patch("openvino.Core", return_value=mock_core):
                    importlib.reload(config_module)
                    expected = ["CPU", "GPU", "NPU", "NVIDIA GPU"]
                    assert config_module.ASR_DEVICE_NAME in expected
                    assert config_module.PREPROCESS_DEVICE_NAME in expected

    def test_hardware_detection_logic_exception(self):
        """Test hardware detection handles exceptions gracefully."""
        with mock.patch.dict(os.environ, {"ASR_DEVICE": "AUTO"}):
            fail_patch = mock.patch(
                "ctranslate2.get_cuda_device_count",
                side_effect=RuntimeError("Hard fail"),
            )
            # Disable the real /dev/dri GPU-node fallback so this test's
            # outcome doesn't depend on whether the machine running it
            # happens to have a real Intel GPU device node present.
            with fail_patch, mock.patch("modules.core.config_helpers._can_use_gpu_node_fallback", return_value=False):
                importlib.reload(config_module)
                assert config_module.DEVICE == "CPU"

    def test_hardware_detection_logic_manual_override(self):
        """Test manual ASR_DEVICE override path."""
        with mock.patch.dict(os.environ, {"ASR_DEVICE": "CPU", "ASR_PREPROCESS_DEVICE": "GPU"}):
            importlib.reload(config_module)
            assert config_module.DEVICE == "CPU"
            assert config_module.PREPROCESS_DEVICE == "GPU"

    def test_system_model_detection(self):
        """Test detection of baked-in system models."""
        with mock.patch("os.path.exists", side_effect=lambda p: "/app/system_models" in p):
            with mock.patch("os.listdir", return_value=["file1"]):
                importlib.reload(config_module)
                assert "/app/system_models" in config_module.MODEL_ID
                assert "/app/system_models" in config_module.UVR_MODEL_DIR

    def test_custom_compute_type(self):
        """Test custom ASR_COMPUTE_TYPE override."""
        with mock.patch.dict(os.environ, {"ASR_COMPUTE_TYPE": "FLOAT32"}):
            importlib.reload(config_module)
            assert config_module.COMPUTE_TYPE == "float32"

    def test_hallucination_phrases_content(self):
        """Test some common phrases exist."""
        assert "thank you for watching" in config_module.HALLUCINATION_PHRASES
        assert "vă mulțumim pentru vizionare" in config_module.HALLUCINATION_PHRASES


def _is_cuda_preprocess_override_warning(call: Any, preprocess_value: str) -> bool:
    if not call.args:
        return False
    return "ASR_DEVICE=CUDA" in call.args[0] and "unsupported" in call.args[0] and preprocess_value in call.args


@pytest.mark.parametrize("preprocess_value", ["GPU", "NPU", "INTEL", "OPENVINO"])
def test_cuda_device_forces_preprocess_device_to_cuda(preprocess_value: str):
    """Without isolation, ASR_DEVICE=CUDA must still force PREPROCESS_DEVICE to CUDA.

    A CUDA context cannot safely share a process with an OpenVINO GPU/NPU context (see
    the guard comment above PREPROCESS_DEVICE's assignment in config.py), so when both
    would live in one interpreter the override still applies and still explains itself.
    The isolated case is covered by
    test_isolated_preprocessing_allows_cross_vendor_devices below.
    """
    original_environ = dict(os.environ)
    try:
        env = {"ASR_DEVICE": "CUDA", "ASR_PREPROCESS_DEVICE": preprocess_value, "ASR_ISOLATE_PREPROCESSING": "0"}
        with mock.patch.dict(os.environ, env):
            with mock.patch("ctranslate2.get_cuda_device_count", return_value=0):
                mock_core = mock.MagicMock()
                mock_core.available_devices = ["CPU"]
                with mock.patch("openvino.Core", return_value=mock_core):
                    with mock.patch("logging.getLogger") as mock_get_logger:
                        mock_logger = mock.MagicMock()
                        mock_get_logger.return_value = mock_logger
                        importlib.reload(config_module)
                        # Other unrelated warnings (e.g. directory-writability fallbacks)
                        # can also fire during a full config reload, so search for the
                        # specific CUDA-preprocessing-override warning rather than
                        # asserting it's the only one logged.
                        matching_calls = [
                            call
                            for call in mock_logger.warning.call_args_list
                            if _is_cuda_preprocess_override_warning(call, preprocess_value)
                        ]
                        assert (config_module.DEVICE, config_module.PREPROCESS_DEVICE, len(matching_calls)) == (
                            "CUDA",
                            "CUDA",
                            1,
                        )
    finally:
        # Reload restores env-derived module globals for later tests (same contract as
        # tests/config_reload_helpers.py:restore_config_after_reload).
        os.environ.clear()
        os.environ.update(original_environ)
        importlib.reload(config_module)


@pytest.mark.parametrize("preprocess_value", ["GPU", "NPU"])
def test_isolated_preprocessing_allows_cross_vendor_devices(preprocess_value: str):
    """With UVR out-of-process the cross-vendor restriction no longer applies.

    The crash it guards against needs both contexts in one interpreter. Once preprocessing
    has its own process, a CUDA ASR engine can coexist with Intel-GPU vocal separation --
    and equally with a ROCm one, which is the only way an AMD host gets accelerated UVR
    alongside a non-AMD ASR engine.
    """
    original_environ = dict(os.environ)
    try:
        env = {"ASR_DEVICE": "CUDA", "ASR_PREPROCESS_DEVICE": preprocess_value, "ASR_ISOLATE_PREPROCESSING": "1"}
        with mock.patch.dict(os.environ, env):
            with mock.patch("ctranslate2.get_cuda_device_count", return_value=0):
                mock_core = mock.MagicMock()
                mock_core.available_devices = ["CPU"]
                with mock.patch("openvino.Core", return_value=mock_core):
                    importlib.reload(config_module)

                    assert config_module.ISOLATE_PREPROCESSING is True
                    assert config_module.DEVICE == "CUDA"
                    assert config_module.PREPROCESS_DEVICE == preprocess_value, "isolation should preserve the requested device"
    finally:
        os.environ.clear()
        os.environ.update(original_environ)
        importlib.reload(config_module)
