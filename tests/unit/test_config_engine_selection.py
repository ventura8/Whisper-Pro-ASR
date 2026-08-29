"""Which ASR engine config resolves, on what hardware, and what may override it.

Split out of test_config.py, which had grown past the project's module-length limit. These
cover engine selection (AUTO vs explicit), the Intel-hardware requirement, hybrid per-unit
engines, and the model-path redirection each engine needs -- everything that decides *what
runs where*, as opposed to the environment parsing and directory resolution left behind.
"""

import importlib
import os
from unittest import mock

import pytest

import modules.core.config as config_module
from modules.core import engine_registry
from tests.config_reload_helpers import npu_cannot_execute, reloaded_config

pytestmark = pytest.mark.usefixtures("restore_config_after_reload")


class TestEngineSelection:
    """AUTO resolution, explicit requests, and the hardware each implies."""

    def test_asr_engine_auto_prefers_cuda_over_intel(self):
        """ASR_ENGINE=AUTO should resolve to FASTER-WHISPER when CUDA is available."""
        env = {"ASR_ENGINE": "AUTO", "ASR_DEVICE": "AUTO"}
        with mock.patch.dict(os.environ, env):
            with mock.patch("ctranslate2.get_cuda_device_count", return_value=1):
                mock_core = mock.MagicMock()
                mock_core.available_devices = ["GPU.0", "NPU.0"]
                with mock.patch("openvino.Core", return_value=mock_core):
                    importlib.reload(config_module)
                    assert config_module.ASR_ENGINE == "FASTER-WHISPER"
                    assert config_module.DEVICE == "CUDA"

    def test_asr_engine_defaults_to_auto_and_resolves_to_faster_whisper_on_intel(self):
        """An unset ASR_ENGINE means AUTO, and AUTO is FASTER-WHISPER on every host.

        Intel hardware used to change the answer to INTEL-WHISPER, which made the engine --
        and therefore the transcript -- a property of the machine a request landed on. The
        Intel engine is still available; it has to be asked for.
        """
        env = {"ASR_DEVICE": "AUTO"}
        with mock.patch.dict(os.environ, env):
            os.environ.pop("ASR_ENGINE", None)
            with mock.patch("ctranslate2.get_cuda_device_count", return_value=0):
                mock_core = mock.MagicMock()
                mock_core.available_devices = ["GPU.0", "CPU"]
                with mock.patch("openvino.Core", return_value=mock_core):
                    importlib.reload(config_module)
                    assert config_module.ASR_ENGINE_SOURCE == "auto"
                    assert config_module.ASR_ENGINE == "FASTER-WHISPER"

    def test_asr_engine_default_keeps_faster_whisper_on_nvidia(self):
        """The AUTO default must not move NVIDIA hosts off Faster-Whisper.

        This covers the hybrid nvidia-intel and full images, where an Intel iGPU is visible
        alongside CUDA and must not win the engine choice.
        """
        env = {"ASR_DEVICE": "AUTO"}
        with mock.patch.dict(os.environ, env):
            os.environ.pop("ASR_ENGINE", None)
            with mock.patch("ctranslate2.get_cuda_device_count", return_value=1):
                mock_core = mock.MagicMock()
                mock_core.available_devices = ["GPU.0", "CPU"]
                with mock.patch("openvino.Core", return_value=mock_core):
                    importlib.reload(config_module)
                    assert config_module.ASR_ENGINE == "FASTER-WHISPER"
                    assert config_module.DEVICE == "CUDA"

    def test_explicit_faster_whisper_still_wins_on_intel_hardware(self):
        """Asking for Faster-Whisper in compose must override the Intel default."""
        env = {"ASR_ENGINE": "FASTER-WHISPER", "ASR_DEVICE": "AUTO"}
        with mock.patch.dict(os.environ, env):
            with mock.patch("ctranslate2.get_cuda_device_count", return_value=0):
                mock_core = mock.MagicMock()
                mock_core.available_devices = ["GPU.0", "CPU"]
                with mock.patch("openvino.Core", return_value=mock_core):
                    importlib.reload(config_module)
                    assert config_module.ASR_ENGINE_SOURCE == "explicit"
                    assert config_module.ASR_ENGINE == "FASTER-WHISPER"

    def test_empty_asr_engine_is_treated_as_auto(self):
        """`ASR_ENGINE=` in an env file must not fail validation."""
        with reloaded_config({"ASR_ENGINE": "", "ASR_DEVICE": "AUTO"}) as reloaded:
            assert reloaded.ASR_ENGINE_SOURCE == "auto"
            assert reloaded.ASR_ENGINE == "FASTER-WHISPER"

    def test_hybrid_engines_enabled_when_both_vendors_present(self):
        """Asked for on a CUDA+Intel host, each unit runs its native engine in its own worker.

        HYBRID_ENGINES is opt-in: it makes the engine depend on which unit a request lands
        on, which is exactly what the single default engine exists to avoid. It has to be
        requested, and this pins that requesting it works.
        """
        env = {"ASR_DEVICE": "AUTO", "HYBRID_ENGINES": "true"}
        with mock.patch.dict(os.environ, env):
            os.environ.pop("ASR_ENGINE", None)
            with mock.patch("ctranslate2.get_cuda_device_count", return_value=1):
                mock_core = mock.MagicMock()
                mock_core.available_devices = ["GPU.0", "CPU"]
                with mock.patch("openvino.Core", return_value=mock_core):
                    importlib.reload(config_module)

                    assert config_module.HYBRID_ENGINES is True
                    assert config_module.engine_for_unit({"type": "CUDA"}) == "FASTER-WHISPER"
                    assert config_module.engine_for_unit({"type": "GPU"}) == "INTEL-WHISPER"
                    assert config_module.engine_for_unit({"type": "NPU"}) == "INTEL-WHISPER"
                    assert set(config_module.engines_in_use()) == {"FASTER-WHISPER", "INTEL-WHISPER"}

    def test_hybrid_gives_each_engine_its_own_weights(self):
        """CT2 and OpenVINO read different formats; each unit must get the right one."""
        env = {"ASR_DEVICE": "AUTO", "HYBRID_ENGINES": "true"}
        with mock.patch.dict(os.environ, env):
            os.environ.pop("ASR_ENGINE", None)
            with mock.patch("ctranslate2.get_cuda_device_count", return_value=1):
                mock_core = mock.MagicMock()
                mock_core.available_devices = ["GPU.0", "CPU"]
                with mock.patch("openvino.Core", return_value=mock_core):
                    importlib.reload(config_module)

                    ct2 = config_module.model_id_for_engine("FASTER-WHISPER")
                    ov = config_module.model_id_for_engine("INTEL-WHISPER")
                    assert ct2 != ov
                    assert ct2.endswith("whisper")
                    assert ov.endswith("whisper-openvino")

    def test_hybrid_disabled_when_engine_is_explicit(self):
        """An explicit ASR_ENGINE is an instruction, not a hint; honour it on every unit."""
        env = {"ASR_ENGINE": "FASTER-WHISPER", "ASR_DEVICE": "AUTO"}
        with mock.patch.dict(os.environ, env):
            with mock.patch("ctranslate2.get_cuda_device_count", return_value=1):
                mock_core = mock.MagicMock()
                mock_core.available_devices = ["GPU.0", "CPU"]
                with mock.patch("openvino.Core", return_value=mock_core):
                    importlib.reload(config_module)

                    assert config_module.HYBRID_ENGINES is False
                    assert config_module.engine_for_unit({"type": "GPU"}) == "FASTER-WHISPER"

    def test_hybrid_disabled_without_isolation(self):
        """Both engines in one process is the documented CUDA/OpenVINO crash."""
        env = {"ASR_DEVICE": "AUTO", "ASR_ISOLATE_ENGINES": "0"}
        with mock.patch.dict(os.environ, env):
            os.environ.pop("ASR_ENGINE", None)
            with mock.patch("ctranslate2.get_cuda_device_count", return_value=1):
                mock_core = mock.MagicMock()
                mock_core.available_devices = ["GPU.0", "CPU"]
                with mock.patch("openvino.Core", return_value=mock_core):
                    importlib.reload(config_module)

                    assert config_module.ISOLATE_ENGINES is False
                    assert config_module.HYBRID_ENGINES is False

    def test_hybrid_disabled_for_custom_weights(self):
        """A custom ASR_MODEL cannot be assumed to exist in both CT2 and OpenVINO form."""
        env = {"ASR_DEVICE": "AUTO", "ASR_MODEL": "openai/whisper-tiny"}
        with mock.patch.dict(os.environ, env):
            os.environ.pop("ASR_ENGINE", None)
            with mock.patch("ctranslate2.get_cuda_device_count", return_value=1):
                mock_core = mock.MagicMock()
                mock_core.available_devices = ["GPU.0", "CPU"]
                with mock.patch("openvino.Core", return_value=mock_core):
                    importlib.reload(config_module)

                    assert config_module.HYBRID_ENGINES is False

    def test_asr_engine_auto_prefers_intel_gpu_over_npu(self):
        """The device tier is still hardware-ordered: GPU outranks NPU. The engine is not."""
        env = {"ASR_ENGINE": "AUTO", "ASR_DEVICE": "AUTO"}
        with mock.patch.dict(os.environ, env):
            with mock.patch("ctranslate2.get_cuda_device_count", return_value=0):
                mock_core = mock.MagicMock()
                mock_core.available_devices = ["NPU.0", "GPU.0", "CPU"]
                with mock.patch("openvino.Core", return_value=mock_core):
                    importlib.reload(config_module)
                    assert config_module.ASR_ENGINE == "FASTER-WHISPER"
                    # GPU wins the tier, but CTranslate2 cannot drive it, so the ASR device
                    # claim settles on the CPU rather than naming a device it cannot use.
                    assert config_module.DEVICE == "CPU"

    def test_asr_engine_auto_on_an_npu_only_host(self):
        """AUTO resolves the default engine, and ASR runs on the CPU."""
        env = {"ASR_ENGINE": "AUTO", "ASR_DEVICE": "AUTO"}
        with mock.patch.dict(os.environ, env):
            with mock.patch("ctranslate2.get_cuda_device_count", return_value=0):
                mock_core = mock.MagicMock()
                mock_core.available_devices = ["NPU.0", "CPU"]
                with mock.patch("openvino.Core", return_value=mock_core), npu_cannot_execute():
                    importlib.reload(config_module)
                    assert config_module.ASR_ENGINE == "FASTER-WHISPER"
                    # ASR on the CPU, and the NPU kept for preprocessing, which it does run.
                    assert config_module.DEVICE == "CPU"
                    assert config_module.PREPROCESS_DEVICE == "NPU"

    def test_asr_engine_invalid_value_fails_fast(self):
        """Invalid ASR_ENGINE values should fail startup with a clear error."""
        with mock.patch.dict(os.environ, {"ASR_ENGINE": "INVALID-ENGINE"}):
            with pytest.raises(ValueError, match="Invalid ASR_ENGINE"):
                importlib.reload(config_module)


class TestConfigHardwareIntelResolution:
    """Config hardware tests for Intel model/device resolution branches."""

    def test_intel_engine_redirection(self):
        """Test that MODEL_ID redirects for INTEL-WHISPER."""
        env = {"ASR_ENGINE": "INTEL-WHISPER", "ASR_MODEL": "Systran/faster-whisper-large-v3"}
        with mock.patch.dict(os.environ, env):
            with mock.patch("ctranslate2.get_cuda_device_count", return_value=0):
                mock_core = mock.MagicMock()
                mock_core.available_devices = ["GPU.0"]
                with mock.patch("openvino.Core", return_value=mock_core):
                    with mock.patch("os.path.exists", side_effect=lambda p: "whisper-openvino" in p):
                        importlib.reload(config_module)
                        assert "whisper-openvino" in config_module.MODEL_ID

    def test_intel_engine_falls_back_to_cache_dir(self):
        """INTEL-WHISPER without a baked model must resolve to the provisioning cache dir.

        This previously resolved to the bare string "OpenVINO", a sentinel handed straight
        to ov_genai.WhisperPipeline with no download behind it.
        """
        env = {"ASR_ENGINE": "INTEL-WHISPER", "ASR_MODEL": "Systran/faster-whisper-large-v3"}
        with mock.patch.dict(os.environ, env):
            with mock.patch("ctranslate2.get_cuda_device_count", return_value=0):
                mock_core = mock.MagicMock()
                mock_core.available_devices = ["GPU.0"]
                with mock.patch("openvino.Core", return_value=mock_core):
                    with mock.patch("os.path.exists", return_value=False):
                        importlib.reload(config_module)
                        assert "whisper-openvino" in config_module.MODEL_ID
                        assert config_module.MODEL_ID != "OpenVINO"

    def test_intel_engine_explicit_falls_back_to_faster_without_intel_hardware(self):
        """INTEL-WHISPER should resolve to Faster-Whisper when no Intel GPU/NPU is detected."""
        env = {"ASR_ENGINE": "INTEL-WHISPER", "ASR_DEVICE": "AUTO"}
        with mock.patch.dict(os.environ, env):
            with mock.patch("ctranslate2.get_cuda_device_count", return_value=0):
                with mock.patch("openvino.Core", side_effect=RuntimeError("No Intel accel")):
                    # Disable the real /dev/dri GPU-node fallback so this test's
                    # outcome doesn't depend on whether the machine running it
                    # happens to have a real Intel GPU device node present.
                    with mock.patch("modules.core.config_helpers._can_use_gpu_node_fallback", return_value=False):
                        importlib.reload(config_module)
                        assert config_module.ASR_ENGINE == "FASTER-WHISPER"

    def test_intel_engine_explicit_auto_device_prefers_intel_hardware(self):
        """Explicit INTEL-WHISPER with ASR_DEVICE=AUTO should propagate detected Intel device."""
        env = {"ASR_ENGINE": "INTEL-WHISPER", "ASR_DEVICE": "AUTO"}
        with mock.patch.dict(os.environ, env):
            with mock.patch("ctranslate2.get_cuda_device_count", return_value=0):
                mock_core = mock.MagicMock()
                mock_core.available_devices = ["GPU.0", "CPU"]
                with mock.patch("openvino.Core", return_value=mock_core):
                    importlib.reload(config_module)
                    assert config_module.ASR_ENGINE == "INTEL-WHISPER"
                    assert config_module.DEVICE == "GPU"


def test_engine_registry_validation_paths():
    """Engine registry should validate supported values and hardware unit input."""
    assert engine_registry.normalize_and_validate_engine("faster-whisper") == engine_registry.ENGINE_FASTER_WHISPER

    with pytest.raises(ValueError, match="Supported values"):
        engine_registry.normalize_and_validate_engine("AUTO")

    with pytest.raises(ValueError, match="non-empty list"):
        engine_registry.resolve_auto_engine([])

    with pytest.raises(ValueError, match="type"):
        engine_registry.resolve_auto_engine([{"id": "CPU"}])

    assert engine_registry.resolve_auto_engine([{"type": "CPU"}]) == (
        engine_registry.ENGINE_FASTER_WHISPER,
        "CPU",
    )
    assert engine_registry.resolve_auto_engine([{"type": "GPU"}, {"type": "CPU"}], "CPU") == (
        engine_registry.ENGINE_FASTER_WHISPER,
        "CPU",
    )
    assert engine_registry.resolve_auto_device([{"type": "OTHER"}]) == "CPU"


class TestHybridIsOffByDefault:
    """A dual-vendor host must not silently run two different engines.

    Hybrid mode was previously enabled wherever it was possible, which made the engine --
    and therefore the decoding behaviour and the transcript -- depend on which accelerator
    the scheduler happened to pick for a request.
    """

    def test_both_vendors_present_but_hybrid_not_requested(self):
        """Both accelerators present is not a request; every unit runs the one resolved engine."""
        env = {"ASR_DEVICE": "AUTO"}
        with mock.patch.dict(os.environ, env):
            os.environ.pop("ASR_ENGINE", None)
            os.environ.pop("HYBRID_ENGINES", None)
            with mock.patch("ctranslate2.get_cuda_device_count", return_value=1):
                mock_core = mock.MagicMock()
                mock_core.available_devices = ["GPU.0", "CPU"]
                with mock.patch("openvino.Core", return_value=mock_core):
                    importlib.reload(config_module)

                    assert config_module.HYBRID_ENGINES is False
                    assert config_module.engine_for_unit({"type": "CUDA"}) == "FASTER-WHISPER"
                    assert config_module.engine_for_unit({"type": "GPU"}) == "FASTER-WHISPER"
                    assert config_module.engines_in_use() == ["FASTER-WHISPER"]

    def test_requesting_it_on_a_single_vendor_host_is_ignored(self):
        """The hard requirement is real hardware, not a preference; asking cannot conjure it."""
        env = {"ASR_DEVICE": "AUTO", "HYBRID_ENGINES": "true"}
        with mock.patch.dict(os.environ, env):
            os.environ.pop("ASR_ENGINE", None)
            with mock.patch("ctranslate2.get_cuda_device_count", return_value=1):
                mock_core = mock.MagicMock()
                mock_core.available_devices = ["CPU"]
                with mock.patch("openvino.Core", return_value=mock_core):
                    importlib.reload(config_module)

                    assert config_module.HYBRID_ENGINES is False
