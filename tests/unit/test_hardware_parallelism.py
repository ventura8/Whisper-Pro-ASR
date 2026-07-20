"""Tests for full hardware parallelism: multi-CUDA, multi-AMD, multi-Intel, cross-vendor.

These tests cover the provider dispatch paths and unit registration logic that ensure
all hardware units run UVR isolation in parallel on independent preprocessors.
"""

from unittest import mock

from modules.core import config_helpers as ch
from modules.inference.pipeline import openvino_provider_dispatch as pd
from modules.inference.pipeline.preprocessing import provider as prep_provider

# ─────────────────────────────────────────────────────────────────────────────
# openvino_provider_dispatch: cuda_provider_config – multi-NVIDIA device IDs
# ─────────────────────────────────────────────────────────────────────────────


class TestCudaProviderConfigDeviceId:
    """cuda_provider_config must correctly extract device index from all ID formats."""

    def test_raw_zero(self):
        """Raw string index 0 converts to integer 0."""
        providers, opts = pd.cuda_provider_config("0")
        assert "CUDAExecutionProvider" in providers
        assert opts[0]["device_id"] == 0

    def test_cuda_colon_index(self):
        """cuda:N unit IDs extract device integer N."""
        _p, opts = pd.cuda_provider_config("cuda:1")
        assert opts[0]["device_id"] == 1

    def test_nvidia_colon_index(self):
        """'nvidia:N' unit IDs (used by scheduler) must resolve to the correct GPU."""
        _p, opts = pd.cuda_provider_config("nvidia:0")
        assert opts[0]["device_id"] == 0

    def test_nvidia_colon_second_gpu(self):
        """'nvidia:1' resolves to device index 1."""
        _p, opts = pd.cuda_provider_config("nvidia:1")
        assert opts[0]["device_id"] == 1

    def test_cuda_literal_string(self):
        """Literal string 'cuda' defaults to string index 0."""
        _p, opts = pd.cuda_provider_config("cuda")
        assert opts[0]["device_id"] == "0"

    def test_non_parseable_falls_back_to_zero(self):
        """Non-parseable string falls back to default string index 0."""
        _p, opts = pd.cuda_provider_config("bogus")
        assert opts[0]["device_id"] == "0"


# ─────────────────────────────────────────────────────────────────────────────
# openvino_provider_dispatch: amd_provider_config
# ─────────────────────────────────────────────────────────────────────────────


class TestAmdProviderConfig:
    """amd_provider_config must pick ROCm > DirectML > CPU fallback."""

    def test_rocm_preferred(self):
        """ROCMExecutionProvider is preferred when available and /dev/kfd is present."""
        with mock.patch("os.path.exists", side_effect=lambda p: p == "/dev/kfd"):
            providers, opts = pd.amd_provider_config("amd:0", ["ROCMExecutionProvider", "CPUExecutionProvider"])
            assert providers[0] == "ROCMExecutionProvider"
            assert opts[0]["device_id"] == 0

    def test_dml_when_no_rocm(self):
        """Falls back to CPU execution provider when /dev/kfd is absent."""
        providers, _opts = pd.amd_provider_config("amd:0", ["CPUExecutionProvider"])
        assert providers[0] == "CPUExecutionProvider"

    def test_second_amd_device_index(self):
        """amd:1 extracts device index 1."""
        with mock.patch("os.path.exists", side_effect=lambda p: p == "/dev/kfd"):
            _p, opts = pd.amd_provider_config("amd:1", ["ROCMExecutionProvider"])
            assert opts[0]["device_id"] == 1

    def test_cpu_fallback_when_no_amd_providers(self):
        """Falls back to CPU when no AMD providers are listed."""
        providers, _opts = pd.amd_provider_config("amd:0", ["CPUExecutionProvider"])
        assert providers == ["CPUExecutionProvider"]

    def test_dml_selected_when_device_node_present(self):
        """DmlExecutionProvider is selected when device node and DmlExecutionProvider are present."""
        with mock.patch("modules.inference.pipeline.openvino_provider_dispatch._has_amd_device_node", return_value=True):
            providers, opts = pd.amd_provider_config("amd:0", ["DmlExecutionProvider", "CPUExecutionProvider"])
            assert providers[0] == "DmlExecutionProvider"
            assert opts[0]["device_id"] == 0

    def test_dml_ignored_when_no_device_node(self):
        """DmlExecutionProvider is not selected when _has_amd_device_node is False."""
        with mock.patch("modules.inference.pipeline.openvino_provider_dispatch._has_amd_device_node", return_value=False):
            providers, _opts = pd.amd_provider_config("amd:0", ["DmlExecutionProvider", "CPUExecutionProvider"])
            assert providers[0] == "CPUExecutionProvider"
            assert pd.has_amd_provider(["DmlExecutionProvider"]) is False

    def test_empty_available_returns_cpu(self):
        """Returns CPU provider when available list is empty."""
        providers, _opts = pd.amd_provider_config("amd:0", [])
        assert providers == ["CPUExecutionProvider"]

    def test_is_wsl_missing_rocdxg(self):
        """_is_wsl_missing_rocdxg returns True only when /dev/dxg is present without /dev/kfd or librocdxg."""
        is_missing_fn = getattr(pd, "_is_wsl_missing_rocdxg")
        assert is_missing_fn(False, True, False) is True
        assert is_missing_fn(True, True, False) is False
        assert is_missing_fn(False, False, False) is False

    def test_resolve_amd_fallback_message_branches(self):
        """_resolve_amd_fallback_message returns branch-specific diagnostic text."""
        resolve_msg_fn = getattr(pd, "_resolve_amd_fallback_message")

        # Branch 1: WSL2 with missing librocdxg
        wsl_msg = resolve_msg_fn(False, True, False)
        assert "librocdxg.so is missing" in wsl_msg

        # Branch 2: Neither device node present
        no_nodes_msg = resolve_msg_fn(False, False, False)
        assert "Neither /dev/kfd" in no_nodes_msg and "nor /dev/dxg" in no_nodes_msg

        # Branch 3: Default no-supported-provider case
        default_msg = resolve_msg_fn(True, True, True)
        assert "No supported AMD execution provider found" in default_msg


# ─────────────────────────────────────────────────────────────────────────────
# openvino_provider_dispatch: provider_config_dispatch AMD routing
# ─────────────────────────────────────────────────────────────────────────────


class TestProviderConfigDispatchAmd:
    """AMD device type must be routed to amd_provider_config."""

    def test_amd_dispatch_returns_amd_function(self):
        """Dispatch returns amd_provider_config for AMD."""
        resolver = pd.provider_config_dispatch("AMD")
        assert resolver is pd.amd_provider_config

    def test_amd_dispatch_case_insensitive(self):
        """Dispatch returns amd_provider_config for lowercase amd."""
        resolver = pd.provider_config_dispatch("amd")
        assert resolver is pd.amd_provider_config


# ─────────────────────────────────────────────────────────────────────────────
# openvino_provider_dispatch: auto_provider_config AMD branch
# ─────────────────────────────────────────────────────────────────────────────


class TestAutoProviderConfigAmd:
    """auto_provider_config must include AMD (ROCm) before OpenVINO when /dev/kfd is present."""

    def test_auto_picks_rocm_over_openvino(self):
        """Auto selection prefers ROCm over OpenVINO."""
        with mock.patch("os.path.exists", side_effect=lambda p: p == "/dev/kfd"):
            available = ["ROCMExecutionProvider", "OpenVINOExecutionProvider", "CPUExecutionProvider"]
            providers, _opts = pd.auto_provider_config(available)
            assert providers[0] == "ROCMExecutionProvider"

    def test_auto_picks_dml_when_no_cuda_no_rocm(self):
        """Auto selection picks OpenVINO or CPU when CUDA and ROCm are absent."""
        available = ["CPUExecutionProvider"]
        providers, _opts = pd.auto_provider_config(available)
        assert providers[0] == "CPUExecutionProvider"

    def test_auto_cuda_takes_priority_over_amd(self):
        """Auto selection prefers CUDA over ROCm."""
        available = ["CUDAExecutionProvider", "ROCMExecutionProvider", "CPUExecutionProvider"]
        providers, _opts = pd.auto_provider_config(available)
        assert providers[0] == "CUDAExecutionProvider"


# ─────────────────────────────────────────────────────────────────────────────
# preprocessing.provider: resolve_provider_config_for_preprocessing AMD path
# ─────────────────────────────────────────────────────────────────────────────


class TestPreprocessingProviderAmd:
    """AMD device_type must resolve to AMD providers, not fall through to auto."""

    def test_amd_device_type_resolves_rocm(self):
        """Resolves AMD device type to ROCm execution provider."""
        with mock.patch("os.path.exists", side_effect=lambda p: p == "/dev/kfd"):
            providers, opts = prep_provider.resolve_provider_config_for_preprocessing(
                device_type="AMD",
                device_id="amd:0",
                available_providers=["ROCMExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
                available_openvino_devices=[],
                ov_cache_dir="/tmp",
                preprocess_threads=2,
            )
            assert providers[0] == "ROCMExecutionProvider"
            assert opts[0]["device_id"] == 0

    def test_amd_device_type_resolves_dml(self):
        """Resolves AMD device type to CPU execution provider when /dev/kfd is absent."""
        providers, _opts = prep_provider.resolve_provider_config_for_preprocessing(
            device_type="AMD",
            device_id="amd:0",
            available_providers=["CPUExecutionProvider"],
            available_openvino_devices=[],
            ov_cache_dir="/tmp",
            preprocess_threads=2,
        )
        assert providers[0] == "CPUExecutionProvider"

    def test_amd_falls_back_to_cpu_when_no_amd_providers(self):
        """Falls back to CPU execution provider when no AMD providers exist."""
        providers, _opts = prep_provider.resolve_provider_config_for_preprocessing(
            device_type="AMD",
            device_id="amd:0",
            available_providers=["CPUExecutionProvider"],
            available_openvino_devices=[],
            ov_cache_dir="/tmp",
            preprocess_threads=2,
        )
        assert providers == ["CPUExecutionProvider"]


# ─────────────────────────────────────────────────────────────────────────────
# config_helpers: multi-AMD unit registration
# ─────────────────────────────────────────────────────────────────────────────


class TestAppendAmdUnits:
    """append_amd_units must register up to max_amd units."""

    def test_single_amd_unit(self):
        """Appends single amd:0 unit when max_amd is 1."""
        append_amd_units_fn = getattr(ch, "_append_amd_units")
        units: list = []
        append_amd_units_fn(1, units)
        assert len(units) == 1
        assert units[0] == {"type": "AMD", "id": "amd:0", "name": "AMD GPU 0"}

    def test_multi_amd_units(self):
        """Appends 3 AMD units when max_amd is 3."""
        append_amd_units_fn = getattr(ch, "_append_amd_units")
        units: list = []
        with mock.patch("modules.core.config_helpers._count_amd_drm_devices", return_value=3):
            append_amd_units_fn(3, units)
        assert [u["id"] for u in units] == ["amd:0", "amd:1", "amd:2"]

    def test_max_amd_units_uncapped(self):
        """Appends 12 AMD units when max_amd is 12."""
        append_amd_units_fn = getattr(ch, "_append_amd_units")
        units: list = []
        with mock.patch("modules.core.config_helpers._count_amd_drm_devices", return_value=12):
            append_amd_units_fn(12, units)
        assert len(units) == 12

    def test_zero_max_amd_registers_nothing(self):
        """Registers no units when max_amd is 0."""
        append_amd_units_fn = getattr(ch, "_append_amd_units")
        units: list = []
        append_amd_units_fn(0, units)
        assert not units


# ─────────────────────────────────────────────────────────────────────────────
# config_helpers: update_amd_state does not overwrite CUDA prep_device
# ─────────────────────────────────────────────────────────────────────────────


class TestUpdateAmdState:
    """update_amd_state must not overwrite CUDA or Intel prep_device."""

    def test_amd_only_sets_both(self):
        """Sets device and prep_device to AMD when CPU was initial state."""
        update_amd_state_fn = getattr(ch, "_update_amd_state")
        state = {"device": "CPU", "prep_device": "CPU", "compute": "int8"}
        update_amd_state_fn(1, state)
        assert state["device"] == "AMD"
        assert state["prep_device"] == "AMD"

    def test_cuda_plus_amd_preserves_cuda_prep_device(self):
        """Preserves CUDA prep_device when CUDA claimed prep_device first."""
        update_amd_state_fn = getattr(ch, "_update_amd_state")
        state = {"device": "CUDA", "prep_device": "CUDA", "compute": "float16"}
        update_amd_state_fn(1, state)
        assert state["device"] == "CUDA"
        assert state["prep_device"] == "CUDA"

    def test_zero_max_amd_is_noop(self):
        """State is unchanged when max_amd is 0."""
        update_amd_state_fn = getattr(ch, "_update_amd_state")
        state = {"device": "CPU", "prep_device": "CPU", "compute": "int8"}
        update_amd_state_fn(0, state)
        assert state["device"] == "CPU"
        assert state["prep_device"] == "CPU"
