"""Tests for AMD sysfs/smi helpers in modules/monitoring/metrics_amd.py."""

from unittest import mock

from modules.core import process_exec
from modules.monitoring import metrics_amd, metrics_discovery


def test_extract_int_from_csv_line_uses_first_digit_column():
    """When the last CSV column is not numeric, the first digit column is used."""
    assert metrics_amd._extract_int_from_csv_line("55, n/a") == 55
    assert metrics_amd._extract_int_from_csv_line("foo, bar") is None


def test_probe_amd_smi_metrics_missing_binaries_returns_empty():
    """No rocm-smi/amd-smi on PATH yields an empty utilization list."""
    with mock.patch("modules.monitoring.metrics_amd.which", return_value=None):
        assert metrics_amd._probe_amd_smi_metrics() == []


def test_probe_amd_smi_metrics_command_error_returns_empty():
    """CLI probe failures must not raise out of the metrics helper."""
    with (
        mock.patch("modules.monitoring.metrics_amd.which", return_value="/usr/bin/rocm-smi"),
        mock.patch(
            "modules.core.process_exec.check_output_text",
            side_effect=process_exec.CommandExecutionError(["rocm-smi"], 1),
        ),
    ):
        assert metrics_amd._probe_amd_smi_metrics() == []


def test_amd_vendor_busy_paths_skips_unreadable_vendor_files():
    """Unreadable DRM vendor nodes are skipped instead of aborting the probe."""
    with (
        mock.patch(
            "modules.monitoring.metrics_amd.glob.glob",
            return_value=["/sys/class/drm/card0/device/vendor"],
        ),
        mock.patch("builtins.open", side_effect=OSError("denied")),
    ):
        assert metrics_amd._amd_vendor_busy_paths() == []


def test_resolve_amd_utilization_inactive_returns_zero():
    """Inactive AMD unit: _inactive_accelerator_zero_result zeroes result and clears held sample."""
    with (
        mock.patch("modules.inference.scheduler.get_service_stats_minimal", return_value={"active_tasks": []}),
        mock.patch("modules.inference.runtime.model_manager.PREPROCESSOR_POOL", {}),
    ):
        assert metrics_discovery._resolve_unit_utilization("AMD", "amd:0") == 0


def test_resolve_amd_utilization_active_task_returns_100():
    """Active AMD task without sysfs: _inactive_accelerator_zero_result returns None -> fallback returns 100."""
    active_amd_task = [{"unit_type": "AMD", "unit_id": "amd:0", "stage": "Vocal Isolation"}]
    with (
        mock.patch("modules.inference.scheduler.get_service_stats_minimal", return_value={"active_tasks": active_amd_task}),
        mock.patch("modules.inference.runtime.model_manager.PREPROCESSOR_POOL", {}),
        mock.patch("modules.monitoring.metrics_discovery._read_first_int_value", return_value=None),
        mock.patch("modules.monitoring.amd_utilization_helpers._probe_amd_smi_metrics", return_value=[]),
    ):
        assert metrics_discovery._resolve_unit_utilization("AMD", "amd:0") == 100


def test_resolve_amd_utilization_sysfs_success():
    """Active AMD task with sysfs metric: reports real sysfs utilization."""
    active_amd_task = [{"unit_type": "AMD", "unit_id": "amd:0", "stage": "Vocal Isolation"}]
    with (
        mock.patch("modules.inference.scheduler.get_service_stats_minimal", return_value={"active_tasks": active_amd_task}),
        mock.patch("modules.inference.runtime.model_manager.PREPROCESSOR_POOL", {}),
        mock.patch("modules.monitoring.metrics_discovery._read_first_int_value", return_value=42),
        mock.patch("modules.monitoring.amd_utilization_helpers._probe_amd_smi_metrics", return_value=[]),
    ):
        assert metrics_discovery._resolve_unit_utilization("AMD", "amd:0") == 42


def test_resolve_amd_utilization_preprocessor_lock_returns_100():
    """Locked AMD preprocessor with no active tasks: _probe_activity_fallback reports 100 via lock path."""
    pm = mock.MagicMock()
    pm.device_type = "AMD"
    pm.device_id = "amd:0"
    pm.unit = {"id": "amd:0", "name": "AMD Radeon"}
    pm.lock.locked.return_value = True

    with (
        mock.patch("modules.inference.scheduler.get_service_stats_minimal", return_value={"active_tasks": []}),
        mock.patch.dict("modules.inference.runtime.model_manager.PREPROCESSOR_POOL", {"amd:0": pm}, clear=True),
        mock.patch("modules.monitoring.metrics_discovery._read_first_int_value", return_value=None),
        mock.patch("modules.monitoring.amd_utilization_helpers._probe_amd_smi_metrics", return_value=[]),
    ):
        assert metrics_discovery._resolve_unit_utilization("AMD", "amd:0") == 100


def test_resolve_unit_utilization_amd():
    """_resolve_unit_utilization dispatching for 'AMD' calls resolve_amd_utilization."""
    active_amd_task = [{"unit_type": "AMD", "unit_id": "amd:0", "stage": "Vocal Isolation"}]
    with (
        mock.patch("modules.inference.scheduler.get_service_stats_minimal", return_value={"active_tasks": active_amd_task}),
        mock.patch("modules.inference.runtime.model_manager.PREPROCESSOR_POOL", {}),
        mock.patch("modules.monitoring.metrics_discovery._read_first_int_value", return_value=None),
        mock.patch("modules.monitoring.amd_utilization_helpers._probe_amd_smi_metrics", return_value=[]),
    ):
        assert metrics_discovery._resolve_unit_utilization("AMD", "amd:0") == 100


def test_probe_amd_smi_metrics_rocm_smi():
    """_probe_amd_smi_metrics correctly parses rocm-smi CSV output."""
    csv_output = "device, GPU use (%)\ncard0, 75%\ncard1, 12%"
    with (
        mock.patch("modules.monitoring.metrics_amd.which", side_effect=lambda x: "/usr/bin/rocm-smi" if x == "rocm-smi" else None),
        mock.patch("modules.core.process_exec.check_output_text", return_value=csv_output) as mock_exec,
    ):
        loads = metrics_amd._probe_amd_smi_metrics()
        assert loads == [75, 12]
        mock_exec.assert_called_once_with(["/usr/bin/rocm-smi", "--showuse", "--csv"], timeout=5.0)


def test_probe_amd_smi_metrics_amd_smi():
    """_probe_amd_smi_metrics correctly invokes amd-smi metric arguments and parses CSV."""
    csv_output = "gpu,usage\n0, 88\n"
    with (
        mock.patch("modules.monitoring.metrics_amd.which", side_effect=lambda x: "/usr/bin/amd-smi" if x == "amd-smi" else None),
        mock.patch("modules.core.process_exec.check_output_text", return_value=csv_output) as mock_exec,
    ):
        loads = metrics_amd._probe_amd_smi_metrics()
        assert loads == [88]
        mock_exec.assert_called_once_with(["/usr/bin/amd-smi", "metric", "--usage", "--csv"], timeout=5.0)


def test_fetch_single_amd_gpu_load_from_cli():
    """resolve_amd_utilization uses CLI metric when index is in range."""
    active_amd_task = [{"unit_type": "AMD", "unit_id": "amd:0", "stage": "Vocal Isolation"}]
    with (
        mock.patch("modules.inference.scheduler.get_service_stats_minimal", return_value={"active_tasks": active_amd_task}),
        mock.patch("modules.inference.runtime.model_manager.PREPROCESSOR_POOL", {}),
        mock.patch("modules.monitoring.amd_utilization_helpers._probe_amd_smi_metrics", return_value=[65]),
    ):
        assert metrics_discovery._resolve_unit_utilization("AMD", "amd:0") == 65


def test_amd_sysfs_paths_maps_vendor_cards_by_index():
    """AMD sysfs probes must map AMD-vendor cards and skip Intel DRM nodes."""
    vendor_files = {
        "/sys/class/drm/card0/device/vendor": "0x8086\n",
        "/sys/class/drm/card1/device/vendor": "0x1002\n",
    }

    def _open_vendor(path, *_args, **_kwargs):
        return mock.mock_open(read_data=vendor_files[path])()

    with (
        mock.patch("modules.monitoring.metrics_amd.glob.glob", return_value=list(vendor_files)),
        mock.patch("builtins.open", side_effect=_open_vendor),
    ):
        assert metrics_amd._amd_sysfs_paths(0) == ["/sys/class/drm/card1/device/gpu_busy_percent"]
        assert metrics_amd._amd_sysfs_paths(1) == []


def test_preprocessor_accelerated_recognizes_amd_providers():
    """Locked AMD separators using ROCm/DirectML count as accelerated."""

    class _Sep:
        onnx_execution_provider = ["ROCMExecutionProvider"]

    class _Pm:
        separator = _Sep()

    assert metrics_discovery._is_preprocessor_accelerated(_Pm()) is True

    class _DmlSep:
        onnx_execution_provider = ["DmlExecutionProvider"]

    class _DmlPm:
        separator = _DmlSep()

    assert metrics_discovery._is_preprocessor_accelerated(_DmlPm()) is True


def test_is_task_using_accelerator_amd_asr_is_not_gpu_work():
    """AMD transcription is CPU-side and must not mark the AMD GPU busy."""
    assert metrics_discovery._is_task_using_accelerator({"stage": "Transcribing (Seg 1)"}, "AMD") is False
    assert metrics_discovery._is_task_using_accelerator({"stage": "Vocal isolation"}, "AMD") is True
