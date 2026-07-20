"""Tests for modules/monitoring/metrics_discovery.py."""

from unittest import mock

import pytest

from modules.core import process_exec
from modules.monitoring import metrics_discovery


@pytest.fixture(autouse=True)
def clear_metric_cache():
    """Ensure a fresh cache for each test."""
    with metrics_discovery._CACHE_LOCK:
        metrics_discovery._METRIC_CACHE.clear()
        metrics_discovery._LAST_REAL_ACCEL_SAMPLES.clear()
    yield


def test_get_nvidia_metrics_success():
    """Test successful NVIDIA metrics retrieval."""
    mock_output = "25, 1024, 8192\n\n30, 2048, 8192"
    with (
        mock.patch("modules.monitoring.metrics_discovery.which", return_value="nvidia-smi"),
        mock.patch("modules.core.process_exec.check_output_text", return_value=mock_output),
    ):
        metrics = metrics_discovery.get_nvidia_metrics()
        assert len(metrics) == 2
        assert metrics[0]["util"] == 25
        assert metrics[1]["util"] == 30


def test_get_nvidia_metrics_failure():
    """Test handling of failed nvidia-smi call."""
    with mock.patch("modules.core.process_exec.check_output_text", side_effect=FileNotFoundError()):
        metrics = metrics_discovery.get_nvidia_metrics()
        assert metrics == []


def test_get_intel_gpu_load_windows_success():
    """Test Intel GPU load on Windows via PowerShell mock."""
    with mock.patch("platform.system", return_value="Windows"):
        with mock.patch("modules.monitoring.metrics_discovery._resolve_windows_powershell", return_value="powershell"):
            with mock.patch("modules.core.process_exec.check_output_text", return_value="45.5\n"):
                with mock.patch("glob.glob", return_value=[]):
                    with mock.patch("modules.inference.scheduler.get_service_stats_minimal", return_value={"active_tasks": []}):
                        load = metrics_discovery.get_intel_gpu_load()
                        assert load == 0


def test_get_intel_gpu_load_windows_success_with_active_gpu_task():
    """Test Intel GPU load parse-success path on Windows when app has active GPU work."""
    mock_stats = {
        "active_tasks": [
            {
                "unit_type": "GPU",
                "unit_id": "GPU.0",
                "unit_name": "Intel Arc",
                "stage": "Vocal Isolation",
            }
        ]
    }
    with mock.patch("platform.system", return_value="Windows"):
        with mock.patch("modules.monitoring.metrics_discovery._resolve_windows_powershell", return_value="powershell"):
            with mock.patch("modules.core.process_exec.check_output_text", return_value="45.5\n"):
                with mock.patch("glob.glob", return_value=[]):
                    with mock.patch("modules.inference.scheduler.get_service_stats_minimal", return_value=mock_stats):
                        load = metrics_discovery.get_intel_gpu_load()
                        assert load == 45


def test_get_intel_gpu_load_sysfs_fail():
    """Test Intel GPU load when sysfs exists but read fails."""
    with mock.patch("glob.glob", return_value=["/sys/class/drm/card0/device/gpu_busy_percent"]):
        with mock.patch("builtins.open", side_effect=IOError("Read fail")):
            with mock.patch("platform.system", return_value="Linux"):
                load = metrics_discovery.get_intel_gpu_load()
                assert load == 0


def test_get_npu_load_sysfs_fail():
    """Test NPU load when sysfs exists but read fails."""
    with mock.patch("glob.glob", return_value=["/sys/class/accel/accel0/device/utilization"]):
        with mock.patch("builtins.open", side_effect=ValueError("Parse fail")):
            with mock.patch("platform.system", return_value="Linux"):
                load = metrics_discovery.get_npu_load()
                assert load == 0


def test_get_intel_gpu_load_scheduler_fallback():
    """Test Intel GPU load fallback via active tasks."""
    mock_stats = {"active_tasks": [{"unit_type": "GPU", "unit_name": "Intel Arc", "stage": "Vocal Isolation"}]}
    with mock.patch("platform.system", return_value="Linux"):
        with mock.patch("glob.glob", return_value=[]):
            with mock.patch("modules.inference.scheduler.get_service_stats_minimal", return_value=mock_stats):
                load = metrics_discovery.get_intel_gpu_load()
                assert load == 100


def test_get_npu_load_scheduler_fallback():
    """Test NPU load fallback via active tasks."""
    mock_stats = {"active_tasks": [{"unit_type": "NPU", "stage": "Vocal Isolation"}]}
    with mock.patch("platform.system", return_value="Linux"):
        with mock.patch("glob.glob", return_value=[]):
            with mock.patch("modules.inference.scheduler.get_service_stats_minimal", return_value=mock_stats):
                load = metrics_discovery.get_npu_load()
                assert load == 100


def test_caching_logic():
    """Test that metrics are cached and not refetched immediately."""
    mock_fetch = mock.MagicMock(return_value=10)

    # First call
    val1 = metrics_discovery._get_cached_metric("test_key", mock_fetch)
    assert val1 == 10
    assert mock_fetch.call_count == 1

    # Second call (should be cached)
    val2 = metrics_discovery._get_cached_metric("test_key", mock_fetch)
    assert val2 == 10
    assert mock_fetch.call_count == 1


# --- Edge Cases and Direct Coverage ---


def test_fetch_intel_gpu_load_linux_sysfs_success():
    """Test successful Intel GPU load reading on Linux sysfs."""
    with mock.patch("glob.glob", return_value=["/sys/class/drm/card0/device/gpu_busy_percent"]):
        with mock.patch("builtins.open", mock.mock_open(read_data="42\n")):
            with mock.patch(
                "modules.inference.scheduler.get_service_stats_minimal",
                return_value={"active_tasks": [{"unit_type": "GPU", "unit_id": "GPU.0", "stage": "Vocal Isolation"}]},
            ):
                res = metrics_discovery._fetch_intel_gpu_load()
                assert res == 42


def test_fetch_npu_load_linux_sysfs_success():
    """Test successful NPU load reading on Linux sysfs."""
    with mock.patch("glob.glob", return_value=["/sys/class/accel/accel0/device/utilization"]):
        with mock.patch("builtins.open", mock.mock_open(read_data="88\n")):
            with mock.patch(
                "modules.inference.scheduler.get_service_stats_minimal",
                return_value={"active_tasks": [{"unit_type": "NPU", "unit_id": "NPU.0", "stage": "Vocal Isolation"}]},
            ):
                res = metrics_discovery._fetch_npu_load()
                assert res == 88


def test_fetch_npu_load_windows_success():
    """Test successful NPU load reading on Windows via PowerShell."""
    with mock.patch("platform.system", return_value="Windows"):
        with mock.patch("glob.glob", return_value=[]):
            with mock.patch("modules.monitoring.metrics_discovery._resolve_windows_powershell", return_value="powershell"):
                with mock.patch("modules.core.process_exec.check_output_text", return_value="12.5\n"):
                    with mock.patch(
                        "modules.inference.scheduler.get_service_stats_minimal",
                        return_value={"active_tasks": [{"unit_type": "NPU", "unit_id": "NPU.0", "stage": "Vocal Isolation"}]},
                    ):
                        res = metrics_discovery._fetch_npu_load()
                        assert res == 12


def test_fetch_single_intel_gpu_load_windows_nonzero_index():
    """Test Windows Intel GPU probing for a nonzero unit index."""
    with mock.patch("platform.system", return_value="Windows"):
        with mock.patch("glob.glob", return_value=[]):
            with mock.patch("modules.monitoring.metrics_discovery._resolve_windows_powershell", return_value="powershell"):
                with mock.patch("modules.core.process_exec.check_output_text", return_value="34.9\n"):
                    with mock.patch(
                        "modules.inference.scheduler.get_service_stats_minimal",
                        return_value={"active_tasks": [{"unit_type": "GPU", "unit_id": "GPU.2", "stage": "Vocal Isolation"}]},
                    ):
                        res = metrics_discovery._fetch_single_intel_gpu_load("GPU.2")
                        assert res == 34


def test_fetch_single_npu_load_windows_nonzero_index():
    """Test Windows NPU probing for a nonzero unit index."""
    with mock.patch("platform.system", return_value="Windows"):
        with mock.patch("glob.glob", return_value=[]):
            with mock.patch("modules.monitoring.metrics_discovery._resolve_windows_powershell", return_value="powershell"):
                with mock.patch("modules.core.process_exec.check_output_text", return_value="18.2\n"):
                    with mock.patch(
                        "modules.inference.scheduler.get_service_stats_minimal",
                        return_value={"active_tasks": [{"unit_type": "NPU", "unit_id": "NPU.3", "stage": "Vocal Isolation"}]},
                    ):
                        res = metrics_discovery._fetch_single_npu_load("NPU.3")
                        assert res == 18


def test_fetch_npu_load_windows_fail():
    """Test PowerShell failure on Windows for NPU."""
    with mock.patch("platform.system", return_value="Windows"):
        with mock.patch("glob.glob", return_value=[]):
            with mock.patch(
                "modules.core.process_exec.check_output_text",
                side_effect=process_exec.CommandExecutionError(["cmd"], 1),
            ):
                res = metrics_discovery._fetch_npu_load()
                assert res == 0


def test_fetch_hybrid_fallback_import_error():
    """Test fallback when scheduler import fails."""
    with mock.patch("glob.glob", return_value=[]):
        with mock.patch("platform.system", return_value="Linux"):
            with mock.patch("builtins.__import__", side_effect=ImportError):
                res = metrics_discovery._fetch_intel_gpu_load()
                assert res == 0


def test_fetch_hybrid_fallback_npu_active():
    """Test NPU fallback when tasks are active."""
    mock_stats = {"active_tasks": [{"unit_type": "NPU", "stage": "Vocal Isolation"}]}
    with mock.patch("glob.glob", return_value=[]):
        with mock.patch("platform.system", return_value="Linux"):
            with mock.patch("modules.inference.scheduler.get_service_stats_minimal", return_value=mock_stats):
                res = metrics_discovery._fetch_npu_load()
                assert res == 100


def test_resolve_index():
    """Test index extraction from unit IDs."""
    assert metrics_discovery._resolve_index("GPU.1") == 1
    assert metrics_discovery._resolve_index("cuda:2") == 2
    assert metrics_discovery._resolve_index("NPU") == 0
    assert metrics_discovery._resolve_index("CPU") == 0


def test_fetch_single_npu_load_sysfs():
    """Test reading load for single NPU via sysfs."""
    with mock.patch("glob.glob", return_value=["/sys/class/accel/accel1/device/utilization"]):
        with mock.patch("builtins.open", mock.mock_open(read_data="75\n")):
            with mock.patch(
                "modules.inference.scheduler.get_service_stats_minimal",
                return_value={"active_tasks": [{"unit_type": "NPU", "unit_id": "NPU.1", "stage": "Vocal Isolation"}]},
            ):
                res = metrics_discovery._fetch_single_npu_load("NPU.1")
                assert res == 75


def test_fetch_single_npu_load_sysfs_zero_is_preserved():
    """Test that a real zero NPU reading is not replaced by synthetic activity."""
    with mock.patch("glob.glob", return_value=["/sys/class/accel/accel1/device/utilization"]):
        with mock.patch("builtins.open", mock.mock_open(read_data="0\n")):
            with mock.patch("modules.inference.scheduler.get_service_stats_minimal", return_value={"active_tasks": []}):
                res = metrics_discovery._fetch_single_npu_load("NPU.1")
                assert res == 0


def test_fetch_single_intel_gpu_load_sysfs():
    """Test reading load for single Intel GPU via sysfs."""
    with mock.patch("glob.glob", return_value=["/sys/class/drm/card2/device/gpu_busy_percent"]):
        with mock.patch("builtins.open", mock.mock_open(read_data="50\n")):
            with mock.patch(
                "modules.inference.scheduler.get_service_stats_minimal",
                return_value={"active_tasks": [{"unit_type": "GPU", "unit_id": "GPU.2", "stage": "Vocal Isolation"}]},
            ):
                res = metrics_discovery._fetch_single_intel_gpu_load("GPU.2")
                assert res == 50


def test_gpu_unit_utilization_is_masked_without_app_accelerator_work():
    """Chart-facing Intel GPU utilization should be hidden when this service is not using the GPU."""
    with mock.patch("glob.glob", return_value=["/sys/class/drm/card2/device/gpu_busy_percent"]):
        with mock.patch("builtins.open", mock.mock_open(read_data="50\n")):
            with mock.patch("modules.inference.scheduler.get_service_stats_minimal", return_value={"active_tasks": []}):
                assert metrics_discovery._resolve_unit_utilization("GPU", "GPU.2") == 0


def test_npu_unit_utilization_is_masked_without_app_accelerator_work():
    """Chart-facing Intel NPU utilization should be hidden when this service is not using the NPU."""
    with mock.patch("glob.glob", return_value=["/sys/class/accel/accel1/device/utilization"]):
        with mock.patch("builtins.open", mock.mock_open(read_data="75\n")):
            with mock.patch("modules.inference.scheduler.get_service_stats_minimal", return_value={"active_tasks": []}):
                assert metrics_discovery._resolve_unit_utilization("NPU", "NPU.1") == 0


def test_fetch_single_intel_gpu_load_sysfs_zero_is_preserved():
    """Test that a real zero Intel GPU reading is not replaced by synthetic activity."""
    with mock.patch("glob.glob", return_value=["/sys/class/drm/card2/device/gpu_busy_percent"]):
        with mock.patch("builtins.open", mock.mock_open(read_data="0\n")):
            with mock.patch("modules.inference.scheduler.get_service_stats_minimal", return_value={"active_tasks": []}):
                res = metrics_discovery._fetch_single_intel_gpu_load("GPU.2")
                assert res == 0


def test_get_intel_gpu_load_zero_is_preserved():
    """Test that the aggregate Intel GPU probe preserves zero instead of falling back."""
    with mock.patch("glob.glob", return_value=["/sys/class/drm/card0/device/gpu_busy_percent"]):
        with mock.patch("builtins.open", mock.mock_open(read_data="0\n")):
            with mock.patch("modules.inference.scheduler.get_service_stats_minimal", return_value={"active_tasks": []}):
                with mock.patch("modules.inference.runtime.model_manager.PREPROCESSOR_POOL", {}):
                    assert metrics_discovery.get_intel_gpu_load() == 0


def test_get_npu_load_zero_is_preserved():
    """Test that the aggregate NPU probe preserves zero instead of falling back."""
    with mock.patch("glob.glob", side_effect=[["/sys/class/accel/accel0/device/utilization"], []]):
        with mock.patch("builtins.open", mock.mock_open(read_data="0\n")):
            with mock.patch("modules.inference.scheduler.get_service_stats_minimal", return_value={"active_tasks": []}):
                with mock.patch("modules.inference.runtime.model_manager.PREPROCESSOR_POOL", {}):
                    assert metrics_discovery.get_npu_load() == 0


def test_get_all_hardware_utilization():
    """Test get_all_hardware_utilization retrieves mapped loads."""
    mock_units = [
        {"type": "CUDA", "id": "cuda:0", "name": "NVIDIA GPU 0"},
        {"type": "GPU", "id": "GPU.1", "name": "Intel Arc GPU"},
        {"type": "NPU", "id": "NPU.2", "name": "Intel NPU 2"},
    ]
    with mock.patch("modules.core.config.HARDWARE_UNITS", mock_units):
        with mock.patch("modules.monitoring.metrics_discovery.get_nvidia_metrics", return_value=[{"util": 35}]):
            with mock.patch("modules.monitoring.metrics_discovery._fetch_single_intel_gpu_load", return_value=45):
                with mock.patch("modules.monitoring.metrics_discovery._fetch_single_npu_load", return_value=85):
                    with mock.patch(
                        "modules.inference.scheduler.get_service_stats_minimal",
                        return_value={
                            "active_tasks": [
                                {"unit_type": "CUDA", "unit_id": "cuda:0", "stage": "Inference"},
                                {"unit_type": "GPU", "unit_id": "GPU.1", "stage": "Vocal Isolation"},
                                {"unit_type": "NPU", "unit_id": "NPU.2", "stage": "Vocal Isolation"},
                            ]
                        },
                    ):
                        # Ensure metric cache is cleared to force fresh fetch
                        with metrics_discovery._CACHE_LOCK:
                            metrics_discovery._METRIC_CACHE.clear()
                        util = metrics_discovery.get_all_hardware_utilization()
                        assert util["cuda:0"] == 35
                        assert util["GPU.1"] == 45
                        assert util["NPU.2"] == 85


def test_fetch_single_npu_load_mismatch():
    """Test NPU single load fallback when unit ID strings have slight mismatch (e.g. NPU vs NPU.0)."""
    # Active task has unit_id = "NPU", but we probe "NPU.0"
    mock_stats = {"active_tasks": [{"unit_id": "NPU", "unit_type": "NPU", "stage": "Vocal Isolation"}]}
    with mock.patch("glob.glob", return_value=[]):
        with mock.patch("platform.system", return_value="Linux"):
            with mock.patch("modules.inference.scheduler.get_service_stats_minimal", return_value=mock_stats):
                res = metrics_discovery._fetch_single_npu_load("NPU.0")
                assert res == 100


def test_fetch_single_intel_gpu_load_mismatch():
    """Test Intel GPU single load fallback when unit ID strings have slight mismatch (e.g. GPU vs GPU.0)."""
    # Active task has unit_id = "GPU", but we probe "GPU.0"
    mock_stats = {"active_tasks": [{"unit_id": "GPU", "unit_type": "GPU", "unit_name": "Intel Arc", "stage": "Vocal Isolation"}]}
    with mock.patch("glob.glob", return_value=[]):
        with mock.patch("platform.system", return_value="Linux"):
            with mock.patch("modules.inference.scheduler.get_service_stats_minimal", return_value=mock_stats):
                res = metrics_discovery._fetch_single_intel_gpu_load("GPU.0")
                assert res == 100


def test_fetch_single_cuda_fallback():
    """Test CUDA single load fallback logic when nvidia-smi is unavailable."""
    # Active task has unit_id = "cuda:0", but we probe "cuda:0"
    mock_stats = {"active_tasks": [{"unit_id": "cuda:0", "unit_type": "CUDA", "stage": "Inference"}]}
    with mock.patch("modules.inference.scheduler.get_service_stats_minimal", return_value=mock_stats):
        res = metrics_discovery._fetch_single_cuda_fallback("cuda:0", 0)
        assert res == 99


def test_is_task_using_accelerator_uvr_and_non_accelerated_stage():
    """UVR stages should count as accelerated, but non-accelerated stages should not."""
    from modules.monitoring.metrics_discovery import _is_task_using_accelerator

    task_uvr = {"stage": "Vocal Isolation"}
    assert _is_task_using_accelerator(task_uvr, "NPU") is True
    assert _is_task_using_accelerator(task_uvr, "GPU") is True
    task_init = {"stage": "Initializing"}
    assert _is_task_using_accelerator(task_init, "NPU") is False
    assert _is_task_using_accelerator(task_init, "GPU") is False


def test_is_task_using_accelerator_non_hardware_stage_is_idle():
    """Tasks in non-UVR/non-inference stages should not light up accelerator charts."""
    from modules.monitoring.metrics_discovery import _is_task_using_accelerator

    task = {"stage": "Standardizing audio"}
    assert _is_task_using_accelerator(task, "CUDA") is False
    assert _is_task_using_accelerator(task, "GPU") is False
    assert _is_task_using_accelerator(task, "NPU") is False


def test_is_task_using_accelerator_cuda_stages():
    """CUDA tasks should use accelerators during ASR and language detection."""
    from modules.monitoring.metrics_discovery import _is_task_using_accelerator

    task_asr = {"stage": "Transcribing (Seg 1)"}
    task_ld = {"stage": "Inference (Language Detection)"}
    assert _is_task_using_accelerator(task_asr, "CUDA") is True
    assert _is_task_using_accelerator(task_ld, "CUDA") is True


def test_supports_asr_stage_on_unit_amd():
    """AMD units support ASR stage tasks."""
    from modules.monitoring.metrics_discovery import _supports_asr_stage_on_unit

    assert _supports_asr_stage_on_unit("AMD") is True


def test_is_task_using_accelerator_intel_engine():
    """Intel Whisper should use NPU/GPU accelerators."""
    from modules.core import config
    from modules.monitoring.metrics_discovery import _is_task_using_accelerator

    task_asr = {"stage": "Transcribing (Seg 1)"}

    with mock.patch.object(config, "ASR_ENGINE", "INTEL-WHISPER"):
        assert _is_task_using_accelerator(task_asr, "NPU") is True
        assert _is_task_using_accelerator(task_asr, "GPU") is True


def test_is_task_using_accelerator_faster_engine():
    """Faster Whisper should fall back to CPU for NPU/GPU accelerators."""
    from modules.core import config
    from modules.monitoring.metrics_discovery import _is_task_using_accelerator

    task_asr = {"stage": "Transcribing (Seg 1)"}
    task_ld = {"stage": "Inference (Language Detection)"}

    with mock.patch.object(config, "ASR_ENGINE", "FASTER-WHISPER"):
        assert _is_task_using_accelerator(task_asr, "NPU") is False
        assert _is_task_using_accelerator(task_asr, "GPU") is False
        assert _is_task_using_accelerator(task_ld, "NPU") is False
        assert _is_task_using_accelerator(task_ld, "GPU") is False


def test_fetch_intel_gpu_load_windows_errors():
    """Test Windows PowerShell exceptions/value errors in Intel GPU load query."""
    with mock.patch("platform.system", return_value="Windows"):
        with mock.patch("glob.glob", return_value=[]):
            with mock.patch(
                "modules.core.process_exec.check_output_text",
                side_effect=process_exec.CommandExecutionError(["cmd"], 1),
            ):
                assert metrics_discovery._fetch_intel_gpu_load() == 0
                assert metrics_discovery._fetch_single_intel_gpu_load("GPU.0") == 0
            with mock.patch("modules.core.process_exec.check_output_text", return_value="invalid\n"):
                assert metrics_discovery._fetch_intel_gpu_load() == 0
                assert metrics_discovery._fetch_single_intel_gpu_load("GPU.0") == 0


def test_fetch_single_npu_load_windows_errors():
    """Test Windows PowerShell exceptions/value errors in single NPU load query."""
    with mock.patch("platform.system", return_value="Windows"):
        with mock.patch("glob.glob", return_value=[]):
            with mock.patch(
                "modules.core.process_exec.check_output_text",
                side_effect=process_exec.CommandExecutionError(["cmd"], 1),
            ):
                assert metrics_discovery._fetch_single_npu_load("NPU.0") == 0
            with mock.patch("modules.core.process_exec.check_output_text", return_value="invalid\n"):
                assert metrics_discovery._fetch_single_npu_load("NPU.0") == 0


def test_attribute_error_fallbacks_for_gpu_and_npu():
    """AttributeError fallbacks should return zero for GPU and NPU queries."""
    with mock.patch("modules.monitoring.metrics_discovery.scheduler", None):
        with mock.patch("modules.monitoring.metrics_discovery.model_manager", None):
            assert metrics_discovery._fetch_intel_gpu_load() == 0
            assert metrics_discovery._fetch_single_intel_gpu_load("GPU.1") == 0
            assert metrics_discovery._fetch_npu_load() == 0
            assert metrics_discovery._fetch_single_npu_load("NPU.1") == 0


def test_attribute_error_fallbacks_for_cuda():
    """AttributeError fallbacks should return zero for CUDA queries."""
    with mock.patch("modules.monitoring.metrics_discovery.scheduler", None):
        with mock.patch("modules.monitoring.metrics_discovery.model_manager", None):
            assert metrics_discovery._fetch_single_cuda_fallback("cuda:0", 0) == 0


def test_single_device_read_errors():
    """Test single GPU/NPU Linux sysfs read failures (IOError/ValueError)."""
    with mock.patch("glob.glob", return_value=["/sys/class/drm/card1/device/gpu_busy_percent"]):
        with mock.patch("builtins.open", side_effect=ValueError("invalid int")):
            assert metrics_discovery._fetch_single_intel_gpu_load("GPU.1") == 0

    with mock.patch("glob.glob", return_value=["/sys/class/accel/accel1/device/utilization"]):
        with mock.patch("builtins.open", side_effect=IOError("read error")):
            assert metrics_discovery._fetch_single_npu_load("NPU.1") == 0


def test_all_hardware_utilization_failures():
    """Test that all_hardware_utilization handles out-of-bounds CUDA index correctly."""
    mock_units = [
        {"type": "CUDA", "id": "cuda:1", "name": "NVIDIA GPU 1"},
    ]
    with mock.patch("modules.core.config.HARDWARE_UNITS", mock_units):
        # get_nvidia_metrics returns empty list (len = 0), but index = 1, triggering IndexError
        with mock.patch("modules.monitoring.metrics_discovery.get_nvidia_metrics", return_value=[]):
            with mock.patch("modules.monitoring.metrics_discovery._fetch_single_cuda_fallback", return_value=55):
                with mock.patch(
                    "modules.inference.scheduler.get_service_stats_minimal",
                    return_value={"active_tasks": [{"unit_type": "CUDA", "unit_id": "cuda:1", "stage": "Inference"}]},
                ):
                    util = metrics_discovery._fetch_all_hardware_utilization()
                    assert util["cuda:1"] == 55


def test_fetch_npu_load_falls_back_to_locked_preprocessor():
    """NPU probe should report busy when preprocessor lock is held without active tasks."""
    pm = mock.MagicMock()
    pm.device_type = "NPU"
    pm.device_id = "NPU.0"
    pm.unit = {"id": "NPU.0", "name": "Intel NPU"}
    pm.lock.locked.return_value = True

    with (
        mock.patch("glob.glob", return_value=[]),
        mock.patch("platform.system", return_value="Linux"),
        mock.patch("modules.inference.scheduler.get_service_stats_minimal", return_value={"active_tasks": []}),
        mock.patch.dict("modules.inference.runtime.model_manager.PREPROCESSOR_POOL", {"NPU.0": pm}, clear=True),
    ):
        assert metrics_discovery._fetch_npu_load() == 100


def test_metrics_task_matching_false_paths():
    """Cover task matching branches for type mismatch and NVIDIA exclusion."""
    task = {"unit_id": "GPU.0", "unit_type": "GPU", "unit_name": "NVIDIA RTX"}
    assert metrics_discovery._task_matches_unit(task, "NPU", unit_id=None, idx=0, exclude_nvidia=False) is False
    assert metrics_discovery._task_matches_unit(task, "GPU", unit_id=None, idx=0, exclude_nvidia=True) is False


def test_metrics_exact_unit_match_helper():
    """Cover exact unit-id match helper."""
    task = {"unit_id": "GPU.0"}
    assert metrics_discovery._has_exact_unit_id_match(task, "GPU.0") is True


def test_metrics_preprocessor_matching_false_paths():
    """Cover preprocessor vendor and unit-id mismatch helper branches."""
    pm = mock.MagicMock()
    pm.device_type = "GPU"
    pm.device_id = "GPU.0"
    pm.unit = {"id": "GPU.0", "name": "NVIDIA Arc"}

    assert metrics_discovery._preprocessor_vendor_matches(pm, exclude_nvidia=True) is False
    assert metrics_discovery._preprocessor_id_matches("GPU.0", "GPU.1", idx=1) is False


def test_metrics_resolve_helpers_cover_cpu_unknown_and_exception_fallbacks():
    """Cover utilization resolver branches for CPU/unknown units and CUDA exception fallback."""
    assert metrics_discovery._resolve_unit_utilization("CPU", "CPU") is None
    assert metrics_discovery._resolve_unit_utilization("UNKNOWN", "X.0") is None

    with (
        mock.patch("modules.monitoring.metrics_discovery.get_nvidia_metrics", return_value=[{}]),
        mock.patch("modules.monitoring.metrics_discovery._fetch_single_cuda_fallback", return_value=77),
        mock.patch(
            "modules.inference.scheduler.get_service_stats_minimal",
            return_value={"active_tasks": [{"unit_type": "CUDA", "unit_id": "cuda:0", "stage": "Inference"}]},
        ),
    ):
        assert metrics_discovery._resolve_cuda_utilization("cuda:0") == 77


def test_metrics_probe_helpers_cover_zero_and_locked_fallback_paths():
    """Cover probe fallbacks when sysfs/windows counters are unavailable and lock fallback is used."""
    with mock.patch("modules.monitoring.metrics_discovery._run_windows_accelerator_counter", return_value=0):
        assert metrics_discovery._probe_sysfs_and_windows([], windows_cmd="noop") == 0

    with (
        mock.patch("modules.monitoring.metrics_discovery._has_active_accelerator_tasks", return_value=False),
        mock.patch("modules.monitoring.metrics_discovery._has_locked_preprocessor", return_value=True),
    ):
        assert metrics_discovery._probe_activity_fallback("GPU.0", 0, "GPU", 100, exclude_nvidia=False) == 100


def test_cuda_probe_holds_recent_nvidia_sample_during_transient_failure():
    """CUDA resolver should return held nvidia-smi sample during transient probe failures."""
    with (
        mock.patch("modules.monitoring.metrics_discovery._monotonic_now", return_value=1000.0),
        mock.patch("modules.monitoring.metrics_discovery.get_nvidia_metrics", return_value=[{"util": 47}]),
        mock.patch("modules.monitoring.metrics_discovery._fetch_single_cuda_fallback", return_value=11),
        mock.patch(
            "modules.inference.scheduler.get_service_stats_minimal",
            return_value={"active_tasks": [{"unit_type": "CUDA", "unit_id": "cuda:0", "stage": "Inference"}]},
        ),
    ):
        first = metrics_discovery._resolve_cuda_utilization("cuda:0")

    with (
        mock.patch(
            "modules.monitoring.metrics_discovery._monotonic_now",
            return_value=1000.0 + (metrics_discovery.REAL_SAMPLE_HOLD_TTL / 2.0),
        ),
        mock.patch("modules.monitoring.metrics_discovery.get_nvidia_metrics", side_effect=RuntimeError("transient")),
        mock.patch("modules.monitoring.metrics_discovery._fetch_single_cuda_fallback", return_value=11),
        mock.patch(
            "modules.inference.scheduler.get_service_stats_minimal",
            return_value={"active_tasks": [{"unit_type": "CUDA", "unit_id": "cuda:0", "stage": "Inference"}]},
        ),
    ):
        second = metrics_discovery._resolve_cuda_utilization("cuda:0")

    assert first == 47
    assert second == 47


def test_cuda_probe_held_nvidia_sample_expires_then_uses_fallback():
    """Held nvidia-smi CUDA sample should expire after TTL and allow fallback."""
    with (
        mock.patch("modules.monitoring.metrics_discovery._monotonic_now", return_value=3000.0),
        mock.patch("modules.monitoring.metrics_discovery.get_nvidia_metrics", return_value=[{"util": 52}]),
        mock.patch("modules.monitoring.metrics_discovery._fetch_single_cuda_fallback", return_value=19),
        mock.patch(
            "modules.inference.scheduler.get_service_stats_minimal",
            return_value={"active_tasks": [{"unit_type": "CUDA", "unit_id": "cuda:0", "stage": "Inference"}]},
        ),
    ):
        first = metrics_discovery._resolve_cuda_utilization("cuda:0")

    with (
        mock.patch(
            "modules.monitoring.metrics_discovery._monotonic_now",
            return_value=3000.0 + metrics_discovery.REAL_SAMPLE_HOLD_TTL + 1.0,
        ),
        mock.patch("modules.monitoring.metrics_discovery.get_nvidia_metrics", side_effect=RuntimeError("transient")),
        mock.patch("modules.monitoring.metrics_discovery._fetch_single_cuda_fallback", return_value=19),
        mock.patch(
            "modules.inference.scheduler.get_service_stats_minimal",
            return_value={"active_tasks": [{"unit_type": "CUDA", "unit_id": "cuda:0", "stage": "Inference"}]},
        ),
    ):
        second = metrics_discovery._resolve_cuda_utilization("cuda:0")

    assert first == 52
    assert second == 19


def test_intel_probe_holds_recent_real_sample_before_activity_fallback():
    """Transient real-probe misses should keep the last real sample instead of flipping to binary fallback."""
    with (
        mock.patch("modules.monitoring.metrics_discovery._read_first_int_value", side_effect=[42, None]),
        mock.patch("modules.monitoring.metrics_discovery._probe_activity_fallback", return_value=100),
    ):
        first = metrics_discovery._fetch_single_accelerator_load(
            unit_id="GPU.0",
            idx=0,
            device_type="GPU",
            sysfs_paths=[],
            windows_cmd=None,
            busy_value=100,
            exclude_nvidia=True,
        )
        second = metrics_discovery._fetch_single_accelerator_load(
            unit_id="GPU.0",
            idx=0,
            device_type="GPU",
            sysfs_paths=[],
            windows_cmd=None,
            busy_value=100,
            exclude_nvidia=True,
        )

    assert first == 42
    assert second == 42


def test_intel_probe_real_sample_hold_expires_to_activity_fallback():
    """Held real samples should expire and then allow synthetic activity fallback."""
    with (
        mock.patch("modules.monitoring.metrics_discovery._read_first_int_value", return_value=55),
        mock.patch("modules.monitoring.metrics_discovery._monotonic_now", return_value=1000.0),
    ):
        first = metrics_discovery._fetch_single_accelerator_load(
            unit_id="NPU.0",
            idx=0,
            device_type="NPU",
            sysfs_paths=[],
            windows_cmd=None,
            busy_value=100,
            exclude_nvidia=False,
        )

    with (
        mock.patch("modules.monitoring.metrics_discovery._read_first_int_value", return_value=None),
        mock.patch("modules.monitoring.metrics_discovery._probe_activity_fallback", return_value=100),
        mock.patch(
            "modules.monitoring.metrics_discovery._monotonic_now",
            return_value=1000.0 + metrics_discovery.REAL_SAMPLE_HOLD_TTL + 1.0,
        ),
    ):
        second = metrics_discovery._fetch_single_accelerator_load(
            unit_id="NPU.0",
            idx=0,
            device_type="NPU",
            sysfs_paths=[],
            windows_cmd=None,
            busy_value=100,
            exclude_nvidia=False,
        )

    assert first == 55
    assert second == 100


# --- AMD Utilization Resolver ---


def test_resolve_amd_utilization_inactive_returns_zero():
    """Inactive AMD unit: _inactive_accelerator_zero_result zeroes result and clears held sample."""
    with (
        mock.patch("modules.inference.scheduler.get_service_stats_minimal", return_value={"active_tasks": []}),
        mock.patch("modules.inference.runtime.model_manager.PREPROCESSOR_POOL", {}),
    ):
        assert metrics_discovery._resolve_amd_utilization("amd:0") == 0


def test_resolve_amd_utilization_active_task_returns_100():
    """Active AMD task: _inactive_accelerator_zero_result returns None → _probe_activity_fallback returns 100."""
    active_amd_task = [{"unit_type": "AMD", "unit_id": "amd:0", "stage": "Vocal Isolation"}]
    with (
        mock.patch("modules.inference.scheduler.get_service_stats_minimal", return_value={"active_tasks": active_amd_task}),
        mock.patch("modules.inference.runtime.model_manager.PREPROCESSOR_POOL", {}),
    ):
        assert metrics_discovery._resolve_amd_utilization("amd:0") == 100


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
    ):
        assert metrics_discovery._resolve_amd_utilization("amd:0") == 100


def test_resolve_unit_utilization_amd():
    """_resolve_unit_utilization dispatching for 'AMD' calls _resolve_amd_utilization."""
    active_amd_task = [{"unit_type": "AMD", "unit_id": "amd:0", "stage": "Vocal Isolation"}]
    with (
        mock.patch("modules.inference.scheduler.get_service_stats_minimal", return_value={"active_tasks": active_amd_task}),
        mock.patch("modules.inference.runtime.model_manager.PREPROCESSOR_POOL", {}),
    ):
        assert metrics_discovery._resolve_unit_utilization("AMD", "amd:0") == 100
