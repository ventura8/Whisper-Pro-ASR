"""Tests for modules/monitoring/metrics_discovery.py."""

import subprocess
from unittest import mock

import pytest

from modules.monitoring import metrics_discovery


@pytest.fixture(autouse=True)
def clear_metric_cache():
    """Ensure a fresh cache for each test."""
    with metrics_discovery._CACHE_LOCK:
        metrics_discovery._METRIC_CACHE.clear()
    yield


def test_get_nvidia_metrics_success():
    """Test successful NVIDIA metrics retrieval."""
    mock_output = "25, 1024, 8192\n\n30, 2048, 8192"
    with mock.patch("subprocess.check_output", return_value=mock_output):
        metrics = metrics_discovery.get_nvidia_metrics()
        assert len(metrics) == 2
        assert metrics[0]["util"] == 25
        assert metrics[1]["util"] == 30


def test_get_nvidia_metrics_failure():
    """Test handling of failed nvidia-smi call."""
    with mock.patch("subprocess.check_output", side_effect=FileNotFoundError()):
        metrics = metrics_discovery.get_nvidia_metrics()
        assert metrics == []


def test_get_intel_gpu_load_windows_success():
    """Test Intel GPU load on Windows via PowerShell mock."""
    with mock.patch("platform.system", return_value="Windows"):
        with mock.patch("subprocess.check_output", return_value="45.5\n"):
            with mock.patch("glob.glob", return_value=[]):
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
    mock_stats = {"active_tasks": [{"unit_type": "GPU", "unit_name": "Intel Arc"}]}
    with mock.patch("platform.system", return_value="Linux"):
        with mock.patch("glob.glob", return_value=[]):
            with mock.patch("modules.inference.scheduler.get_service_stats_minimal", return_value=mock_stats):
                load = metrics_discovery.get_intel_gpu_load()
                assert load == 100


def test_get_npu_load_scheduler_fallback():
    """Test NPU load fallback via active tasks."""
    mock_stats = {"active_tasks": [{"unit_type": "NPU"}]}
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
            res = metrics_discovery._fetch_intel_gpu_load()
            assert res == 42


def test_fetch_npu_load_linux_sysfs_success():
    """Test successful NPU load reading on Linux sysfs."""
    with mock.patch("glob.glob", return_value=["/sys/class/accel/accel0/device/utilization"]):
        with mock.patch("builtins.open", mock.mock_open(read_data="88\n")):
            res = metrics_discovery._fetch_npu_load()
            assert res == 88


def test_fetch_npu_load_windows_success():
    """Test successful NPU load reading on Windows via PowerShell."""
    with mock.patch("platform.system", return_value="Windows"):
        with mock.patch("glob.glob", return_value=[]):
            with mock.patch("subprocess.check_output", return_value="12.5\n"):
                res = metrics_discovery._fetch_npu_load()
                assert res == 12


def test_fetch_npu_load_windows_fail():
    """Test PowerShell failure on Windows for NPU."""
    with mock.patch("platform.system", return_value="Windows"):
        with mock.patch("glob.glob", return_value=[]):
            with mock.patch("subprocess.check_output", side_effect=subprocess.CalledProcessError(1, "cmd")):
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
    mock_stats = {"active_tasks": [{"unit_type": "NPU"}]}
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
            res = metrics_discovery._fetch_single_npu_load("NPU.1")
            assert res == 75


def test_fetch_single_intel_gpu_load_sysfs():
    """Test reading load for single Intel GPU via sysfs."""
    with mock.patch("glob.glob", return_value=["/sys/class/drm/card2/device/gpu_busy_percent"]):
        with mock.patch("builtins.open", mock.mock_open(read_data="50\n")):
            res = metrics_discovery._fetch_single_intel_gpu_load("GPU.2")
            assert res == 50


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
    mock_stats = {"active_tasks": [{"unit_id": "NPU", "unit_type": "NPU"}]}
    with mock.patch("glob.glob", return_value=[]):
        with mock.patch("platform.system", return_value="Linux"):
            with mock.patch("modules.inference.scheduler.get_service_stats_minimal", return_value=mock_stats):
                res = metrics_discovery._fetch_single_npu_load("NPU.0")
                assert res == 100


def test_fetch_single_intel_gpu_load_mismatch():
    """Test Intel GPU single load fallback when unit ID strings have slight mismatch (e.g. GPU vs GPU.0)."""
    # Active task has unit_id = "GPU", but we probe "GPU.0"
    mock_stats = {"active_tasks": [{"unit_id": "GPU", "unit_type": "GPU", "unit_name": "Intel Arc"}]}
    with mock.patch("glob.glob", return_value=[]):
        with mock.patch("platform.system", return_value="Linux"):
            with mock.patch("modules.inference.scheduler.get_service_stats_minimal", return_value=mock_stats):
                res = metrics_discovery._fetch_single_intel_gpu_load("GPU.0")
                assert res == 100


def test_fetch_single_cuda_fallback():
    """Test CUDA single load fallback logic when nvidia-smi is unavailable."""
    # Active task has unit_id = "cuda:0", but we probe "cuda:0"
    mock_stats = {"active_tasks": [{"unit_id": "cuda:0", "unit_type": "CUDA"}]}
    with mock.patch("modules.inference.scheduler.get_service_stats_minimal", return_value=mock_stats):
        res = metrics_discovery._fetch_single_cuda_fallback("cuda:0", 0)
        assert res == 99


def test_is_task_using_accelerator():
    """Test _is_task_using_accelerator logic across different stages and engines."""
    from modules.core import config
    from modules.monitoring.metrics_discovery import _is_task_using_accelerator

    # 1. Vocal Isolation / UVR stage (always uses accelerator)
    task_uvr = {"stage": "Vocal Isolation"}
    assert _is_task_using_accelerator(task_uvr, "NPU") is True
    assert _is_task_using_accelerator(task_uvr, "GPU") is True

    # 2. Unknown/initializing stages (defaults to True for backward compatibility/safety)
    task_init = {"stage": "Initializing"}
    assert _is_task_using_accelerator(task_init, "NPU") is True

    # 3. Transcription/Inference/Language Detection stages
    task_asr = {"stage": "Transcribing (Seg 1)"}
    task_ld = {"stage": "Inference (Language Detection)"}

    # CUDA always runs on accelerator during ASR
    assert _is_task_using_accelerator(task_asr, "CUDA") is True
    assert _is_task_using_accelerator(task_ld, "CUDA") is True

    # NPU/GPU with INTEL-WHISPER runs on accelerator
    with mock.patch.object(config, "ASR_ENGINE", "INTEL-WHISPER"):
        assert _is_task_using_accelerator(task_asr, "NPU") is True
        assert _is_task_using_accelerator(task_asr, "GPU") is True

    # NPU/GPU with FASTER-WHISPER falls back to CPU
    with mock.patch.object(config, "ASR_ENGINE", "FASTER-WHISPER"):
        assert _is_task_using_accelerator(task_asr, "NPU") is False
        assert _is_task_using_accelerator(task_asr, "GPU") is False
        assert _is_task_using_accelerator(task_ld, "NPU") is False
        assert _is_task_using_accelerator(task_ld, "GPU") is False


def test_fetch_intel_gpu_load_windows_errors():
    """Test Windows PowerShell exceptions/value errors in Intel GPU load query."""
    with mock.patch("platform.system", return_value="Windows"):
        with mock.patch("glob.glob", return_value=[]):
            with mock.patch("subprocess.check_output", side_effect=subprocess.CalledProcessError(1, "cmd")):
                assert metrics_discovery._fetch_intel_gpu_load() == 0
                assert metrics_discovery._fetch_single_intel_gpu_load("GPU.0") == 0
            with mock.patch("subprocess.check_output", return_value="invalid\n"):
                assert metrics_discovery._fetch_intel_gpu_load() == 0
                assert metrics_discovery._fetch_single_intel_gpu_load("GPU.0") == 0


def test_fetch_single_npu_load_windows_errors():
    """Test Windows PowerShell exceptions/value errors in single NPU load query."""
    with mock.patch("platform.system", return_value="Windows"):
        with mock.patch("glob.glob", return_value=[]):
            with mock.patch("subprocess.check_output", side_effect=subprocess.CalledProcessError(1, "cmd")):
                assert metrics_discovery._fetch_single_npu_load("NPU.0") == 0
            with mock.patch("subprocess.check_output", return_value="invalid\n"):
                assert metrics_discovery._fetch_single_npu_load("NPU.0") == 0


def test_attribute_error_fallbacks():
    """Test that AttributeError in scheduler or model_manager is handled gracefully."""
    with mock.patch("modules.monitoring.metrics_discovery.scheduler", None):
        with mock.patch("modules.monitoring.metrics_discovery.model_manager", None):
            # Test GPU load fallbacks
            assert metrics_discovery._fetch_intel_gpu_load() == 0
            assert metrics_discovery._fetch_single_intel_gpu_load("GPU.1") == 0

            # Test NPU load fallbacks
            assert metrics_discovery._fetch_npu_load() == 0
            assert metrics_discovery._fetch_single_npu_load("NPU.1") == 0

            # Test CUDA fallback
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
                util = metrics_discovery._fetch_all_hardware_utilization()
                assert util["cuda:1"] == 55
