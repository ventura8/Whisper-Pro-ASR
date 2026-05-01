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
    mock_stats = {
        "active_tasks": [{"unit_type": "GPU", "unit_name": "Intel Arc"}]
    }
    with mock.patch("platform.system", return_value="Linux"):
        with mock.patch("glob.glob", return_value=[]):
            with mock.patch("modules.inference.scheduler.get_service_stats_minimal", return_value=mock_stats):
                load = metrics_discovery.get_intel_gpu_load()
                assert load == 99


def test_get_npu_load_scheduler_fallback():
    """Test NPU load fallback via active tasks."""
    mock_stats = {
        "active_tasks": [{"unit_type": "NPU"}]
    }
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
