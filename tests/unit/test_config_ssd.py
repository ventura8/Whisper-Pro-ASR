"""Tests for SSD-related optimization settings in modules/config.py.

Split out from test_config.py: TestConfigSSD as its own concern/file.
"""

import importlib
import os
import tempfile
from typing import Any
from unittest import mock

import pytest

import modules.core.config as config_module


@pytest.mark.usefixtures("restore_config_after_reload")
class TestConfigSSD:
    """Tests for SSD optimization settings."""

    def test_temp_dir_default(self):
        """Test TEMP_DIR defaults to system temp."""
        with mock.patch.dict(os.environ, {}, clear=True):
            importlib.reload(config_module)
            assert config_module.TEMP_DIR == tempfile.gettempdir()

    def test_temp_dir_from_env(self, tmp_path):
        """Test TEMP_DIR from WHISPER_TEMP_DIR env."""
        with mock.patch.dict(os.environ, {"WHISPER_TEMP_DIR": str(tmp_path)}):
            importlib.reload(config_module)
            assert config_module.TEMP_DIR == str(tmp_path)

    def test_temp_min_free_default(self):
        """Test TEMP_DIR_MIN_FREE_BYTES defaults to 2048MB."""
        with mock.patch.dict(os.environ, {}, clear=True):
            importlib.reload(config_module)
            assert config_module.TEMP_DIR_MIN_FREE_BYTES == 2048 * 1024 * 1024

    def test_temp_min_free_from_env(self):
        """Test TEMP_DIR_MIN_FREE_BYTES from env."""
        with mock.patch.dict(os.environ, {"WHISPER_TEMP_MIN_FREE_MB": "100"}):
            importlib.reload(config_module)
            assert config_module.TEMP_DIR_MIN_FREE_BYTES == 100 * 1024 * 1024

    def test_get_temp_dir_disk_usage_fail(self):
        """Test get_temp_dir fallback when disk_usage fails."""
        with mock.patch("shutil.disk_usage", side_effect=OSError("Disk fail")):
            # Should return PERSISTENT_TEMP_DIR
            assert config_module.get_temp_dir() == config_module.PERSISTENT_TEMP_DIR

    def test_validate_thread_concurrency_warning(self):
        """Test the over-provisioning warning in validate_thread_concurrency."""
        env = {"ASR_THREADS": "8", "ASR_PREPROCESS_THREADS": "8", "FFMPEG_THREADS": "8", "CPU_CORE_LIMIT": "4"}
        with mock.patch.dict(os.environ, env):
            with mock.patch("logging.getLogger") as mock_get_logger:
                mock_logger = mock.MagicMock()
                mock_get_logger.return_value = mock_logger
                importlib.reload(config_module)
                # Should log a warning about over-provisioning
                warning_calls = mock_logger.warning.call_args_list
                assert any("OVER-PROVISIONING" in str(call) for call in warning_calls)

    def test_get_parallel_unit_limit_cuda_fallback(self):
        """Test hardware unit limit fallback when accelerator registry lacks CUDA."""
        with mock.patch("modules.core.config.HARDWARE_UNITS", [{"type": "CPU", "id": "CPU", "name": "Host CPU"}]):
            assert config_module.get_parallel_limit("CUDA") == 1

    def test_update_env(self):
        """Test update_env function."""
        config_module.update_env("TEST_KEY_DUMMY", "DUMMY_VAL")
        assert os.environ.get("TEST_KEY_DUMMY") == "DUMMY_VAL"

    def test_get_parallel_limit_various(self):
        """Test get_parallel_limit for CPU and GPU/NPU."""
        # 1. CPU
        assert config_module.get_parallel_limit("CPU") == config_module.CPU_PARALLEL_LIMIT

        # 2. GPU/NPU from capped hardware registry
        with mock.patch(
            "modules.core.config.HARDWARE_UNITS",
            [
                {"type": "GPU", "id": "GPU.0", "name": "Intel GPU 0"},
                {"type": "GPU", "id": "GPU.1", "name": "Intel GPU 1"},
                {"type": "NPU", "id": "NPU.0", "name": "Intel NPU 0"},
            ],
        ):
            assert config_module.get_parallel_limit("GPU") == 2
            assert config_module.get_parallel_limit("NPU") == 1

    def test_permission_error_fallbacks(self):
        """Test that PermissionError during directory creation defaults to fallbacks."""
        env = {
            "WHISPER_TEMP_DIR": "/fail_temp",
            "WHISPER_PERSISTENT_DIR": "/fail_persist",
            "WHISPER_STATE_DIR": "/fail_state",
            "OV_CACHE_DIR": "/fail_cache",
        }

        def mock_makedirs(path: str, *_args: Any, **_kwargs: Any) -> None:
            if any(fail_path in str(path) for fail_path in ["/fail_temp", "/fail_persist", "/fail_state", "/fail_cache"]):
                raise PermissionError("Permission denied")

        with mock.patch.dict(os.environ, env):
            with (
                mock.patch("os.makedirs", side_effect=mock_makedirs),
                # _resolve_writable_dir also probes with a real create+delete
                # (_is_path_writable, defined in modules.core.config itself).
                # Mocking that name directly doesn't work here: importlib.reload()
                # re-executes config.py's top-level `def _is_path_writable(...)`
                # statement, which clobbers this patch before _resolve_writable_dir
                # ever calls it. Mock the lower-level open()/os.remove() primitives
                # _is_path_writable itself uses instead -- those live outside the
                # reloaded module and survive the reload, making the fallback
                # chain deterministic regardless of the real filesystem
                # permissions of whatever process runs this test (e.g.
                # /app/model_cache is not actually writable in the CI container).
                mock.patch("modules.core.config.open", mock.mock_open(), create=True),
                mock.patch("os.remove"),
            ):
                importlib.reload(config_module)
                # STATE_DIR/LOG_DIR intentionally reuse PERSISTENT_DIR's own
                # resolved fallback when it's writable (see the comment above
                # STATE_DIR's resolution in config.py: this keeps task history
                # on the persistent bind mount rather than tmpfs). Since
                # WHISPER_PERSISTENT_DIR's primary candidate (/fail_persist)
                # fails here, PERSISTENT_DIR itself falls back to
                # OV_CACHE_DIR/.state, which succeeds and is what STATE_DIR
                # (a candidate list of [PERSISTENT_DIR, "./test_state"]) picks
                # up as its first writable candidate — "./test_state" is only
                # reached if that also fails, which isn't this scenario.
                expected_persistent_fallback = os.path.normpath(os.path.abspath(os.path.join(config_module.OV_CACHE_DIR, ".state")))
                assert (
                    config_module.TEMP_DIR,
                    os.path.normpath(config_module.PERSISTENT_DIR),
                    config_module.STATE_DIR,
                    config_module.LOG_DIR,
                    os.path.normpath(config_module.PERSISTENT_TEMP_DIR),
                ) == (
                    tempfile.gettempdir(),
                    expected_persistent_fallback,
                    expected_persistent_fallback,
                    expected_persistent_fallback,
                    os.path.normpath(os.path.join(config_module.OV_CACHE_DIR, "temp")),
                )

    def test_validate_thread_concurrency_error(self):
        """Test exception handling in validate_thread_concurrency."""
        # Force a TypeError by passing None
        with mock.patch("modules.core.config.PREPROCESS_THREADS", None):
            # Should not raise any exception due to try-except
            config_module.validate_thread_concurrency()

    def test_capping_preprocess_threads_hardware_limit(self):
        """Test capping preprocess threads to hardware limit message."""
        env = {"ASR_PREPROCESS_DEVICE": "GPU", "ASR_PREPROCESS_THREADS": "8", "CPU_CORE_LIMIT": "2"}
        with mock.patch.dict(os.environ, env):
            # We mock core.available_devices to return GPU
            mock_core = mock.MagicMock()
            mock_core.available_devices = ["GPU.0"]
            with mock.patch("openvino.Core", return_value=mock_core):
                importlib.reload(config_module)
                # PREPROCESS_THREADS should be capped to CPU_CORE_LIMIT (2) since requested (8) > 2
                assert config_module.PREPROCESS_THREADS == 2

    def test_npu_full_device_name_exception(self):
        """Test NPU name fallback on property query exception."""
        with mock.patch.dict(os.environ, {"ASR_DEVICE": "AUTO"}):
            with mock.patch("ctranslate2.get_cuda_device_count", return_value=0):
                mock_core = mock.MagicMock()
                mock_core.available_devices = ["NPU"]
                mock_core.get_property.side_effect = RuntimeError("Property error")
                with mock.patch("openvino.Core", return_value=mock_core):
                    importlib.reload(config_module)
                    # Verify unit was added with fallback name "Intel NPU"
                    assert any(u["name"] == "Intel NPU" for u in config_module.HARDWARE_UNITS)

    def test_intel_accelerator_detection_exception(self):
        """Test that Intel accelerator detection failure logs skipping message."""
        with mock.patch.dict(os.environ, {"ASR_DEVICE": "AUTO"}):
            with mock.patch("ctranslate2.get_cuda_device_count", return_value=0):
                with mock.patch("openvino.Core", side_effect=RuntimeError("OpenVINO Core init error")):
                    # Disable the real /dev/dri GPU-node fallback so this test's
                    # outcome doesn't depend on whether the machine running it
                    # happens to have a real Intel GPU device node present.
                    with mock.patch("modules.core.config_helpers._can_use_gpu_node_fallback", return_value=False):
                        importlib.reload(config_module)
                        # Skip logs error and falls back to CPU
                        assert config_module.DEVICE == "CPU"

    def test_get_custom_mount_points(self):
        """Test parsing of /proc/mounts to automatically detect custom directories."""
        # 1. When /proc/mounts does not exist, return empty list
        with mock.patch("os.path.exists", return_value=False):
            assert not config_module.get_custom_mount_points()

        # 2. When /proc/mounts exists, correctly parse custom mount points and ignore system ones
        fake_mounts = """sysfs /sys sysfs rw,nosuid,nodev,noexec,relatime 0 0
proc /proc proc rw,nosuid,nodev,noexec,relatime 0 0
udev /dev devtmpfs rw,nosuid,noexec,relatime,size=8000492k,nr_inodes=2000123,mode=755 0 0
/dev/sda1 /tv ext4 rw,relatime 0 0
/dev/sda2 /movies ext4 rw,relatime 0 0
/dev/sda3 /app/data ext4 rw,relatime 0 0
shm /dev/shm tmpfs rw,nosuid,nodev,noexec,relatime,size=65536k 0 0
"""
        with (
            mock.patch("os.path.exists", return_value=True),
            mock.patch("builtins.open", mock.mock_open(read_data=fake_mounts)),
        ):
            mounts = config_module.get_custom_mount_points()
            assert {"/tv", "/movies"}.issubset(set(mounts))
            assert {"/sys", "/proc", "/dev/shm"}.isdisjoint(set(mounts))
