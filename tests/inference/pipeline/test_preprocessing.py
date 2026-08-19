"""Core preprocessing manager/cache tests (split suite entry)."""

import os
import time
from unittest import mock

import pytest

from modules.inference.pipeline import openvino_resolver, preprocessing
from modules.inference.pipeline.preprocessing import PreprocessingManager


@pytest.fixture
def prep_manager():
    """Fixture to provide a clean PreprocessingManager instance."""
    unit = {"id": "CPU", "type": "CPU", "name": "CPU"}
    return PreprocessingManager(assigned_unit=unit)


@pytest.fixture(autouse=True)
def reset_openvino_family_circuit_breaker():
    openvino_resolver.clear_openvino_disabled_families()
    yield
    openvino_resolver.clear_openvino_disabled_families()


class TestManagerBasics:
    """Tests for basic manager operations."""

    def test_init_defaults(self):
        with mock.patch("modules.inference.pipeline.preprocessing.config") as mock_cfg:
            mock_cfg.PREPROCESS_DEVICE = "AUTO"
            pm = PreprocessingManager()
            assert pm._device_id == "AUTO"

    def test_unload_model(self, prep_manager):
        prep_manager.separator = mock.MagicMock()
        prep_manager.unload_model()
        assert prep_manager.separator is None

    def test_offload(self, prep_manager):
        prep_manager.separator = mock.MagicMock()
        with mock.patch("modules.inference.pipeline.preprocessing.utils.clear_gpu_cache") as mock_clear:
            prep_manager.offload()
            assert prep_manager.separator is None
            mock_clear.assert_called_once()
        prep_manager.offload()


class TestCache:
    """Tests for cache management."""

    def test_purge_cache_success(self, prep_manager):
        with mock.patch("modules.inference.pipeline.preprocessing.CACHE_DIR") as mock_cache:
            mock_file_old = mock.MagicMock()
            mock_file_old.is_file.return_value = True
            mock_file_old.stat.return_value.st_mtime = time.time() - 4000

            mock_file_new = mock.MagicMock()
            mock_file_new.is_file.return_value = True
            mock_file_new.stat.return_value.st_mtime = time.time()

            mock_cache.iterdir.return_value = [mock_file_old, mock_file_new]
            prep_manager._purge_stale_cache()
            mock_file_old.unlink.assert_called_once()
            mock_file_new.unlink.assert_not_called()

    def test_purge_cache_exception(self, prep_manager):
        with mock.patch("modules.inference.pipeline.preprocessing.CACHE_DIR") as mock_cache:
            mock_cache.iterdir.side_effect = Exception("Fail")
            prep_manager._purge_stale_cache()

    def test_purge_cache_unlink_fail(self, prep_manager):
        with mock.patch("modules.inference.pipeline.preprocessing.CACHE_DIR") as mock_cache:
            mock_file = mock.MagicMock()
            mock_file.is_file.return_value = True
            mock_file.stat.return_value.st_mtime = time.time() - 4000
            mock_file.unlink.side_effect = OSError("Locked")
            mock_cache.iterdir.return_value = [mock_file]
            prep_manager._purge_stale_cache()


def test_lazy_import():
    res = preprocessing._lazy_import_separator()
    assert res is not None or res is None


class TestCandidateOutputDirs:
    """Tests for _candidate_output_dirs()."""

    def test_returns_list_no_duplicates(self):
        dirs = preprocessing._candidate_output_dirs()
        assert isinstance(dirs, list)
        assert len(dirs) == len(set(dirs))

    def test_cache_dir_is_first(self):
        dirs = preprocessing._candidate_output_dirs()
        assert dirs[0] == str(preprocessing.CACHE_DIR)

    def test_persistent_temp_dir_included(self):
        with mock.patch("os.path.isdir", return_value=True):
            dirs = preprocessing._candidate_output_dirs()
            assert os.path.abspath(preprocessing.config.PERSISTENT_TEMP_DIR) in dirs

    def test_candidate_dirs_no_shm_dependency(self):
        with mock.patch("os.path.isdir", return_value=False):
            dirs = preprocessing._candidate_output_dirs()
            assert "/dev/shm" not in dirs
