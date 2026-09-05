"""Tests for modules/core/mount_helpers.py error and edge-case branches."""

import os
from unittest import mock

import pytest

from modules.core import mount_helpers
from modules.core.mount_helpers import _extract_mount_point, _is_custom_mount_point


def test_is_path_writable_returns_false_when_tempfile_denied():
    """A PermissionError from the probe write marks the directory non-writable."""
    with mock.patch("modules.core.mount_helpers.tempfile.TemporaryFile", side_effect=PermissionError("denied")):
        assert mount_helpers.is_path_writable("/some/dir") is False


def test_resolve_writable_dir_skips_empty_candidates():
    """Empty candidate strings are skipped before any makedirs attempt."""
    with (
        mock.patch("modules.core.mount_helpers.os.makedirs") as makedirs,
        mock.patch("modules.core.mount_helpers.is_path_writable", return_value=True),
    ):
        result = mount_helpers.resolve_writable_dir("Cache", ["", "/data/cache"], "/fallback")

    assert result == "/data/cache"
    makedirs.assert_called_once_with("/data/cache", exist_ok=True)


def test_get_custom_mount_points_returns_empty_on_os_error():
    """An unreadable /proc/mounts degrades to an empty list rather than raising."""
    with (
        mock.patch("modules.core.mount_helpers.os.path.exists", return_value=True),
        mock.patch("modules.core.mount_helpers._system_mount_roots", side_effect=OSError("boom")),
    ):
        assert not mount_helpers.get_custom_mount_points()


def test_extract_mount_point_returns_none_for_malformed_line():
    """A /proc/mounts line without a mount-point field yields None."""
    assert _extract_mount_point("/dev/sda1") is None


def test_is_custom_mount_point_rejects_system_root():
    """A mount point that is itself a system root is not custom."""
    assert _is_custom_mount_point("/proc", {"/proc", "/sys"}) is False


def test_resolve_writable_dir_raises_when_fallback_not_writable():
    """When candidates fail and fallback is non-writable, raise RuntimeError."""
    with (
        mock.patch("modules.core.mount_helpers.os.makedirs"),
        mock.patch("modules.core.mount_helpers.is_path_writable", return_value=False),
    ):
        with pytest.raises(RuntimeError, match="fallback directory /fallback is not writable"):
            mount_helpers.resolve_writable_dir("Cache", ["/data/cache"], "/fallback")


def test_resolve_temp_dir_threshold_min_free_bytes():
    """When free space is below min_free_bytes, fallback to persistent dir."""
    tmpfs_usage = mock.MagicMock(free=50 * 1024 * 1024)
    persistent_usage = mock.MagicMock(free=200 * 1024 * 1024)
    with mock.patch("modules.core.mount_helpers.shutil.disk_usage", side_effect=[tmpfs_usage, persistent_usage]):
        # min_free_bytes = 100MB > 50MB free
        res = mount_helpers.resolve_temp_dir("/tmp", "/persistent", min_free_bytes=100 * 1024 * 1024, required_bytes=0)
        assert res == "/persistent"


def test_resolve_temp_dir_warns_but_still_answers_when_neither_has_headroom(caplog):
    """Report the shortage; do not refuse to name a directory.

    This replaces test_resolve_temp_dir_raises_when_no_directory_has_required_space,
    which asserted a RuntimeError here. That raise was deliberate in v1.3.0, but it
    contradicted two older tests and the function's own documented contract ("a graceful
    fallback to persistent storage instead of an ENOSPC crash"), and those two had been
    failing ever since.

    Three reasons the warning is the better contract. resolve_temp_dir only *selects* a
    directory, and it runs while config is resolved on the request path -- raising turned
    a full disk into a hard failure on every request rather than a degraded one. The
    threshold is max(min_free_bytes, 1.5 * required_bytes): desired headroom, not the
    space the work needs, so being under it does not mean the write would fail. And here
    required_bytes is 0 -- the caller never said how much it needs -- so there is nothing
    to conclude is impossible.

    If the data genuinely does not fit, the write fails with a real ENOSPC naming the
    file, which is more use than a config-time error that cannot name one.
    """
    import logging  # pylint: disable=import-outside-toplevel

    low_space = mock.MagicMock(free=50 * 1024 * 1024)
    with mock.patch("modules.core.mount_helpers.shutil.disk_usage", return_value=low_space):
        with caplog.at_level(logging.WARNING):
            res = mount_helpers.resolve_temp_dir("/tmp", "/persistent", min_free_bytes=100 * 1024 * 1024)

    assert res == "/persistent"
    assert any("Neither temp directory" in record.message for record in caplog.records)


def test_resolve_temp_dir_threshold_required_bytes_headroom():
    """When required_bytes * 1.5 exceeds free space, fallback to persistent dir."""
    # Free space: 140MB. required_bytes = 100MB -> headroom = 150MB -> threshold = 150MB > 140MB
    tmpfs_usage = mock.MagicMock(free=140 * 1024 * 1024)
    persistent_usage = mock.MagicMock(free=200 * 1024 * 1024)
    with mock.patch("modules.core.mount_helpers.shutil.disk_usage", side_effect=[tmpfs_usage, persistent_usage]):
        res = mount_helpers.resolve_temp_dir("/tmp", "/persistent", min_free_bytes=10 * 1024 * 1024, required_bytes=100 * 1024 * 1024)
        assert res == "/persistent"


def test_resolve_temp_dir_boundary_equality_retains_temp_dir():
    """When free space equals the exact threshold, retain temp_dir."""
    # Free space: 150MB. required_bytes = 100MB -> headroom = 150MB -> threshold = 150MB == 150MB free
    mock_usage = mock.MagicMock(free=150 * 1024 * 1024)
    with mock.patch("modules.core.mount_helpers.shutil.disk_usage", return_value=mock_usage):
        res = mount_helpers.resolve_temp_dir("/tmp", "/persistent", min_free_bytes=100 * 1024 * 1024, required_bytes=100 * 1024 * 1024)
        assert res == "/tmp"


def test_resolve_temp_dir_retains_temp_dir_when_free_exceeds_threshold():
    """When free space strictly exceeds threshold, retain temp_dir."""
    mock_usage = mock.MagicMock(free=200 * 1024 * 1024)
    with mock.patch("modules.core.mount_helpers.shutil.disk_usage", return_value=mock_usage):
        res = mount_helpers.resolve_temp_dir("/tmp", "/persistent", min_free_bytes=100 * 1024 * 1024, required_bytes=100 * 1024 * 1024)
        assert res == "/tmp"


def test_resolve_temp_dir_handles_os_error():
    """When shutil.disk_usage raises OSError, fallback to persistent dir."""
    with mock.patch("modules.core.mount_helpers.shutil.disk_usage", side_effect=OSError("disk failure")):
        res = mount_helpers.resolve_temp_dir("/tmp", "/persistent", min_free_bytes=100 * 1024 * 1024)
        assert res == "/persistent"


def test_resolve_temp_dir_relative_persistent_dir():
    """Relative persistent directory must resolve to an absolute path."""
    with mock.patch("modules.core.mount_helpers.shutil.disk_usage", side_effect=OSError("fallback")):
        res = mount_helpers.resolve_temp_dir("./temp", "./model_cache", min_free_bytes=100 * 1024 * 1024)
        assert os.path.isabs(res)
        assert res == os.path.abspath("./model_cache")
