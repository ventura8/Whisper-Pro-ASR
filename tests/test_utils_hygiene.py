"""Tests for storage hygiene in utils.py"""
from unittest import mock
from modules import utils


def test_track_file_success():
    """Verify that a valid file is correctly tracked."""
    with mock.patch("os.path.exists", return_value=True), \
            mock.patch("os.path.isfile", return_value=True):
        utils.get_tracked_files().clear()
        utils.track_file("test.wav")
        assert "test.wav" in utils.get_tracked_files()
        utils.get_tracked_files().clear()


def test_track_file_invalid():
    """Verify that a non-existent file is not tracked."""
    with mock.patch("os.path.exists", return_value=False):
        utils.get_tracked_files().clear()
        utils.track_file("test.wav")
        assert "test.wav" not in utils.get_tracked_files()


def test_cleanup_tracked_files():
    """Verify that tracked files are correctly cleaned up."""
    with mock.patch("os.path.exists", return_value=True), \
            mock.patch("os.path.isfile", return_value=True), \
            mock.patch("modules.utils.secure_remove") as mock_remove:
        utils.get_tracked_files().clear()
        utils.track_file("file1.wav")
        utils.track_file("file2.wav")
        utils.cleanup_tracked_files()
        assert mock_remove.call_count == 2
        assert len(utils.get_tracked_files()) == 0
