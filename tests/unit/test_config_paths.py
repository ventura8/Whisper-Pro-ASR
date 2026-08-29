"""Runtime path and approved-root resolution."""

import os
from unittest import mock

import pytest

from modules.core import config_paths


class TestAutoApprovingBindMounts:
    """Which bind mounts become approved roots for caller-supplied local paths.

    APPROVED_ROOTS decides what a request may ask the service to read off disk, so the grant
    is worth being able to see and to refuse.
    """

    def test_mounts_are_approved_by_default(self):
        """Local-path optimisation depends on it; defaulting off breaks most deployments."""
        with (
            mock.patch.dict(os.environ, {}, clear=False),
            mock.patch.object(config_paths, "get_custom_mount_points", return_value=["/media"]),
        ):
            os.environ.pop("WHISPER_AUTO_APPROVE_MOUNTS", None)
            assert config_paths._auto_approved_mount_roots() == ["/media"]

    @pytest.mark.parametrize("value", ["false", "0", "no", "off", "FALSE", " Off "])
    def test_the_opt_out_leaves_only_explicitly_configured_roots(self, value):
        """An operator who wants request reads confined to WHISPER_APPROVED_ROOTS can have it."""
        with (
            mock.patch.dict(os.environ, {"WHISPER_AUTO_APPROVE_MOUNTS": value}),
            mock.patch.object(config_paths, "get_custom_mount_points", return_value=["/media"]),
        ):
            assert not config_paths._auto_approved_mount_roots()

    def test_the_grant_is_logged_so_it_can_be_reviewed(self, caplog):
        """An implicit authorisation nobody can see is one nobody can review."""
        with (
            mock.patch.dict(os.environ, {"WHISPER_AUTO_APPROVE_MOUNTS": "true"}),
            mock.patch.object(config_paths, "get_custom_mount_points", return_value=["/media", "/tv"]),
            caplog.at_level("INFO"),
        ):
            config_paths._auto_approved_mount_roots()
        assert "/media" in caplog.text and "/tv" in caplog.text

    def test_nothing_is_logged_when_there_is_nothing_to_approve(self, caplog):
        """A container with no extra bind mounts should not emit a misleading grant line."""
        with (
            mock.patch.dict(os.environ, {"WHISPER_AUTO_APPROVE_MOUNTS": "true"}),
            mock.patch.object(config_paths, "get_custom_mount_points", return_value=[]),
            caplog.at_level("INFO"),
        ):
            config_paths._auto_approved_mount_roots()
        assert "Auto-approved" not in caplog.text
