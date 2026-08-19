"""Additional targeted coverage for modules/core/utils_helpers.py."""

import os
import tempfile
import threading
from unittest import mock

from modules.core import utils_helpers


def test_mark_standard_ffmpeg_start_and_end_updates_count_and_notifies():
    """Standard FFmpeg counters should increment/decrement and notify waiters."""
    cond = threading.Condition()
    state = {"count": 0}

    with mock.patch.object(cond, "notify_all") as mock_notify:
        getattr(utils_helpers, "_mark_standard_ffmpeg_start")(False, cond, state)
        getattr(utils_helpers, "_mark_standard_ffmpeg_end")(False, cond, state)

    assert state["count"] == 0
    mock_notify.assert_called_once_with()


def test_compute_ffmpeg_timeout_zero_duration_uses_floor():
    """Zero-duration timeout should use the fixed floor value."""
    assert getattr(utils_helpers, "_compute_ffmpeg_timeout")(0) == 300.0


def test_update_ffmpeg_progress_state_without_out_time_keeps_progress():
    """Progress state should remain unchanged when FFmpeg line has no out_time_ms."""
    state = {"last_logged_pct": -10.0, "last_stage_pct": -1, "final_speed": "N/A", "last_yield_time": 0.0}
    getattr(utils_helpers, "_update_ffmpeg_progress_state")("frame=1", 10.0, lambda x: x, state)
    assert state["last_stage_pct"] == -1
    assert state["last_logged_pct"] == -10.0


def test_handle_duration_progress_invalid_line_returns_previous_state():
    """Invalid progress payload should preserve existing stage/logged values."""
    stage, logged = getattr(utils_helpers, "_handle_duration_progress")("out_time_ms=bad", 10.0, (7, 20.0), str, None)
    assert (stage, logged) == (7, 20.0)


def test_publish_ffmpeg_stage_if_needed_updates_and_yields_once():
    """Publishing should update scheduler stage and invoke yield callback once."""
    yield_calls = []
    with (
        mock.patch("modules.core.utils_helpers._update_scheduler_ffmpeg_stage") as mock_update,
        mock.patch("modules.core.utils_helpers._run_optional_yield", side_effect=lambda cb: cb() if cb else None),
    ):
        new_pct = getattr(utils_helpers, "_publish_ffmpeg_stage_if_needed")(17.9, 1, lambda: yield_calls.append(1))
    assert new_pct == 17
    mock_update.assert_called_once_with(17)
    assert len(yield_calls) == 1


def test_update_scheduler_ffmpeg_stage_swallows_import_error():
    """Scheduler update helper should suppress import-time failures."""
    with mock.patch("importlib.import_module", side_effect=ImportError):
        getattr(utils_helpers, "_update_scheduler_ffmpeg_stage")(50)


def test_parse_ffmpeg_progress_without_duration_yields_and_tracks_speed():
    """Streaming parser should track speed and periodically yield without known duration."""
    process = mock.MagicMock()
    process.stdout.readline.side_effect = ["speed=1.2x\n", "out_time_ms=100\n", ""]
    yield_calls = []

    with mock.patch("modules.core.utils_helpers.time.time", side_effect=[2.0, 3.5]):
        final_speed = utils_helpers.parse_ffmpeg_progress(process, duration=0, format_duration=str, yield_cb=lambda: yield_calls.append(1))

    assert final_speed == "1.2x"
    assert len(yield_calls) == 1


def test_secure_remove_ignores_remove_errors():
    """secure_remove should ignore OS errors when deleting temp files."""
    with (
        mock.patch("os.path.exists", return_value=True),
        mock.patch("os.remove", side_effect=OSError("locked")),
    ):
        utils_helpers.secure_remove("file.tmp")


def test_remove_temporary_asset_entry_removes_directory_branch():
    """Temporary asset helper should remove target directory entries."""
    with (
        mock.patch("os.path.isfile", return_value=False),
        mock.patch("os.path.isdir", return_value=True),
        mock.patch("shutil.rmtree") as mock_rmtree,
    ):
        getattr(utils_helpers, "_remove_temporary_asset_entry")("/tmp", "preprocessing")
        mock_rmtree.assert_called_once_with(os.path.normpath("/tmp/preprocessing"))


def test_validate_audio_true_for_existing_nonempty_file():
    """validate_audio should return True for existing non-empty files."""
    with tempfile.NamedTemporaryFile(delete=False) as fh:
        fh.write(b"abc")
        path = fh.name
    try:
        assert utils_helpers.validate_audio(path) is True
    finally:
        os.remove(path)
