"""Tests for Bazarr path-as-key parsing and failure history recording."""

from unittest import mock

from modules.api.support import request_utils as routes_utils
from modules.api.support.local_path import extract_path_from_mapping_keys, normalize_bazarr_request_params
from modules.inference.runtime import model_manager


def test_extract_local_path_from_bazarr_json_path_key():
    """Some Bazarr clients send the media path as a JSON object key instead of local_path."""
    mock_req = mock.MagicMock()
    mock_req.query_params = {}
    form_data = {
        "/tv/Doc - In Your Hands/Season 3/Doc (IT) - S03E01 - Awakenings WEBDL-1080p.mkv": "",
    }
    path = routes_utils.extract_local_path(None, form_data, mock_req)
    assert path == "/tv/Doc - In Your Hands/Season 3/Doc (IT) - S03E01 - Awakenings WEBDL-1080p.mkv"


def test_extract_path_from_mapping_keys_ignores_non_media_keys():
    """Only absolute media-like paths should be recovered from mapping keys."""
    data = {"provider": "whisper", "language": "en"}
    assert extract_path_from_mapping_keys(data) is None


def test_normalize_bazarr_request_params_promotes_path_key_to_local_path():
    """Audit/history payloads should store local_path instead of path-as-key objects."""
    path = "/tv/Doc - In Your Hands/Season 3/Doc (IT) - S03E01 - Awakenings WEBDL-1080p.mkv"
    normalized = normalize_bazarr_request_params({path: ""})
    assert normalized == {"local_path": path}


def test_record_task_failure_persists_error_and_logs():
    """Failed tasks should expose error payloads and execution logs in history."""
    with (
        mock.patch("modules.inference.scheduler.task_helpers.update_task_metadata") as update_mock,
        mock.patch("modules.inference.scheduler.task_helpers.logger.error") as log_mock,
    ):
        model_manager.record_task_failure("No audio source provided", 400, context="LD")

    update_mock.assert_called_once()
    kwargs = update_mock.call_args.kwargs
    assert kwargs["status"] == "failed"
    assert kwargs["result"]["error"] == "No audio source provided"
    assert kwargs["response_json"]["status_code"] == 400
    log_mock.assert_called_once()
