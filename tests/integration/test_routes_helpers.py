"""Helper-route tests split from test_routes.py."""

import asyncio
from unittest import mock

import pytest

from modules.api.routes.asr import _parse_bool_param, build_response, get_request_params
from modules.api.support.request_utils import extract_local_path, prepare_source_path


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_prepare_source_path_from_query(self):
        """Test prepare_source_path resolution from query."""
        with mock.patch("os.path.exists", return_value=True):
            with pytest.raises(ValueError, match="Path not accessible"):
                prepare_source_path(local_path="/test/path.mp3")

    def test_prepare_source_path_from_form(self):
        """Test prepare_source_path resolution from form."""
        mock_req = mock.MagicMock()
        mock_req.query_params = {}
        form_data = {"local_path": "/form/path.mp3"}

        extracted = extract_local_path(None, form_data, mock_req)
        with mock.patch("os.path.exists", return_value=True):
            with pytest.raises(ValueError, match="Path not accessible"):
                prepare_source_path(local_path=extracted)

    def test_get_request_params_defaults(self):
        """Test default request parameters."""
        mock_request = mock.MagicMock()
        mock_request.query_params = {}
        mock_request.url.path = "/asr"
        params = asyncio.run(get_request_params(mock_request, {}))
        assert params["output_format"] == "srt"
        assert params["task"] == "transcribe"

    def test_get_request_params_custom(self):
        """Test custom request parameters."""
        mock_request = mock.MagicMock()
        mock_request.query_params = {"output": "json", "language": "es", "task": "translate", "batch_size": "4"}
        mock_request.url.path = "/asr"
        params = asyncio.run(get_request_params(mock_request, {}))
        assert params["output_format"] == "json"
        assert params["language"] == "es"
        assert params["task"] == "translate"
        assert params["batch_size"] == 4

    def test_build_response_json(self):
        """Test JSON response building."""
        result = {"text": "Hello", "segments": []}
        params = {"output_format": "json"}
        stats = {"active_sessions": 0}
        response = build_response(result, params, stats, "/fake/path", 100.0)
        assert response.headers["content-type"] == "application/json"

    def test_build_response_srt(self):
        """Test SRT response building."""
        result = {"text": "Hello", "segments": [], "language": "en"}
        params = {"output_format": "srt"}
        stats = {"active_sessions": 0}
        response = build_response(result, params, stats, "/fake/path", 100.0)
        assert "text/plain" in response.headers["content-type"]
        assert response.headers["Content-Disposition"] == "attachment; filename=\"path.en-ai.srt\"; filename*=UTF-8''path.en-ai.srt"

    def test_build_response_unicode_filename(self):
        """Test response building with unicode filename to ensure no encoding issues occur."""
        result = {"text": "Hello", "segments": [], "language": "en"}
        params = {"output_format": "srt"}
        stats = {"active_sessions": 0}
        unicode_path = "/movies/Liceenii Extemporal la dirigenție (1987) DVD-R.mkv"
        response = build_response(result, params, stats, unicode_path, 100.0)
        assert "text/plain" in response.headers["content-type"]
        cd_header = response.headers["Content-Disposition"]
        assert 'filename="Liceenii Extemporal la dirigenie (1987) DVD-R.en-ai.srt"' in cd_header
        assert "filename*=UTF-8''Liceenii%20Extemporal%20la%20dirigen%C8%9Bie%20%281987%29%20DVD-R.en-ai.srt" in cd_header


@pytest.mark.parametrize(
    ("query_params", "expected"),
    [
        (
            {
                "initial_prompt": "testprompt",
                "vad_filter": "false",
                "word_timestamps": "true",
                "max_line_width": "40",
                "max_line_count": "2",
            },
            {
                "initial_prompt": "testprompt",
                "vad_filter": False,
                "word_timestamps": True,
                "max_line_width": 40,
                "max_line_count": 2,
            },
        ),
        (
            {},
            {
                "initial_prompt": None,
                "vad_filter": True,
                "word_timestamps": False,
                "max_line_width": None,
                "max_line_count": None,
            },
        ),
        (
            {"max_line_width": "invalid", "max_line_count": "invalid"},
            {"max_line_width": None, "max_line_count": None},
        ),
    ],
)
def test_routes_extract_new_params(query_params, expected):
    """Verify that ASR routes correctly parse new parameters."""
    mock_request = mock.MagicMock()
    mock_request.query_params = query_params
    mock_request.url.path = "/asr"
    params = asyncio.run(get_request_params(mock_request, {}))
    for key, value in expected.items():
        assert params[key] == value


@pytest.mark.parametrize(
    ("qp", "fd", "expected"),
    [
        ({"vad_filter": "1"}, {}, True),
        ({"vad_filter": "yes"}, {}, True),
        ({}, {"vad_filter": "1"}, True),
        ({}, {"vad_filter": "yes"}, True),
        ({"vad_filter": "true"}, {}, True),
        ({}, {"vad_filter": "true"}, True),
    ],
)
def test_parse_bool_param_spellings(qp, fd, expected):
    """Verify _parse_bool_param handles '1' and 'yes' from query_params and form_data."""
    assert _parse_bool_param(qp, fd, "vad_filter", False) is expected
