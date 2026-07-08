"""Request logging edge-case tests for ASR route."""

from unittest import mock


class TestRequestLogging:
    """Tests for request body logging exception handling."""

    def test_body_logging_json_exception(self, routes_client):
        """Test body logging handles JSON parsing exceptions gracefully."""
        with mock.patch("modules.api.routes_asr.model_manager") as mock_mm:
            mock_mm.is_engine_initialized.return_value = True
            response = routes_client.post(
                "/asr",
                content="invalid json {",
                headers={"Content-Type": "application/json"},
            )
            assert response.status_code in [400, 500]

    def test_body_logging_form_exception(self, routes_client):
        """Test body logging handles form parsing exceptions gracefully."""
        with mock.patch("modules.api.routes_asr.model_manager") as mock_mm:
            mock_mm.is_engine_initialized.return_value = True
            response = routes_client.post("/asr", data={})
            assert response.status_code in [400, 500]
