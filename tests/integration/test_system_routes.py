"""Integration tests for system-level routes."""
# pylint: disable=redefined-outer-name,import-outside-toplevel
import json
from unittest import mock
import pytest
from whisper_pro_asr import create_app


def _unpack_response(resp):
    """Helper to handle Flask Response objects or tuples."""
    if isinstance(resp, tuple):
        return resp[0], resp[1]
    # If resp is a string (e.g. from direct function call return), mock a status_code
    if isinstance(resp, str):
        return resp, 200
    return resp, resp.status_code


@pytest.fixture
def client():
    """Setup Flask test client."""
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as test_client:
        yield test_client


def test_root_json(client):
    """Test health check JSON response."""
    response = client.get('/', headers={'Accept': 'application/json'})
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "status" in data
    assert data["status"] == "healthy"


def test_root_html(client):
    """Test dashboard HTML response."""
    with mock.patch("modules.monitoring.dashboard.get_dashboard_html", return_value="<html></html>"):
        response = client.get('/', headers={'Accept': 'text/html'})
        assert response.status_code == 200
        assert b"<html>" in response.data


def test_status_endpoint(client):
    """Test system status endpoint."""
    mock_data = {
        "telemetry": {"cpu": 10},
        "engines": {"whisper": {"status": "ready"}},
        "system": {"ram_usage_pct": 50, "gpu_load_pct": 0}
    }
    with mock.patch("modules.monitoring.dashboard.get_status_data", return_value=mock_data):
        response = client.get('/status')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "telemetry" in data
        assert "engines" in data


def test_settings_get(client):
    """Test retrieving system settings."""
    response = client.get('/system/settings')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "ASR_MODEL" in data


def test_settings_update(client):
    """Test updating system settings."""
    # We need to mock config and model reloading
    with mock.patch("modules.config.update_env") as mock_update, \
            mock.patch("modules.inference.model_manager.load_model") as mock_load:

        payload = {"ASR_MODEL": "small"}
        response = client.post('/system/settings',
                               data=json.dumps(payload),
                               content_type='application/json')

        assert response.status_code == 200
        mock_update.assert_called_once()
        mock_load.assert_called_once()


def test_history_endpoint(client):
    """Test task history retrieval."""
    with mock.patch("modules.monitoring.history_manager.get_history") as mock_history:
        mock_history.return_value = [{"task_id": "test"}]
        response = client.get('/system/history')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data) == 1
        assert data[0]["task_id"] == "test"


def test_stats_endpoint(client):
    """Test cumulative stats endpoint."""
    mock_data = {
        "telemetry": {"sessions": 5},
        "system": {"ram_usage_pct": 50, "gpu_load_pct": 0}
    }
    with mock.patch("modules.monitoring.dashboard.get_status_data", return_value=mock_data):
        response = client.get('/system/stats')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "telemetry" in data


def test_download_logs(client):
    """Test log download endpoint."""
    # Create a dummy log file
    import os
    from modules import config

    log_path = os.path.join(config.LOG_DIR, "whisper_pro.log")
    os.makedirs(config.LOG_DIR, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("test log")

    response = client.get('/logs/download')
    assert response.status_code == 200
    assert b"test log" in response.data


def test_clear_history(client):
    """Test history clearing endpoint."""
    with mock.patch("modules.monitoring.history_manager.clear_history") as mock_clear:
        response = client.post('/system/history/clear')
        assert response.status_code == 200
        mock_clear.assert_called_once()


def test_cleanup_trigger(client):
    """Test manual cleanup trigger."""
    with mock.patch("modules.utils.purge_temporary_assets") as mock_cleanup:
        response = client.post('/system/cleanup')
        assert response.status_code == 200
        mock_cleanup.assert_called_once()


def test_system_routes_telemetry_alias(client):
    """Cover the telemetry/system alias logic in /status."""
    mock_data = {"telemetry": {"cpu": 5}}
    with mock.patch("modules.monitoring.dashboard.get_status_data", return_value=mock_data):
        response = client.get('/status')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["system"]["cpu"] == 5


def test_system_routes_render_dashboard(client):
    """Cover render_dashboard endpoint."""
    with mock.patch("modules.monitoring.dashboard.get_dashboard_html", return_value="<h1>Dashboard</h1>"):
        response = client.get('/dashboard')
        assert response.status_code == 200
        assert b"Dashboard" in response.data


def test_system_routes_download_logs_error(client):
    """Cover exception in download_logs."""
    with mock.patch("os.path.exists", return_value=True), \
            mock.patch("modules.api.routes_system.send_from_directory", side_effect=Exception("Disk Error")):
        response = client.get('/logs/download')
        assert response.status_code == 500
        data = json.loads(response.data)
        assert "Disk Error" in data["error"]


def test_system_routes_update_settings_missing_data(client):
    """Cover missing data in update_settings."""
    response = client.post('/system/settings', data=json.dumps({}), content_type='application/json')
    assert response.status_code == 400


def test_system_routes_update_settings_all_fields(client):
    """Cover all updateable fields in settings."""
    payload = {
        "ASR_DEVICE": "cuda",
        "telemetry_retention_hours": 12,
        "log_retention_days": 7
    }
    with mock.patch("modules.config.update_env") as mock_upd, \
            mock.patch("modules.inference.model_manager.load_model"):
        response = client.post('/system/settings', data=json.dumps(payload), content_type='application/json')
        assert response.status_code == 200
        assert mock_upd.call_count == 3
