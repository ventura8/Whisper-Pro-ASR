"""Tests for modules/monitoring/dashboard.py."""
from unittest import mock
from modules.monitoring import dashboard


def test_get_status_data():
    """Test aggregation of status data."""
    mock_service_stats = {"active_sessions": 2}
    mock_system_stats = {"cpu_percent": 10}

    with mock.patch("modules.monitoring.telemetry.get_service_stats", return_value=mock_service_stats):
        with mock.patch("modules.utils.get_system_telemetry", return_value=mock_system_stats):
            data = dashboard.get_status_data()
            assert data["active_sessions"] == 2
            assert data["system"]["cpu_percent"] == 10


def test_get_dashboard_html():
    """Test dashboard HTML retrieval."""
    with mock.patch("modules.monitoring.dashboard_ui.get_dashboard_html", return_value="<html>"):
        html = dashboard.get_dashboard_html()
        assert html == "<html>"
