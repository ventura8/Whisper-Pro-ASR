"""
Service Monitoring Dashboard Template
"""
from modules.monitoring import telemetry
from modules import utils
from modules.monitoring import dashboard_ui


def get_status_data():
    """Collects system and service metrics."""
    stats = telemetry.get_service_stats()
    stats.update({
        "system": utils.get_system_telemetry()
    })
    return stats


def get_dashboard_html():
    """Returns the rendered HTML for the monitoring dashboard."""
    return dashboard_ui.get_dashboard_html()
