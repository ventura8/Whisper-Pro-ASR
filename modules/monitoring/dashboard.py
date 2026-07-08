"""
Service Monitoring Dashboard Template
"""

from modules.core import utils
from modules.monitoring import analytics_ui, dashboard_ui, telemetry


def get_status_data():
    """Collects system and service metrics."""
    stats = telemetry.get_service_stats()
    stats.update({"system": utils.get_system_telemetry()})
    return stats


def get_dashboard_html():
    """Returns the rendered HTML for the monitoring dashboard."""
    return dashboard_ui.get_dashboard_html()


def get_analytics_html():
    """Returns the rendered HTML for the analytics dashboard."""
    return analytics_ui.get_analytics_html()
