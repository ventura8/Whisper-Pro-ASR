"""
Dashboard UI Components
"""
import os


def get_dashboard_html():
    """Returns the rendered HTML for the monitoring dashboard."""
    template_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()
