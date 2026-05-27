"""
HTML template for the Whisper Pro Analytics Dashboard.
"""
import os


def get_analytics_html() -> str:
    """Returns the rendered HTML for the analytics page."""
    template_path = os.path.join(os.path.dirname(__file__), "analytics.html")
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()
