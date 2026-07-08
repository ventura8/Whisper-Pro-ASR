"""
HTML template for the Whisper Pro Analytics Dashboard.
"""

import os


def get_analytics_html() -> str:
    """Returns the rendered HTML for the analytics page."""
    base_dir = os.path.dirname(__file__)
    templates_dir = os.path.join(base_dir, "templates")

    template_path = os.path.join(templates_dir, "analytics.html")
    css_path = os.path.join(templates_dir, "analytics.css")
    js_path = os.path.join(templates_dir, "analytics.js")

    with open(template_path, "r", encoding="utf-8") as f:
        html = f.read()
    with open(css_path, "r", encoding="utf-8") as f:
        css = f.read()
    with open(js_path, "r", encoding="utf-8") as f:
        js = f.read()

    return html.replace("/* {{ANALYTICS_CSS}} */", css).replace("// {{ANALYTICS_JS}}", js)
