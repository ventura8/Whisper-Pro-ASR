"""
Dashboard UI Components
"""

import os


def get_dashboard_html():
    """Returns the rendered HTML for the monitoring dashboard."""
    base_dir = os.path.dirname(__file__)
    templates_dir = os.path.join(base_dir, "templates")

    template_path = os.path.join(templates_dir, "dashboard.html")
    css_path = os.path.join(templates_dir, "dashboard.css")

    with open(template_path, "r", encoding="utf-8") as f:
        html = f.read()
    with open(css_path, "r", encoding="utf-8") as f:
        css = f.read()

    js_files = ["dashboard_state.js", "dashboard_utils.js", "dashboard_charts.js", "dashboard_main.js"]
    js_contents = []
    for js_file in js_files:
        js_path = os.path.join(templates_dir, js_file)
        with open(js_path, "r", encoding="utf-8") as f:
            js_contents.append(f.read())

    combined_js = "\n\n".join(js_contents)

    return html.replace("/* {{DASHBOARD_CSS}} */", css).replace("// {{DASHBOARD_JS}}", combined_js)
