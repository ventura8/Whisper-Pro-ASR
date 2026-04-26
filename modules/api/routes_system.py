"""
System and Diagnostic Routes for Whisper Pro ASR
"""
import logging
import os
from flask import Blueprint, request, jsonify, send_from_directory  # pylint: disable=import-error
from modules import config, utils
from modules.monitoring import dashboard, history_manager

bp = Blueprint('system', __name__)
logger = logging.getLogger(__name__)


@bp.route('/', methods=['GET'])
def root():
    """
    Service Health Check / Dashboard
    ---
    tags:
      - System
    summary: Check service status or view dashboard.
    description: Returns JSON health status or HTML dashboard depending on Accept header.
    responses:
      200:
        description: Service is healthy.
    """
    if 'text/html' in request.headers.get('Accept', ''):
        return dashboard.get_dashboard_html()

    logger.info("[System] Health check (JSON): OK")
    return jsonify({
        "message": "Whisper ASR Webservice is working",
        "status": "healthy",
        "app": config.APP_NAME,
        "version": config.VERSION,
        "dashboard": f"{request.host_url}dashboard"
    })


@bp.route('/status', methods=['GET'])
@bp.route('/system/stats', methods=['GET'])
def status():
    """
    Hardware and Model Diagnostics
    ---
    tags:
      - System
    summary: Get real-time telemetry and model state.
    description: Returns CPU, Memory, GPU/NPU utilization and active session counts.
    responses:
      200:
        description: Current system metrics.
    """
    stats = dashboard.get_status_data()
    # Add 'telemetry' alias for test compatibility if 'system' is present
    if "system" in stats:
        stats["telemetry"] = stats["system"]
    elif "telemetry" in stats:
        stats["system"] = stats["telemetry"]
    else:
        # Fallback for minimal mocks in tests
        stats["telemetry"] = {}
        stats["system"] = {}

    # Add 'engines' for test compatibility
    if "engines" not in stats:
        stats["engines"] = {}

    logger.debug("[System] Status check: %d active, %d queued",
                 stats.get("active_sessions", 0), stats.get("queued_sessions", 0))
    return jsonify(stats)


@bp.route('/history', methods=['GET'])
@bp.route('/system/history', methods=['GET'])
def get_history():
    """
    Retrieve full task history
    ---
    tags:
      - System
    summary: Get the list of recently completed and active tasks.
    responses:
      200:
        description: A list of task objects.
    """
    return jsonify(history_manager.get_history())


@bp.route('/system/history/clear', methods=['POST'])
def clear_history():
    """
    Purge all task history
    ---
    tags:
      - System
    summary: Clear all task records from the history manager.
    responses:
      200:
        description: History cleared successfully.
    """
    history_manager.clear_history()
    return jsonify({"status": "success", "message": "History cleared"})


@bp.route('/system/cleanup', methods=['POST'])
def trigger_cleanup():
    """
    Manually trigger temporary asset cleanup
    ---
    tags:
      - System
    summary: Force removal of old temporary audio files.
    responses:
      200:
        description: Cleanup routine triggered.
    """
    utils.purge_temporary_assets()
    return jsonify({"status": "success", "message": "Cleanup triggered"})


@bp.route('/dashboard', methods=['GET'])
def render_dashboard():
    """
    Direct Dashboard Access
    ---
    tags:
      - System
    summary: View the HTML monitoring dashboard.
    responses:
      200:
        description: Rendered HTML dashboard.
    """
    return dashboard.get_dashboard_html()


@bp.route('/logs/download', methods=['GET'])
def download_logs():
    """
    System Log Export
    ---
    tags:
      - System
    summary: Download the system log file.
    responses:
      200:
        description: Log file stream.
      404:
        description: Log file not found.
    """
    log_dir = config.LOG_DIR
    log_name = "whisper_pro.log"
    log_path = os.path.join(log_dir, log_name)

    # Fallback to TEMP_DIR for tests or environments without explicit log volumes
    if not os.path.exists(log_path):
        log_path = os.path.join(config.TEMP_DIR, log_name)
        log_dir = config.TEMP_DIR

    if not os.path.exists(log_path):
        logger.error("[System] Log download failed: File not found at %s", log_path)
        return jsonify({"error": "Log file not found"}), 404

    try:
        return send_from_directory(
            log_dir,
            log_name,
            as_attachment=True,
            mimetype='text/plain'
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("[System] Log download error: %s", e)
        return jsonify({"error": str(e)}), 500


@bp.route('/settings', methods=['GET', 'POST'])
@bp.route('/system/settings', methods=['GET', 'POST'])
def update_settings():
    """
    Dynamic Service Configuration
    ---
    tags:
      - System
    summary: View or update service settings at runtime.
    parameters:
      - name: body
        in: body
        required: false
        schema:
          type: object
          properties:
            ASR_MODEL:
              type: string
            ASR_DEVICE:
              type: string
            telemetry_retention_hours:
              type: integer
            log_retention_days:
              type: integer
    responses:
      200:
        description: Settings updated or current settings returned.
    """
    if request.method == 'GET':
        # Return current settings for test/UI compatibility
        return jsonify({
            "ASR_MODEL": config.ASR_MODEL,
            "ASR_DEVICE": config.ASR_DEVICE,
            "TELEMETRY_RETENTION_HOURS": os.environ.get('TELEMETRY_RETENTION_HOURS', 24)
        })

    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        if 'ASR_MODEL' in data:
            config.update_env('ASR_MODEL', data['ASR_MODEL'])
            logger.info("[Settings] ASR Model updated to %s", data['ASR_MODEL'])

        if 'ASR_DEVICE' in data:
            config.update_env('ASR_DEVICE', data['ASR_DEVICE'])
            logger.info("[Settings] ASR Device updated to %s", data['ASR_DEVICE'])

        if 'telemetry_retention_hours' in data:
            config.update_env('TELEMETRY_RETENTION_HOURS', data['telemetry_retention_hours'])
            logger.info("[Settings] Telemetry retention updated to %sh",
                        data['telemetry_retention_hours'])

        if 'log_retention_days' in data:
            config.update_env('LOG_RETENTION_DAYS', data['log_retention_days'])
            logger.info("[Settings] Log retention updated to %sd", data['log_retention_days'])

        # Reload model pool with new settings
        from modules.inference import model_manager  # pylint: disable=import-outside-toplevel
        model_manager.load_model()

        return jsonify({"status": "success", "message": "Settings updated"})
    except Exception as e:  # pylint: disable=broad-exception-caught
        return jsonify({"error": str(e)}), 500


@bp.route('/help', methods=['GET'])
def help_endpoint():
    """
    API Discovery
    ---
    tags:
      - System
    summary: Discover available endpoints.
    responses:
      200:
        description: List of endpoints.
    """
    return jsonify({
        "app": config.APP_NAME,
        "version": config.VERSION,
        "endpoints": ["/status", "/asr", "/detect-language", "/dashboard", "/logs/download", "/settings"],
        "docs": f"{request.host_url}docs"
    })
