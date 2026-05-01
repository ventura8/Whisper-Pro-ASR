"""
Language Detection Routes for Whisper Pro ASR
"""
import logging
import time
from flask import Blueprint, jsonify, request  # pylint: disable=import-error
from modules.inference import model_manager, language_detection
from modules import utils
from modules.api import routes_utils

bp = Blueprint('detect', __name__)
logger = logging.getLogger(__name__)


@bp.route('/detect-language', methods=['POST'])
def detect_language():
    """
    Automated Language Identification
    ---
    tags:
      - Identification
    summary: Identify the primary language of an audio stream.
    description: |
      Analyzes the first 30 seconds of audio using a high-precision voting consensus.
      Returns the ISO 639-1 language code and a confidence score.
    consumes:
      - multipart/form-data
    produces:
      - application/json
    parameters:
      - name: audio_file
        in: formData
        type: file
        description: Binary audio/video file for language detection.
      - name: local_path
        in: formData
        type: string
        description: Absolute path to a local file (if mapped in volumes).
    responses:
      200:
        description: Detection successful.
      400:
        description: No audio source provided or media corrupted.
      503:
        description: Model engine not initialized.
    """
    if not model_manager.is_engine_initialized():
        return "Model not loaded", 503

    model_manager.increment_active_session()
    params = {**request.args.to_dict(), **request.form.to_dict()}
    utils.THREAD_CONTEXT.request_json = params
    utils.THREAD_CONTEXT.caller_info = {
        "ip": request.remote_addr,
        "user_agent": request.headers.get('User-Agent', 'Unknown')
    }
    start_time = time.time()
    filename = routes_utils.get_display_name_early()

    try:
        with model_manager.early_task_registration(task_type="Language Detection", filename=filename, is_priority=True):
            try:
                source_path, _, err = routes_utils.initialize_task_context(
                    is_priority=True)
                if err:
                    return err

                model_manager.update_task_progress(None, "Analyzing Stream")
                result = language_detection.run_voting_detection(
                    source_path, model_manager, start_time=start_time)

                _log_detection_result(result, start_time)
                model_manager.update_task_metadata(result=result)
                return jsonify(result)
            except Exception as e:
                model_manager.update_task_metadata(result={"error": str(e)})
                # pylint: disable=broad-exception-raised
                raise e
    except (ValueError, RuntimeError, IOError) as e:
        return routes_utils.handle_error(e, "LD")
    except Exception as e:  # pylint: disable=broad-exception-caught
        return routes_utils.handle_error(e, "LD")
    finally:
        routes_utils.cleanup_files()
        model_manager.decrement_active_session()


def _log_detection_result(result, start_time):
    """Log identification details."""
    elapsed = time.time() - start_time
    detected_lang = result.get('detected_language', 'unknown')
    detected_conf = result.get('confidence', 0) * 100

    candidates = result.get('voting_details') or result.get('all_probabilities') or {}
    scans = result.get('segment_count', 1)
    top_3 = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:3]
    cand_str = ", ".join([f"{k}:{v*100:.1f}%" for k, v in top_3])

    perf = result.get('performance', {})
    q_dur = utils.format_duration(perf.get('queue_sec', 0))
    m_dur = utils.format_duration(perf.get('montage_sec', 0))
    s_dur = utils.format_duration(perf.get('isolation_sec', 0))
    i_dur = utils.format_duration(perf.get('inference_sec', 0))
    perf_str = f"Queue:{q_dur} | Montage:{m_dur} | Isolation:{s_dur} | Inference:{i_dur}"

    logger.info("LD Completed | Lang: %s (%.1f%%) | Segments: %d | Rank: %s | Phases: %s | Total: %s",
                detected_lang, detected_conf, scans, cand_str, perf_str, utils.format_duration(elapsed))
