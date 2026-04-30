"""
ASR Transcription Routes for Whisper Pro ASR
"""
import logging
import time
import os
from flask import Blueprint, request, jsonify, Response  # pylint: disable=import-error
from modules import config
from modules.inference import model_manager, language_detection
from modules import utils
from modules.api import routes_utils

bp = Blueprint('asr', __name__)
logger = logging.getLogger(__name__)


@bp.route('/asr', methods=['POST', 'GET'])
@bp.route('/v1/audio/transcriptions', methods=['POST'])
@bp.route('/v1/audio/translations', methods=['POST'])
def transcribe():
    """
    High-Precision Audio Transcription (ASR)
    ---
    tags:
      - Transcription
    summary: Convert speech to text with hardware acceleration (OpenAI Compatible).
    description: |
      Processes media files into high-accuracy SRT, VTT, or JSON.
      Supports native OpenVINO (Intel) and CUDA (NVIDIA) acceleration.
      The /v1/audio/transcriptions and /v1/audio/translations endpoints are OpenAI-API compatible.
    consumes:
      - multipart/form-data
    produces:
      - text/plain
      - application/json
    parameters:
      - name: audio_file
        in: formData
        type: file
        description: Binary audio/video file for upload.
      - name: local_path
        in: formData
        type: string
        description: Absolute path to a local file (if mapped in volumes).
      - name: task
        in: formData
        type: string
        enum: [transcribe, translate]
        default: transcribe
        description: Whether to transcribe or translate to English.
      - name: language
        in: formData
        type: string
        description: ISO 639-1 language code (e.g., 'en', 'fr'). Auto-detected if omitted.
      - name: response_format
        in: formData
        type: string
        enum: [srt, json, verbose_json, vtt, txt, tsv]
        default: srt
        description: Desired output format.
      - name: batch_size
        in: formData
        type: integer
        default: 1
        description: Number of parallel segments for inference.
    responses:
      200:
        description: Transcription successfully completed.
      400:
        description: Invalid request parameters or corrupted file.
      503:
        description: Model engine not initialized.
    """
    if request.method == 'GET':
        status = "ready" if model_manager.is_engine_initialized() else "not_ready"
        return Response(status, mimetype='text/plain')

    if not model_manager.is_engine_initialized():
        return "Model not loaded", 503

    model_manager.increment_active_session()
    params = _get_request_params()
    utils.THREAD_CONTEXT.caller_info = {
        "ip": request.remote_addr,
        "user_agent": request.headers.get('User-Agent', 'Unknown')
    }
    utils.THREAD_CONTEXT.request_json = params

    start_time = time.time()
    temp_path, clean_wav = None, None
    filename = routes_utils.get_display_name_early()

    task_type = "Translation" if params.get('task') == 'translate' else "Transcription"
    try:
        with model_manager.early_task_registration(task_type=task_type, filename=filename):
            result, source_path, err = _perform_transcription_task(params, task_type)
            if err:
                return err
            return _build_response(result, params, {'total': time.time() - start_time}, source_path, start_time)
    except (ValueError, RuntimeError, IOError) as e:
        return routes_utils.handle_error(e)
    except Exception as e:  # pylint: disable=broad-exception-caught
        return routes_utils.handle_error(e)
    finally:
        routes_utils.cleanup_files(temp_path, clean_wav)
        model_manager.decrement_active_session()


def _perform_transcription_task(params, task_type):
    """Inner logic for transcription execution."""
    try:
        source_path, _, err = routes_utils.initialize_task_context(is_priority=False)
        if err:
            return None, None, err

        model_manager.update_task_progress(None, f"{task_type} Initializing")
        lang = _detect_lang_if_needed(params.get('language'), source_path)

        clean_wav = None
        if not config.ENABLE_VOCAL_SEPARATION:
            clean_wav, err = routes_utils.get_clean_wav_or_error(source_path)
            if err:
                return None, source_path, err

        target_audio = clean_wav if clean_wav else source_path
        result = model_manager.run_transcription(target_audio, lang, params['task'])

        if result:
            model_manager.update_task_metadata(result=result)
        return result, source_path, None
    except Exception as e:
        model_manager.update_task_metadata(result={"error": str(e)})
        raise e


def _get_request_params():
    """Extract parameters from request."""
    params = {
        'task': request.args.get('task') or request.form.get('task') or 'transcribe',
        'language': request.args.get('language') or request.form.get('source_lang') or request.form.get('language'),
        'output_format': (
            request.args.get('response_format') or
            request.args.get('output') or
            request.form.get('output') or
            request.form.get('response_format') or
            'srt'
        )
    }
    if '/translations' in request.path:
        params['task'] = 'translate'

    try:
        bs = request.args.get('batch_size') or request.form.get('batch_size')
        params['batch_size'] = int(bs) if bs else config.DEFAULT_BATCH_SIZE
    except (ValueError, TypeError):
        params['batch_size'] = config.DEFAULT_BATCH_SIZE
    return params


def _detect_lang_if_needed(lang, path):
    """Run detection if language is unknown."""
    if not lang:
        model_manager.update_task_progress(0, "Language Detection")
        res = language_detection.run_voting_detection(path, model_manager)
        return res.get('detected_language')
    return lang


def _build_response(result, params, stats, path, start):
    """Format final response."""
    stats['total'] = time.time() - start
    stats['video_dur'] = result.get('video_duration_sec', 0.0)
    stats['language'] = result.get('language')

    perf = result.get('performance', {})
    logger.info("ASR Completed | Lang: %s | Total: %s | Queue: %.2fs | Isolation: %.2fs | Inference: %.2fs",
                stats['language'], utils.format_duration(stats['total']),
                perf.get('queue_sec', 0), perf.get('isolation_sec', 0), perf.get('inference_sec', 0))

    fmt = params['output_format'].lower()
    if fmt == 'json':
        return jsonify(result)

    content = ""
    if fmt == 'vtt':
        content = utils.generate_vtt(result)
    elif fmt == 'txt':
        content = utils.generate_txt(result)
    else:
        content = utils.generate_srt(result)

    resp = Response(content, mimetype='text/plain')
    fname = os.path.splitext(os.path.basename(path))[0]
    resp.headers["Content-Disposition"] = f'attachment; filename="{fname}.{fmt}"'
    return resp
