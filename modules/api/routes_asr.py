"""
ASR Transcription Routes for Whisper Pro ASR
"""
import logging
import time
import os
import json
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
      - name: diarize
        in: formData
        type: boolean
        default: false
        description: Whether to perform speaker diarization.
      - name: min_speakers
        in: formData
        type: integer
        description: Minimum number of speakers.
      - name: max_speakers
        in: formData
        type: integer
        description: Maximum number of speakers.
      - name: hf_token
        in: formData
        type: string
        description: Hugging Face authentication token (overrides env).
      - name: initial_prompt
        in: formData
        type: string
        description: Text to guide transcription spelling/style.
      - name: vad_filter
        in: formData
        type: boolean
        default: true
        description: Whether to enable voice activity detection.
      - name: word_timestamps
        in: formData
        type: boolean
        default: false
        description: Generate word-level timestamps.
      - name: max_line_width
        in: formData
        type: integer
        description: Maximum characters per subtitle line.
      - name: max_line_count
        in: formData
        type: integer
        description: Maximum lines per subtitle block.
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
    filename = routes_utils.get_display_name_early()

    task_type = "Translation" if params.get('task') == 'translate' else "Transcription"
    logger.info("    Task: %s | Format: %s | Lang: %s",
                task_type.upper(), params.get('output_format', 'srt').upper(),
                params.get('language') or 'auto-detect')
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
        routes_utils.cleanup_files()
        model_manager.decrement_active_session()


def _perform_transcription_task(params, task_type):
    """Inner logic for transcription execution."""
    try:
        source_path, _, err = routes_utils.initialize_task_context(is_priority=False)
        if err:
            return None, None, err

        model_manager.update_task_progress(None, f"{task_type} Initializing")

        # --- Stage 1: FFmpeg normalization ---
        clean_wav = None
        if not config.ENABLE_VOCAL_SEPARATION:
            clean_wav, err = routes_utils.get_clean_wav_or_error(source_path)
            if err:
                return None, source_path, err

        # --- Stage 2: Language detection on clean audio (post-FFmpeg, pre-vocal separation) ---
        # Use the clean WAV when available for the most accurate fast detection.
        # Falls back to source_path when vocal separation is enabled (ffmpeg runs inside UVR).
        detection_target = clean_wav if clean_wav else source_path
        lang = _detect_lang_if_needed(params.get('language'), detection_target)

        # --- Stage 3: run_transcription (vocal separation → inference) ---
        target_audio = clean_wav if clean_wav else source_path
        result = model_manager.run_transcription(
            target_audio,
            lang,
            params['task'],
            diarize=params.get('diarize', False),
            min_speakers=params.get('min_speakers'),
            max_speakers=params.get('max_speakers'),
            hf_token=params.get('hf_token'),
            initial_prompt=params.get('initial_prompt'),
            vad_filter=params.get('vad_filter', True),
            word_timestamps=params.get('word_timestamps', False)
        )

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

    # Speaker Diarization parameters
    diarize_val = request.args.get('diarize') or request.form.get('diarize') or 'false'
    params['diarize'] = str(diarize_val).lower() == 'true'

    try:
        min_spk = request.args.get('min_speakers') or request.form.get('min_speakers')
        params['min_speakers'] = int(min_spk) if min_spk else None
    except (ValueError, TypeError):
        params['min_speakers'] = None

    try:
        max_spk = request.args.get('max_speakers') or request.form.get('max_speakers')
        params['max_speakers'] = int(max_spk) if max_spk else None
    except (ValueError, TypeError):
        params['max_speakers'] = None

    params['hf_token'] = request.args.get('hf_token') or request.form.get('hf_token') or request.headers.get('X-HF-Token')

    # New ASR parameters
    params['initial_prompt'] = request.args.get('initial_prompt') or request.form.get('initial_prompt')

    vad_val = request.args.get('vad_filter') or request.form.get('vad_filter') or 'true'
    params['vad_filter'] = str(vad_val).lower() == 'true'

    word_ts_val = request.args.get('word_timestamps') or request.form.get('word_timestamps') or 'false'
    params['word_timestamps'] = str(word_ts_val).lower() == 'true'

    try:
        mlw = request.args.get('max_line_width') or request.form.get('max_line_width')
        params['max_line_width'] = int(mlw) if mlw else None
    except (ValueError, TypeError):
        params['max_line_width'] = None

    try:
        mlc = request.args.get('max_line_count') or request.form.get('max_line_count')
        params['max_line_count'] = int(mlc) if mlc else None
    except (ValueError, TypeError):
        params['max_line_count'] = None

    return params


def _detect_lang_if_needed(lang, path):
    """Run voting language detection on the provided audio path if no language was specified.

    This is called on the post-FFmpeg normalised WAV (when available) so that
    the detector operates on clean, standardised audio rather than the raw
    source upload, improving identification accuracy before vocal separation begins.
    """
    if not lang:
        model_manager.update_task_progress(0, "Language Detection")
        logger.info("[LD] Running language detection on post-FFmpeg audio: %s",
                    os.path.basename(path))
        res = language_detection.run_voting_detection(path, model_manager)
        loggable = {k: v for k, v in res.items() if k != 'logs'}
        logger.info("LD Response JSON: %s", json.dumps(loggable, ensure_ascii=False, indent=None))
        return res.get('detected_language')
    return lang


def _build_response(result, params, stats, path, start):
    """Format final response."""
    stats['total'] = time.time() - start
    stats['video_dur'] = result.get('video_duration_sec', 0.0)
    stats['language'] = result.get('language')

    perf = result.get('performance', {})
    task_type = "TRANSLATE" if params.get('task') == 'translate' else "TRANSCRIBE"
    logger.info("ASR Completed | Type: %s | Lang: %s | Total: %s | Queue: %.2fs | Isolation: %.2fs | Inference: %.2fs",
                task_type, stats['language'], utils.format_duration(stats['total']),
                perf.get('queue_sec', 0), perf.get('isolation_sec', 0), perf.get('inference_sec', 0))

    fmt = params['output_format'].lower()
    if fmt == 'json':
        return jsonify(result)

    content = ""
    if fmt == 'vtt':
        content = utils.generate_vtt(
            result,
            max_line_width=params.get('max_line_width'),
            max_line_count=params.get('max_line_count')
        )
    elif fmt == 'txt':
        content = utils.generate_txt(result)
    elif fmt == 'tsv':
        content = utils.generate_tsv(result)
    else:
        content = utils.generate_srt(
            result,
            max_line_width=params.get('max_line_width'),
            max_line_count=params.get('max_line_count')
        )

    resp = Response(content, mimetype='text/plain')
    fname = os.path.splitext(os.path.basename(path))[0]
    resp.headers["Content-Disposition"] = f'attachment; filename="{fname}.{fmt}"'
    return resp
