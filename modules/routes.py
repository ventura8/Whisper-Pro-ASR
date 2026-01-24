"""
Flask API Endpoints for ASR and Language Detection

This module defines the RESTful interface for the Whisper Pro ASR service,
including health checks, hardware status, language identification, and 
multi-format transcription.
"""
# pylint: disable=broad-exception-caught
import logging
import os
import tempfile
import traceback
import time
import uuid

from flask import Blueprint, request, jsonify, Response  # pylint: disable=import-error
from . import config, model_manager, utils, language_detection

# Blueprint for modular routing
bp = Blueprint('api', __name__)
logger = logging.getLogger(__name__)

# --- [CORE SERVICE ENDPOINTS] ---


@bp.route('/', methods=['GET'])
def root():
    """
    Service Health Check
    ---
    tags:
      - System
    summary: Verify service availability.
    description: |
      Exhaustive heartbeat endpoint providing basic service identity and status.
    produces:
      - application/json
    responses:
      200:
        description: Service is reachable and healthy.
        schema:
          type: object
          required: [message, status, app, version]
          properties:
            message:
              type: string
              description: Human-readable healthy confirmation.
              example: "Whisper ASR Webservice is working"
            status:
              type: string
              description: Current operational status.
              example: "healthy"
            app:
              type: string
              description: Application name.
              example: "Whisper Pro ASR"
            version:
              type: string
              description: Version string.
              example: "1.0.0"
    """
    logger.info("[System] Health check: OK")
    return jsonify({
        "message": "Whisper ASR Webservice is working",
        "status": "healthy",
        "app": config.APP_NAME,
        "version": config.VERSION
    })


@bp.route('/status', methods=['GET'])
def status():
    """
    Hardware and Model Diagnostics
    ---
    tags:
      - System
    summary: Retrieve detailed system state.
    description: |
      Returns the current application version, loaded ASR model identity, and 
      hardware engine status.
    produces:
      - application/json
    responses:
      200:
        description: Detailed system status.
        schema:
          type: object
          required: [version, model, status]
          properties:
            version:
              type: string
              description: Semantic application version.
              example: "1.0.0"
            model:
              type: string
              description: Relative or absolute path to the Whisper model weights.
              example: "Systran/faster-whisper-large-v3"
            status:
              type: string
              description: Readiness state of the model manager.
              enum: [loaded, failed, warming_up]
              example: "loaded"
    """
    # pylint: disable=no-member
    model_status = "loaded" if model_manager.WHISPER else "failed"
    logger.info("[System] Status check: model=%s, version=%s",
                model_status, config.VERSION)
    return jsonify({
        "version": config.VERSION,
        "model": config.MODEL_ID,
        "status": model_status
    })


@bp.route('/admin', methods=['GET', 'POST'])
def admin():
    """
    Administrative Interface
    ---
    tags:
      - Management
    summary: Service administration portal.
    description: |
      Administrative endpoint for service management. Currently returns a reachability
      status and basic diagnostic links.
    produces:
      - application/json
    responses:
      200:
        description: Administration service is reachable.
        schema:
          type: object
          required: [status, authenticated, endpoints]
          properties:
            status:
              type: string
              description: Heartbeat status of the admin module.
              example: "OK"
            authenticated:
              type: boolean
              description: Whether the request was authenticated.
              example: false
            endpoints:
              type: array
              description: List of available management endpoints.
              items:
                type: string
              example: ["/status", "/asr", "/detect-language"]
    """
    logger.info("Admin service endpoint accessed.")
    return jsonify({
        "status": "OK",
        "authenticated": False,
        "endpoints": ["/status", "/asr", "/detect-language"]
    })


@bp.route('/detect-language', methods=['POST'])
def detect_language():
    """
    Automated Language Identification
    ---
    tags:
      - Identification
    summary: Detect source language of media.
    description: |
      Analyzes an audio file to identify the spoken language.
      Uses Smart Sampling and multi-zone voting for high precision (99%+ accuracy).

      **Prioritization**:
      1. **Local Path**: Direct container path if provided.
      2. **Upload**: Binary fallback if path is invalid.
    consumes:
      - multipart/form-data
    produces:
      - application/json
    parameters:
      - name: audio_file
        in: formData
        type: file
        description: Binary file upload. Used as a fallback if local_path is missing or invalid.
      - name: file
        in: formData
        type: file
        description: Binary file upload alias.
      - name: local_path
        in: formData
        type: string
        description: Absolute server-side file path. Preferred for performance (Volume mapping).
      - name: original_path
        in: formData
        type: string
        description: Alias for local_path.
      - name: file_path
        in: formData
        type: string
        description: Alias for local_path.
      - name: video_file
        in: formData
        type: string
        description: Alias for local_path.
      - name: encode
        in: formData
        type: boolean
        default: true
        description: Encode audio first through FFmpeg. Included for API compatibility.
    responses:
      200:
        description: Language identified successfully.
        schema:
          type: object
          required: [detected_language, confidence, chunks_processed, language, language_code]
          properties:
            detected_language:
              type: string
              description: Winning language Name.
              example: "english"
            language:
              type: string
              description: Identity of the detected language (Full Name).
              example: "english"
            language_code:
              type: string
              description: Identity of the detected language code.
              example: "en"
            confidence:
              type: number
              description: Score between 0.0 and 1.0.
              example: 0.998
            chunks_processed:
              type: integer
              description: Total scans performed before voting consensus.
              example: 5
            voting_details:
              type: object
              description: Exhaustive map of candidate probabilities across all scanned zones.
              additionalProperties:
                type: number
              example: {"en": 0.95, "fr": 0.02, "de": 0.01}
            all_probabilities:
              type: object
              description: Raw candidate map from the initial inference pass.
              additionalProperties:
                type: number
              example: {"en": 0.998, "fr": 0.001}
      400:
        description: Malformed request - neither a valid local path nor an uploaded file was found.
      503:
        description: Inference engine unavailable or still warming up.
    """
    start_time = time.time()
    logger.info("[LD] Starting language detection request...")

    if model_manager.WHISPER is None:
        return "Model not loaded. Check server logs.", 503

    # Acquire priority lock (pauses ongoing ASR tasks)
    model_manager.request_priority()
    temp_path, clean_wav = None, None
    try:
        source_path, temp_path = _prepare_source_path()
        if not source_path:
            return "No input provided", 400

        # Execution Phase (Multi-Zone Voting)
        # Optimization: Pass the source_path (original video/audio) directly.
        # language_detection will use FFmpeg to extract chunks efficiently.
        result = language_detection.run_voting_detection(
            source_path, model_manager)

        if result and "detected_language" in result:
            code = result["detected_language"]
            result["detected_language"] = utils.LANGUAGES.get(
                code, code).lower()

        _log_detection_result(result, start_time)
        return jsonify(result)

    except ValueError as err:
        logger.warning("[LD] Input validation failed: %s", err)
        return str(err), 400
    except FileNotFoundError as err:
        logger.error("Language detection failed (Missing File): %s", err)
        return str(err), 404
    except Exception as err:
        logger.error("Language detection failed (Runtime Error): %s", err)
        return f"Error: {str(err)}", 500
    finally:
        # Crucial: Always release lock
        model_manager.release_priority()
        _cleanup_files(temp_path, clean_wav)


# --- [TRANSCRIPTION ENDPOINTS] ---

@bp.route('/asr', methods=['POST', 'GET'])
@bp.route('/v1/audio/transcriptions', methods=['POST'])
@bp.route('/v1/audio/translations', methods=['POST'])
def transcribe():
    """
    High-Precision Audio Transcription (ASR)
    ---
    tags:
      - Transcription
    summary: Convert speech to text/subtitles with hardware acceleration.
    description: |
      Processes media files into high-accuracy SRT or JSON. 
      Supports native OpenVINO/CUDA acceleration and inherits full OpenAI compatibility.
      GET requests return 'ready' if the engine is initialized.

      **Prioritization**:
      1. **Local Path**: Direct container path if provided.
      2. **Upload**: Binary fallback if path is invalid.
    consumes:
      - multipart/form-data
    produces:
      - text/plain
      - application/json
    parameters:
      - name: audio_file
        in: formData
        type: file
        description: Binary media file. Used as a fallback if local_path is missing or invalid.
      - name: file
        in: formData
        type: file
        description: Binary media file alias.
      - name: local_path
        in: formData
        type: string
        description: Absolute server-side path. Preferred for performance (Volume mapping).
      - name: task
        in: formData
        type: string
        enum: [transcribe, translate]
        default: transcribe
        description: Whether to perform transcription or translation to English.
      - name: language
        in: formData
        type: string
        description: ISO-639-1 language code. Automated detection if omitted.
        example: "en"
      - name: response_format
        in: formData
        type: string
        enum: [srt, json, verbose_json, vtt, txt, tsv]
        default: srt
        description: Desired output format. 'verbose_json' maps to 'json'.
      - name: batch_size
        in: formData
        type: integer
        default: 1
        minimum: 1
        maximum: 8
        description: Number of parallel segments to process (Resource intensive).
      - name: original_path
        in: formData
        type: string
        description: Alias for local_path.
      - name: file_path
        in: formData
        type: string
        description: Alias for local_path.
      - name: video_file
        in: formData
        type: string
        description: Alias for local_path.
      - name: source_lang
        in: formData
        type: string
        description: Alias for language parameter.
      - name: output
        in: formData
        type: string
        enum: [srt, json, verbose_json, vtt, txt, tsv]
        description: Alias for response_format.
      - name: encode
        in: formData
        type: boolean
        default: true
        description: Encode audio first through FFmpeg. Included for API compatibility.
    responses:
      200:
        description: Transcription successfully completed.
        schema:
          type: object
          required: 
            - text
            - chunks
            - language
            - language_probability
            - video_duration_sec
            - transcription_duration
          properties:
            text:
              type: string
              description: Normalized full transcript text.
              example: "Hello world"
            chunks:
              type: array
              description: Chronological list of identified audio segments and their timestamps.
              items:
                type: object
                properties:
                  text:
                    type: string
                    description: Transcribed text of the segment.
                  start:
                    type: number
                    description: Start time in seconds.
                  end:
                    type: number
                    description: End time in seconds.
                  timestamp:
                    type: array
                    items:
                      type: number
                    description: Start and end relative timestamps.
                    example: [0.0, 5.0]
                  probability:
                    type: number
                    description: Confidence score for this segment (0-1).
            segments:
              type: array
              description: Alias for 'chunks' (Compatibility field).
              items:
                type: object
                properties:
                  text:
                    type: string
                    description: Transcribed text of the segment.
                  start:
                    type: number
                    description: Start time in seconds.
                  end:
                    type: number
                    description: End time in seconds.
                  timestamp:
                    type: array
                    items:
                      type: number
                    description: Start and end relative timestamps.
                    example: [0.0, 5.0]
                  probability:
                    type: number
                    description: Confidence score for this segment (0-1).
            language:
              type: string
              description: ISO-639-1 code of the detected or specified language.
              example: "en"
            language_probability:
              type: number
              description: Confidence score of the language identification.
              example: 0.99
            video_duration_sec:
              type: number
              description: Total length of the processed media in seconds.
              example: 120.5
            transcription_duration:
              type: number
              description: Raw inference time in seconds.
              example: 4.2
            post_processing_duration:
              type: number
              description: Time spent on filters/formatting.
              example: 0.1
            extra_preprocess_duration:
              type: number
              description: Time spent on vocal isolation (if enabled).
              example: 1.5
      400:
        description: Invalid request - missing file or unsupported parameters.
      503:
        description: Inference engine unavailable or resource depleted.
    """
    # pylint: disable=too-many-locals
    resp, done = _check_transcribe_preconditions()
    if done:
        return resp

    # Passive parameter extraction - matches OpenAI / Standard behaviors
    params = _get_all_request_params()

    # Unpack parameters
    language = params.get('language')
    task = params.get('task', 'transcribe')
    batch_size = params.get('batch_size', config.DEFAULT_BATCH_SIZE)
    output_format = params.get('output_format', 'srt')

    request_start_time = time.time()
    preprocess_start = time.time()

    temp_path, clean_wav = None, None
    try:
        # Source Resolution (Optimized Path Search + Upload Fallback)
        source_path, temp_path = _prepare_source_path()
        if not source_path:
            return "No valid audio source or file provided.", 400

        # Dynamic Language Detection (Triggered if language is missing from request)
        # Optimization: Perform detection on source_path BEFORE full normalization
        model_manager.wait_for_priority()
        language = _detect_language_if_needed(language, source_path)

        # Audio Standardization Phase (FFmpeg)
        clean_wav, err_resp = _get_clean_wav_or_error(source_path)
        if err_resp:
            return err_resp

        stats = {'preprocess_dur': time.time() - preprocess_start}

        # ASR Inference Phase
        transcribe_start = time.time()
        result = model_manager.run_transcription(
            clean_wav, language, task, batch_size)
        stats['transcribe_dur'] = time.time() - transcribe_start

        # Post-Processing Phase (Hallucination removal & Timing Correction)
        stats['video_dur'] = result.get('video_duration_sec', 0.0)
        stats['post_dur'] = result.get('post_processing_duration', 0.0)

        # Consolidate timing stats for the final summary log
        stats['total_proc_dur'] = time.time() - request_start_time
        stats['transcribe_dur'] = max(
            0, stats['transcribe_dur'] - stats['post_dur'])
        extra_prep = result.get('extra_preprocess_duration', 0.0)
        stats['preprocess_dur'] += extra_prep
        stats['transcribe_dur'] = max(0, stats['transcribe_dur'] - extra_prep)

        stats['language'] = result.get('language')
        stats['language_prob'] = result.get('language_probability', 0.0)
        stats['chunk_count'] = len(result.get('chunks', []))
        stats['text_len'] = len(result.get('text', ''))

        _log_request_stats(stats)

        # Output Generation
        return _format_transcription_response(result, output_format)

    except Exception as err:
        status_code = 500
        if isinstance(err, ValueError):
            logger.warning("[ASR] Local path missing: %s", err)
            status_code = 400
        elif isinstance(err, FileNotFoundError):
            status_code = 404
        else:
            logger.error("TRANSCRIPTION CRITICAL: %s\n%s",
                         err, traceback.format_exc())

        msg = str(err) if status_code != 500 else f"Service Error: {str(err)}"
        return msg, status_code

    finally:
        _cleanup_files(temp_path, clean_wav)


# --- [HELPER LOGIC & DISK UTILITIES] ---

def _get_clean_wav_or_error(source_path):
    """Normalize input media to 16kHz mono WAV for high-compatibility inference."""
    logger.info("[Prep] Normalizing audio stream (FFmpeg)...")
    start = time.time()

    # Validation Phase: Check for null-padding before calling FFmpeg
    try:
        if os.path.exists(source_path):
            with open(source_path, 'rb') as f:
                header = f.read(1024)
                if len(header) > 0 and all(b == 0 for b in header):
                    logger.error(
                        "[Prep] Input file is corrupted: contains only null bytes (zeros).")
                    err_msg = (
                        "Input file is corrupted (only null bytes). "
                        "Check client-side transcoding (Bazarr/App)."
                    )
                    return None, (err_msg, 400)
    except Exception:
        pass

    clean_wav = utils.convert_to_wav(source_path)
    if not clean_wav:
        return None, ("FFmpeg conversion failed - invalid or corrupted media format", 400)

    logger.info(
        "[Prep] Standardization completed in %s",
        utils.format_duration(time.time() - start)
    )
    return clean_wav, None


def _cleanup_files(*args):
    """Securely remove temporary processing assets from the filesystem."""
    for f_path in args:
        if f_path and os.path.exists(f_path):
            try:
                os.remove(f_path)
            except Exception:
                pass


def _check_transcribe_preconditions():
    """Verify engine readiness before accepting heavy workloads."""
    # pylint: disable=no-member
    if request.method == 'GET':
        m_status = "ready" if model_manager.WHISPER else "not_ready"
        return Response(m_status, mimetype='text/plain'), True

    if model_manager.WHISPER is None:
        return ("Model not loaded. Service is warming up or failed to start.", 503), True

    return None, False


def _format_transcription_response(result, output_format):
    """Encode the final result into the requested delivery format."""

    # 1. JSON Responses
    if output_format == 'json':
        # Add 'segments' alias for compatibility with other Whisper APIs
        if result and 'chunks' in result:
            result['segments'] = result['chunks']
        return jsonify(result)

    # 2. Text-based Formats with file download headers
    content = ""
    mime_type = "text/plain"
    ext = "txt"

    if output_format == 'vtt':
        content = utils.generate_vtt(result)
        mime_type = "text/vtt"
        ext = "vtt"
    elif output_format == 'srt':
        content = utils.generate_srt(result)
        mime_type = "text/plain"
        ext = "srt"
    elif output_format == 'tsv':
        content = utils.generate_tsv(result)
        mime_type = "text/tab-separated-values"
        ext = "tsv"
    elif output_format == 'txt':
        content = utils.generate_txt(result)
        mime_type = "text/plain"
        ext = "txt"
    else:
        # Default fallback
        content = utils.generate_srt(result)
        ext = "srt"

    # Create response with Content-Disposition for proper filename handling in browsers/Bazarr
    resp = Response(content, mimetype=mime_type)
    filename = f"transcription.{ext}"
    resp.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    return resp


def _prepare_source_path():
    """Resolve input media - 1. Local path mapping (Optimization), 2. Upload (Standard Fallback)."""
    raw_path = _get_raw_path()
    if raw_path:
        p = _resolve_local_path(raw_path)
        if p:
            return p, None

    # 2. Upload Processing (Standard Whisper Behavior / Fallback)
    tmp_path, temp_path = _handle_upload()
    if tmp_path:
        return tmp_path, temp_path

    if raw_path:
        raise ValueError(
            f"Path not accessible: {raw_path} (Volumes unmapped and no audio data attached)")

    return None, None


def _get_raw_path():
    """Extract raw path from request arguments or form data."""
    path_keys = ['video_file', 'local_path', 'file_path', 'original_path']
    for key in path_keys:
        val = request.args.get(key) or request.form.get(key)
        if val:
            return val
    return None


def _resolve_local_path(raw_path):
    """Check if the provided path exists locally."""
    clean_path = raw_path.strip().strip('"').strip("'")
    candidates = [
        clean_path,                  # Expectation: Volumes are mapped 1:1
        clean_path.replace('+', ' '),  # Decoded spaces
    ]

    for p in candidates:
        if p and os.path.exists(p):
            logger.info("[System] Optimization: Using Local Path -> %s", p)
            return p

    logger.debug(
        "[System] Optimization missed (Volume unmapped): %s. checking for upload...", raw_path)
    return None


def _handle_upload():
    """Handle binary file upload."""
    audio_file = request.files.get('audio_file') or request.files.get('file')
    if not audio_file:
        return None, None

    logger.info("[System] Ingesting remote data: %s", audio_file.filename)
    tmp_path = None
    try:
        # SAFETY: Ensure we're at the beginning of the binary stream
        if hasattr(audio_file.stream, 'seek'):
            try:
                audio_file.stream.seek(0)
            except Exception: # pylint: disable=broad-exception-caught
                pass

        # Save to temporary storage with extension hint
        ext = os.path.splitext(audio_file.filename)[
            1] if audio_file.filename else ".tmp"
        if len(ext) > 6:
            ext = ".tmp"

        # Generate unique safe path manually to avoid file handle conflicts
        tmp_path = os.path.join(
            tempfile.gettempdir(), f"upload_{uuid.uuid4().hex}{ext}")

        # Read into memory to verify content immediately
        file_data = audio_file.read()
        f_size = len(file_data)

        if f_size == 0:
            raise ValueError("Remote data stream is empty (0 bytes received).")

        # Check for null bytes (corruption/transcoding failure)
        if f_size > 1024 and all(b == 0 for b in file_data[:1024]):
            raise ValueError(
                "Input file is corrupted (contains only null bytes).")

        with open(tmp_path, 'wb') as f:
            f.write(file_data)

        logger.info("[System] Remote source ingestion successful: %d bytes", f_size)
        return tmp_path, tmp_path
    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise e

    return None, None


def _get_all_request_params():
    """Extract all metadata parameters from URL and Form without double-parsing."""
    params = {}

    # Extract Local Path
    params['local_path'] = _get_raw_path()

    # Extract Output Format
    output_keys = ['output', 'response_format']
    params['output_format'] = 'srt'
    for key in output_keys:
        val = request.args.get(key) or request.form.get(key)
        if val:
            params['output_format'] = val.lower()
            break

    # Extract Language
    lang_keys = ['language', 'source_lang']
    params['language'] = None
    for key in lang_keys:
        val = request.args.get(key) or request.form.get(key)
        if val:
            params['language'] = val
            break

    # Extract Task
    params['task'] = request.args.get('task') or request.form.get('task') or 'transcribe'
    if '/translations' in request.path:
        params['task'] = 'translate'

    # Extract Batch Size
    try:
        bs_str = request.args.get('batch_size') or request.form.get('batch_size')
        params['batch_size'] = int(bs_str) if bs_str else config.DEFAULT_BATCH_SIZE
    except (ValueError, TypeError):
        params['batch_size'] = config.DEFAULT_BATCH_SIZE

    # Extract Encode (Compatibility)
    encode_str = request.args.get('encode') or request.form.get('encode')
    params['encode'] = str(encode_str).lower() != 'false' if encode_str else True

    return params


def _get_transcription_options():
    """Extract transcription options from the request params."""
    params = _get_all_request_params()

    # Priority order for output format parameters
    output_format = params.get('output_format', 'srt')

    if output_format == 'verbose_json':
        output_format = 'json'

    language = params.get('language')
    task = params.get('task', 'transcribe')
    batch_size = params.get('batch_size', config.DEFAULT_BATCH_SIZE)

    return output_format, language, task, batch_size


def _detect_language_if_needed(language, clean_wav):
    """Trigger automated language identification if the source language is unknown."""
    if not language:
        logger.info(
            "[LD] Metadata: Language missing. Initiating consensus scan...")
        lang_result = language_detection.run_voting_detection(
            clean_wav, model_manager)
        language = lang_result.get('detected_language')
        confidence = lang_result.get('confidence', 0)
        logger.info("[LD] High-confidence winner: %s (Prob: %.1f%%)",
                    language, confidence * 100)
    return language


def _log_detection_result(result, start_time):
    """Log identification lifecycle details."""
    elapsed = time.time() - start_time
    detected_lang = result.get('detected_language', 'unknown')
    detected_conf = result.get('confidence', 0) * 100
    chunks = result.get('chunks_processed', 0)

    # Candidate ranking for better visibility
    candidates = result.get('voting_details') or result.get(
        'all_probabilities') or {}
    top_3 = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:3]
    cand_str = ", ".join([f"{k}:{v*100:.1f}%" for k, v in top_3])

    logger.info(
        "LD Completed | Lang: %s (%.1f%%) | Latency: %s | Chunks: %d | Rank: %s",
        detected_lang, detected_conf, utils.format_duration(
            elapsed), chunks, cand_str
    )


def _log_request_stats(stats):
    """Log comprehensive processing summary for the transcription task."""
    lang_info = f"{stats.get('language', 'unknown')}"
    if 'language_prob' in stats:
        lang_info += f" ({stats['language_prob']*100:.1f}%)"

    logger.info(
        "ASR Completed | Lang: %s | Source: %s | Total: %s | "
        "Pre: %s | Core: %s | Post: %s | Segments: %d",
        lang_info,
        utils.format_duration(stats['video_dur']),
        utils.format_duration(stats['total_proc_dur']),
        utils.format_duration(stats['preprocess_dur']),
        utils.format_duration(stats['transcribe_dur']),
        utils.format_duration(stats['post_dur']),
        stats.get('chunk_count', 0)
    )
