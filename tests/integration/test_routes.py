"""Tests for modules/routes.py"""

import json
import io
from unittest import mock
from flask import Flask
from modules.api import routes_asr
from modules.api.routes_utils import cleanup_files, prepare_source_path
from modules.api.routes_asr import get_request_params, build_response
from modules import config


def test_root_get(routes_client):
    """Test GET / returns health status."""
    response = routes_client.get('/')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'healthy'


class TestStatusEndpoint:
    """Tests for the /status endpoint."""

    def test_status_model_loaded(self, routes_client):
        """Test /status when model is loaded."""
        with mock.patch('modules.api.routes_system.dashboard') as mock_db:
            mock_db.get_status_data.return_value = {
                'active_sessions': 0,
                'queued_sessions': 0,
                'hardware': []
            }
            response = routes_client.get('/status')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'active_sessions' in data

    def test_status_model_not_loaded(self, routes_client):
        """Test /status when model stats fail."""
        with mock.patch('modules.api.routes_system.dashboard') as mock_db:
            mock_db.get_status_data.return_value = {
                'active_sessions': 0,
                'queued_sessions': 0,
                'hardware': []
            }
            response = routes_client.get('/status')
            assert response.status_code == 200


class TestDetectLanguageEndpoint:
    """Tests for /detect-language endpoint."""

    def test_detect_language_no_model(self, routes_client):
        """Test detect-language when model not loaded."""
        with mock.patch('modules.api.routes_detect.model_manager') as mock_mm:
            mock_mm.is_engine_initialized.return_value = False
            response = routes_client.post('/detect-language')
            assert response.status_code == 503

    def test_detect_language_no_input(self, routes_client):
        """Test detect-language with no input."""
        with mock.patch('modules.api.routes_detect.model_manager') as mock_mm:
            mock_mm.is_engine_initialized.return_value = True
            response = routes_client.post('/detect-language')
            assert response.status_code == 400
            assert b"No audio source provided" in response.data

    def test_detect_language_file_not_found(self, routes_client):
        """Test detect-language with non-existent file."""
        with mock.patch('modules.api.routes_detect.model_manager') as mock_mm:
            mock_mm.is_engine_initialized.return_value = True
            response = routes_client.post(
                '/detect-language?local_path=/nonexistent/file.mp3')
            assert response.status_code == 400

    def test_detect_language_success(self, routes_client, tmp_path):
        """Test successful language detection."""
        test_file = tmp_path / "test.mp3"
        test_file.write_bytes(b"fake audio data")

        with mock.patch('modules.api.routes_detect.model_manager') as mock_mm:
            mock_mm.is_engine_initialized.return_value = True
            with mock.patch('modules.inference.language_detection.run_voting_detection') as mock_ld:
                mock_ld.return_value = {
                    'detected_language': 'en',
                    'language': 'en',
                    'language_code': 'en',
                    'confidence': 0.95
                }
                response = routes_client.post(
                    f'/detect-language?local_path={test_file}')
                assert response.status_code == 200
                data = json.loads(response.data)
                assert data['detected_language'] == 'en'
                assert data['language'] == 'en'
                assert data['language_code'] == 'en'

    def test_detect_language_ffmpeg_fails(self, routes_client, tmp_path):
        """Test detect-language when the voting engine fails."""
        test_file = tmp_path / "test.mp3"
        test_file.write_bytes(b"fake audio data")

        with mock.patch('modules.api.routes_detect.model_manager') as mock_mm:
            mock_mm.is_engine_initialized.return_value = True
            with mock.patch('modules.inference.language_detection.run_voting_detection') as mock_ld:
                mock_ld.side_effect = Exception("Detection error")
                response = routes_client.post(
                    f'/detect-language?local_path={test_file}')
                assert response.status_code == 500
                assert b"Error" in response.data

    def test_detect_language_exception(self, routes_client, tmp_path):
        """Test detect-language handles exceptions."""
        test_file = tmp_path / "test.mp3"
        test_file.write_bytes(b"fake audio data")

        with mock.patch('modules.api.routes_detect.model_manager') as mock_mm:
            mock_mm.is_engine_initialized.return_value = True
            with mock.patch('modules.inference.language_detection.run_voting_detection') as mock_ld:
                mock_ld.side_effect = Exception("Detection error")
                response = routes_client.post(
                    f'/detect-language?local_path={test_file}')
                assert response.status_code == 500
                assert b"Error" in response.data


class TestASREndpoint:
    """Tests for /asr endpoint."""

    def test_asr_get_ready(self, routes_client):
        """Test GET /asr when model is ready."""
        with mock.patch('modules.api.routes_asr.model_manager') as mock_mm:
            mock_mm.is_engine_initialized.return_value = True
            response = routes_client.get('/asr')
            assert response.status_code == 200
            assert b"ready" in response.data

    def test_asr_get_not_ready(self, routes_client):
        """Test GET /asr when model not ready."""
        with mock.patch('modules.api.routes_asr.model_manager') as mock_mm:
            mock_mm.is_engine_initialized.return_value = False
            response = routes_client.get('/asr')
            assert response.status_code == 200
            assert b"not_ready" in response.data

    def test_asr_post_no_model(self):
        """Test POST /asr when model not loaded."""
        with mock.patch('modules.api.routes_asr.model_manager') as mock_mm:
            mock_mm.is_engine_initialized.return_value = False

            test_app = Flask(__name__)
            test_app.register_blueprint(routes_asr.bp)
            test_app.config['TESTING'] = True
            test_client = test_app.test_client()

            response = test_client.post('/asr')
            assert response.status_code == 503

    def test_asr_post_no_input(self, routes_client):
        """Test POST /asr with no input."""
        with mock.patch('modules.api.routes_asr.model_manager') as mock_mm:
            mock_mm.is_engine_initialized.return_value = True
            response = routes_client.post('/asr')
            assert response.status_code == 400

    def test_asr_post_file_not_found(self, routes_client):
        """Test POST /asr with non-existent file."""
        with mock.patch('modules.api.routes_asr.model_manager') as mock_mm:
            mock_mm.is_engine_initialized.return_value = True
            response = routes_client.post('/asr?local_path=/nonexistent/file.mp3')
            assert response.status_code == 400

    def test_asr_post_success_srt(self, routes_client, tmp_path):
        """Test successful ASR with SRT output."""
        test_file = tmp_path / "test.mp3"
        test_file.write_bytes(b"fake audio data")

        with mock.patch('modules.api.routes_asr.model_manager') as mock_mm:
            mock_mm.is_engine_initialized.return_value = True
            mock_mm.run_transcription.return_value = {
                'text': 'Hello',
                'segments': [{'timestamp': (0.0, 1.0), 'text': 'Hello'}]
            }
            with mock.patch('modules.utils.convert_to_wav') as mock_wav:
                mock_wav.return_value = str(test_file)
                response = routes_client.post(
                    f'/asr?local_path={test_file}&output=srt')
                assert response.status_code == 200
                assert b"Hello" in response.data

    def test_asr_post_success_json(self, routes_client, tmp_path):
        """Test successful ASR with JSON output."""
        test_file = tmp_path / "test.mp3"
        test_file.write_bytes(b"fake audio data")

        with mock.patch('modules.api.routes_asr.model_manager') as mock_mm:
            mock_mm.is_engine_initialized.return_value = True
            mock_mm.run_transcription.return_value = {
                'text': 'Hello',
                'segments': []
            }
            with mock.patch('modules.utils.convert_to_wav') as mock_wav:
                mock_wav.return_value = str(test_file)
                response = routes_client.post(
                    f'/asr?local_path={test_file}&output=json')
                assert response.status_code == 200
                data = json.loads(response.data)
                assert data['text'] == 'Hello'

    def test_asr_post_ffmpeg_fails(self, routes_client, tmp_path):
        """Test ASR when FFmpeg conversion fails."""
        test_file = tmp_path / "test.mp3"
        test_file.write_bytes(b"fake audio data")

        with mock.patch('modules.api.routes_asr.model_manager') as mock_mm:
            mock_mm.is_engine_initialized.return_value = True
            with mock.patch('modules.utils.convert_to_wav') as mock_wav:
                mock_wav.return_value = None
                response = routes_client.post(f'/asr?local_path={test_file}')
                assert response.status_code == 400
                assert b"FFmpeg" in response.data

    def test_asr_post_exception(self, routes_client, tmp_path):
        """Test ASR handles transcription exceptions."""
        test_file = tmp_path / "test.mp3"
        test_file.write_bytes(b"fake audio data")

        with mock.patch('modules.api.routes_asr.model_manager') as mock_mm:
            mock_mm.is_engine_initialized.return_value = True
            mock_mm.run_transcription.side_effect = Exception(
                "Transcription error")
            with mock.patch('modules.utils.convert_to_wav') as mock_wav:
                mock_wav.return_value = str(test_file)
                response = routes_client.post(f'/asr?local_path={test_file}')
                assert response.status_code == 500
                assert b"Error" in response.data

    def test_asr_post_exception_generic(self, routes_client, tmp_path):
        """Test ASR handles generic exceptions."""
        test_file = tmp_path / "test.mp3"
        test_file.write_bytes(b"fake audio data")

        with mock.patch('modules.api.routes_asr.model_manager') as mock_mm:
            mock_mm.is_engine_initialized.return_value = True
            with mock.patch('modules.api.routes_utils.prepare_source_path',
                            return_value=(test_file, None, "test.mp3")):
                with mock.patch('os.path.exists', return_value=True):
                    with mock.patch('modules.utils.convert_to_wav',
                                    return_value="clean.wav"):
                        mock_mm.run_transcription.side_effect = Exception("Generic error")
                        response = routes_client.post('/asr?local_path=test.mp3')
                        assert response.status_code == 500

    def test_asr_various_outputs(self, routes_client, tmp_path):
        """Test ASR with various output formats."""
        test_file = tmp_path / "test.mp3"
        test_file.write_bytes(b"fake audio data")

        with mock.patch('modules.api.routes_asr.model_manager') as mock_mm:
            mock_mm.is_engine_initialized.return_value = True
            mock_mm.run_transcription.return_value = {
                'text': 'Hello world',
                'segments': [{'timestamp': (0.0, 1.0), 'text': 'Hello world'}]
            }
            with mock.patch('modules.api.routes_utils.prepare_source_path',
                            return_value=(test_file, None, "test.mp3")):
                with mock.patch('os.path.exists', return_value=True):
                    with mock.patch('modules.utils.convert_to_wav',
                                    return_value="clean.wav"):
                        # VTT
                        resp = routes_client.post(
                            '/asr?local_path=test.mp3&output=vtt')
                        assert resp.status_code == 200

                        # TXT
                        resp = routes_client.post(
                            '/asr?local_path=test.mp3&output=txt')
                        assert resp.status_code == 200
                        assert b'Hello world' in resp.data

    def test_asr_upload_corrupt(self, routes_client):
        """Test ASR with uploaded corrupted file."""
        corrupt_data = b"\x00" * 2048
        # We need to mock os.path.getsize and os.remove and open
        with mock.patch("os.path.join", return_value="fake.mp3"), \
                mock.patch("os.path.getsize", return_value=2048), \
                mock.patch("builtins.open", mock.mock_open(read_data=b"\x00"*2048)), \
                mock.patch("os.remove"):
            response = routes_client.post(
                '/asr',
                data={'audio_file': (io.BytesIO(corrupt_data), 'corrupt.mp3')},
                content_type='multipart/form-data'
            )
            assert response.status_code == 400
            assert b"corrupted" in response.data

    def test_asr_upload_empty(self, routes_client):
        """Test ASR with empty uploaded file."""
        with mock.patch("os.path.getsize", return_value=0), \
                mock.patch("os.remove"):
            response = routes_client.post(
                '/asr',
                data={'audio_file': (io.BytesIO(b""), 'empty.mp3')},
                content_type='multipart/form-data'
            )
            assert response.status_code == 400
            assert b"empty" in response.data.lower()

    def test_cleanup_files_coverage(self):
        """Test cleanup_files helper coverage."""
        with mock.patch("os.path.exists", return_value=True):
            with mock.patch("os.remove") as mock_remove:
                cleanup_files("file1.wav", "file2.wav")
                assert mock_remove.call_count == 2

    def test_get_all_request_params_edge_cases(self):
        """Test get_request_params default fallbacks."""
        test_app = Flask(__name__)
        with test_app.test_request_context('/asr?batch_size=not_an_int'):
            params = get_request_params()
            assert params['batch_size'] == config.DEFAULT_BATCH_SIZE

        with test_app.test_request_context('/asr?task=translate'):
            params = get_request_params()
            assert params['task'] == 'translate'


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_prepare_source_path_from_query(self):
        """Test prepare_source_path resolution from query."""
        test_app = Flask(__name__)
        with test_app.test_request_context('/asr?local_path=/test/path.mp3'):
            with mock.patch('os.path.exists', return_value=True):
                path, _temp, display = prepare_source_path()
                assert path == '/test/path.mp3'
                assert display == 'path.mp3'

    def test_prepare_source_path_from_form(self):
        """Test prepare_source_path resolution from form."""
        test_app = Flask(__name__)
        with test_app.test_request_context(
            '/asr',
            method='POST',
            data={'local_path': '/form/path.mp3'}
        ):
            with mock.patch('os.path.exists', return_value=True):
                path, _temp, _display = prepare_source_path()
                assert path == '/form/path.mp3'

    def test_get_request_params_defaults(self):
        """Test default request parameters."""
        test_app = Flask(__name__)
        with test_app.test_request_context('/asr'):
            params = get_request_params()
            assert params['output_format'] == 'srt'
            assert params['task'] == 'transcribe'

    def test_get_request_params_custom(self):
        """Test custom request parameters."""
        test_app = Flask(__name__)
        test_url = '/asr?output=json&language=es&task=translate&batch_size=4'
        with test_app.test_request_context(test_url):
            params = get_request_params()
            assert params['output_format'] == 'json'
            assert params['language'] == 'es'
            assert params['task'] == 'translate'
            assert params['batch_size'] == 4

    def test_build_response_json(self):
        """Test JSON response building."""
        test_app = Flask(__name__)
        with test_app.app_context():
            result = {'text': 'Hello', 'segments': []}
            params = {'output_format': 'json'}
            stats = {'active_sessions': 0}
            response = build_response(result, params, stats, '/fake/path', 100.0)
            assert response.content_type == 'application/json'

    def test_build_response_srt(self):
        """Test SRT response building."""
        test_app = Flask(__name__)
        with test_app.app_context():
            result = {'text': 'Hello', 'segments': []}
            params = {'output_format': 'srt'}
            stats = {'active_sessions': 0}
            response = build_response(result, params, stats, '/fake/path', 100.0)
            assert 'text/plain' in response.content_type
            assert response.headers['Content-Disposition'] == 'attachment; filename="path.srt"; filename*=UTF-8\'\'path.srt'

    def test_build_response_unicode_filename(self):
        """Test response building with unicode filename to ensure no encoding issues occur."""
        test_app = Flask(__name__)
        with test_app.app_context():
            result = {'text': 'Hello', 'segments': []}
            params = {'output_format': 'srt'}
            stats = {'active_sessions': 0}
            unicode_path = '/movies/Liceenii Extemporal la dirigenție (1987) DVD-R.mkv'
            response = build_response(result, params, stats, unicode_path, 100.0)
            assert 'text/plain' in response.content_type
            cd_header = response.headers['Content-Disposition']
            assert 'filename="Liceenii Extemporal la dirigenie (1987) DVD-R.srt"' in cd_header
            assert "filename*=UTF-8''Liceenii%20Extemporal%20la%20dirigen%C8%9Bie%20%281987%29%20DVD-R.srt" in cd_header


class TestRequestLogging:
    """Tests for request body logging exception handling."""

    def test_body_logging_json_exception(self, routes_client):
        """Test body logging handles JSON parsing exceptions gracefully."""
        with mock.patch('modules.api.routes_asr.model_manager') as mock_mm:
            mock_mm.is_engine_initialized.return_value = True
            # Send a POST with invalid JSON that will trigger the exception path
            response = routes_client.post(
                '/asr',
                data='invalid json {',
                content_type='application/json'
            )
            # Request should still be processed
            assert response.status_code in [400, 500]

    def test_body_logging_form_exception(self, routes_client):
        """Test body logging handles form parsing exceptions gracefully."""
        with mock.patch('modules.api.routes_asr.model_manager') as mock_mm:
            mock_mm.is_engine_initialized.return_value = True
            # Send request with form data
            response = routes_client.post('/asr', data={})
            # Request should still be processed
            assert response.status_code in [400, 500]


def test_routes_extract_new_params():
    """Verify that ASR routes correctly parse new parameters."""
    app = Flask(__name__)
    app.register_blueprint(routes_asr.bp)

    # Test full parameters extraction
    with app.test_request_context(
        '/asr?initial_prompt=testprompt&vad_filter=false&word_timestamps=true&max_line_width=40&max_line_count=2'
    ):
        params = routes_asr.get_request_params()
        assert params['initial_prompt'] == 'testprompt'
        assert params['vad_filter'] is False
        assert params['word_timestamps'] is True
        assert params['max_line_width'] == 40
        assert params['max_line_count'] == 2

    # Test default values when omitted
    with app.test_request_context('/asr'):
        params = routes_asr.get_request_params()
        assert params['initial_prompt'] is None
        assert params['vad_filter'] is True
        assert params['word_timestamps'] is False
        assert params['max_line_width'] is None
        assert params['max_line_count'] is None

    # Test malformed width/count integers fallback to None
    with app.test_request_context('/asr?max_line_width=invalid&max_line_count=invalid'):
        params = routes_asr.get_request_params()
        assert params['max_line_width'] is None
        assert params['max_line_count'] is None
