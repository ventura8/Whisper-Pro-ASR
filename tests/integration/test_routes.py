"""Tests for modules/routes.py"""
# pylint: disable=redefined-outer-name, unused-argument, import-outside-toplevel
# pylint: disable=too-few-public-methods
import json
from unittest import mock
import pytest
from flask import Flask

# Mock heavy ML dependencies before importing routes
from modules.api import routes_asr, routes_detect, routes_system
from modules.inference import scheduler


@pytest.fixture(autouse=True)
def reset_state():
    """Clear scheduler state between tests."""
    from modules.inference.scheduler import SchedulerState
    scheduler.STATE = SchedulerState()
    yield
    scheduler.STATE = SchedulerState()


@pytest.fixture
def app(reset_state):
    """Create test Flask app with mocked model_manager."""
    with mock.patch('modules.api.routes_asr.model_manager') as mock_mm_asr, \
            mock.patch('modules.api.routes_detect.model_manager') as mock_mm_det, \
            mock.patch('modules.inference.language_detection.run_voting_detection') as mock_ld:

        mock_mm_asr.is_engine_initialized.return_value = True
        mock_mm_det.is_engine_initialized.return_value = True

        mock_ld.return_value = {
            'detected_language': 'en',
            'language': 'en',
            'language_code': 'en',
            'confidence': 0.95
        }

        mock_mm_asr.run_transcription.return_value = {
            'text': 'Hello world',
            'segments': [{'timestamp': (0.0, 1.0), 'text': 'Hello world'}]
        }

        test_app = Flask(__name__)
        test_app.register_blueprint(routes_system.bp)
        test_app.register_blueprint(routes_asr.bp)
        test_app.register_blueprint(routes_detect.bp)
        test_app.config['TESTING'] = True
        yield test_app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


class TestRootEndpoint:
    """Tests for the root endpoint."""

    def test_root_get(self, client):
        """Test GET / returns health status."""
        response = client.get('/')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'


class TestStatusEndpoint:
    """Tests for the /status endpoint."""

    def test_status_model_loaded(self, client):
        """Test /status when model is loaded."""
        with mock.patch('modules.api.routes_system.dashboard') as mock_db:
            mock_db.get_status_data.return_value = {
                'active_sessions': 0,
                'queued_sessions': 0,
                'hardware': []
            }
            response = client.get('/status')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'active_sessions' in data

    def test_status_model_not_loaded(self, client):
        """Test /status when model stats fail."""
        with mock.patch('modules.api.routes_system.dashboard') as mock_db:
            mock_db.get_status_data.return_value = {
                'active_sessions': 0,
                'queued_sessions': 0,
                'hardware': []
            }
            response = client.get('/status')
            assert response.status_code == 200


class TestDetectLanguageEndpoint:
    """Tests for /detect-language endpoint."""

    def test_detect_language_no_model(self, client):
        """Test detect-language when model not loaded."""
        with mock.patch('modules.api.routes_detect.model_manager') as mock_mm:
            mock_mm.is_engine_initialized.return_value = False
            response = client.post('/detect-language')
            assert response.status_code == 503

    def test_detect_language_no_input(self, client):
        """Test detect-language with no input."""
        with mock.patch('modules.api.routes_detect.model_manager') as mock_mm:
            mock_mm.is_engine_initialized.return_value = True
            response = client.post('/detect-language')
            assert response.status_code == 400
            assert b"No audio source provided" in response.data

    def test_detect_language_file_not_found(self, client):
        """Test detect-language with non-existent file."""
        with mock.patch('modules.api.routes_detect.model_manager') as mock_mm:
            mock_mm.is_engine_initialized.return_value = True
            response = client.post(
                '/detect-language?local_path=/nonexistent/file.mp3')
            assert response.status_code == 400

    def test_detect_language_success(self, client, tmp_path):
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
                response = client.post(
                    f'/detect-language?local_path={test_file}')
                assert response.status_code == 200
                data = json.loads(response.data)
                assert data['detected_language'] == 'en'
                assert data['language'] == 'en'
                assert data['language_code'] == 'en'

    def test_detect_language_ffmpeg_fails(self, client, tmp_path):
        """Test detect-language when the voting engine fails."""
        test_file = tmp_path / "test.mp3"
        test_file.write_bytes(b"fake audio data")

        with mock.patch('modules.api.routes_detect.model_manager') as mock_mm:
            mock_mm.is_engine_initialized.return_value = True
            with mock.patch('modules.inference.language_detection.run_voting_detection') as mock_ld:
                mock_ld.side_effect = Exception("Detection error")
                response = client.post(
                    f'/detect-language?local_path={test_file}')
                assert response.status_code == 500
                assert b"Error" in response.data

    def test_detect_language_exception(self, client, tmp_path):
        """Test detect-language handles exceptions."""
        test_file = tmp_path / "test.mp3"
        test_file.write_bytes(b"fake audio data")

        with mock.patch('modules.api.routes_detect.model_manager') as mock_mm:
            mock_mm.is_engine_initialized.return_value = True
            with mock.patch('modules.inference.language_detection.run_voting_detection') as mock_ld:
                mock_ld.side_effect = Exception("Detection error")
                response = client.post(
                    f'/detect-language?local_path={test_file}')
                assert response.status_code == 500
                assert b"Error" in response.data


class TestASREndpoint:
    """Tests for /asr endpoint."""

    def test_asr_get_ready(self, client):
        """Test GET /asr when model is ready."""
        with mock.patch('modules.api.routes_asr.model_manager') as mock_mm:
            mock_mm.is_engine_initialized.return_value = True
            response = client.get('/asr')
            assert response.status_code == 200
            assert b"ready" in response.data

    def test_asr_get_not_ready(self, client):
        """Test GET /asr when model not ready."""
        with mock.patch('modules.api.routes_asr.model_manager') as mock_mm:
            mock_mm.is_engine_initialized.return_value = False
            response = client.get('/asr')
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

    def test_asr_post_no_input(self, client):
        """Test POST /asr with no input."""
        with mock.patch('modules.api.routes_asr.model_manager') as mock_mm:
            mock_mm.is_engine_initialized.return_value = True
            response = client.post('/asr')
            assert response.status_code == 400

    def test_asr_post_file_not_found(self, client):
        """Test POST /asr with non-existent file."""
        with mock.patch('modules.api.routes_asr.model_manager') as mock_mm:
            mock_mm.is_engine_initialized.return_value = True
            response = client.post('/asr?local_path=/nonexistent/file.mp3')
            assert response.status_code == 400

    def test_asr_post_success_srt(self, client, tmp_path):
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
                response = client.post(
                    f'/asr?local_path={test_file}&output=srt')
                assert response.status_code == 200
                assert b"Hello" in response.data

    def test_asr_post_success_json(self, client, tmp_path):
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
                response = client.post(
                    f'/asr?local_path={test_file}&output=json')
                assert response.status_code == 200
                data = json.loads(response.data)
                assert data['text'] == 'Hello'

    def test_asr_post_ffmpeg_fails(self, client, tmp_path):
        """Test ASR when FFmpeg conversion fails."""
        test_file = tmp_path / "test.mp3"
        test_file.write_bytes(b"fake audio data")

        with mock.patch('modules.api.routes_asr.model_manager') as mock_mm:
            mock_mm.is_engine_initialized.return_value = True
            with mock.patch('modules.utils.convert_to_wav') as mock_wav:
                mock_wav.return_value = None
                response = client.post(f'/asr?local_path={test_file}')
                assert response.status_code == 400
                assert b"FFmpeg" in response.data

    def test_asr_post_exception(self, client, tmp_path):
        """Test ASR handles transcription exceptions."""
        test_file = tmp_path / "test.mp3"
        test_file.write_bytes(b"fake audio data")

        with mock.patch('modules.api.routes_asr.model_manager') as mock_mm:
            mock_mm.is_engine_initialized.return_value = True
            mock_mm.run_transcription.side_effect = Exception(
                "Transcription error")
            with mock.patch('modules.utils.convert_to_wav') as mock_wav:
                mock_wav.return_value = str(test_file)
                response = client.post(f'/asr?local_path={test_file}')
                assert response.status_code == 500
                assert b"Error" in response.data

    def test_asr_post_exception_generic(self, client, tmp_path):
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
                        response = client.post('/asr?local_path=test.mp3')
                        assert response.status_code == 500

    def test_asr_various_outputs(self, client, tmp_path):
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
                        resp = client.post(
                            '/asr?local_path=test.mp3&output=vtt')
                        assert resp.status_code == 200

                        # TXT
                        resp = client.post(
                            '/asr?local_path=test.mp3&output=txt')
                        assert resp.status_code == 200
                        assert b'Hello world' in resp.data

    def test_asr_upload_corrupt(self, client):
        """Test ASR with uploaded corrupted file."""
        from io import BytesIO
        corrupt_data = b"\x00" * 2048
        # We need to mock os.path.getsize and os.remove and open
        with mock.patch("os.path.join", return_value="fake.mp3"), \
                mock.patch("os.path.getsize", return_value=2048), \
                mock.patch("builtins.open", mock.mock_open(read_data=b"\x00"*2048)), \
                mock.patch("os.remove"):
            response = client.post(
                '/asr',
                data={'audio_file': (BytesIO(corrupt_data), 'corrupt.mp3')},
                content_type='multipart/form-data'
            )
            assert response.status_code == 400
            assert b"corrupted" in response.data

    def test_asr_upload_empty(self, client):
        """Test ASR with empty uploaded file."""
        from io import BytesIO
        with mock.patch("os.path.getsize", return_value=0), \
                mock.patch("os.remove"):
            response = client.post(
                '/asr',
                data={'audio_file': (BytesIO(b""), 'empty.mp3')},
                content_type='multipart/form-data'
            )
            assert response.status_code == 400
            assert b"empty" in response.data.lower()

    def test_cleanup_files_coverage(self):
        """Test _cleanup_files helper coverage."""
        from modules.api.routes_utils import cleanup_files as _cleanup_files
        with mock.patch("os.path.exists", return_value=True):
            with mock.patch("os.remove") as mock_remove:
                _cleanup_files("file1.wav", "file2.wav")
                assert mock_remove.call_count == 2

    def test_get_all_request_params_edge_cases(self):
        """Test _get_request_params default fallbacks."""
        from modules.api.routes_asr import _get_request_params
        test_app = Flask(__name__)
        with test_app.test_request_context('/asr?batch_size=not_an_int'):
            params = _get_request_params()
            from modules import config
            assert params['batch_size'] == config.DEFAULT_BATCH_SIZE

        with test_app.test_request_context('/asr?task=translate'):
            params = _get_request_params()
            assert params['task'] == 'translate'


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_prepare_source_path_from_query(self):
        """Test prepare_source_path resolution from query."""
        from modules.api.routes_utils import prepare_source_path
        test_app = Flask(__name__)
        with test_app.test_request_context('/asr?local_path=/test/path.mp3'):
            with mock.patch('os.path.exists', return_value=True):
                path, _temp, display = prepare_source_path()
                assert path == '/test/path.mp3'
                assert display == 'path.mp3'

    def test_prepare_source_path_from_form(self):
        """Test prepare_source_path resolution from form."""
        from modules.api.routes_utils import prepare_source_path
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
        from modules.api.routes_asr import _get_request_params
        test_app = Flask(__name__)
        with test_app.test_request_context('/asr'):
            params = _get_request_params()
            assert params['output_format'] == 'srt'
            assert params['task'] == 'transcribe'

    def test_get_request_params_custom(self):
        """Test custom request parameters."""
        from modules.api.routes_asr import _get_request_params
        test_app = Flask(__name__)
        test_url = '/asr?output=json&language=es&task=translate&batch_size=4'
        with test_app.test_request_context(test_url):
            params = _get_request_params()
            assert params['output_format'] == 'json'
            assert params['language'] == 'es'
            assert params['task'] == 'translate'
            assert params['batch_size'] == 4

    def test_build_response_json(self):
        """Test JSON response building."""
        from modules.api.routes_asr import _build_response
        test_app = Flask(__name__)
        with test_app.app_context():
            result = {'text': 'Hello', 'segments': []}
            params = {'output_format': 'json'}
            stats = {'active_sessions': 0}
            response = _build_response(result, params, stats, '/fake/path', 100.0)
            assert response.content_type == 'application/json'

    def test_build_response_srt(self):
        """Test SRT response building."""
        from modules.api.routes_asr import _build_response
        test_app = Flask(__name__)
        with test_app.app_context():
            result = {'text': 'Hello', 'segments': []}
            params = {'output_format': 'srt'}
            stats = {'active_sessions': 0}
            response = _build_response(result, params, stats, '/fake/path', 100.0)
            assert 'text/plain' in response.content_type


class TestRequestLogging:
    """Tests for request body logging exception handling."""

    def test_body_logging_json_exception(self, client):
        """Test body logging handles JSON parsing exceptions gracefully."""
        with mock.patch('modules.api.routes_asr.model_manager') as mock_mm:
            mock_mm.is_engine_initialized.return_value = True
            # Send a POST with invalid JSON that will trigger the exception path
            response = client.post(
                '/asr',
                data='invalid json {',
                content_type='application/json'
            )
            # Request should still be processed
            assert response.status_code in [400, 500]

    def test_body_logging_form_exception(self, client):
        """Test body logging handles form parsing exceptions gracefully."""
        with mock.patch('modules.api.routes_asr.model_manager') as mock_mm:
            mock_mm.is_engine_initialized.return_value = True
            # Send request with form data
            response = client.post('/asr', data={})
            # Request should still be processed
            assert response.status_code in [400, 500]
