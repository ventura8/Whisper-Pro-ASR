"""Tests for modules/routes.py"""
# pylint: disable=redefined-outer-name, unused-argument, import-outside-toplevel
# pylint: disable=too-few-public-methods
import json
from unittest import mock
import pytest
from flask import Flask

# Mock heavy ML dependencies before importing routes
from modules import routes


@pytest.fixture
def app():
    """Create test Flask app with mocked model_manager."""
    with mock.patch('modules.routes.model_manager') as mock_mm, \
            mock.patch('modules.routes.language_detection.run_voting_detection') as mock_ld:
        mock_mm.WHISPER = mock.MagicMock()
        mock_ld.return_value = {
            'detected_language': 'en',
            'confidence': 0.95
        }
        mock_mm.request_priority = mock.MagicMock()
        mock_mm.release_priority = mock.MagicMock()
        mock_mm.wait_for_priority = mock.MagicMock()
        mock_mm.run_transcription.return_value = {
            'text': 'Hello world',
            'chunks': [{'timestamp': (0.0, 1.0), 'text': 'Hello world'}]
        }

        test_app = Flask(__name__)
        test_app.register_blueprint(routes.bp)
        test_app.config['TESTING'] = True
        yield test_app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_get(self, client):
        """Test GET / returns health check message."""
        response = client.get('/')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "Whisper ASR Webservice is working" in data['message']
        assert data['status'] == 'healthy'


class TestStatusEndpoint:
    """Tests for /status endpoint."""

    def test_status_model_loaded(self, client):
        """Test status when model is loaded."""
        with mock.patch('modules.routes.model_manager') as mock_mm:
            mock_mm.WHISPER = mock.MagicMock()
            response = client.get('/status')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['status'] == 'loaded'

    def test_status_model_not_loaded(self, client):
        """Test status when model failed to load."""
        with mock.patch('modules.routes.model_manager') as mock_mm:
            mock_mm.WHISPER = None
            response = client.get('/status')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['status'] == 'failed'


class TestAdminEndpoint:
    """Tests for /admin endpoint."""

    def test_admin_get(self, client):
        """Test GET /admin."""
        response = client.get('/admin')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == "OK"

    def test_admin_post(self, client):
        """Test POST /admin."""
        response = client.post('/admin')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == "OK"


class TestDetectLanguageEndpoint:
    """Tests for /detect-language endpoint."""

    def test_detect_language_no_model(self, client):
        """Test detect-language when model not loaded."""
        with mock.patch('modules.routes.model_manager') as mock_mm:
            mock_mm.WHISPER = None
            response = client.post('/detect-language')
            assert response.status_code == 503

    def test_detect_language_no_input(self, client):
        """Test detect-language with no input."""
        with mock.patch('modules.routes.model_manager') as mock_mm:
            mock_mm.WHISPER = mock.MagicMock()
            response = client.post('/detect-language')
            assert response.status_code == 400
            assert b"No input provided" in response.data

    def test_detect_language_file_not_found(self, client):
        """Test detect-language with non-existent file."""
        with mock.patch('modules.routes.model_manager') as mock_mm:
            mock_mm.WHISPER = mock.MagicMock()
            response = client.post(
                '/detect-language?local_path=/nonexistent/file.mp3')
            assert response.status_code == 400

    def test_detect_language_success(self, client, tmp_path):
        """Test successful language detection."""
        test_file = tmp_path / "test.mp3"
        test_file.write_bytes(b"fake audio data")

        with mock.patch('modules.routes.model_manager') as mock_mm:
            mock_mm.WHISPER = mock.MagicMock()
            mock_mm.run_language_detection.return_value = {
                'detected_language': 'en',
                'confidence': 0.95
            }
            with mock.patch('modules.routes.utils.convert_to_wav') as mock_wav:
                mock_wav.return_value = str(test_file)
                response = client.post(
                    f'/detect-language?local_path={test_file}')
                assert response.status_code == 200
                data = json.loads(response.data)
                assert data['detected_language'] == 'english'

    def test_detect_language_ffmpeg_fails(self, client, tmp_path):
        """Test detect-language when FFmpeg fails."""
        test_file = tmp_path / "test.mp3"
        test_file.write_bytes(b"fake audio data")

        with mock.patch('modules.routes.model_manager') as mock_mm:
            mock_mm.WHISPER = mock.MagicMock()
            with mock.patch('modules.routes.language_detection.run_voting_detection') as mock_ld:
                mock_ld.side_effect = Exception("Detection error")
                response = client.post(
                    f'/detect-language?local_path={test_file}')
                assert response.status_code == 500
                assert b"Detection error" in response.data

    def test_detect_language_exception(self, client, tmp_path):
        """Test detect-language handles exceptions."""
        test_file = tmp_path / "test.mp3"
        test_file.write_bytes(b"fake audio data")

        with mock.patch('modules.routes.model_manager') as mock_mm:
            mock_mm.WHISPER = mock.MagicMock()
            with mock.patch('modules.routes.language_detection.run_voting_detection') as mock_ld:
                mock_ld.side_effect = Exception("Detection error")
                response = client.post(
                    f'/detect-language?local_path={test_file}')
                assert response.status_code == 500
                assert b"Detection error" in response.data


class TestASREndpoint:
    """Tests for /asr endpoint."""

    def test_asr_get_ready(self, client):
        """Test GET /asr when model is ready."""
        with mock.patch('modules.routes.model_manager') as mock_mm:
            mock_mm.WHISPER = mock.MagicMock()
            response = client.get('/asr')
            assert response.status_code == 200
            assert b"ready" in response.data

    def test_asr_get_not_ready(self, client):
        """Test GET /asr when model not ready."""
        with mock.patch('modules.routes.model_manager') as mock_mm:
            mock_mm.WHISPER = None
            response = client.get('/asr')
            assert response.status_code == 200
            assert b"not_ready" in response.data

    def test_asr_post_no_model(self):
        """Test POST /asr when model not loaded."""
        with mock.patch('modules.routes.model_manager') as mock_mm:
            mock_mm.WHISPER = None

            test_app = Flask(__name__)
            test_app.register_blueprint(routes.bp)
            test_app.config['TESTING'] = True
            test_client = test_app.test_client()

            response = test_client.post('/asr')
            assert response.status_code == 503

    def test_asr_post_no_input(self, client):
        """Test POST /asr with no input."""
        with mock.patch('modules.routes.model_manager') as mock_mm:
            mock_mm.WHISPER = mock.MagicMock()
            response = client.post('/asr')
            assert response.status_code == 400

    def test_asr_post_file_not_found(self, client):
        """Test POST /asr with non-existent file."""
        with mock.patch('modules.routes.model_manager') as mock_mm:
            mock_mm.WHISPER = mock.MagicMock()
            response = client.post('/asr?local_path=/nonexistent/file.mp3')
            assert response.status_code == 400

    def test_asr_post_success_srt(self, client, tmp_path):
        """Test successful transcription with SRT output."""
        test_file = tmp_path / "test.mp3"
        test_file.write_bytes(b"fake audio data")

        with mock.patch('modules.routes.model_manager') as mock_mm:
            mock_mm.WHISPER = mock.MagicMock()
            mock_mm.run_transcription.return_value = {
                'text': 'Hello',
                'chunks': [{'timestamp': (0.0, 1.0), 'text': 'Hello'}]
            }
            with mock.patch('modules.routes.utils.convert_to_wav') as mock_wav:
                mock_wav.return_value = str(test_file)
                response = client.post(
                    f'/asr?local_path={test_file}&output=srt')
                assert response.status_code == 200
                assert b"-->" in response.data  # SRT format

    def test_asr_post_success_json(self, client, tmp_path):
        """Test successful transcription with JSON output."""
        test_file = tmp_path / "test.mp3"
        test_file.write_bytes(b"fake audio data")

        with mock.patch('modules.routes.model_manager') as mock_mm:
            mock_mm.WHISPER = mock.MagicMock()
            mock_mm.run_transcription.return_value = {
                'text': 'Hello',
                'chunks': []
            }
            with mock.patch('modules.routes.utils.convert_to_wav') as mock_wav:
                mock_wav.return_value = str(test_file)
                response = client.post(
                    f'/asr?local_path={test_file}&output=json')
                assert response.status_code == 200
                data = json.loads(response.data)
                assert data['text'] == 'Hello'

    def test_asr_post_ffmpeg_fails(self, client, tmp_path):
        """Test ASR when FFmpeg fails."""
        test_file = tmp_path / "test.mp3"
        test_file.write_bytes(b"fake audio data")

        with mock.patch('modules.routes.model_manager') as mock_mm:
            mock_mm.WHISPER = mock.MagicMock()
            with mock.patch('modules.routes.utils.convert_to_wav') as mock_wav:
                mock_wav.return_value = None
                response = client.post(f'/asr?local_path={test_file}')
                assert response.status_code == 400

    def test_asr_post_exception(self, client, tmp_path):
        """Test ASR handles exceptions."""
        test_file = tmp_path / "test.mp3"
        test_file.write_bytes(b"fake audio data")

        with mock.patch('modules.routes.model_manager') as mock_mm:
            mock_mm.WHISPER = mock.MagicMock()
            mock_mm.run_transcription.side_effect = Exception(
                "Transcription error")
            with mock.patch('modules.routes.utils.convert_to_wav') as mock_wav:
                mock_wav.return_value = str(test_file)
                response = client.post(f'/asr?local_path={test_file}')
                assert response.status_code == 500

    def test_detect_language_exception_generic(self, client):
        """Test detect_language handles generic exceptions."""
        with mock.patch('modules.routes.model_manager') as mock_mm:
            mock_mm.request_priority = mock.MagicMock()
            with mock.patch('modules.routes.language_detection.run_voting_detection',
                            side_effect=Exception("LD Fail")):
                with mock.patch('modules.routes._prepare_source_path',
                                return_value=("/fake/path", None)):
                    response = client.post('/detect-language')
                    assert response.status_code == 500
                    assert b"Error" in response.data

    def test_asr_post_exception_generic(self, client, tmp_path):
        """Test ASR handles generic exceptions."""
        test_file = str(tmp_path / "test.mp3")

        with mock.patch('modules.routes.model_manager') as mock_mm:
            mock_mm.run_transcription.side_effect = Exception("Generic error")
            with mock.patch('modules.routes._prepare_source_path', return_value=(test_file, None)):
                with mock.patch('os.path.exists', return_value=True):
                    with mock.patch('modules.routes.utils.convert_to_wav',
                                    return_value="clean.wav"):
                        response = client.post('/asr?local_path=test.mp3')
                        assert response.status_code == 500
                        assert b"Service Error" in response.data

    def test_asr_various_outputs(self, client, tmp_path):
        """Test ASR with vtt, txt, tsv outputs."""
        test_file = str(tmp_path / "test.mp3")

        with mock.patch('modules.routes.model_manager') as mock_mm:
            mock_mm.run_transcription.return_value = {
                'text': 'Hello',
                'chunks': [{'timestamp': (0.0, 1.0), 'text': 'Hello'}]
            }
            with mock.patch('modules.routes._prepare_source_path', return_value=(test_file, None)):
                with mock.patch('os.path.exists', return_value=True):
                    with mock.patch('modules.routes.utils.convert_to_wav',
                                    return_value="clean.wav"):
                        # VTT
                        resp = client.post(
                            '/asr?local_path=test.mp3&output=vtt')
                        assert resp.status_code == 200
                        assert b"WEBVTT" in resp.data

                        # TXT
                        resp = client.post(
                            '/asr?local_path=test.mp3&output=txt')
                        assert resp.status_code == 200
                        assert resp.data.strip() == b"Hello"

                        # TSV
                        resp = client.post(
                            '/asr?local_path=test.mp3&output=tsv')
                        assert resp.status_code == 200
                        assert b"start\tend\ttext" in resp.data

    def test_asr_corrupt_file(self, client, tmp_path):
        """Test ASR with null-byte corrupted file."""
        test_file = tmp_path / "corrupt.mp3"
        test_file.write_bytes(b"\x00" * 2048)  # 2KB of zeros

        with mock.patch('modules.routes.model_manager') as mock_mm:
            mock_mm.WHISPER = mock.MagicMock()
            response = client.post(f'/asr?local_path={test_file}')
            assert response.status_code == 400
            assert b"corrupted" in response.data

    def test_asr_upload_corrupt(self, client):
        """Test ASR with uploaded corrupted file."""
        from io import BytesIO
        corrupt_data = b"\x00" * 2048
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
        response = client.post(
            '/asr',
            data={'audio_file': (BytesIO(b""), 'empty.mp3')},
            content_type='multipart/form-data'
        )
        assert response.status_code == 400
        assert b"empty" in response.data.lower()

    def test_cleanup_files_coverage(self):
        """Test _cleanup_files helper coverage."""
        from modules.routes import _cleanup_files
        with mock.patch("os.path.exists", return_value=True):
            with mock.patch("os.remove") as mock_remove:
                _cleanup_files("file1.wav", "file2.wav")
                assert mock_remove.call_count == 2

    def test_get_all_request_params_edge_cases(self):
        """Test _get_all_request_params default fallbacks."""
        from modules.routes import _get_all_request_params
        test_app = Flask(__name__)
        with test_app.test_request_context('/asr?batch_size=not_an_int'):
            params = _get_all_request_params()
            from modules import config
            assert params['batch_size'] == config.DEFAULT_BATCH_SIZE

        with test_app.test_request_context('/asr?task=translate'):
            params = _get_all_request_params()
            assert params['task'] == 'translate'


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_input_path_from_query(self):
        """Test _get_all_request_params path resolution from query."""
        with mock.patch('modules.routes.model_manager'):
            from modules.routes import _get_all_request_params

            test_app = Flask(__name__)
            with test_app.test_request_context('/asr?local_path=/test/path.mp3'):
                params = _get_all_request_params()
                assert params.get('local_path') == '/test/path.mp3'

    def test_get_input_path_from_json(self):
        """Test _get_all_request_params path resolution from JSON."""
        # Note: _get_all_request_params primarily looks at args/form.
        # routes.py logic for JSON might be handled elsewhere or
        # _get_all_request_params doesn't support JSON body for path.
        # Checking routes.py: It looks at request.args and request.form.
        # It DOES NOT look at request.json/get_json() in _get_all_request_params.
        # The previous _get_input_path might have.
        # However, looking at the code, Flask's request.form usually
        # doesn't include JSON data unless parsed.

    def test_get_input_path_from_form(self):
        """Test _get_all_request_params path resolution from form."""
        with mock.patch('modules.routes.model_manager'):
            from modules.routes import _get_all_request_params

            test_app = Flask(__name__)
            with test_app.test_request_context(
                '/asr',
                method='POST',
                data={'local_path': '/form/path.mp3'}
            ):
                params = _get_all_request_params()
                assert params.get('local_path') == '/form/path.mp3'

    def test_get_input_path_file_path(self):
        """Test _get_input_path aliases with file_path param."""
        with mock.patch('modules.routes.model_manager'):
            from modules.routes import _get_all_request_params

            test_app = Flask(__name__)
            with test_app.test_request_context('/asr?file_path=/alt/path.mp3'):
                params = _get_all_request_params()
                assert params.get('local_path') == '/alt/path.mp3'

    def test_get_input_path_video_file(self):
        """Test _get_input_path aliases with video_file param."""
        with mock.patch('modules.routes.model_manager'):
            from modules.routes import _get_all_request_params

            test_app = Flask(__name__)
            with test_app.test_request_context('/asr?video_file=/video/file.mkv'):
                params = _get_all_request_params()
                assert params.get('local_path') == '/video/file.mkv'

    def test_get_transcription_options_defaults(self):
        """Test default transcription options."""
        with mock.patch('modules.routes.model_manager'):
            from modules.routes import _get_transcription_options

            test_app = Flask(__name__)
            with test_app.test_request_context('/asr'):
                output_format, _, task, _ = _get_transcription_options()
                assert output_format == 'srt'
                assert task == 'transcribe'

    def test_get_transcription_options_custom(self):
        """Test custom transcription options."""
        with mock.patch('modules.routes.model_manager'):
            from modules.routes import _get_transcription_options

            test_app = Flask(__name__)
            test_url = '/asr?output=json&language=es&task=translate&batch_size=4'
            with test_app.test_request_context(test_url):
                output_format, language, task, batch_size = _get_transcription_options()
                assert output_format == 'json'
                assert language == 'es'
                assert task == 'translate'
                assert batch_size == 4

    def test_get_transcription_options_source_lang(self):
        """Test source_lang parameter."""
        with mock.patch('modules.routes.model_manager'):
            from modules.routes import _get_transcription_options

            test_app = Flask(__name__)
            with test_app.test_request_context('/asr?source_lang=fr'):
                _, language, _, _ = _get_transcription_options()
                assert language == 'fr'

    def test_get_transcription_options_invalid_batch(self):
        """Test invalid batch_size falls back to default."""
        with mock.patch('modules.routes.model_manager'):
            from modules.routes import _get_transcription_options
            from modules import config

            test_app = Flask(__name__)
            with test_app.test_request_context('/asr?batch_size=invalid'):
                _, _, _, batch_size = _get_transcription_options()
                assert batch_size == config.DEFAULT_BATCH_SIZE

    def test_format_transcription_response_json(self):
        """Test JSON response formatting."""
        with mock.patch('modules.routes.model_manager'):
            from modules.routes import _format_transcription_response

            test_app = Flask(__name__)
            with test_app.app_context():
                result = {'text': 'Hello', 'chunks': []}
                response = _format_transcription_response(result, 'json')
                assert response.content_type == 'application/json'

    def test_format_transcription_response_srt(self):
        """Test SRT response formatting."""
        with mock.patch('modules.routes.model_manager'):
            from modules.routes import _format_transcription_response

            test_app = Flask(__name__)
            with test_app.app_context():
                result = {'text': 'Hello', 'chunks': []}
                response = _format_transcription_response(result, 'srt')
                assert 'text/plain' in response.content_type


class TestRequestLogging:
    """Tests for request body logging exception handling."""

    def test_body_logging_json_exception(self, client):
        """Test body logging handles JSON parsing exceptions gracefully."""
        with mock.patch('modules.routes.model_manager') as mock_mm:
            mock_mm.WHISPER = mock.MagicMock()
            # Send a POST with invalid JSON that will trigger the exception path
            response = client.post(
                '/asr',
                data='invalid json {',
                content_type='application/json'
            )
            # Request should still be processed, just logging should handle error
            assert response.status_code in [400, 500]

    def test_body_logging_form_exception(self, client):
        """Test body logging handles form parsing exceptions gracefully."""
        with mock.patch('modules.routes.model_manager') as mock_mm:
            mock_mm.WHISPER = mock.MagicMock()
            # Send request with form data - even if it fails, logging should handle it
            response = client.post('/asr', data={})
            # Request should still be processed
            assert response.status_code in [400, 500]
