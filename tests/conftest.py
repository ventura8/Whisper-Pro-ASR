"""Shared test fixtures and global mocks for the Whisper Pro ASR test suite."""
import argparse
from unittest import mock
from flask import Flask
import pytest

# 1. Apply global mocks before any project imports to prevent real module loading
from tests.mock_setup import mock_torch

# 2. Now safe to import project modules
from modules.api import routes_asr, routes_detect, routes_system
import modules.inference.preprocessing as prep_module
from modules.inference.vad import reset_vad_state
from modules.inference.scheduler import SchedulerState
from modules.inference import model_manager, scheduler
from whisper_pro_asr import create_app


def mock_tensor_with_shape(*shape):
    """Create a mock tensor with a specific shape attribute."""
    t = mock.MagicMock()
    t.shape = shape
    t.dtype = 'float32'
    t.device = 'cpu'
    return t


mock_torch.cat = lambda _t, dim=0: mock_tensor_with_shape(8, 512)
mock_torch.zeros = lambda *args, **kwargs: mock_tensor_with_shape(*args)
mock_torch.full = lambda size, _v, **kwargs: mock_tensor_with_shape(*size)

# Pytest fixtures for test isolation


@pytest.fixture
def client():
    """Flask test client with full orchestration mocks."""
    app = create_app()
    with mock.patch('modules.monitoring.dashboard.psutil', create=True) as mock_psu, \
            mock.patch('modules.monitoring.dashboard.utils') as mock_utils:

        mock_psu.cpu_percent.return_value = 10.0
        mock_psu.cpu_count.return_value = 8
        mock_psu.virtual_memory.return_value = argparse.Namespace(
            percent=50.0,
            used=8 * (1024**3),
            total=16 * (1024**3)
        )

        mock_utils.get_system_telemetry.return_value = {
            "cpu_percent": 10.0,
            "app_cpu_percent": 5.0,
            "memory_percent": 50.0,
            "memory_used_gb": 8.0,
            "memory_total_gb": 16.0,
            "app_memory_gb": 1.0
        }

        app.config['TESTING'] = True
        with app.test_client() as test_client:
            test_client.application_instance = app
            yield test_client


@pytest.fixture(autouse=True)
def reset_module_state():
    """Reset module-level state between tests to prevent test pollution."""
    # Force reset module state before test
    model_manager.MODEL_POOL = {}
    model_manager.PREPROCESSOR_POOL = {}
    prep_module.Separator = None
    prep_module.ort = None
    scheduler.STATE = SchedulerState()
    reset_vad_state()

    yield

    # Also reset after test
    model_manager.MODEL_POOL = {}
    model_manager.PREPROCESSOR_POOL = {}
    prep_module.Separator = None
    prep_module.ort = None
    scheduler.STATE = SchedulerState()
    reset_vad_state()


@pytest.fixture
def routes_app():
    """Create test Flask app with mocked model_manager for API routes testing."""
    with mock.patch('modules.api.routes_asr.model_manager') as mock_mm_asr, \
            mock.patch('modules.api.routes_detect.model_manager') as mock_mm_det, \
            mock.patch('modules.inference.language_detection.run_voting_detection') as mock_ld:

        mock_mm_asr.is_engine_initialized.return_value = True
        mock_mm_det.is_engine_initialized.return_value = True

        mock_ld.return_value = {
            'confidence': 0.95, 'detected_language': 'en', 'language': 'en', 'language_code': 'en'
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
def routes_client(request):
    """Create test client for routes_app."""
    app = request.getfixturevalue("routes_app")
    return app.test_client()
