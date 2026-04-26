"""Global pytest configuration and universal mocks for ML dependencies."""
import sys
from unittest import mock
import pytest
import modules.preprocessing as prep_module
import modules.vad as vad_module

# Create universal mocks for ML dependencies
# This prevents different test files from conflicting and avoids
# RuntimeError: function '_has_torch_function' already has a docstring
mock_torch = mock.MagicMock()
mock_torch.__path__ = []  # Mark as a package for subpackage imports

# Mock subpackages to avoid "torch is not a package" errors
mock_torch_nn = mock.MagicMock()
mock_torch_nn.__path__ = []
mock_torch_nn_functional = mock.MagicMock()


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
    from whisper_pro_asr import app
    with mock.patch('modules.dashboard.psutil') as mock_psu, \
            mock.patch('modules.dashboard._PROCESS_OBJ') as mock_proc:

        mock_psu.cpu_percent.return_value = 10.0
        mock_psu.cpu_count.return_value = 8
        from argparse import Namespace
        mock_psu.virtual_memory.return_value = Namespace(
            percent=50.0,
            used=8 * (1024**3),
            total=16 * (1024**3)
        )

        mock_proc.memory_info.return_value = Namespace(rss=1 * (1024**3))
        mock_proc.cpu_percent.return_value = 5.0

        app.config['TESTING'] = True
        with app.test_client() as test_client:
            yield test_client


@pytest.fixture(autouse=True)
def reset_module_state():
    """Reset module-level state between tests to prevent test pollution."""
    # pylint: disable=protected-access

    # Force reset module state before test
    prep_module._INSTANCE = None
    prep_module.Separator = None
    prep_module.ort = None
    vad_module._MODEL = None
    vad_module._UTILS = None
    # dash_module._PROCESS_OBJ = None

    yield

    # Also reset after test
    prep_module._INSTANCE = None
    prep_module.Separator = None
    prep_module.ort = None
    vad_module._MODEL = None
    vad_module._UTILS = None


# Apply global mocks before any tests run
# NOTE: numpy is NOT mocked because soundfile depends on it
sys.modules['torch'] = mock_torch
sys.modules['torch.nn'] = mock_torch_nn
sys.modules['torch.nn.functional'] = mock_torch_nn_functional
sys.modules['torchaudio'] = mock.MagicMock()
sys.modules['transformers'] = mock.MagicMock()
sys.modules['optimum'] = mock.MagicMock()
sys.modules['optimum.intel'] = mock.MagicMock()
sys.modules['openvino'] = mock.MagicMock()
sys.modules['openvino.runtime'] = mock.MagicMock()
sys.modules['openvino_genai'] = mock.MagicMock()
sys.modules['librosa'] = mock.MagicMock()
sys.modules['df'] = mock.MagicMock()
sys.modules['df.enhance'] = mock.MagicMock()
sys.modules['demucs'] = mock.MagicMock()
sys.modules['demucs.apply'] = mock.MagicMock()
sys.modules['demucs.pretrained'] = mock.MagicMock()

# Faster-Whisper mock
mock_faster_whisper = mock.MagicMock()
mock_faster_whisper.WhisperModel = mock.MagicMock()
mock_faster_whisper.BatchedInferencePipeline = mock.MagicMock()
sys.modules['faster_whisper'] = mock_faster_whisper
sys.modules['flasgger'] = mock.MagicMock()

# Audio separator mock
sys.modules['audio_separator'] = mock.MagicMock()
sys.modules['audio_separator.separator'] = mock.MagicMock()

# Soundfile mock (for tests that don't need real audio)
mock_soundfile = mock.MagicMock()
mock_soundfile.info = mock.MagicMock(
    return_value=mock.MagicMock(duration=10.0))
sys.modules['soundfile'] = mock_soundfile

# CTranslate2 mock
mock_ctranslate2 = mock.MagicMock()
mock_ctranslate2.get_cuda_device_count = mock.MagicMock(return_value=0)
sys.modules['ctranslate2'] = mock_ctranslate2

# Utility mocks
mock_psutil = mock.MagicMock()
mock_process = mock.MagicMock()
mock_psutil.cpu_percent.return_value = 10.0
mock_psutil.cpu_count.return_value = 8
mock_psutil.virtual_memory.return_value.percent = 50.0
mock_psutil.virtual_memory.return_value.used = 8 * (1024**3)
mock_psutil.virtual_memory.return_value.total = 16 * (1024**3)
mock_process.cpu_percent.return_value = 10.0
mock_process.memory_info.return_value = mock.MagicMock(rss=100 * 1024 * 1024)
mock_psutil.Process.return_value = mock_process
sys.modules['psutil'] = mock_psutil
sys.modules['tqdm'] = mock.MagicMock()
sys.modules['pydub'] = mock.MagicMock()
sys.modules['pydub.AudioSegment'] = mock.MagicMock()
sys.modules['requests'] = mock.MagicMock()
sys.modules['ffmpeg'] = mock.MagicMock()
sys.modules['ffmpeg_python'] = mock.MagicMock()
