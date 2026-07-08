"""
Global mocks setup for Whisper Pro ASR tests.
This module is imported first in conftest.py to intercept imports of heavy/ML libraries.
"""

import sys
from unittest import mock

# 1. Core ML dependency mocks
mock_torch = mock.MagicMock()
mock_torch.__path__ = []
mock_torch_nn = mock.MagicMock()
mock_torch_nn.__path__ = []
mock_torch_nn_functional = mock.MagicMock()

sys.modules["torch"] = mock_torch
sys.modules["torch.nn"] = mock_torch_nn
sys.modules["torch.nn.functional"] = mock_torch_nn_functional
sys.modules["torchaudio"] = mock.MagicMock()
sys.modules["transformers"] = mock.MagicMock()
sys.modules["optimum"] = mock.MagicMock()
sys.modules["optimum.intel"] = mock.MagicMock()
sys.modules["openvino"] = mock.MagicMock()
sys.modules["openvino.runtime"] = mock.MagicMock()
sys.modules["openvino_genai"] = mock.MagicMock()
sys.modules["librosa"] = mock.MagicMock()
sys.modules["df"] = mock.MagicMock()
sys.modules["df.enhance"] = mock.MagicMock()
sys.modules["demucs"] = mock.MagicMock()
sys.modules["demucs.apply"] = mock.MagicMock()
sys.modules["demucs.pretrained"] = mock.MagicMock()

# 2. Faster Whisper mocks
mock_fw = mock.MagicMock()
mock_fw.WhisperModel = mock.MagicMock()
mock_fw.BatchedInferencePipeline = mock.MagicMock()
mock_fw_audio = mock.MagicMock()
mock_fw_vad = mock.MagicMock()
mock_fw.audio = mock_fw_audio
mock_fw.vad = mock_fw_vad
sys.modules["faster_whisper"] = mock_fw
sys.modules["faster_whisper.audio"] = mock_fw_audio
sys.modules["faster_whisper.vad"] = mock_fw_vad

# 3. Flasgger mock
sys.modules["flasgger"] = mock.MagicMock()

# 4. Audio separator mock
sys.modules["audio_separator"] = mock.MagicMock()
sys.modules["audio_separator.separator"] = mock.MagicMock()

# 5. Soundfile mock
mock_soundfile = mock.MagicMock()
mock_soundfile.info = mock.MagicMock(return_value=mock.MagicMock(duration=10.0))
sys.modules["soundfile"] = mock_soundfile

# 6. CTranslate2 mock
mock_ctranslate2 = mock.MagicMock()
mock_ctranslate2.get_cuda_device_count = mock.MagicMock(return_value=0)
sys.modules["ctranslate2"] = mock_ctranslate2

# 7. Utility and System mocks
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
sys.modules["psutil"] = mock_psutil

sys.modules["tqdm"] = mock.MagicMock()
sys.modules["pydub"] = mock.MagicMock()
sys.modules["pydub.AudioSegment"] = mock.MagicMock()
sys.modules["requests"] = mock.MagicMock()
sys.modules["ffmpeg"] = mock.MagicMock()
sys.modules["ffmpeg_python"] = mock.MagicMock()
