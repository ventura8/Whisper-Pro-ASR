"""
Global mocks setup for Whisper Pro ASR tests.
This module is imported first in conftest.py to intercept imports of heavy/ML libraries.

The rule for what belongs here: mock a module only when it is genuinely not installed in
the test image. Everything below is absent because installing it would mean shipping torch,
CTranslate2, OpenVINO or ROCm into a lint-and-unit image -- gigabytes, for a container that
never runs a model. Those tests run against the real stacks on hardware instead, through
the `real_asr` suites in tests/real_audio and tests/integration.

A mock for a module that *is* installed is worse than no test: it silently replaces the
thing under test. tqdm, requests, soundfile, psutil and ffmpeg were all mocked here while
being fully installed, and the tqdm entry was hiding a genuine failure -- huggingface_hub
could not import through the MagicMock, so the one test checking our snapshot_download call
against the real installed signature never ran. They now use the real modules; tests that
need particular values patch them locally, where the substitution is visible.
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

# 6. CTranslate2 mock
mock_ctranslate2 = mock.MagicMock()
mock_ctranslate2.get_cuda_device_count = mock.MagicMock(return_value=0)
sys.modules["ctranslate2"] = mock_ctranslate2

sys.modules["pydub"] = mock.MagicMock()
sys.modules["pydub.AudioSegment"] = mock.MagicMock()
