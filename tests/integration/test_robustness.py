"""System-wide robustness and edge-case validation for core modules."""

import asyncio
import importlib
import os
import shutil
import sys
import tempfile
from collections import namedtuple
from unittest import mock

from modules.api import routes_system
from modules.core import bootstrap, config, utils
from modules.inference import language_detection, model_manager, scheduler
from modules.monitoring import dashboard_ui, metrics_discovery


def test_hardware_path_resolution():
    """Test bootstrap module hardware path resolution logic."""

    class PathTracker(list):
        """Minimal sys.path stand-in that records insert calls."""

        def __init__(self):
            super().__init__()
            self.insert_calls = []

        def insert(self, index, value):
            self.insert_calls.append((index, value))
            super().insert(index, value)

    # 1. Test NVIDIA path
    with mock.patch.dict(os.environ, {"ASR_DEVICE": "CUDA"}):
        with mock.patch("os.path.exists", return_value=True):
            fake_path = PathTracker()
            with mock.patch.object(sys, "path", fake_path):
                with mock.patch("importlib.reload"):
                    bootstrap.initialize_hardware_path()
                    assert (0, "/app/libs/nvidia") in fake_path.insert_calls

    # 2. Test Intel path
    with mock.patch.dict(os.environ, {"ASR_DEVICE": "INTEL"}):
        with mock.patch("os.path.exists", return_value=True):
            fake_path = PathTracker()
            with mock.patch.object(sys, "path", fake_path):
                with mock.patch("importlib.reload"):
                    bootstrap.initialize_hardware_path()
                    assert (0, "/app/libs/intel") in fake_path.insert_calls

    # 3. Test Intel path fallback when OpenVINO probe cannot be performed
    with mock.patch.dict(os.environ, {"ASR_DEVICE": "auto"}):

        def mock_exists(path):
            return path in ["/dev/accel", "/app/libs/intel"]

        with mock.patch("os.path.exists", side_effect=mock_exists):
            fake_path = PathTracker()
            with mock.patch.object(sys, "path", fake_path):
                with mock.patch("openvino.Core", side_effect=RuntimeError("probe failed")):
                    with mock.patch("importlib.reload"):
                        bootstrap.initialize_hardware_path()
                        assert (0, "/app/libs/intel") in fake_path.insert_calls

    # 4. Test Intel path auto-detection via OpenVINO fallback query
    with mock.patch.dict(os.environ, {"ASR_DEVICE": "auto"}):

        def mock_exists_ov(path):
            return path == "/app/libs/intel"

        with mock.patch("os.path.exists", side_effect=mock_exists_ov):
            mock_core = mock.MagicMock()
            mock_core.available_devices = ["NPU"]
            with mock.patch("openvino.Core", return_value=mock_core):
                fake_path = PathTracker()
                with mock.patch.object(sys, "path", fake_path):
                    with mock.patch("importlib.reload"):
                        bootstrap.initialize_hardware_path()
                        assert (0, "/app/libs/intel") in fake_path.insert_calls

    # 5. Test Unknown platform fallback
    with mock.patch("platform.system", return_value="Unknown"):
        bootstrap.initialize_hardware_path()


def test_media_utilities_resilience():
    """Target gaps in media utilities and formatting logic."""
    # 1. clear_gpu_cache exception path
    with mock.patch("modules.core.utils.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.empty_cache.side_effect = Exception("CUDA Fail")
        utils.clear_gpu_cache()

    # 2. FFmpeg progress parsing failures and speed parsing
    process = mock.MagicMock()
    process.stdout.readline.side_effect = ["out_time_ms=invalid\n", "speed= 1.25x\n", ""]
    speed = utils.parse_ffmpeg_progress(process, 10.0)
    assert speed == "1.25x"

    # 3. Subtitle generation fallbacks
    res_text = {"text": "hello"}
    assert "00:00:00,000" in utils.generate_srt(res_text)
    assert "[No dialogue detected]" in utils.generate_srt({})
    assert "[No dialogue detected]" in utils.generate_srt({"text": ""})

    res_none = {"segments": [{"text": "hello", "start": None, "end": None}]}
    assert "00:00:00,000" in utils.generate_srt(res_none)

    assert "WEBVTT" in utils.generate_vtt(res_text)
    assert "[No dialogue detected]" in utils.generate_vtt({})
    assert "[No dialogue detected]" in utils.generate_vtt({"text": ""})

    res_vtt_none = {"segments": [{"text": "hello", "start": None, "end": None}]}
    assert "00:00:00.000" in utils.generate_vtt(res_vtt_none)

    assert "start\tend\ttext" == utils.generate_tsv({})
    assert "" == utils.generate_txt({})

    # 4. Filesystem operations resilience
    utils.secure_remove("")
    with mock.patch("os.path.exists", return_value=True):
        with mock.patch("os.remove", side_effect=Exception("Remove Fail")):
            utils.secure_remove("dummy.path")

    # 5. Cleanup routine error handling
    with mock.patch("os.path.exists", return_value=True):
        with mock.patch("os.walk", return_value=[("/tmp", [], ["old.tmp"])]):
            with mock.patch("os.path.getmtime", return_value=0):
                with mock.patch("os.remove", side_effect=Exception("Prune Fail")):
                    utils.cleanup_old_files("/tmp", days=1)

    # 6. Temporary asset purging
    temp_dir = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(temp_dir, "preprocessing"))
        with mock.patch.dict(os.environ, {"WHISPER_TEMP_DIR": temp_dir}):
            utils.purge_temporary_assets()
        with mock.patch("os.listdir", side_effect=OSError("List Fail")):
            with mock.patch.dict(os.environ, {"WHISPER_TEMP_DIR": temp_dir}):
                utils.purge_temporary_assets()
    finally:
        shutil.rmtree(temp_dir)

    # 7. Media validation failures
    assert utils.convert_to_wav(None) is None
    with mock.patch("os.path.exists", return_value=True):
        with mock.patch("os.path.getsize", return_value=0):
            assert utils.convert_to_wav("empty.wav") is None

    assert utils.get_pretty_model_name(None) == "Unknown Engine"


def test_engine_resource_management():
    """Test model management and hardware locking edge cases."""
    # 1. Language detection fallback paths
    model = mock.MagicMock()
    model.detect_language.return_value = ("en", 0.9, [("en", 0.9)])
    audio = mock.MagicMock()
    audio.astype.return_value = audio
    res = model_manager.run_language_detection_core(model, audio, skip_vad=True)
    assert res["language"] == "en"

    model.detect_language.side_effect = Exception("Detect Fail")
    model.transcribe.return_value = ([], mock.MagicMock(language="en", language_probability=0.8, all_language_probs=[]))
    res = model_manager.run_language_detection_core(model, audio, skip_vad=True)
    assert res["language"] == "en"

    # 2. Aggressive offload and reclamation failures
    model_mock = mock.MagicMock()
    model_mock_2 = mock.MagicMock(spec=["pipeline"])
    pm_mock = mock.MagicMock()
    pm_mock.unload_model.side_effect = Exception("UVR Fail")

    model_manager.MODEL_POOL = {"CPU": model_mock, "GPU": model_mock_2}
    model_manager.PREPROCESSOR_POOL = {"CPU": pm_mock}
    model_mock.unload.side_effect = Exception("Unload Fail")

    mock_ct2 = mock.MagicMock()
    mock_ct2.clear_caches.side_effect = RuntimeError("CT2 Fail")
    with mock.patch.dict("modules.inference.model_manager._ENGINES", {"ctranslate2": mock_ct2}):
        with mock.patch("modules.core.utils.get_system_telemetry", return_value={}):
            model_manager.unload_models()

    assert len(model_manager.MODEL_POOL) == 0

    # 3. Libc/Platform specific reclamation failures
    with mock.patch("ctypes.CDLL", side_effect=OSError("No libc")):
        with mock.patch("modules.core.utils.get_system_telemetry", return_value={}):
            model_manager.unload_models()

    # 4. Scheduler metadata updates for untracked threads
    with mock.patch("threading.get_ident", return_value=999999):
        scheduler.update_task_metadata(foo="bar")
    scheduler.STATE.preemptible_units.clear()
    assert scheduler.get_preemptible_unit() is None

    # 5. Status helpers
    model_manager.MODEL_POOL = {"CPU": mock.MagicMock()}
    assert model_manager.is_engine_actually_loaded() is True
    model_manager.MODEL_POOL = {}
    assert model_manager.is_engine_actually_loaded() is False


def test_hardware_monitoring_fallbacks():
    """Test resilience when hardware monitoring tools are missing."""
    with metrics_discovery.CACHE_LOCK:
        metrics_discovery.METRIC_CACHE.clear()

    with mock.patch("subprocess.check_output", side_effect=FileNotFoundError("No tool")):
        assert metrics_discovery.get_npu_load() == 0
        assert metrics_discovery.get_intel_gpu_load() == 0
        assert metrics_discovery.get_nvidia_metrics() == []


def test_language_sampling_edge_cases():
    """Test empty results in language voting aggregation."""
    assert language_detection.aggregate_language_probs([]) == {}


def test_system_config_validation():
    """Test temp directory resolution and environment parsing resilience."""
    # 1. Temp dir resolution when primary storage is full
    Usage = namedtuple("Usage", ["free"])
    with mock.patch("shutil.disk_usage", return_value=Usage(free=0)):
        path = config.get_temp_dir(required_bytes=1000000)
        assert path == config.PERSISTENT_TEMP_DIR

    # 2. Environment parsing exception paths
    with mock.patch.dict(os.environ, {"MAX_CPU_UNITS": "INVALID"}):
        importlib.reload(config)
        assert config.MAX_CPU <= 999


def test_system_routes_logic_gaps():
    """Directly test system routes logic using mock Request objects."""
    # 1. root with HTML
    mock_request = mock.MagicMock()
    mock_request.headers = {"accept": "text/html"}
    with mock.patch("modules.monitoring.dashboard.get_dashboard_html", return_value="<html>"):
        resp = routes_system.root(mock_request)
        assert "<html>" in resp.body.decode()

    # 2. download_logs fail paths
    with mock.patch("modules.api.routes_system.config") as mock_conf:
        mock_conf.LOG_DIR = "/non/existent"
        mock_conf.TEMP_DIR = "/non/existent/temp"
        with mock.patch("os.path.exists", return_value=False):
            resp = routes_system.download_logs()
            assert resp.status_code == 404

    # 3. update_settings POST paths
    mock_post_request = mock.AsyncMock()
    mock_post_request.json.return_value = {
        "ASR_MODEL": "test_model",
        "ASR_DEVICE": "CPU",
        "telemetry_retention_hours": 12,
        "log_retention_days": 5,
    }

    with mock.patch("modules.core.config.update_env"):
        with mock.patch("modules.inference.model_manager.load_model"):
            resp = asyncio.run(routes_system.update_settings(mock_post_request))
            assert resp["status"] == "success"

    # 4. help_endpoint
    mock_help_request = mock.MagicMock()
    mock_help_request.base_url = "http://localhost/"
    resp = routes_system.help_endpoint(mock_help_request)
    assert "endpoints" in resp


def test_dashboard_ui_coverage():
    """Simple test for dashboard_ui to hit the long template string."""
    html = dashboard_ui.get_dashboard_html()
    assert "<!DOCTYPE html>" in html
    assert "Whisper Pro Dashboard" in html
