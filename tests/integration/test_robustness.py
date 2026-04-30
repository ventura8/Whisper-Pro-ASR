"""System-wide robustness and edge-case validation for core modules."""
# pylint: disable=protected-access
import os
import sys
import shutil
import tempfile
import importlib
from collections import namedtuple
from unittest import mock
from flask import Flask
from modules import utils, bootstrap, config
from modules.api import routes_system
from modules.inference import model_manager, scheduler, language_detection
from modules.monitoring import metrics_discovery, dashboard_ui


def test_hardware_path_resolution():
    """Test bootstrap module hardware path resolution logic."""
    # 1. Test NVIDIA path
    with mock.patch.dict(os.environ, {"ASR_DEVICE": "CUDA"}):
        with mock.patch("os.path.exists", return_value=True):
            fake_path = []
            with mock.patch.object(sys, "path", fake_path):
                with mock.patch("importlib.reload"):
                    bootstrap.initialize_hardware_path()
                    assert "/app/libs/nvidia" in fake_path

    # 2. Test Intel path
    with mock.patch.dict(os.environ, {"ASR_DEVICE": "INTEL"}):
        with mock.patch("os.path.exists", return_value=True):
            fake_path = []
            with mock.patch.object(sys, "path", fake_path):
                with mock.patch("importlib.reload"):
                    bootstrap.initialize_hardware_path()
                    assert "/app/libs/intel" in fake_path

    # 3. Test Unknown platform fallback
    with mock.patch("platform.system", return_value="Unknown"):
        bootstrap.initialize_hardware_path()


def test_media_utilities_resilience():
    """Target gaps in media utilities and formatting logic."""
    # 1. clear_gpu_cache exception path
    with mock.patch("modules.utils.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.empty_cache.side_effect = Exception("CUDA Fail")
        utils.clear_gpu_cache()

    # 2. FFmpeg progress parsing failures
    process = mock.MagicMock()
    process.stdout.readline.side_effect = ["out_time_ms=invalid\n", ""]
    utils._parse_ffmpeg_progress(process, 10.0)

    # 3. Subtitle generation fallbacks
    res_text = {"text": "hello"}
    assert "00:00:00,000" in utils.generate_srt(res_text)
    assert "[No dialogue detected]" == utils.generate_srt({})
    assert "[No dialogue detected]" == utils.generate_srt({"text": ""})

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
    model_mock_2 = mock.MagicMock(spec=['pipeline'])
    pm_mock = mock.MagicMock()
    pm_mock.unload_model.side_effect = Exception("UVR Fail")

    model_manager._MODEL_POOL = {"CPU": model_mock, "GPU": model_mock_2}
    model_manager._PREPROCESSOR_POOL = {"CPU": pm_mock}
    model_mock.unload.side_effect = Exception("Unload Fail")

    mock_ct2 = mock.MagicMock()
    mock_ct2.clear_caches.side_effect = RuntimeError("CT2 Fail")
    with mock.patch.dict("modules.inference.model_manager._ENGINES", {"ctranslate2": mock_ct2}):
        with mock.patch("modules.utils.get_system_telemetry", return_value={}):
            model_manager.unload_models()

    assert len(model_manager._MODEL_POOL) == 0

    # 3. Libc/Platform specific reclamation failures
    with mock.patch("ctypes.CDLL", side_effect=OSError("No libc")):
        with mock.patch("modules.utils.get_system_telemetry", return_value={}):
            model_manager.unload_models()

    # 4. Scheduler metadata updates for untracked threads
    with mock.patch("threading.get_ident", return_value=999999):
        scheduler.update_task_metadata(foo="bar")
    scheduler.STATE.preemptible_units.clear()
    assert scheduler.get_preemptible_unit() is None

    # 5. Status helpers
    model_manager._MODEL_POOL = {"CPU": mock.MagicMock()}
    assert model_manager.is_engine_actually_loaded() is True
    model_manager._MODEL_POOL = {}
    assert model_manager.is_engine_actually_loaded() is False


def test_hardware_monitoring_fallbacks():
    """Test resilience when hardware monitoring tools are missing."""
    with metrics_discovery._CACHE_LOCK:
        metrics_discovery._METRIC_CACHE.clear()

    with mock.patch("subprocess.check_output", side_effect=FileNotFoundError("No tool")):
        assert metrics_discovery.get_npu_load() == 0
        assert metrics_discovery.get_intel_gpu_load() == 0
        assert metrics_discovery.get_nvidia_metrics() == []


def test_language_sampling_edge_cases():
    """Test empty results in language voting aggregation."""
    assert language_detection._aggregate_language_probs([]) == {}


def test_system_config_validation():
    """Test temp directory resolution and environment parsing resilience."""
    # 1. Temp dir resolution when primary storage is full
    Usage = namedtuple('Usage', ['free'])
    with mock.patch("shutil.disk_usage", return_value=Usage(free=0)):
        path = config.get_temp_dir(required_bytes=1000000)
        assert path == config.PERSISTENT_TEMP_DIR

    # 2. Environment parsing exception paths
    with mock.patch.dict(os.environ, {"MAX_CPU_UNITS": "INVALID"}):
        importlib.reload(config)
        assert config.MAX_CPU <= 999


def test_system_routes_logic_gaps():
    """Directly test system routes logic using a mock request context."""
    app = Flask(__name__)

    # 1. root with HTML
    with app.test_request_context(headers={"Accept": "text/html"}):
        with mock.patch("modules.monitoring.dashboard.get_dashboard_html", return_value="<html>"):
            assert routes_system.root() == "<html>"

    # 2. download_logs fail paths
    with app.test_request_context():
        with mock.patch("modules.api.routes_system.config") as mock_conf:
            mock_conf.LOG_DIR = "/non/existent"
            mock_conf.TEMP_DIR = "/non/existent/temp"
            with mock.patch("os.path.exists", return_value=False):
                resp, code = routes_system.download_logs()
                assert code == 404

    # 3. update_settings POST paths
    with app.test_request_context(method='POST'):
        # Valid update
        with mock.patch("flask.request") as mock_req:
            mock_req.method = 'POST'
            mock_req.json = {
                "ASR_MODEL": "test_model",
                "ASR_DEVICE": "CPU",
                "telemetry_retention_hours": 12,
                "log_retention_days": 5
            }
            with mock.patch("modules.config.update_env"):
                with mock.patch("modules.inference.model_manager.load_model"):
                    resp = routes_system.update_settings()
                    # Check status success if it's a JSON response
                    if hasattr(resp, 'json'):
                        assert resp.json["status"] == "success"

    # 4. help_endpoint
    with app.test_request_context():
        resp = routes_system.help_endpoint()
        if hasattr(resp, 'json'):
            assert "endpoints" in resp.json


def test_dashboard_ui_coverage():
    """Simple test for dashboard_ui to hit the long template string."""
    html = dashboard_ui.get_dashboard_html()
    assert "<!DOCTYPE html>" in html
    assert "Whisper Pro Dashboard" in html
