"""Tests for modules/logging_setup.py"""

import importlib
import logging
import os
from unittest import mock

from starlette.datastructures import QueryParams

import whisper_pro_asr
from modules.core import config, logging_setup
from modules.core.logging_setup import (
    LOGGERS_TO_FILTER,
    IgnoreSpecificWarnings,
    _banner_config_lines,
    _get_device_properties,
    _get_real_model_name,
    _unique_device_props,
    log_banner,
)


def _logged_messages(mock_logger: mock.MagicMock) -> str:
    messages = []
    for call in mock_logger.info.call_args_list:
        args = call[0]
        if len(args) > 1:
            try:
                messages.append(args[0] % args[1:])
            except TypeError:
                messages.append(str(args[0]))
        else:
            messages.append(str(args[0]))
    return "\n".join(messages)


class TestIgnoreSpecificWarnings:
    """Test suite for IgnoreSpecificWarnings filter."""

    filter = None

    def setup_method(self):
        """Set up test fixtures."""
        # Import the filter class
        self.filter = IgnoreSpecificWarnings()

    def _create_record(self, message):
        """Helper to create a log record with a message."""
        record = logging.LogRecord(name="test", level=logging.WARNING, pathname="test.py", lineno=1, msg=message, args=(), exc_info=None)
        return record

    def test_filter_allows_normal_messages(self):
        """Test that normal messages pass through."""
        record = self._create_record("Normal log message")
        assert self.filter.filter(record) is True

    def test_filter_blocks_default_values_modified(self):
        """Test filtering of 'default values have been modified' warning."""
        record = self._create_record("generation_config default values have been modified")
        assert self.filter.filter(record) is False

    def test_filter_blocks_custom_logits_processor(self):
        """Test filtering of custom logits processor warning."""
        record = self._create_record("A custom logits processor of type MyProcessor")
        assert self.filter.filter(record) is False

    def test_filter_blocks_segment_length_experimental(self):
        """Test filtering of chunk_length_s experimental warning."""
        record = self._create_record("Using chunk_length_s is very experimental feature")
        assert self.filter.filter(record) is False

    def test_filter_blocks_device_use_cpu(self):
        """Test filtering of 'Device set to use cpu' warning."""
        record = self._create_record("Device set to use cpu for inference")
        assert self.filter.filter(record) is False

    def test_filter_blocks_use_cpu_alternative(self):
        """Test filtering of 'use cpu' variation."""
        record = self._create_record("Will use cpu instead")
        assert self.filter.filter(record) is False

    def test_filter_blocks_development_server(self):
        """Test filtering of Flask development server warning."""
        record = self._create_record("This is a development server. Do not use in production.")
        assert self.filter.filter(record) is False

    def test_filter_case_insensitive(self):
        """Test that filtering is case insensitive."""
        record = self._create_record("DEFAULT VALUES HAVE BEEN MODIFIED")
        assert self.filter.filter(record) is False

    def test_ignore_warnings_repr(self):
        """Cover line 52."""
        assert repr(IgnoreSpecificWarnings()) == "IgnoreSpecificWarnings()"


def test_contextual_filter_uses_system_when_filename_none():
    """ContextualFilter should map None filename values to System context."""
    filt = logging_setup.ContextualFilter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )

    logging_setup.utils.THREAD_CONTEXT.reset()
    logging_setup.utils.THREAD_CONTEXT.filename = None

    assert filt.filter(record) is True
    assert getattr(record, "task_ctx", "") == ""


class TestLogBanner:
    """Test suite for log_banner function."""

    def test_log_banner_logs_version(self):
        """Test that banner includes version name."""
        with mock.patch("modules.core.logging_setup.logger") as mock_logger:
            log_banner()

            assert "Whisper Pro ASR" in _logged_messages(mock_logger)

    def test_log_banner_model_locally_found(self):
        """Test banner shows 'Locally Found' when model exists."""
        with mock.patch("modules.core.logging_setup.os.path.exists") as mock_exists:
            mock_exists.return_value = True
            with mock.patch("modules.core.logging_setup.os.listdir") as mock_listdir:
                mock_listdir.return_value = ["dummy_file"]
                with mock.patch("modules.core.logging_setup.logger") as mock_logger:
                    log_banner()

                    assert "Locally Found" in _logged_messages(mock_logger)

    def test_log_banner_model_hugging_face(self):
        """Test banner shows 'Hugging Face' when model not local."""
        with mock.patch("modules.core.logging_setup.os.path.exists") as mock_exists:
            mock_exists.return_value = False
            with mock.patch("modules.core.logging_setup.logger") as mock_logger:
                log_banner()

                assert "Hugging Face" in _logged_messages(mock_logger)

    def test_log_banner_shows_configuration(self):
        """Test banner shows configuration section."""
        with mock.patch("modules.core.logging_setup.logger") as mock_logger:
            log_banner()

            full_log = _logged_messages(mock_logger)

            assert all(
                token in full_log
                for token in [
                    "[ENGINE CONFIG]",
                    "Whisper Model ID",
                    "Vocal Separator Model ID",
                    "[HARDWARE INFO]",
                    "Pipeline target",
                    "ASR Runtime",
                    "Preprocess Device",
                    "Beam Size",
                ]
            )

    def test_log_banner_shows_thread_info(self):
        """Test banner shows thread configuration."""
        with mock.patch("modules.core.logging_setup.config") as mock_conf:
            mock_conf.ASR_THREADS = 4
            mock_conf.PREPROCESS_THREADS = 8
            mock_conf.FFMPEG_THREADS = 2
            mock_conf.APP_NAME = "Whisper Intel XPU"
            mock_conf.VERSION = "1.0.1"
            mock_conf.MODEL_ID = "test-model"
            mock_conf.OV_CACHE_DIR = "/tmp/cache"
            mock_conf.DEVICE = "CPU"
            mock_conf.COMPUTE_TYPE = "int8"
            mock_conf.ASR_ENGINE_DEVICE = "cpu"
            mock_conf.ASR_ENGINE_COMPUTE_TYPE = "int8"
            mock_conf.PREPROCESS_DEVICE_NAME = "CPU"
            mock_conf.DEFAULT_BEAM_SIZE = 5
            mock_conf.ENABLE_VOCAL_SEPARATION = True

            with mock.patch("modules.core.logging_setup.logger") as mock_logger:
                log_banner()

                full_log = _logged_messages(mock_logger)

                assert all(token in full_log for token in ["ASR=4", "Preprocess=8", "FFmpeg=2"])


class TestGetDeviceProperties:
    """Test suite for _get_device_properties function."""

    def test_get_device_properties_no_openvino(self):
        """Test _get_device_properties when OpenVINO is not available."""
        with mock.patch("importlib.import_module", side_effect=ImportError):
            device_name, info = _get_device_properties("NPU")
            assert device_name == "NPU"
            assert not info

    def test_get_device_properties_with_openvino(self):
        """Test _get_device_properties when OpenVINO Core returns multiple properties."""
        mock_core = mock.MagicMock()
        mock_core.available_devices = ["NPU"]
        mock_core.get_property.side_effect = lambda dev, prop: {
            "FULL_DEVICE_NAME": "Intel(R) AI Boost",
            "SUPPORTED_PROPERTIES": ["DEVICE_ARCHITECTURE", "NPU_DRIVER_VERSION", "DEVICE_UUID"],
            "DEVICE_ARCHITECTURE": "NPU3720",
            "NPU_DRIVER_VERSION": "32.0.100.3104",
            "DEVICE_UUID": "abc-123",
        }.get(prop, None)

        mock_ov = mock.MagicMock()
        mock_ov.Core.return_value = mock_core
        with mock.patch("importlib.import_module", return_value=mock_ov):
            device_name, info = _get_device_properties("NPU")
            assert device_name == "Intel(R) AI Boost"
            # Labels: Architecture, Driver Version, Uuid
            assert any("Architecture" in line and "NPU3720" in line for line in info)
            assert any("Driver Version" in line and "32.0.100.3104" in line for line in info)
            assert any("Uuid" in line and "abc-123" in line for line in info)

    def test_get_device_properties_substring_match(self):
        """Cover lines 133-135."""
        mock_core = mock.MagicMock()
        mock_core.available_devices = ["GPU.0", "CPU"]
        # Force full name lookup to fail so it returns real_device
        mock_core.get_property.side_effect = Exception("No property")

        mock_ov = mock.MagicMock()
        mock_ov.Core.return_value = mock_core
        with mock.patch("importlib.import_module", return_value=mock_ov):
            name, _ = _get_device_properties("GPU")
            assert name == "GPU.0"

    def test_get_device_properties_with_range_values(self):
        """Test _get_device_properties formats list and boolean values correctly."""
        mock_core = mock.MagicMock()
        mock_core.available_devices = ["NPU"]
        mock_core.get_property.side_effect = lambda dev, prop: {
            "FULL_DEVICE_NAME": "Intel(R) AI Boost",
            "SUPPORTED_PROPERTIES": ["RANGE_FOR_STREAMS", "NPU_BACKEND_IS_READY"],
            "RANGE_FOR_STREAMS": [1, 2, 4],
            "NPU_BACKEND_IS_READY": True,
        }.get(prop, None)

        mock_ov = mock.MagicMock()
        mock_ov.Core.return_value = mock_core
        with mock.patch("importlib.import_module", return_value=mock_ov):
            _, info = _get_device_properties("NPU")
            # Labels: Range For Streams, Backend Is Ready
            assert any("Range For Streams" in line and "1, 2, 4" in line for line in info)
            assert any("Backend Is Ready" in line and "Yes" in line for line in info)

    def test_get_device_properties_property_exception(self):
        """Test _get_device_properties handles property exceptions gracefully."""
        mock_core = mock.MagicMock()
        mock_core.available_devices = ["NPU"]
        mock_core.get_property.side_effect = Exception("Property not found")

        mock_ov = mock.MagicMock()
        mock_ov.Core.return_value = mock_core
        with mock.patch("importlib.import_module", return_value=mock_ov):
            device_name, info = _get_device_properties("NPU")
            assert device_name == "NPU"
            assert not info

    def test_get_device_properties_skips_internal_props(self):
        """Test _get_device_properties skips internal properties like SUPPORTED_PROPERTIES."""
        mock_core = mock.MagicMock()
        mock_core.available_devices = ["NPU"]
        mock_core.get_property.side_effect = lambda dev, prop: {
            "FULL_DEVICE_NAME": "Intel(R) AI Boost",
            "SUPPORTED_PROPERTIES": [
                p.upper()
                for p in [
                    "supported_properties",
                    "full_device_name",
                    "device_id",
                    "caching_properties",
                    "supported_config_keys",
                    "device_arch",
                ]
            ],
            "DEVICE_ARCH": "NPU4000",
        }.get(prop, None)

        mock_ov = mock.MagicMock()
        mock_ov.Core.return_value = mock_core
        with mock.patch("importlib.import_module", return_value=mock_ov):
            _, info = _get_device_properties("NPU")
            # Only DEVICE_ARCH should show, others are skipped
            assert any("Arch" in line for line in info)
            assert not any("Supported Properties" in line for line in info)
            assert not any("Full Device Name" in line for line in info)

    def test_get_device_properties_property_value_exception(self):
        """Test _get_device_properties handles per-property exceptions gracefully."""
        mock_core = mock.MagicMock()
        mock_core.available_devices = ["NPU"]

        def prop_getter(_dev, prop):
            if prop == "FULL_DEVICE_NAME":
                return "Intel NPU"
            if prop == "SUPPORTED_PROPERTIES":
                return ["DEVICE_ARCH", "FAILING_PROP"]
            if prop == "FAILING_PROP":
                raise RuntimeError("Cannot read property")
            return "Value"

        mock_core.get_property.side_effect = prop_getter

        mock_ov = mock.MagicMock()
        mock_ov.Core.return_value = mock_core
        with mock.patch("importlib.import_module", return_value=mock_ov):
            device_name, info = _get_device_properties("NPU")
            assert device_name == "Intel NPU"
            # DEVICE_ARCH should show, FAILING_PROP should silently fail
            assert any("Arch" in line for line in info)

    def test_log_banner_with_npu_info(self):
        """Test banner shows hardware info section."""
        with mock.patch("modules.core.logging_setup.logger") as mock_logger:
            with mock.patch("modules.core.logging_setup.config") as mock_conf:
                mock_conf.APP_NAME = "Whisper Pro ASR"
                mock_conf.VERSION = "1.0.1"
                mock_conf.MODEL_ID = "test-model"
                mock_conf.OV_CACHE_DIR = "/tmp/cache"
                mock_conf.DEVICE = "NPU"
                mock_conf.ASR_DEVICE_NAME = "NPU"
                mock_conf.PREPROCESS_DEVICE_NAME = "GPU"
                mock_conf.COMPUTE_TYPE = "int8"
                mock_conf.ASR_ENGINE_DEVICE = "cpu"
                mock_conf.ASR_ENGINE_COMPUTE_TYPE = "int8"
                mock_conf.DEFAULT_BEAM_SIZE = 5
                mock_conf.ASR_THREADS = 4
                mock_conf.PREPROCESS_THREADS = 8
                mock_conf.FFMPEG_THREADS = 2
                mock_conf.ENABLE_VOCAL_SEPARATION = True

                with mock.patch.dict(
                    os.environ,
                    {
                        "INTEL_OPENVINO_DIR": "/opt/intel/openvino",
                        "LD_LIBRARY_PATH": "/opt/intel/openvino/runtime/lib/intel64",
                        "LIBVA_DRIVER_NAME": "iHD",
                        "ONEAPI_DEVICE_SELECTOR": "level_zero:gpu",
                        "ZE_AFFINITY_MASK": "0",
                        "OCL_ICD_VENDORS": "/etc/OpenCL/vendors",
                    },
                    clear=False,
                ):
                    with mock.patch("importlib.import_module") as mock_import_module:
                        mock_core = mock.MagicMock()
                        mock_core.available_devices = ["GPU.0", "NPU.0", "CPU"]
                        mock_core.get_property.return_value = "Intel(R) AI Boost"
                        mock_ov = mock.MagicMock()
                        mock_ov.Core.return_value = mock_core
                        mock_import_module.return_value = mock_ov

                        log_banner()

                full_log = _logged_messages(mock_logger)

                assert all(token in full_log for token in ["[HARDWARE INFO]", "NPU"])
                assert "INTEL RUNTIME ENV" in full_log
                assert "LIBVA_DRIVER_NAME" in full_log
                assert "OpenVINO devices" in full_log

    def test_log_level_info_by_default(self):
        """Test that log level is INFO when DEBUG is false."""
        with mock.patch.dict(os.environ, {"DEBUG": "false"}, clear=True):
            importlib.reload(logging_setup)

            assert logging_setup.LOG_LEVEL == logging.INFO

    def test_log_level_debug_when_debug_mode(self):
        """Test that log level is DEBUG when DEBUG is true."""
        with mock.patch.dict(os.environ, {"DEBUG": "true"}):
            # Need to reload config first since logging_setup imports it
            importlib.reload(config)
            importlib.reload(logging_setup)
            logging_setup.setup_logging()

            assert logging_setup.LOG_LEVEL == logging.DEBUG

    def test_loggers_to_filter_list(self):
        """Test that LOGGERS_TO_FILTER contains expected loggers."""

        assert "transformers" in LOGGERS_TO_FILTER
        assert "optimum" in LOGGERS_TO_FILTER
        assert "openvino" in LOGGERS_TO_FILTER
        assert "werkzeug" in LOGGERS_TO_FILTER


class TestHardwareInfo:
    """Tests for hardware information utilities."""

    def test_get_real_model_name_intel_baked(self):
        """Cover lines 161-162."""
        with mock.patch("modules.core.config.ASR_ENGINE", "INTEL-WHISPER"):
            with mock.patch("modules.core.config.MODEL_ID", config.OV_MODEL_BAKED):
                name = _get_real_model_name()
                assert "OpenVINO" in name

    def test_get_real_model_name_faster_baked(self):
        """Cover line 166."""
        with mock.patch("modules.core.config.MODEL_ID", config.SYS_WHISPER_PATH):
            name = _get_real_model_name()
            assert "Systran" in name

    def test_unique_device_props(self):
        """Cover lines 229-230."""
        props = ["A", "B", "A", "C"]
        res = _unique_device_props(props[:2], props[2:])
        assert res == ["A", "B", "C"]

    def test_banner_config_lines_intel(self):
        """Cover line 238, 267-269."""
        with mock.patch("modules.core.config.ASR_ENGINE", "INTEL-WHISPER"):
            with mock.patch("modules.core.config.DEVICE", "GPU"):
                cfg = {
                    "asr_display": "GPU.0",
                    "prep_display": "CPU",
                    "unique_props": ["Prop1", "Prop2"],
                    "model_status": "OK",
                    "cache_status": "OK",
                    "threads": "ASR=1",
                    "resource_pool": "GPU.0, CPU",
                }
                lines = _banner_config_lines(cfg)
                assert any("OpenVINO (GPU)" in line for line in lines)
                assert any("[DEVICE PROPERTIES]" in line for line in lines)
                assert any("Prop1" in line for line in lines)


def test_sanitize_query_params():
    """Verify sanitize_query_params retains only safe fields and URL-encodes keys/values."""
    qp = QueryParams([("hf_token", "secret123"), ("beam_size", "5 & more"), ("special key", "value")])
    assert whisper_pro_asr.sanitize_query_params(qp) == "beam_size=5%20%26%20more"
