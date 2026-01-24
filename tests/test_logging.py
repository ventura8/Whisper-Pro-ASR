"""Tests for modules/logging_setup.py"""
# pylint: disable=import-outside-toplevel, attribute-defined-outside-init
# pylint: disable=consider-using-from-import
import os
import logging
from unittest import mock


class TestIgnoreSpecificWarnings:
    """Test suite for IgnoreSpecificWarnings filter."""

    def setup_method(self):
        """Set up test fixtures."""
        # Import the filter class
        from modules.logging_setup import IgnoreSpecificWarnings
        self.filter = IgnoreSpecificWarnings()

    def _create_record(self, message):
        """Helper to create a log record with a message."""
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="test.py",
            lineno=1,
            msg=message,
            args=(),
            exc_info=None
        )
        return record

    def test_filter_allows_normal_messages(self):
        """Test that normal messages pass through."""
        record = self._create_record("Normal log message")
        assert self.filter.filter(record) is True

    def test_filter_blocks_default_values_modified(self):
        """Test filtering of 'default values have been modified' warning."""
        record = self._create_record(
            "generation_config default values have been modified")
        assert self.filter.filter(record) is False

    def test_filter_blocks_custom_logits_processor(self):
        """Test filtering of custom logits processor warning."""
        record = self._create_record(
            "A custom logits processor of type MyProcessor")
        assert self.filter.filter(record) is False

    def test_filter_blocks_chunk_length_experimental(self):
        """Test filtering of chunk_length_s experimental warning."""
        record = self._create_record(
            "Using chunk_length_s is very experimental feature")
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
        record = self._create_record(
            "This is a development server. Do not use in production.")
        assert self.filter.filter(record) is False

    def test_filter_case_insensitive(self):
        """Test that filtering is case insensitive."""
        record = self._create_record("DEFAULT VALUES HAVE BEEN MODIFIED")
        assert self.filter.filter(record) is False

    def test_ignore_warnings_repr(self):
        """Cover line 52."""
        from modules.logging_setup import IgnoreSpecificWarnings
        assert repr(IgnoreSpecificWarnings()) == "IgnoreSpecificWarnings()"


class TestLogBanner:
    """Test suite for log_banner function."""

    def test_log_banner_logs_version(self):
        """Test that banner includes version name."""
        with mock.patch("modules.logging_setup.logger") as mock_logger:
            from modules.logging_setup import log_banner
            log_banner()

            # Collect all logged messages, handle lazy formatting
            logged_messages = []
            for call in mock_logger.info.call_args_list:
                args = call[0]
                if len(args) > 1:
                    # Basic reconstruction for test purposes
                    try:
                        logged_messages.append(args[0] % args[1:])
                    except TypeError:
                        logged_messages.append(str(args[0]))
                else:
                    logged_messages.append(str(args[0]))
            full_log = "\n".join(logged_messages)

            assert "Whisper Pro ASR" in full_log

    def test_log_banner_model_locally_found(self):
        """Test banner shows 'Locally Found' when model exists."""
        with mock.patch("modules.logging_setup.os.path.exists") as mock_exists:
            mock_exists.return_value = True
            with mock.patch("modules.logging_setup.os.listdir") as mock_listdir:
                mock_listdir.return_value = ["dummy_file"]
                with mock.patch("modules.logging_setup.logger") as mock_logger:
                    from modules.logging_setup import log_banner
                    log_banner()

                    logged_messages = []
                    for call in mock_logger.info.call_args_list:
                        args = call[0]
                        if len(args) > 1:
                            try:
                                logged_messages.append(args[0] % args[1:])
                            except TypeError:
                                logged_messages.append(str(args[0]))
                        else:
                            logged_messages.append(str(args[0]))
                    full_log = "\n".join(logged_messages)

                    assert "Locally Found" in full_log

    def test_log_banner_model_hugging_face(self):
        """Test banner shows 'Hugging Face' when model not local."""
        with mock.patch("modules.logging_setup.os.path.exists") as mock_exists:
            mock_exists.return_value = False
            with mock.patch("modules.logging_setup.logger") as mock_logger:
                from modules.logging_setup import log_banner
                log_banner()

                logged_messages = []
                for call in mock_logger.info.call_args_list:
                    args = call[0]
                    if len(args) > 1:
                        try:
                            logged_messages.append(args[0] % args[1:])
                        except TypeError:
                            logged_messages.append(str(args[0]))
                    else:
                        logged_messages.append(str(args[0]))
                full_log = "\n".join(logged_messages)

                assert "Hugging Face" in full_log

    def test_log_banner_shows_configuration(self):
        """Test banner shows configuration section."""
        with mock.patch("modules.logging_setup.logger") as mock_logger:
            from modules.logging_setup import log_banner
            log_banner()

            logged_messages = []
            for call in mock_logger.info.call_args_list:
                args = call[0]
                if len(args) > 1:
                    try:
                        logged_messages.append(args[0] % args[1:])
                    except TypeError:
                        logged_messages.append(str(args[0]))
                else:
                    logged_messages.append(str(args[0]))
            full_log = "\n".join(logged_messages)

            assert "[ENGINE CONFIG]" in full_log
            assert "Whisper Model ID" in full_log
            assert "Vocal Separator Model ID" in full_log
            assert "[HARDWARE INFO]" in full_log
            assert "Pipeline target" in full_log
            assert "ASR Runtime" in full_log
            assert "Preprocess Device" in full_log
            assert "Beam Size" in full_log

    def test_log_banner_shows_thread_info(self):
        """Test banner shows thread configuration."""
        with mock.patch("modules.logging_setup.config") as mock_conf:
            mock_conf.ASR_THREADS = 4
            mock_conf.PREPROCESS_THREADS = 8
            mock_conf.FFMPEG_THREADS = 2
            mock_conf.APP_NAME = "Whisper Intel XPU"
            mock_conf.VERSION = "1.0.0"
            mock_conf.MODEL_ID = "test-model"
            mock_conf.OV_CACHE_DIR = "/tmp/cache"
            mock_conf.DEVICE = "CPU"
            mock_conf.COMPUTE_TYPE = "int8"
            mock_conf.ASR_ENGINE_DEVICE = "cpu"
            mock_conf.ASR_ENGINE_COMPUTE_TYPE = "int8"
            mock_conf.PREPROCESS_DEVICE_NAME = "CPU"
            mock_conf.DEFAULT_BEAM_SIZE = 5
            mock_conf.ENABLE_VOCAL_SEPARATION = True

            with mock.patch("modules.logging_setup.logger") as mock_logger:
                from modules.logging_setup import log_banner
                log_banner()

                logged_messages = []
                for call in mock_logger.info.call_args_list:
                    args = call[0]
                    if len(args) > 1:
                        try:
                            logged_messages.append(args[0] % args[1:])
                        except TypeError:
                            logged_messages.append(str(args[0]))
                    else:
                        logged_messages.append(str(args[0]))
                full_log = "\n".join(logged_messages)

                assert "ASR=4" in full_log
                assert "Preprocess=8" in full_log
                assert "FFmpeg=2" in full_log


class TestGetDeviceProperties:
    """Test suite for _get_device_properties function."""

    def test_get_device_properties_no_openvino(self):
        """Test _get_device_properties when OpenVINO is not available."""
        with mock.patch.dict("sys.modules", {"openvino": None}):
            from modules.logging_setup import _get_device_properties
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
        with mock.patch.dict("sys.modules", {"openvino": mock_ov}):
            from modules.logging_setup import _get_device_properties
            device_name, info = _get_device_properties("NPU")
            assert device_name == "Intel(R) AI Boost"
            # Labels: Architecture, Driver Version, Uuid
            assert any(
                "Architecture" in line and "NPU3720" in line for line in info)
            assert any(
                "Driver Version" in line and "32.0.100.3104" in line for line in info)
            assert any("Uuid" in line and "abc-123" in line for line in info)

    def test_get_device_properties_substring_match(self):
        """Cover lines 133-135."""
        mock_core = mock.MagicMock()
        mock_core.available_devices = ["GPU.0", "CPU"]
        # Force full name lookup to fail so it returns real_device
        mock_core.get_property.side_effect = Exception("No property")

        mock_ov = mock.MagicMock()
        mock_ov.Core.return_value = mock_core

        with mock.patch.dict("sys.modules", {"openvino": mock_ov}):
            from modules.logging_setup import _get_device_properties
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
        with mock.patch.dict("sys.modules", {"openvino": mock_ov}):
            from modules.logging_setup import _get_device_properties
            _, info = _get_device_properties("NPU")
            # Labels: Range For Streams, Backend Is Ready
            assert any(
                "Range For Streams" in line and "1, 2, 4" in line for line in info)
            assert any(
                "Backend Is Ready" in line and "Yes" in line for line in info)

    def test_get_device_properties_property_exception(self):
        """Test _get_device_properties handles property exceptions gracefully."""
        mock_core = mock.MagicMock()
        mock_core.available_devices = ["NPU"]
        mock_core.get_property.side_effect = Exception("Property not found")

        mock_ov = mock.MagicMock()
        mock_ov.Core.return_value = mock_core
        with mock.patch.dict("sys.modules", {"openvino": mock_ov}):
            from modules.logging_setup import _get_device_properties
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
                "SUPPORTED_PROPERTIES", "FULL_DEVICE_NAME", "DEVICE_ID",
                "CACHING_PROPERTIES", "SUPPORTED_CONFIG_KEYS", "DEVICE_ARCH"
            ],
            "DEVICE_ARCH": "NPU4000",
        }.get(prop, None)

        mock_ov = mock.MagicMock()
        mock_ov.Core.return_value = mock_core
        with mock.patch.dict("sys.modules", {"openvino": mock_ov}):
            from modules.logging_setup import _get_device_properties
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
        with mock.patch.dict("sys.modules", {"openvino": mock_ov}):
            from modules.logging_setup import _get_device_properties
            device_name, info = _get_device_properties("NPU")
            assert device_name == "Intel NPU"
            # DEVICE_ARCH should show, FAILING_PROP should silently fail
            assert any("Arch" in line for line in info)

    def test_log_banner_with_npu_info(self):
        """Test banner shows hardware info section."""
        with mock.patch("modules.logging_setup.logger") as mock_logger:
            with mock.patch("modules.logging_setup.config") as mock_conf:
                mock_conf.APP_NAME = "Whisper Pro ASR"
                mock_conf.VERSION = "1.0.0"
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

                from modules.logging_setup import log_banner
                log_banner()

                logged_messages = []
                for call in mock_logger.info.call_args_list:
                    args = call[0]
                    if len(args) > 1:
                        try:
                            logged_messages.append(args[0] % args[1:])
                        except TypeError:
                            logged_messages.append(str(args[0]))
                    else:
                        logged_messages.append(str(args[0]))
                full_log = "\n".join(logged_messages)

                assert "[HARDWARE INFO]" in full_log
                assert "NPU" in full_log

    def test_log_level_info_by_default(self):
        """Test that log level is INFO when DEBUG is false."""
        with mock.patch.dict(os.environ, {"DEBUG": "false"}, clear=True):
            import importlib
            import modules.logging_setup as logging_setup
            importlib.reload(logging_setup)

            assert logging_setup.LOG_LEVEL == logging.INFO

    def test_log_level_debug_when_debug_mode(self):
        """Test that log level is DEBUG when DEBUG is true."""
        with mock.patch.dict(os.environ, {"DEBUG": "true"}):
            import importlib
            # Need to reload config first since logging_setup imports it
            import modules.config as config_module
            importlib.reload(config_module)
            import modules.logging_setup as logging_setup
            importlib.reload(logging_setup)

            assert logging_setup.LOG_LEVEL == logging.DEBUG

    def test_loggers_to_filter_list(self):
        """Test that LOGGERS_TO_FILTER contains expected loggers."""
        from modules.logging_setup import LOGGERS_TO_FILTER

        assert "transformers" in LOGGERS_TO_FILTER
        assert "optimum" in LOGGERS_TO_FILTER
        assert "openvino" in LOGGERS_TO_FILTER
        assert "werkzeug" in LOGGERS_TO_FILTER


class TestHardwareInfo:
    """Tests for hardware information utilities."""

    def test_get_real_model_name_intel_baked(self):
        """Cover lines 161-162."""
        from modules import config
        with mock.patch("modules.config.ASR_ENGINE", "INTEL-WHISPER"):
            with mock.patch("modules.config.MODEL_ID", config.OV_MODEL_BAKED):
                from modules.logging_setup import _get_real_model_name
                name = _get_real_model_name()
                assert "OpenVINO" in name

    def test_get_real_model_name_faster_baked(self):
        """Cover line 166."""
        from modules import config
        with mock.patch("modules.config.MODEL_ID", config.SYS_WHISPER_PATH):
            from modules.logging_setup import _get_real_model_name
            name = _get_real_model_name()
            assert "Systran" in name

    def test_unique_device_props(self):
        """Cover lines 229-230."""
        from modules.logging_setup import _unique_device_props
        props = ["A", "B", "A", "C"]
        res = _unique_device_props(props[:2], props[2:])
        assert res == ["A", "B", "C"]

    def test_banner_config_lines_intel(self):
        """Cover line 238, 267-269."""
        from modules.logging_setup import _banner_config_lines
        with mock.patch("modules.config.ASR_ENGINE", "INTEL-WHISPER"):
            with mock.patch("modules.config.DEVICE", "GPU"):
                cfg = {
                    "asr_display": "GPU.0",
                    "prep_display": "CPU",
                    "unique_props": ["Prop1", "Prop2"],
                    "model_status": "OK",
                    "cache_status": "OK",
                    "threads": "ASR=1",
                }
                lines = _banner_config_lines(cfg)
                assert any("OpenVINO (GPU)" in l for l in lines)
                assert any("[DEVICE PROPERTIES]" in l for l in lines)
                assert any("Prop1" in l for l in lines)
