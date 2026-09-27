"""Branch coverage tests for modules/core/logging_setup.py."""

import logging
from unittest import mock

from modules.core import logging_setup


def test_contextual_filter_includes_step_info_when_present(monkeypatch):
    """Contextual filter should include step info in task context when present."""
    filt = logging_setup.ContextualFilter()
    record = logging.LogRecord("test", logging.INFO, "p", 1, "msg", (), None)

    logging_setup.utils.THREAD_CONTEXT.reset()
    # raising=False: after reset() neither attribute exists yet; monkeypatch deletes both again
    # on teardown instead of leaving this test's task context behind.
    monkeypatch.setattr(logging_setup.utils.THREAD_CONTEXT, "filename", "file.wav", raising=False)
    monkeypatch.setattr(logging_setup.utils.THREAD_CONTEXT, "step_info", "(1/3)", raising=False)

    assert filt.filter(record) is True
    assert getattr(record, "task_ctx", "") == "[file.wav] (1/3)"
    assert repr(filt) == "ContextualFilter()"


def test_werkzeug_status_filter_demotes_to_debug_in_debug_mode():
    """Status polling logs should be demoted instead of dropped when root logger is DEBUG."""
    record = logging.LogRecord("werkzeug", logging.INFO, "p", 1, "GET /status HTTP/1.1", (), None)
    filt = logging_setup.WerkzeugStatusFilter()

    root = logging.getLogger()
    previous_level = root.level
    root.setLevel(logging.DEBUG)
    try:
        assert filt.filter(record) is True
        assert record.levelno == logging.DEBUG
        assert record.levelname == "DEBUG"
        assert repr(filt) == "WerkzeugStatusFilter()"
    finally:
        root.setLevel(previous_level)


def test_log_buffer_handler_emit_swallows_format_errors():
    """Log buffer handler should tolerate formatting/runtime issues without raising."""
    handler = logging_setup.LogBufferHandler()
    record = logging.LogRecord("x", logging.INFO, "p", 1, "x", (), None)

    with mock.patch.object(handler, "format", side_effect=ValueError("boom")):
        handler.emit(record)


def test_resolve_log_buffer_target_key_prefers_task_then_thread():
    """Log buffer routing should prefer task keys and then fallback to thread keys."""
    resolver = getattr(logging_setup, "_resolve_log_buffer_target_key")
    with logging_setup.TASK_LOGS_LOCK:
        logging_setup.TASK_LOGS.clear()
        logging_setup.TASK_LOGS["task-1"] = []
        logging_setup.TASK_LOGS[99] = []

        assert resolver("task-1", 99) == "task-1"
        assert resolver(None, 99) == 99
        assert resolver("missing", 7) is None


def test_get_file_handler_returns_none_on_initialization_error():
    """File handler factory should return None when filesystem setup fails."""
    holder = getattr(logging_setup, "_FILE_HANDLER_HOLDER")
    holder[0] = None
    with (
        mock.patch("modules.core.logging_setup.os.makedirs", side_effect=OSError("nope")),
        mock.patch("builtins.print") as mock_print,
    ):
        assert logging_setup.get_file_handler() is None
    mock_print.assert_called_once()


def test_update_log_retention_logs_error_on_invalid_value():
    """Dynamic retention update should log errors for invalid values."""
    fake_handler = mock.MagicMock()
    with (
        mock.patch("modules.core.logging_setup.get_file_handler", return_value=fake_handler),
        mock.patch("modules.core.logging_setup.logger") as mock_logger,
    ):
        logging_setup.update_log_retention("not-an-int")
        mock_logger.exception.assert_called_once()


def test_openvino_device_and_probe_lines_error_paths():
    """OpenVINO diagnostics line builders should handle import and property errors."""
    get_openvino_devices_line = getattr(logging_setup, "_get_openvino_available_devices_line")
    get_openvino_probe_lines = getattr(logging_setup, "_get_openvino_target_probe_lines")

    with mock.patch("modules.core.logging_setup.importlib.import_module", side_effect=ImportError("missing")):
        line = get_openvino_devices_line()
        assert "unavailable" in line

    ov_module = mock.MagicMock()
    ov_core = mock.MagicMock()
    ov_core.get_property.side_effect = RuntimeError("bad-target")
    ov_module.Core.return_value = ov_core

    with (
        mock.patch.dict("os.environ", {"INTEL_DEEP_OV_PROBE": "true"}, clear=False),
        mock.patch("modules.core.logging_setup.importlib.import_module", return_value=ov_module),
    ):
        lines = get_openvino_probe_lines()
        assert lines
        assert "[OPENVINO TARGET PROBE]" in lines[0]
        assert any("unavailable" in line for line in lines[1:])


def test_log_banner_config_lines_without_optional_sections():
    """Banner rendering should skip optional sections cleanly when absent."""
    banner_config_lines = getattr(logging_setup, "_banner_config_lines")
    cfg = {
        "threads": "ASR=1 | Preprocess=1 | FFmpeg=1",
        "asr_display": "CPU",
        "prep_display": "CPU",
        "resource_pool": "CPU",
        "unique_props": [],
        "intel_env": [],
        "openvino_devices": "  OpenVINO devices               : <none>",
        "openvino_probe": [],
        "model_status": "OK",
        "cache_status": "OK",
    }
    lines = banner_config_lines(cfg)
    assert "  [DEVICE PROPERTIES]" not in lines
