"""Tests for Intel runtime bootstrap diagnostics."""

from types import SimpleNamespace
from unittest import mock

from modules.core import bootstrap


def _fake_core():
    """Return a fake OpenVINO Core object."""
    return SimpleNamespace(available_devices=["CPU", "GPU"])


def _fake_providers():
    """Return fake ONNX Runtime providers."""
    return ["OpenVINOExecutionProvider", "CPUExecutionProvider"]


_FAKE_OPENVINO = SimpleNamespace(
    __version__="2026.2.1",
    __file__="/opt/intel/openvino/python/openvino/__init__.py",
    Core=_fake_core,
)
_FAKE_ONNXRUNTIME = SimpleNamespace(
    __file__="/app/libs/intel/onnxruntime/__init__.py",
    get_available_providers=_fake_providers,
)


def _fake_import_module(name: str):
    """Return fake runtime modules by import name."""
    if name == "onnxruntime":
        return _FAKE_ONNXRUNTIME
    if name == "openvino":
        return _FAKE_OPENVINO
    raise ImportError(name)


def test_log_intel_runtime_diagnostics_reports_runtime_and_devices():
    """Intel diagnostics should log provider, device, and node visibility details."""
    logger = mock.MagicMock()
    log_diagnostics = getattr(bootstrap, "_log_intel_runtime_diagnostics")

    with (
        mock.patch.object(bootstrap.importlib, "import_module", side_effect=_fake_import_module),
        mock.patch.object(bootstrap.os, "getenv", return_value="/opt/intel/openvino"),
        mock.patch.object(bootstrap.os.path, "exists", side_effect=lambda p: p in {"/dev/dri", "/opt/intel/openvino"}),
    ):
        log_diagnostics(logger)

    debug_messages = [call.args[0] for call in logger.debug.call_args_list]
    assert "Intel ONNX Runtime diagnostics: file=%s providers=%s" in debug_messages
    assert "OpenVINO diagnostics: version=%s file=%s devices=%s INTEL_OPENVINO_DIR=%s" in debug_messages
    assert "Intel device nodes: /dev/accel/accel0=%s /dev/dri=%s /opt/intel/openvino=%s" in debug_messages


def test_safe_ort_providers_returns_empty_when_provider_query_fails():
    """Provider diagnostics should tolerate provider query failures."""
    ort = SimpleNamespace(get_available_providers=mock.Mock(side_effect=RuntimeError("provider query failed")))
    safe_providers = getattr(bootstrap, "_safe_ort_providers")

    assert not safe_providers(ort)


def test_detect_nvidia_hardware_ignores_ctranslate2_without_runtime_signal():
    """AUTO bootstrap should not treat bundled CUDA libs as proof of NVIDIA hardware on Intel-only hosts."""
    detect_nvidia_hardware = getattr(bootstrap, "_detect_nvidia_hardware")
    with mock.patch.object(bootstrap.os.path, "exists", return_value=False):
        assert detect_nvidia_hardware() is False


def test_detect_nvidia_hardware_uses_device_nodes_when_present():
    """AUTO bootstrap should detect NVIDIA only from real device nodes."""
    detect_nvidia_hardware = getattr(bootstrap, "_detect_nvidia_hardware")
    with mock.patch.object(bootstrap.os.path, "exists", side_effect=lambda path: path == "/dev/nvidia0"):
        assert detect_nvidia_hardware() is True


def test_resolve_target_library_prefers_intel_for_auto_preprocess_when_intel_available():
    """Bootstrap should prefer Intel ONNX path when AUTO preprocessing can use Intel hardware."""
    resolve_target_library = getattr(bootstrap, "_resolve_target_library")
    with mock.patch.object(bootstrap.os.path, "exists", side_effect=lambda path: path in {"/app/libs/intel", "/app/libs/nvidia"}):
        target, reason = resolve_target_library("auto", "auto", False, True, False)

    assert target == "/app/libs/intel"
    assert reason == "Intel OpenVINO"


def test_has_intel_drm_vendor_handles_oserror_and_finds_intel():
    """DRM vendor probe should skip unreadable entries and detect Intel vendor id."""

    bad_vendor = mock.MagicMock()
    bad_vendor.read_text.side_effect = OSError("unreadable")
    intel_vendor = mock.MagicMock()
    intel_vendor.read_text.return_value = "0x8086\n"

    with mock.patch.object(bootstrap.Path, "glob", return_value=[bad_vendor, intel_vendor]):
        assert getattr(bootstrap, "_has_intel_drm_vendor")() is True


def test_detect_intel_linux_nodes_uses_vendor_probe_when_nodes_missing():
    """Linux Intel node fallback should use DRM vendor probe when device nodes do not exist."""
    with (
        mock.patch.object(bootstrap.os.path, "exists", return_value=False),
        mock.patch.object(bootstrap, "_has_intel_drm_vendor", return_value=True),
    ):
        assert getattr(bootstrap, "_detect_intel_linux_nodes")() is True


def test_parse_proc_status_line_handles_invalid_and_valid_rows():
    """Proc status parser should ignore malformed lines and parse key/value lines."""
    parse_line = getattr(bootstrap, "_parse_proc_status_line")
    assert parse_line("Malformed") == ("", "")
    assert parse_line("CapEff:\t00000000") == ("CapEff", "00000000")


def test_read_process_security_status_extracts_expected_keys():
    """Process security parser should extract known keys and keep defaults for absent keys."""
    with mock.patch.object(
        bootstrap,
        "_iter_proc_status_entries",
        return_value=[("CapEff", "ffff"), ("Seccomp", "2"), ("Other", "x")],
    ):
        assert getattr(bootstrap, "_read_process_security_status")() == ("ffff", "2", "unknown")


def test_optional_openvino_probe_toggle_paths():
    """Optional target probe should no-op when disabled and invoke probe when enabled."""
    logger = mock.MagicMock()
    optional_probe = getattr(bootstrap, "_log_optional_openvino_target_probe")

    with (
        mock.patch.object(bootstrap.os, "getenv", return_value="false"),
        mock.patch.object(bootstrap, "_log_openvino_target_probe") as mock_probe,
    ):
        optional_probe(logger)
        mock_probe.assert_not_called()
        logger.debug.assert_called_once()

    logger.reset_mock()
    with (
        mock.patch.object(bootstrap.os, "getenv", return_value="true"),
        mock.patch.object(bootstrap, "_log_openvino_target_probe") as mock_probe,
    ):
        optional_probe(logger)
        mock_probe.assert_called_once_with(logger)


def test_device_open_probe_reports_missing_and_open_failure():
    """Device open probe should return explicit status for missing and failed-open nodes."""
    probe = getattr(bootstrap, "_device_open_probe")

    with mock.patch.object(bootstrap.os.path, "exists", return_value=False):
        assert probe("/dev/dri/renderD128") == "missing"

    with (
        mock.patch.object(bootstrap.os.path, "exists", return_value=True),
        mock.patch.object(bootstrap.os, "open", side_effect=OSError(13, "denied")),
        mock.patch.object(bootstrap.os, "O_CLOEXEC", 0, create=True),
    ):
        assert "open_failed" in probe("/dev/dri/renderD128")


def test_log_intel_access_diagnostics_warns_on_non_root_without_device_access():
    """Access diagnostics should warn when non-root process cannot open Intel devices."""
    logger = mock.MagicMock()
    access_diag = getattr(bootstrap, "_log_intel_access_diagnostics")

    with (
        mock.patch.object(bootstrap, "_device_open_probe", return_value="missing"),
        mock.patch.object(bootstrap.os, "getuid", return_value=1000, create=True),
        mock.patch.object(bootstrap.os, "getgroups", return_value=[1000, 44], create=True),
    ):
        access_diag(logger)

    logger.warning.assert_called_once()


def test_activate_target_library_clears_cached_onnxruntime_before_reload():
    """Activation should evict cached onnxruntime module before verification import."""
    activate = getattr(bootstrap, "_activate_target_library")
    logger = mock.MagicMock()
    with (
        mock.patch.object(bootstrap.os.path, "exists", return_value=True),
        mock.patch.object(bootstrap.importlib, "invalidate_caches"),
        mock.patch.object(bootstrap, "_log_onnxruntime_load"),
        mock.patch.dict(bootstrap.sys.modules, {"onnxruntime": object()}, clear=False),
    ):
        activate(logger, "/app/libs/intel", "Default")
        assert "onnxruntime" not in bootstrap.sys.modules


def test_log_openvino_target_probe_success_and_target_level_errors():
    """OpenVINO target probe should report per-target failures while keeping overall probe successful."""
    logger = mock.MagicMock()
    core = mock.MagicMock()

    def _prop_side_effect(target: str, _prop: str):
        if target == "NPU":
            raise RuntimeError("npu unavailable")
        return f"name-{target}"

    core.get_property.side_effect = _prop_side_effect
    ov = mock.MagicMock()
    ov.Core.return_value = core

    with mock.patch.object(bootstrap.importlib, "import_module", return_value=ov):
        getattr(bootstrap, "_log_openvino_target_probe")(logger)

    logger.info.assert_called_once()
    assert "OpenVINO target probe" in logger.info.call_args[0][0]


def test_log_openvino_target_probe_handles_import_failure():
    """OpenVINO target probe should log a warning when runtime import fails."""
    logger = mock.MagicMock()
    with mock.patch.object(bootstrap.importlib, "import_module", side_effect=ImportError("missing")):
        getattr(bootstrap, "_log_openvino_target_probe")(logger)
    logger.warning.assert_called_once()


def test_get_process_identity_returns_unknown_on_os_without_uid_api():
    """Process identity helper should return unknown tuple when uid/gid APIs are unavailable."""
    with mock.patch.object(bootstrap.os, "getuid", side_effect=AttributeError("no uid"), create=True):
        assert getattr(bootstrap, "_get_process_identity")() == (-1, -1, [])


def test_iter_proc_status_entries_returns_empty_on_read_failure():
    """Proc status iterator should return empty list when status file cannot be read."""
    with mock.patch("builtins.open", side_effect=OSError("denied")):
        assert not getattr(bootstrap, "_iter_proc_status_entries")()


def test_log_intel_node_details_handles_stat_failure_and_success_paths():
    """Node details logger should include both stat errors and successful stat metadata."""
    logger = mock.MagicMock()

    ok_node = mock.MagicMock()
    ok_node.__str__.return_value = "/dev/dri/renderD128"
    ok_stat = mock.MagicMock(st_mode=0o100660, st_uid=1000, st_gid=44)
    ok_node.stat.return_value = ok_stat

    bad_node = mock.MagicMock()
    bad_node.__str__.return_value = "/dev/accel/accel0"
    bad_node.stat.side_effect = OSError("broken")

    with (
        mock.patch.object(bootstrap, "glob", side_effect=[[str(ok_node)], [str(bad_node)]]),
        mock.patch.object(bootstrap, "Path", side_effect=lambda p: ok_node if p == str(ok_node) else bad_node),
    ):
        getattr(bootstrap, "_log_intel_node_details")(logger)

    assert logger.debug.call_count >= 2


def test_read_sysfs_file_returns_na_on_oserror():
    """Sysfs reader should return n/a for unreadable files."""
    path = mock.MagicMock()
    path.read_text.side_effect = OSError("denied")
    assert getattr(bootstrap, "_read_sysfs_file")(path) == "n/a"


def test_log_sysfs_class_nodes_resolve_failure_and_none_branch():
    """Sysfs logger should handle unresolved driver links and empty class matches."""
    logger = mock.MagicMock()

    device_path = mock.MagicMock()
    driver_link = mock.MagicMock()
    device_path.__truediv__.side_effect = [driver_link, mock.MagicMock(), mock.MagicMock()]
    device_path.__str__.return_value = "/sys/class/drm/renderD128/device"
    driver_link.exists.return_value = True
    driver_link.resolve.side_effect = OSError("unresolved")

    with (
        mock.patch.object(bootstrap, "glob", side_effect=[["/sys/class/drm/renderD128/device"], []]),
        mock.patch.object(bootstrap, "Path", return_value=device_path),
        mock.patch.object(bootstrap, "_read_sysfs_file", return_value="0x8086"),
    ):
        log_sysfs = getattr(bootstrap, "_log_sysfs_class_nodes")
        log_sysfs(logger, "drm-render", "/sys/class/drm/renderD*/device")
        log_sysfs(logger, "accel", "/sys/class/accel/accel*/device")

    assert logger.debug.call_count >= 2


def test_device_open_probe_success_closes_fd_and_access_diag_handles_uid_failure():
    """Device open probe should close descriptors on success; access diagnostics should tolerate uid lookup failures."""
    probe = getattr(bootstrap, "_device_open_probe")
    logger = mock.MagicMock()

    with (
        mock.patch.object(bootstrap.os.path, "exists", return_value=True),
        mock.patch.object(bootstrap.os, "open", return_value=42),
        mock.patch.object(bootstrap.os, "close") as mock_close,
        mock.patch.object(bootstrap.os, "O_CLOEXEC", 0, create=True),
    ):
        assert probe("/dev/dri/renderD128") == "open_ok"
        mock_close.assert_called_once_with(42)

    with (
        mock.patch.object(bootstrap, "_device_open_probe", return_value="open_ok"),
        mock.patch.object(bootstrap.os, "getuid", side_effect=AttributeError("missing"), create=True),
    ):
        getattr(bootstrap, "_log_intel_access_diagnostics")(logger)
    logger.warning.assert_not_called()


def test_detect_amd_hardware_wsl_without_driver_returns_false():
    """_detect_amd_hardware returns False when /dev/dxg exists without AMD WSL driver."""
    detect_amd = getattr(bootstrap, "_detect_amd_hardware")
    logger = mock.MagicMock()
    with (
        mock.patch.object(bootstrap.os.path, "exists", side_effect=lambda p: p == "/dev/dxg"),
        mock.patch.object(bootstrap, "_has_amd_drm_vendor", return_value=False),
        mock.patch("modules.core.config_helpers._is_amd_wsl_driver_present", return_value=False),
    ):
        assert detect_amd(logger) is False
