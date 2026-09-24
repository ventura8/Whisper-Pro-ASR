"""Tests for the provisioning helpers in scripts/preload_model.py.

This runs at image build time, so its failure mode matters more than its success one: a
model that is absent, truncated or unconvertible has to stop the build with a clear error
rather than leave a directory the runtime engine cannot load. The OpenVINO control flow is
covered by test_preload_openvino.py; this covers the cache, download and integrity paths.
"""

import hashlib
import importlib.util
import runpy
import sys
from pathlib import Path
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_preload_module():
    """Import scripts/preload_model.py by path with its heavy imports stubbed.

    Same approach as test_preload_openvino.py: torch and audio_separator are imported at
    module scope, neither is needed here, and both are absent from a lean environment.
    """
    stubs = {
        "torch": mock.MagicMock(),
        "audio_separator": mock.MagicMock(),
        "audio_separator.separator": mock.MagicMock(),
    }
    spec = importlib.util.spec_from_file_location("_preload_helpers_under_test", REPO_ROOT / "scripts" / "preload_model.py")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


@pytest.fixture(name="preload")
def _preload():
    """A freshly imported provisioning module with no build cache configured."""
    module = _load_preload_module()
    module.CACHE_DIR = None
    return module


def _seed(directory, name="model.bin", payload=b"weights"):
    """Create a directory holding one file, and return it."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_bytes(payload)
    return directory


def test_cache_path_is_none_without_a_cache_dir(preload):
    """No --cache-dir means no build cache, not a path under None."""
    assert preload._cache_path("whisper") is None


def test_cache_path_joins_the_configured_cache(preload, tmp_path):
    """Each artifact gets its own subdirectory of the build cache."""
    preload.CACHE_DIR = str(tmp_path)

    assert preload._cache_path("whisper") == str(tmp_path / "whisper")


def test_replace_directory_overwrites_an_existing_target(preload, tmp_path):
    """Restoring must not merge into whatever was there before."""
    source = _seed(tmp_path / "src", name="new.bin")
    target = _seed(tmp_path / "dst", name="stale.bin")

    preload._replace_directory(str(source), str(target))

    assert [p.name for p in target.iterdir()] == ["new.bin"]


def test_cache_directory_is_a_no_op_without_a_cache(preload, tmp_path):
    """Seeding the build cache is optional; without one nothing is written."""
    source = _seed(tmp_path / "src")

    preload._cache_directory(str(source), "whisper")

    assert list(tmp_path.iterdir()) == [source]


def test_cache_directory_seeds_and_replaces(preload, tmp_path):
    """A second build overwrites the cached copy rather than merging with it."""
    preload.CACHE_DIR = str(tmp_path / "cache")
    source = _seed(tmp_path / "src", name="new.bin")
    _seed(tmp_path / "cache" / "whisper", name="stale.bin")

    preload._cache_directory(str(source), "whisper")

    assert [p.name for p in (tmp_path / "cache" / "whisper").iterdir()] == ["new.bin"]


def test_restore_returns_false_without_a_cache(preload, tmp_path):
    """Nothing to restore from is not an error."""
    assert preload._restore_directory_from_cache("whisper", str(tmp_path / "dst"), "Whisper") is False


def test_restore_returns_false_when_the_entry_is_absent(preload, tmp_path):
    """A cache that exists but holds nothing for this artifact."""
    preload.CACHE_DIR = str(tmp_path / "cache")
    (tmp_path / "cache").mkdir()

    assert preload._restore_directory_from_cache("whisper", str(tmp_path / "dst"), "Whisper") is False


def test_restore_rejects_a_cached_copy_that_fails_validation(preload, tmp_path):
    """A truncated cached model must not be restored over a rebuild.

    The cache survives across builds, so an invalid entry would otherwise be restored
    forever and every build would inherit the same broken model.
    """
    preload.CACHE_DIR = str(tmp_path / "cache")
    _seed(tmp_path / "cache" / "whisper")

    restored = preload._restore_directory_from_cache("whisper", str(tmp_path / "dst"), "Whisper", validator=lambda _d: False)

    assert restored is False
    assert not (tmp_path / "dst").exists()


def test_restore_copies_a_valid_cached_copy(preload, tmp_path):
    """The point of the build cache: skip the download entirely."""
    preload.CACHE_DIR = str(tmp_path / "cache")
    _seed(tmp_path / "cache" / "whisper", name="model.bin")

    restored = preload._restore_directory_from_cache("whisper", str(tmp_path / "dst"), "Whisper", validator=lambda _d: True)

    assert restored is True
    assert (tmp_path / "dst" / "model.bin").exists()


def test_run_subprocess_command_reports_success(preload):
    """A zero exit is success; the child's output is logged as it arrives."""
    process = mock.MagicMock()
    process.stdout = iter(["converting\n", "done\n"])
    process.returncode = 0

    with mock.patch.object(preload.subprocess, "Popen", return_value=process):
        assert preload._run_subprocess_command(["optimum-cli"]) is True

    process.wait.assert_called_once()


def test_run_subprocess_command_reports_failure(preload):
    """A non-zero exit must not read as success, or the build ships an unconverted model."""
    process = mock.MagicMock()
    process.stdout = iter([])
    process.returncode = 2

    with mock.patch.object(preload.subprocess, "Popen", return_value=process):
        assert preload._run_subprocess_command(["optimum-cli"]) is False


def test_run_subprocess_command_waits_even_when_reading_fails(preload):
    """Without the wait the child is left running and its exit code never collected."""
    process = mock.MagicMock()
    process.stdout = mock.MagicMock()
    process.stdout.__iter__ = mock.Mock(side_effect=OSError("stream closed"))
    process.returncode = 0

    with mock.patch.object(preload.subprocess, "Popen", return_value=process):
        with pytest.raises(OSError):
            preload._run_subprocess_command(["optimum-cli"])

    process.wait.assert_called_once()


def test_export_skips_when_optimum_is_absent(preload):
    """optimum-cli is not in the runtime image, so this path is normally skipped."""
    with mock.patch.object(preload.shutil, "which", return_value=None):
        assert preload._export_openvino_whisper() is False


def test_export_requires_the_result_to_validate(preload):
    """A command that exits zero but produces an unloadable IR is still a failure."""
    with mock.patch.object(preload.shutil, "which", return_value="/usr/bin/optimum-cli"):
        with mock.patch.object(preload, "_run_subprocess_command", return_value=True):
            with mock.patch.object(preload, "verify_ov_model", return_value=False):
                assert preload._export_openvino_whisper() is False


def test_export_seeds_the_cache_on_success(preload):
    """A successful conversion is expensive; the next build should not repeat it."""
    with mock.patch.object(preload.shutil, "which", return_value="/usr/bin/optimum-cli"):
        with mock.patch.object(preload, "_run_subprocess_command", return_value=True):
            with mock.patch.object(preload, "verify_ov_model", return_value=True):
                with mock.patch.object(preload, "_cache_directory") as cache:
                    assert preload._export_openvino_whisper() is True

    cache.assert_called_once()


def test_export_converts_to_fp16_for_automatic_speech_recognition(preload):
    """The export flags are the contract with the Intel engine."""
    with mock.patch.object(preload.shutil, "which", return_value="/usr/bin/optimum-cli"):
        with mock.patch.object(preload, "_run_subprocess_command", return_value=False) as run:
            preload._export_openvino_whisper()

    cmd = run.call_args[0][0]
    assert cmd[:3] == ["optimum-cli", "export", "openvino"]
    assert cmd[cmd.index("--weight-format") + 1] == "fp16"
    assert cmd[cmd.index("--task") + 1] == "automatic-speech-recognition"


def test_export_survives_an_unexpected_exception(preload):
    """A conversion crash falls through to the download path rather than aborting."""
    with mock.patch.object(preload.shutil, "which", return_value="/usr/bin/optimum-cli"):
        with mock.patch.object(preload, "_run_subprocess_command", side_effect=RuntimeError("boom")):
            assert preload._export_openvino_whisper() is False


def test_ensure_ct2_skips_a_model_that_is_already_valid(preload):
    """The common case on a rebuild."""
    with mock.patch.object(preload.model_integrity, "verify_ct2_model_dir", return_value=True):
        with mock.patch.object(preload, "_download_ct2_whisper") as download:
            preload._ensure_ct2_whisper()

    download.assert_not_called()


def test_ensure_ct2_restores_from_cache_before_downloading(preload):
    """The cache exists to avoid the download."""
    with mock.patch.object(preload.model_integrity, "verify_ct2_model_dir", return_value=False):
        with mock.patch.object(preload, "_restore_directory_from_cache", return_value=True):
            with mock.patch.object(preload, "_download_ct2_whisper") as download:
                preload._ensure_ct2_whisper()

    download.assert_not_called()


def test_ensure_ct2_exits_when_the_download_fails(preload):
    """Provisioning runs at build time, where a clear failure is cheap."""
    with mock.patch.object(preload.model_integrity, "verify_ct2_model_dir", return_value=False):
        with mock.patch.object(preload, "_restore_directory_from_cache", return_value=False):
            with mock.patch.object(preload, "_download_ct2_whisper", return_value=False):
                with pytest.raises(SystemExit) as exit_info:
                    preload._ensure_ct2_whisper()

    assert exit_info.value.code == 1


def test_download_ct2_seeds_the_cache_only_on_success(preload):
    """A failed download must not poison the cache with a partial model."""
    with mock.patch.object(preload.model_provisioning, "ensure_ct2_whisper", return_value=False):
        with mock.patch.object(preload, "_cache_directory") as cache:
            assert preload._download_ct2_whisper() is False

    cache.assert_not_called()


def test_ensure_uvr_delegates_to_shared_provisioning(preload):
    """The download policy lives in model_provisioning, which the runtime path also uses.

    This module reimplemented the fallback, checksum and retry, so a policy change had to
    be made twice and the preloader silently kept the older behaviour when it was not.
    """
    with mock.patch.object(preload.model_integrity, "verify_onnx_model_file", return_value=False):
        with mock.patch.object(preload, "_restore_directory_from_cache", return_value=False):
            with mock.patch.object(preload.model_provisioning, "ensure_uvr_model", return_value=True) as ensure:
                with mock.patch.object(preload, "_cache_directory"):
                    preload._ensure_uvr_model()

    assert ensure.call_args[0][1] == preload.config.PERSISTENT_TEMP_DIR


def test_ensure_uvr_exits_when_provisioning_fails(preload):
    """Same build-time contract as the other models."""
    with mock.patch.object(preload.model_integrity, "verify_onnx_model_file", return_value=False):
        with mock.patch.object(preload, "_restore_directory_from_cache", return_value=False):
            with mock.patch.object(preload.model_provisioning, "ensure_uvr_model", return_value=False):
                with pytest.raises(SystemExit):
                    preload._ensure_uvr_model()


def test_ensure_uvr_skips_a_valid_model(preload):
    """Already provisioned and checksum-clean."""
    with mock.patch.object(preload.model_integrity, "verify_onnx_model_file", return_value=True):
        with mock.patch.object(preload.model_provisioning, "ensure_uvr_model") as ensure:
            preload._ensure_uvr_model()

    ensure.assert_not_called()


def test_ensure_vad_exits_when_every_retry_fails(preload):
    """A missing VAD model would otherwise surface as silent mis-segmentation at runtime."""
    with mock.patch.object(preload.model_integrity, "verify_onnx_model_file", return_value=False):
        with mock.patch.object(preload, "_restore_directory_from_cache", return_value=False):
            with mock.patch.object(preload.model_integrity, "download_with_integrity_retry", return_value=False):
                with pytest.raises(SystemExit):
                    preload._ensure_vad_model()


def test_ensure_vad_seeds_the_cache_on_success(preload):
    """The VAD model is small but fetched from GitHub; caching avoids the round trip."""
    with mock.patch.object(preload.model_integrity, "verify_onnx_model_file", return_value=False):
        with mock.patch.object(preload, "_restore_directory_from_cache", return_value=False):
            with mock.patch.object(preload.model_integrity, "download_with_integrity_retry", return_value=True):
                with mock.patch.object(preload, "_cache_directory") as cache:
                    preload._ensure_vad_model()

    cache.assert_called_once()


def _vad_response(chunks):
    """A requests response context manager yielding `chunks`."""
    response = mock.MagicMock()
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    response.iter_content.return_value = iter(chunks)
    return response


def test_silero_download_writes_the_model_when_the_checksum_matches(preload, tmp_path, monkeypatch):
    """The happy path, with the digest pinned to the tag the constant documents."""
    payload = b"onnx-bytes"

    monkeypatch.setattr(preload, "SILERO_VAD_MODEL_SHA256", hashlib.sha256(payload).hexdigest())
    target = tmp_path / "vad" / "silero_vad.onnx"
    requests_stub = mock.MagicMock()
    requests_stub.get.return_value = _vad_response([payload])

    with mock.patch.dict(sys.modules, {"requests": requests_stub}):
        preload._download_silero_vad_direct(str(target))

    assert target.read_bytes() == payload


def test_silero_download_rejects_a_mismatched_checksum(preload, tmp_path):
    """A moved artifact must fail loudly rather than install unknown bytes.

    The URL pins a tag rather than master precisely so this cannot happen silently.
    """
    target = tmp_path / "vad" / "silero_vad.onnx"
    requests_stub = mock.MagicMock()
    requests_stub.get.return_value = _vad_response([b"wrong-bytes"])

    with mock.patch.dict(sys.modules, {"requests": requests_stub}):
        with pytest.raises(RuntimeError, match="checksum mismatch"):
            preload._download_silero_vad_direct(str(target))

    assert not target.exists()


def test_silero_download_leaves_no_temporary_behind(preload, tmp_path):
    """The temp file sits in the target directory, so one left behind ships in the image."""
    target = tmp_path / "vad" / "silero_vad.onnx"
    requests_stub = mock.MagicMock()
    requests_stub.get.return_value = _vad_response([b"wrong-bytes"])

    with mock.patch.dict(sys.modules, {"requests": requests_stub}):
        with pytest.raises(RuntimeError):
            preload._download_silero_vad_direct(str(target))

    assert [p.name for p in (tmp_path / "vad").iterdir()] == []


def test_silero_download_skips_empty_chunks(preload, tmp_path, monkeypatch):
    """Keep-alive chunks arrive empty and must not enter the digest."""
    payload = b"onnx-bytes"

    monkeypatch.setattr(preload, "SILERO_VAD_MODEL_SHA256", hashlib.sha256(payload).hexdigest())
    target = tmp_path / "vad" / "silero_vad.onnx"
    requests_stub = mock.MagicMock()
    requests_stub.get.return_value = _vad_response([b"onnx-", b"", b"bytes"])

    with mock.patch.dict(sys.modules, {"requests": requests_stub}):
        preload._download_silero_vad_direct(str(target))

    assert target.read_bytes() == payload


def test_download_openvino_genai_reports_a_failed_validation(preload):
    """A download that lands but does not validate is not success."""
    hub = mock.MagicMock()
    with mock.patch.dict(sys.modules, {"huggingface_hub": hub}):
        with mock.patch.object(preload, "verify_ov_model", return_value=False):
            assert preload._download_openvino_genai() is False


def test_download_openvino_genai_seeds_the_cache_on_success(preload):
    """The IR is ~22GB smaller than raw weights but still worth caching."""
    hub = mock.MagicMock()
    with mock.patch.dict(sys.modules, {"huggingface_hub": hub}):
        with mock.patch.object(preload, "verify_ov_model", return_value=True):
            with mock.patch.object(preload, "_cache_directory") as cache:
                assert preload._download_openvino_genai() is True

    cache.assert_called_once()


def test_download_openvino_genai_reports_an_exception_as_failure(preload):
    """A network error must return False so the caller can report the real problem."""
    hub = mock.MagicMock()
    hub.snapshot_download.side_effect = RuntimeError("connection reset")

    with mock.patch.dict(sys.modules, {"huggingface_hub": hub}):
        assert preload._download_openvino_genai() is False


def test_openvino_already_available_falls_back_to_the_cache(preload):
    """Present-and-valid, or restorable -- either skips the download."""
    with mock.patch.object(preload, "verify_ov_model", return_value=False):
        with mock.patch.object(preload, "_restore_directory_from_cache", return_value=True) as restore:
            assert preload._openvino_model_already_available() is True

    restore.assert_called_once()


def test_preload_uvr_and_vad_delegate(preload):
    """The numbered stages are thin wrappers; the work is in the _ensure_ helpers."""
    with mock.patch.object(preload, "_ensure_uvr_model") as uvr:
        preload.preload_uvr()
    with mock.patch.object(preload, "_ensure_vad_model") as vad:
        preload.preload_vad()

    uvr.assert_called_once()
    vad.assert_called_once()


def test_download_ct2_seeds_the_cache_on_success(preload):
    """A successful provision is cached so the next build skips the download."""
    with mock.patch.object(preload.model_provisioning, "ensure_ct2_whisper", return_value=True):
        with mock.patch.object(preload, "_cache_directory") as cache:
            assert preload._download_ct2_whisper() is True

    cache.assert_called_once()


def test_ensure_uvr_restores_from_cache_before_provisioning(preload):
    """The cached UVR model is validated by the same checksum the download uses."""
    with mock.patch.object(preload.model_integrity, "verify_onnx_model_file", return_value=False):
        with mock.patch.object(preload, "_restore_directory_from_cache", return_value=True):
            with mock.patch.object(preload.model_provisioning, "ensure_uvr_model") as ensure:
                preload._ensure_uvr_model()

    ensure.assert_not_called()


def test_ensure_vad_skips_a_valid_model(preload):
    """Already provisioned and matching the pinned digest."""
    with mock.patch.object(preload.model_integrity, "verify_onnx_model_file", return_value=True):
        with mock.patch.object(preload.model_integrity, "download_with_integrity_retry") as download:
            preload._ensure_vad_model()

    download.assert_not_called()


def test_ensure_vad_restores_from_cache_before_downloading(preload):
    """Same ordering as the other artifacts."""
    with mock.patch.object(preload.model_integrity, "verify_onnx_model_file", return_value=False):
        with mock.patch.object(preload, "_restore_directory_from_cache", return_value=True):
            with mock.patch.object(preload.model_integrity, "download_with_integrity_retry") as download:
                preload._ensure_vad_model()

    download.assert_not_called()


def test_the_vad_retry_downloads_through_the_direct_fetch(preload):
    """download_with_integrity_retry drives the direct fetch, which pins the checksum."""
    captured = {}

    def _capture(download_fn, **kwargs):
        captured["fn"] = download_fn
        return True

    with mock.patch.object(preload.model_integrity, "verify_onnx_model_file", return_value=False):
        with mock.patch.object(preload, "_restore_directory_from_cache", return_value=False):
            with mock.patch.object(preload.model_integrity, "download_with_integrity_retry", side_effect=_capture):
                with mock.patch.object(preload, "_cache_directory"):
                    preload._ensure_vad_model()

    with mock.patch.object(preload, "_download_silero_vad_direct") as direct:
        captured["fn"]()

    direct.assert_called_once()


def test_verify_ov_model_delegates_to_shared_integrity(preload):
    """One definition of "a loadable OpenVINO IR", shared with the runtime path."""
    with mock.patch.object(preload.model_integrity, "verify_openvino_model_dir", return_value=True) as verify:
        assert preload.verify_ov_model("/app/system_models/whisper-openvino") is True

    verify.assert_called_once_with("/app/system_models/whisper-openvino")


def test_openvino_already_available_when_the_ir_is_valid(preload):
    """Present and loadable: nothing to download."""
    with mock.patch.object(preload, "verify_ov_model", return_value=True):
        with mock.patch.object(preload, "_restore_directory_from_cache") as restore:
            assert preload._openvino_model_already_available() is True

    restore.assert_not_called()


def test_running_as_a_script_provisions_every_model(tmp_path):
    """The __main__ block wires the flags to the three stages.

    Executed through runpy so the argument parsing and the CACHE_DIR defaulting are
    covered as they actually run. The shared provisioning modules are stubbed, which makes
    every "already valid" check truthy, so each stage short-circuits immediately.
    """
    # __version__ is set explicitly: MagicMock does not synthesize dunder attributes, and
    # the banner logs the torch version before anything else runs.
    torch_stub = mock.MagicMock()
    torch_stub.__version__ = "2.13.0"
    stubs = {
        "torch": torch_stub,
        "audio_separator": mock.MagicMock(),
        "audio_separator.separator": mock.MagicMock(),
        "modules.core": mock.MagicMock(),
    }
    script = REPO_ROOT / "scripts" / "preload_model.py"

    with mock.patch.dict(sys.modules, stubs):
        with mock.patch.object(sys, "argv", ["preload_model.py", "--cache-dir", str(tmp_path), "--skip-intel-whisper"]):
            namespace = runpy.run_path(str(script), run_name="__main__")

    assert namespace["CACHE_DIR"] == str(tmp_path)
    assert namespace["SKIP_INTEL_WHISPER"] is True
