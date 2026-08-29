"""Tests for runtime model provisioning (models are downloaded, not baked)."""

import ast
import hashlib
import inspect
import os
import pathlib
from types import SimpleNamespace
from unittest import mock

import pytest

from modules.core import model_provisioning


@pytest.fixture(autouse=True)
def reset_provisioning_state():
    """Provisioning state is module-global; reset it around every test."""
    model_provisioning.MODEL_READY.clear()
    model_provisioning.PROVISION_COMPLETE.clear()
    model_provisioning.PROVISION_STARTED.clear()
    yield
    model_provisioning.MODEL_READY.clear()
    model_provisioning.PROVISION_COMPLETE.clear()
    model_provisioning.PROVISION_STARTED.clear()


def _config(**overrides):
    """Build a config double with the given overrides."""
    base = {
        "ASR_ENGINE": "FASTER-WHISPER",
        "MODEL_ID": "/app/model_cache/whisper",
        "OV_MODEL_PATH": "/app/model_cache/whisper-openvino",
        "UVR_MODEL_DIR": "/app/model_cache/preprocessing_models",
        "PERSISTENT_TEMP_DIR": "/tmp/whisper",
        "ENABLE_VOCAL_SEPARATION": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_gate_is_inert_until_provisioning_starts():
    """Tests and embedded usage never start provisioning; the queue gate must stay open."""
    assert model_provisioning.should_gate_tasks() is False


def test_gate_closes_while_downloading_and_opens_when_ready():
    """The scheduler gate holds tasks only between start and completion."""
    model_provisioning.PROVISION_STARTED.set()
    assert model_provisioning.should_gate_tasks() is True

    model_provisioning.MODEL_READY.set()
    assert model_provisioning.should_gate_tasks() is False


def test_provision_models_marks_ready_on_success():
    """A successful pass sets the ready event and reports 100%."""
    with mock.patch.object(model_provisioning, "ensure_ct2_whisper", return_value=True) as ensure:
        assert model_provisioning.provision_models(_config()) is True

    ensure.assert_called_once_with("/app/model_cache/whisper")
    assert model_provisioning.is_ready() is True
    assert model_provisioning.get_progress()["percent"] == 100


def test_provision_models_opens_gate_on_failure():
    """A failed download must not queue tasks forever; the gate opens and the error sticks."""
    model_provisioning.PROVISION_STARTED.set()
    with mock.patch.object(model_provisioning, "ensure_ct2_whisper", return_value=False):
        assert model_provisioning.provision_models(_config()) is False

    assert model_provisioning.should_gate_tasks() is False
    assert model_provisioning.get_progress()["error"] is not None


def test_provision_models_opens_gate_when_download_raises():
    """An exception from a download is contained and still releases the gate."""
    model_provisioning.PROVISION_STARTED.set()
    with mock.patch.object(model_provisioning, "ensure_ct2_whisper", side_effect=OSError("network down")):
        assert model_provisioning.provision_models(_config()) is False

    assert model_provisioning.should_gate_tasks() is False
    error = model_provisioning.get_progress()["error"]
    assert error is not None and "network down" in str(error)


def test_a_failure_is_not_reported_as_ready_and_can_be_retried():
    """The gate opening and the models being present are different facts.

    They used to share one event, so a failed download reported is_ready() True -- and made
    every later provision_models() call return early, so a retry after the network came back
    did nothing at all while the service went on claiming the models were ready.
    """
    model_provisioning.PROVISION_STARTED.set()
    with mock.patch.object(model_provisioning, "ensure_ct2_whisper", return_value=False):
        assert model_provisioning.provision_models(_config()) is False

    assert model_provisioning.should_gate_tasks() is False, "a dead download must not queue tasks forever"
    assert model_provisioning.is_ready() is False, "a failed provision has not produced any model"

    with mock.patch.object(model_provisioning, "ensure_ct2_whisper", return_value=True) as retry:
        assert model_provisioning.provision_models(_config()) is True

    retry.assert_called_once()
    assert model_provisioning.is_ready() is True


def test_provision_models_is_single_flight():
    """A second call while already provisioned must not re-download."""
    model_provisioning.MODEL_READY.set()
    with mock.patch.object(model_provisioning, "ensure_ct2_whisper") as ensure:
        assert model_provisioning.provision_models(_config()) is True

    ensure.assert_not_called()


def test_intel_engine_provisions_the_openvino_ir():
    """INTEL-WHISPER needs the OpenVINO IR rather than the CTranslate2 weights."""
    with (
        mock.patch.object(model_provisioning, "ensure_openvino_whisper", return_value=True) as ensure_ov,
        mock.patch.object(model_provisioning, "ensure_ct2_whisper") as ensure_ct2,
    ):
        assert model_provisioning.provision_models(_config(ASR_ENGINE="INTEL-WHISPER")) is True

    ensure_ov.assert_called_once_with("/app/model_cache/whisper-openvino")
    ensure_ct2.assert_not_called()


def test_hybrid_host_provisions_both_weight_formats():
    """A hybrid host runs both engines, so half its pool would fail with only one model."""
    config = _config(
        ASR_ENGINE="FASTER-WHISPER",
        HYBRID_ENGINES=True,
        MODEL_ID_BY_ENGINE={
            "FASTER-WHISPER": "/app/model_cache/whisper",
            "INTEL-WHISPER": "/app/model_cache/whisper-openvino",
        },
        engines_in_use=lambda: ["FASTER-WHISPER", "INTEL-WHISPER"],
    )

    with (
        mock.patch.object(model_provisioning, "ensure_openvino_whisper", return_value=True) as ensure_ov,
        mock.patch.object(model_provisioning, "ensure_ct2_whisper", return_value=True) as ensure_ct2,
    ):
        assert model_provisioning.provision_models(config) is True

    ensure_ct2.assert_called_once_with("/app/model_cache/whisper")
    ensure_ov.assert_called_once_with("/app/model_cache/whisper-openvino")


def test_single_engine_host_provisions_only_its_own_weights():
    """Non-hybrid deployments must not pay to download a model they will never load."""
    config = _config(ASR_ENGINE="FASTER-WHISPER", engines_in_use=lambda: ["FASTER-WHISPER"])

    with (
        mock.patch.object(model_provisioning, "ensure_openvino_whisper") as ensure_ov,
        mock.patch.object(model_provisioning, "ensure_ct2_whisper", return_value=True) as ensure_ct2,
    ):
        assert model_provisioning.provision_models(config) is True

    ensure_ct2.assert_called_once_with("/app/model_cache/whisper")
    ensure_ov.assert_not_called()


def test_vocal_separation_adds_the_uvr_asset():
    """UVR is provisioned only when vocal separation is enabled."""
    with (
        mock.patch.object(model_provisioning, "ensure_ct2_whisper", return_value=True),
        mock.patch.object(model_provisioning, "ensure_uvr_model", return_value=True) as ensure_uvr,
    ):
        assert model_provisioning.provision_models(_config(ENABLE_VOCAL_SEPARATION=True)) is True

    ensure_uvr.assert_called_once_with("/app/model_cache/preprocessing_models", "/tmp/whisper")


def test_start_background_provisioning_sets_started_flag():
    """Launching provisioning arms the scheduler gate."""
    with mock.patch.object(model_provisioning, "provision_models", return_value=True):
        thread = model_provisioning.start_background_provisioning(_config())
        # Asserted, not tested: a conditional join would silently skip synchronising and
        # leave the flag assertion below racing the thread that sets it.
        assert thread is not None, "a cold start must spawn a provisioning thread"
        thread.join(timeout=5)
        assert not thread.is_alive(), "provisioning thread did not finish within 5s"

    assert model_provisioning.PROVISION_STARTED.is_set() is True


def test_start_background_provisioning_skips_when_already_ready():
    """A warm start must not spawn a provisioning thread at all."""
    model_provisioning.MODEL_READY.set()
    assert model_provisioning.start_background_provisioning(_config()) is None


def _is_snapshot_download_call(node: ast.AST) -> bool:
    """Return whether ``node`` is a ``<something>.snapshot_download(...)`` call."""
    return isinstance(node, ast.Call) and isinstance(getattr(node, "func", None), ast.Attribute) and node.func.attr == "snapshot_download"


class TestOpenVinoDownloadCallIsValid:
    """The IR download must call snapshot_download with arguments it actually accepts.

    huggingface_hub 1.x removed ``local_dir_use_symlinks`` and its signature takes no
    ``**kwargs``, so passing it raised TypeError before a single byte was fetched. The
    retry helper swallowed that into a generic "download failed", and because the Intel
    engine's weights are only fetched here, a fresh Intel deployment could never provision
    them at all. Machines with a cached copy kept working, which is why it went unnoticed.
    """

    def _snapshot_download_calls(self) -> list[set[str]]:
        """Return the keyword names of EVERY snapshot_download call in the provisioner.

        Every call, not the first one found. Returning early meant a second download added
        later -- the OpenVINO IR and the CTranslate2 weights are fetched by separate calls --
        was never checked at all, which is precisely how a rejected keyword would slip back
        in on the path this class exists to guard.

        Read from the source rather than by calling it, so this guard runs in the test
        image, which does not ship huggingface_hub.
        """
        source = pathlib.Path(model_provisioning.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = [{kw.arg for kw in node.keywords if kw.arg} for node in ast.walk(tree) if _is_snapshot_download_call(node)]
        assert calls, "no snapshot_download call found in model_provisioning"
        return calls

    def test_the_removed_parameter_is_not_passed(self):
        """No call may pass the parameter huggingface_hub 1.x removed."""
        for keywords in self._snapshot_download_calls():
            assert "local_dir_use_symlinks" not in keywords, "huggingface_hub 1.x rejects this and takes no **kwargs"

    def test_the_download_still_targets_the_requested_directory(self):
        """Every call must still name both the repo and the directory to fetch it into."""
        for keywords in self._snapshot_download_calls():
            assert {"repo_id", "local_dir"} <= keywords

    def test_every_keyword_exists_in_the_installed_signature(self):
        """When huggingface_hub is importable, check the calls against its real signature."""
        hub = pytest.importorskip("huggingface_hub", reason="huggingface_hub is not installed in this image")
        accepted = set(inspect.signature(hub.snapshot_download).parameters)
        for keywords in self._snapshot_download_calls():
            assert keywords <= accepted


class TestProgressReporting:
    """What `/status` shows while weights are downloading."""

    def test_the_stage_and_percent_are_recorded(self):
        """The stage and percent are recorded."""
        model_provisioning._set_progress("Downloading Whisper", percent=42)
        progress = model_provisioning.get_progress()
        assert progress["stage"] == "Downloading Whisper"
        assert progress["percent"] == 42

    def test_a_percent_of_none_leaves_the_previous_one_alone(self):
        """Stage changes and progress ticks arrive separately; a stage-only update must not
        reset the bar to zero."""
        model_provisioning._set_progress("Downloading", percent=70)
        model_provisioning._set_progress("Verifying")
        assert model_provisioning.get_progress()["percent"] == 70

    def test_get_progress_returns_a_copy_not_the_live_dict(self):
        """Get progress returns a copy not the live dict."""
        snapshot = model_provisioning.get_progress()
        snapshot["stage"] = "tampered"
        assert model_provisioning.get_progress()["stage"] != "tampered"


class TestStreamToFile:
    """The direct UVR download: checksum before publish, nothing partial left behind."""

    def _requests(self, monkeypatch, chunks, status_error=None):
        """Build a fake `requests` module whose GET yields the given chunks."""

        class Response:
            """Response behaviour."""

            def __enter__(self):
                """Context-manager protocol, so `with requests.get(...)` works."""
                return self

            def __exit__(self, *_a):
                """Never suppresses; a transport error must reach the caller."""
                return False

            def raise_for_status(self):
                """Raise the scripted HTTP error, if any."""
                if status_error:
                    raise status_error

            def iter_content(self, chunk_size):
                """Yield the scripted body chunks."""
                return iter(chunks)

        monkeypatch.setattr(model_provisioning.importlib, "import_module", lambda _n: SimpleNamespace(get=lambda *a, **k: Response()))

    def test_a_matching_checksum_publishes_the_file_atomically(self, monkeypatch, tmp_path):
        """A matching checksum publishes the file atomically."""
        payload = b"model-bytes"
        self._requests(monkeypatch, [payload])
        target = tmp_path / "uvr.onnx"

        model_provisioning._stream_to_file("https://example/uvr.onnx", str(target), hashlib.sha256(payload).hexdigest())

        assert target.read_bytes() == payload
        assert not list(tmp_path.glob("model_dl_*")), "the staging file must not survive"

    def test_empty_chunks_are_skipped_rather_than_corrupting_the_digest(self, monkeypatch, tmp_path):
        """`iter_content` yields keep-alive blanks; hashing them would fail every download."""
        payload = b"abc"
        self._requests(monkeypatch, [b"a", b"", b"bc"])
        target = tmp_path / "uvr.onnx"

        model_provisioning._stream_to_file("https://example/uvr.onnx", str(target), hashlib.sha256(payload).hexdigest())
        assert target.read_bytes() == payload

    def test_a_checksum_mismatch_raises_and_leaves_no_file_behind(self, monkeypatch, tmp_path):
        """The whole point of staging: a corrupt or substituted download must never be
        published under the name the runtime loads."""
        self._requests(monkeypatch, [b"wrong-bytes"])
        target = tmp_path / "uvr.onnx"

        with pytest.raises(RuntimeError, match="Checksum mismatch"):
            model_provisioning._stream_to_file("https://example/uvr.onnx", str(target), "0" * 64)

        assert not target.exists()
        assert not list(tmp_path.glob("model_dl_*"))

    def test_an_http_error_propagates_and_leaves_nothing(self, monkeypatch, tmp_path):
        """An http error propagates and leaves nothing."""
        self._requests(monkeypatch, [], status_error=RuntimeError("404"))
        target = tmp_path / "uvr.onnx"

        with pytest.raises(RuntimeError, match="404"):
            model_provisioning._stream_to_file("https://example/uvr.onnx", str(target), "0" * 64)
        assert not target.exists()


class TestWhisperAssetSelection:
    """Which weights each engine actually needs -- and which it must not be given."""

    def test_openai_whisper_provisions_nothing_here(self):
        """It downloads its own checkpoint on first load, so provisioning would fetch several
        gigabytes of CTranslate2 weights it can never read."""
        assert model_provisioning._whisper_asset_for_engine(_config(), "OPENAI-WHISPER") is None

    def test_the_intel_engine_is_pointed_at_the_openvino_ir(self):
        """The intel engine is pointed at the openvino ir."""
        label, _provision = model_provisioning._whisper_asset_for_engine(_config(), "INTEL-WHISPER")
        assert label == "OpenVINO Whisper"

    def test_every_other_engine_gets_the_ct2_weights(self):
        """Every other engine gets the ct2 weights."""
        label, _provision = model_provisioning._whisper_asset_for_engine(_config(), "FASTER-WHISPER")
        assert label == "Whisper"

    def test_the_cache_directory_is_the_destination_not_the_repo_id(self):
        """`ensure_ct2_whisper(target_dir, model_id)` was called with the model value first,
        so ASR_MODEL="Systran/faster-whisper-medium" became the download *directory* -- a
        literal "Systran/..." path with the default repo downloaded into it."""
        captured = {}
        config = _config(MODEL_ID="Systran/faster-whisper-medium", CT2_CACHE_DIR="/app/model_cache/whisper")

        with mock.patch.object(model_provisioning, "ensure_ct2_whisper", lambda *a: captured.setdefault("args", a)):
            model_provisioning._ct2_provisioner(config, "FASTER-WHISPER")()

        assert captured["args"] == ("/app/model_cache/whisper", "Systran/faster-whisper-medium")

    def test_default_weights_pass_only_the_destination(self):
        """For the default model the repo id and the cache dir resolve to the same path, and
        a destination is not a repository id."""
        captured = {}
        config = _config(MODEL_ID="/app/model_cache/whisper", CT2_CACHE_DIR="/app/model_cache/whisper")

        with mock.patch.object(model_provisioning, "ensure_ct2_whisper", lambda *a: captured.setdefault("args", a)):
            model_provisioning._ct2_provisioner(config, "FASTER-WHISPER")()

        assert captured["args"] == ("/app/model_cache/whisper",)

    def test_a_config_without_ct2_cache_dir_falls_back_to_model_id(self):
        """A config predating CT2_CACHE_DIR -- or a test double standing in for one."""
        captured = {}
        # _config() deliberately carries no CT2_CACHE_DIR, which is the shape being tested.
        config = _config(MODEL_ID="/app/model_cache/whisper")

        with mock.patch.object(model_provisioning, "ensure_ct2_whisper", lambda *a: captured.setdefault("args", a)):
            model_provisioning._ct2_provisioner(config, "FASTER-WHISPER")()

        assert captured["args"] == ("/app/model_cache/whisper",)

    def test_hybrid_mode_picks_the_repo_for_the_named_engine(self):
        """Hybrid mode picks the repo for the named engine."""
        config = _config(
            MODEL_ID="/app/model_cache/whisper",
            CT2_CACHE_DIR="/app/model_cache/whisper",
            HYBRID_ENGINES=True,
            MODEL_ID_BY_ENGINE={"FASTER-WHISPER": "Systran/faster-whisper-large-v3"},
        )
        assert model_provisioning._ct2_repo_id(config, "FASTER-WHISPER") == "Systran/faster-whisper-large-v3"

    def test_a_non_hybrid_host_always_uses_the_single_resolved_model(self):
        """A non hybrid host always uses the single resolved model."""
        config = _config(MODEL_ID="/m", HYBRID_ENGINES=False, MODEL_ID_BY_ENGINE={"FASTER-WHISPER": "other"})
        assert model_provisioning._ct2_repo_id(config, "FASTER-WHISPER") == "/m"


class TestUvrDownloadFallback:
    """audio-separator first, direct download second."""

    def test_the_separator_path_is_tried_first(self, monkeypatch, tmp_path):
        """The separator path is tried first."""
        loaded = {}

        class Separator:
            """Separator behaviour."""

            def __init__(self, model_file_dir, output_dir):
                """Record what the separator was pointed at."""
                loaded["dir"] = model_file_dir

            def load_model(self, name):
                """Record (or fail) the model load the provisioner asks for."""
                loaded["model"] = name

        monkeypatch.setattr(model_provisioning.importlib, "import_module", lambda _n: SimpleNamespace(Separator=Separator))
        model_provisioning._uvr_via_separator(str(tmp_path), str(tmp_path))

        assert loaded["dir"] == str(tmp_path)
        assert loaded["model"] == model_provisioning.UVR_MODEL

    def test_the_scratch_directory_is_removed_even_when_loading_fails(self, monkeypatch, tmp_path):
        """The scratch directory is removed even when loading fails.

        The path is captured from the separator's own ``output_dir`` rather than hardcoded.
        A literal "temp_uvr_preload" asserts that *a* directory of that name is absent, which
        is trivially true if the provisioner ever renames or relocates its scratch space --
        the cleanup check would keep passing while nothing was being cleaned up.
        """
        observed = {}

        class Separator:
            """Separator behaviour."""

            def __init__(self, model_file_dir, output_dir):
                """Record what the separator was pointed at, and prove the directory exists."""
                observed["scratch"] = output_dir
                observed["existed"] = os.path.isdir(output_dir)

            def load_model(self, name):
                """Record (or fail) the model load the provisioner asks for."""
                raise RuntimeError("no network")

        monkeypatch.setattr(model_provisioning.importlib, "import_module", lambda _n: SimpleNamespace(Separator=Separator))

        with pytest.raises(RuntimeError):
            model_provisioning._uvr_via_separator(str(tmp_path), str(tmp_path))

        assert observed["existed"] is True, "the separator must be handed a scratch directory that exists"
        assert not os.path.isdir(observed["scratch"]), "the configured scratch directory must be removed"


class TestTheUvrDownloadRecoveryPath:
    """``ensure_uvr_model`` prefers audio-separator and falls back to a direct fetch.

    The fallback is the whole point of the pair: the separator resolves the model through
    its own registry, which is the path that breaks when the registry moves or the network
    refuses it, and a deployment with no vocal isolation is a silently degraded one rather
    than a failed boot.
    """

    def _run_download(self, tmp_path, separator_side_effect):
        """Invoke the closure that download_with_integrity_retry would have called."""
        captured = {}

        def fake_retry(*, download_fn, validator_fn, target_path, max_retries, description):
            captured["target_path"] = target_path
            captured["validator"] = validator_fn
            download_fn()
            return True

        with (
            mock.patch.object(model_provisioning, "_uvr_via_separator", side_effect=separator_side_effect) as separator,
            mock.patch.object(model_provisioning, "_stream_to_file") as stream,
            mock.patch.object(model_provisioning.model_integrity, "download_with_integrity_retry", fake_retry),
        ):
            result = model_provisioning.ensure_uvr_model(str(tmp_path), str(tmp_path))
        return result, separator, stream, captured

    def test_the_separator_path_is_used_when_it_succeeds(self, tmp_path):
        """audio-separator resolves the model through its own registry when it can."""
        result, separator, stream, captured = self._run_download(tmp_path, None)
        assert result is True
        separator.assert_called_once()
        stream.assert_not_called()
        assert captured["target_path"].endswith(model_provisioning.UVR_MODEL)

    def test_a_separator_failure_falls_back_to_the_direct_download(self, tmp_path):
        """The registry is the fragile half; the direct URL is the recovery."""
        _result, _separator, stream, _captured = self._run_download(tmp_path, RuntimeError("registry gone"))
        stream.assert_called_once()
        assert stream.call_args.args[0] == model_provisioning.UVR_DIRECT_URL
        assert stream.call_args.args[2] == model_provisioning.UVR_MODEL_SHA256

    def test_every_handled_download_error_reaches_the_fallback(self, tmp_path):
        """The handler uses the module's own error tuple, so the two cannot drift apart."""
        for error in model_provisioning._DOWNLOAD_ERRORS:
            _result, _separator, stream, _captured = self._run_download(tmp_path, error("boom"))
            stream.assert_called_once()

    def test_an_unhandled_error_is_not_swallowed(self, tmp_path):
        """A broad handler here would turn Ctrl-C into a silent retry."""
        with pytest.raises(KeyboardInterrupt):
            self._run_download(tmp_path, KeyboardInterrupt())
