"""Tests for runtime model provisioning (models are downloaded, not baked)."""

import ast
import inspect
import pathlib
from types import SimpleNamespace
from unittest import mock

import pytest

from modules.core import model_provisioning


@pytest.fixture(autouse=True)
def reset_provisioning_state():
    """Provisioning state is module-global; reset it around every test."""
    model_provisioning.MODEL_READY.clear()
    model_provisioning.PROVISION_STARTED.clear()
    yield
    model_provisioning.MODEL_READY.clear()
    model_provisioning.PROVISION_STARTED.clear()


def _config(**overrides):
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


class TestOpenVinoDownloadCallIsValid:
    """The IR download must call snapshot_download with arguments it actually accepts.

    huggingface_hub 1.x removed ``local_dir_use_symlinks`` and its signature takes no
    ``**kwargs``, so passing it raised TypeError before a single byte was fetched. The
    retry helper swallowed that into a generic "download failed", and because the Intel
    engine's weights are only fetched here, a fresh Intel deployment could never provision
    them at all. Machines with a cached copy kept working, which is why it went unnoticed.
    """

    def _snapshot_download_keywords(self) -> set[str]:
        """Return the keyword names the provisioner passes to snapshot_download.

        Read from the source rather than by calling it, so this guard runs in the test
        image, which does not ship huggingface_hub.
        """
        source = pathlib.Path(model_provisioning.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            func = getattr(node, "func", None)
            if isinstance(node, ast.Call) and isinstance(func, ast.Attribute) and func.attr == "snapshot_download":
                return {kw.arg for kw in node.keywords if kw.arg}
        raise AssertionError("no snapshot_download call found in model_provisioning")

    def test_the_removed_parameter_is_not_passed(self):
        keywords = self._snapshot_download_keywords()
        assert "local_dir_use_symlinks" not in keywords, "huggingface_hub 1.x rejects this and takes no **kwargs"

    def test_the_download_still_targets_the_requested_directory(self):
        assert {"repo_id", "local_dir"} <= self._snapshot_download_keywords()

    def test_every_keyword_exists_in_the_installed_signature(self):
        """When huggingface_hub is importable, check the call against its real signature."""
        hub = pytest.importorskip("huggingface_hub", reason="huggingface_hub is not installed in this image")
        accepted = set(inspect.signature(hub.snapshot_download).parameters)
        assert self._snapshot_download_keywords() <= accepted
