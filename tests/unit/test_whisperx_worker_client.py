"""Tests for the WhisperX binding over the shared worker channel.

The client used to own a private copy of the worker transport; it now delegates to
``worker_channel``, so the lifecycle/shutdown/lock invariants are tested once in
tests/unit/test_worker_channel_unit.py. What remains WhisperX-specific -- and therefore
belongs here -- is the binding itself: which worker it spawns, its error type, and that
the public functions other modules call still route to the channel.
"""

# pylint: disable=protected-access
# The unit under test is the module's internals. Reaching them by name is the point
# of these tests, not an accident: the public surface is a thin wrapper and testing
# only through it would leave the rules below unpinned.

from unittest import mock

import pytest

from modules.inference.engines import whisperx_worker
from modules.inference.engines import whisperx_worker_client as client
from modules.inference.engines.worker_channel import WorkerError


def test_channel_is_bound_to_the_whisperx_worker():
    """The channel must spawn the WhisperX worker, named so it is identifiable in ps."""
    channel = client.channel()
    assert channel._worker_main is whisperx_worker.worker_main
    assert channel._name == "whisperx-worker"


def test_error_type_is_a_worker_error_subclass():
    """Callers catching either the shared or the WhisperX-specific type must both work."""
    assert issubclass(client.WhisperXWorkerError, WorkerError)
    assert client.channel()._error_cls is client.WhisperXWorkerError


def test_call_delegates_to_the_channel():
    """The client is a thin forwarder; the channel does the work."""
    with mock.patch.object(client._CHANNEL, "call_with_generation", return_value=("ok", 2)) as mock_call:
        assert client.call("transcribe", audio_path="clip.wav") == "ok"
    mock_call.assert_called_once_with("transcribe", audio_path="clip.wav")


def test_call_with_generation_delegates_and_returns_the_stamp():
    """The generation stamp is passed back to the caller unchanged."""
    with mock.patch.object(client._CHANNEL, "call_with_generation", return_value=("handle", 5)) as mock_call:
        assert client.call_with_generation("load_model", model_id="tiny") == ("handle", 5)
    mock_call.assert_called_once_with("load_model", model_id="tiny")


def test_generation_delegates():
    """Reading the generation asks the channel rather than caching it."""
    with mock.patch.object(client._CHANNEL, "generation", return_value=11):
        assert client.generation() == 11


def test_shutdown_delegates():
    """Shutdown is the channel's to perform."""
    with mock.patch.object(client._CHANNEL, "shutdown") as mock_shutdown:
        client.shutdown()
    mock_shutdown.assert_called_once()


def test_worker_errors_surface_as_the_whisperx_type():
    """Callers catch the WhisperX error type, so the channel must raise that one."""
    with mock.patch.object(client._CHANNEL, "call_with_generation", side_effect=client.WhisperXWorkerError("boom")):
        with pytest.raises(client.WhisperXWorkerError, match="boom"):
            client.call("ping")
