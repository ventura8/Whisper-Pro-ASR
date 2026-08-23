"""Unit tests for the remaining handler/plumbing functions in
modules/inference/engines/whisperx_worker.py not already covered by
test_whisperx_engine_behavior.py (worker_main's request loop, the
isolated-lib-path activation, _get_whisperx, and every handler that talks
to a (mocked) whisperx module: _load_model, _load_audio, _load_align_model,
_align, _load_diarization_pipeline, _run_diarization, _assign_word_speakers).

whisperx itself is not installed in this environment (see the module's own
docstring: it's only available inside the isolated worker subprocess at
runtime), so every test here mocks _get_whisperx() rather than importing
the real package.
"""

import pickle
from types import SimpleNamespace
from unittest import mock

import pytest

from modules.inference.engines import whisperx_worker as worker
from modules.inference.engines.whisperx_worker import (
    _activate_isolated_lib_path,
    _align,
    _assign_word_speakers,
    _build_handlers,
    _dispatch,
    _get_whisperx,
    _load_align_model,
    _load_audio,
    _load_diarization_pipeline,
    _load_model,
    _put,
    _release,
    _resolve_audio,
    _run_diarization,
    _send_response,
    _try_candidate_detect_language,
)


def test_get_whisperx_imports_via_importlib():
    """_get_whisperx should resolve the module through importlib, not a literal import."""
    fake_module = object()
    with mock.patch("modules.inference.engines.whisperx_worker.importlib.import_module", return_value=fake_module) as mock_import:
        result = _get_whisperx()
    mock_import.assert_called_once_with("whisperx")
    assert result is fake_module


def test_activate_isolated_lib_path_prepends_when_dir_exists_and_absent(monkeypatch: pytest.MonkeyPatch):
    """The isolated lib dir should be prepended to sys.path when present and not already there."""
    fake_path: list[str] = []
    monkeypatch.setenv("WHISPERX_LIB_PATH", "/fake/whisperx/lib")
    monkeypatch.setattr(worker.os.path, "isdir", lambda path: path == "/fake/whisperx/lib")
    monkeypatch.setattr(worker, "sys", SimpleNamespace(path=fake_path))
    _activate_isolated_lib_path()
    assert fake_path == ["/fake/whisperx/lib"]


def test_activate_isolated_lib_path_is_idempotent(monkeypatch: pytest.MonkeyPatch):
    """Calling it again when the dir is already on sys.path must not duplicate it."""
    fake_path = ["/fake/whisperx/lib", "/other"]
    monkeypatch.setenv("WHISPERX_LIB_PATH", "/fake/whisperx/lib")
    monkeypatch.setattr(worker.os.path, "isdir", lambda path: True)
    monkeypatch.setattr(worker, "sys", SimpleNamespace(path=fake_path))
    _activate_isolated_lib_path()
    assert fake_path == ["/fake/whisperx/lib", "/other"]


def test_activate_isolated_lib_path_skips_missing_dir(monkeypatch: pytest.MonkeyPatch):
    """A configured but nonexistent lib dir must not be added to sys.path."""
    fake_path: list[str] = []
    monkeypatch.setenv("WHISPERX_LIB_PATH", "/does/not/exist")
    monkeypatch.setattr(worker.os.path, "isdir", lambda path: False)
    monkeypatch.setattr(worker, "sys", SimpleNamespace(path=fake_path))
    _activate_isolated_lib_path()
    assert not fake_path


def _fake_conn(*recv_values: object) -> mock.MagicMock:
    conn = mock.MagicMock()
    conn.recv.side_effect = list(recv_values)
    return conn


def test_worker_main_processes_one_request_then_stops_on_none():
    """worker_main should dispatch each received request and stop on a None sentinel."""
    conn = _fake_conn({"id": 1, "cmd": "ping", "args": {}}, None)
    with mock.patch.object(worker, "_activate_isolated_lib_path"):
        worker.worker_main(conn)
    sent = [call.args[0] for call in conn.send.call_args_list]
    assert sent == [{"id": 1, "ok": True, "result": "pong"}]


def test_worker_main_stops_cleanly_on_eof():
    """A dead pipe (EOFError on recv) should stop the loop without sending anything."""
    conn = mock.MagicMock()
    conn.recv.side_effect = EOFError()
    with mock.patch.object(worker, "_activate_isolated_lib_path"):
        worker.worker_main(conn)
    conn.send.assert_not_called()


def test_worker_main_stops_cleanly_on_oserror():
    """A dead pipe (OSError on recv) should stop the loop without sending anything."""
    conn = mock.MagicMock()
    conn.recv.side_effect = OSError("pipe closed")
    with mock.patch.object(worker, "_activate_isolated_lib_path"):
        worker.worker_main(conn)
    conn.send.assert_not_called()


def test_worker_main_activates_isolated_lib_path_before_processing():
    """worker_main must activate the isolated sys.path before handling any request."""
    conn = _fake_conn(None)
    with mock.patch.object(worker, "_activate_isolated_lib_path") as mock_activate:
        worker.worker_main(conn)
    mock_activate.assert_called_once()


def test_send_response_falls_back_to_error_reply_when_first_send_unpicklable():
    """_send_response must recover from a conn.send() PicklingError by sending a
    serialized error reply instead of letting the exception kill the worker's
    request loop."""
    conn = mock.MagicMock()
    conn.send.side_effect = [pickle.PicklingError("cannot pickle object"), None]

    _send_response(conn, {"id": 7, "ok": True, "result": object()})

    assert conn.send.call_count == 2
    fallback = conn.send.call_args_list[1].args[0]
    assert fallback["id"] == 7
    assert fallback["ok"] is False
    assert "PicklingError" in fallback["error"]


def test_dispatch_unknown_command_reports_error_not_raises():
    """A cmd not present in handlers should surface as an error response, not raise."""
    response = _dispatch({}, {"id": 5, "cmd": "nope", "args": {}})
    assert response["ok"] is False
    assert response["id"] == 5


def test_put_generates_unique_handles_and_stores_object():
    """_put should assign a distinct handle per object and store it for later lookup."""
    objects = {}
    obj_a, obj_b = object(), object()
    handle_a = _put(objects, obj_a)
    handle_b = _put(objects, obj_b)
    assert handle_a != handle_b
    assert objects[handle_a] is obj_a
    assert objects[handle_b] is obj_b


def test_load_model_calls_whisperx_load_model_and_stores_handle():
    """_load_model should call whisperx.load_model and store the resulting model."""
    fake_whisperx = mock.MagicMock()
    fake_model = object()
    fake_whisperx.load_model.return_value = fake_model
    objects = {}
    with mock.patch.object(worker, "_get_whisperx", return_value=fake_whisperx):
        handle = _load_model(objects, model_id="m", device="cpu", compute_type="int8")
    fake_whisperx.load_model.assert_called_once_with("m", device="cpu", compute_type="int8")
    assert objects[handle] is fake_model


def test_resolve_audio_prefers_provided_array_without_touching_whisperx():
    """A preloaded audio_array should short-circuit before ever calling _get_whisperx."""
    with mock.patch.object(worker, "_get_whisperx") as mock_get:
        result = _resolve_audio(audio_path="unused.wav", audio_array="preloaded")
    assert result == "preloaded"
    mock_get.assert_not_called()


def test_resolve_audio_loads_from_path_when_no_array():
    """With no audio_array, it should fall back to whisperx.load_audio(audio_path)."""
    fake_whisperx = mock.MagicMock()
    fake_whisperx.load_audio.return_value = "decoded-audio"
    with mock.patch.object(worker, "_get_whisperx", return_value=fake_whisperx):
        result = _resolve_audio(audio_path="clip.wav", audio_array=None)
    fake_whisperx.load_audio.assert_called_once_with("clip.wav")
    assert result == "decoded-audio"


def test_try_candidate_detect_language_swallows_expected_failures():
    """A candidate whose detect_language raises a known failure type should yield None."""
    candidate = mock.MagicMock()
    candidate.detect_language.side_effect = RuntimeError("boom")
    assert _try_candidate_detect_language(candidate, "audio") is None


def test_load_audio_handler_stores_decoded_audio_handle():
    """_load_audio should decode via whisperx and store the resulting audio handle."""
    fake_whisperx = mock.MagicMock()
    fake_whisperx.load_audio.return_value = "decoded-audio"
    objects = {}
    with mock.patch.object(worker, "_get_whisperx", return_value=fake_whisperx):
        handle = _load_audio(objects, path="clip.wav")
    assert objects[handle] == "decoded-audio"


def test_load_align_model_handler_stores_model_and_metadata_tuple():
    """_load_align_model should store the (model, metadata) tuple whisperx returns."""
    fake_whisperx = mock.MagicMock()
    fake_whisperx.load_align_model.return_value = ("align-model", {"meta": True})
    objects = {}
    with mock.patch.object(worker, "_get_whisperx", return_value=fake_whisperx):
        handle = _load_align_model(objects, lang_code="en", device="cpu")
    fake_whisperx.load_align_model.assert_called_once_with(language_code="en", device="cpu")
    assert objects[handle] == ("align-model", {"meta": True})


def test_align_handler_calls_whisperx_align_with_stored_handles():
    """_align should resolve its align/audio handles and call whisperx.align with them."""
    fake_whisperx = mock.MagicMock()
    fake_whisperx.align.return_value = {"aligned": True}
    objects = {
        "align-h": ("align-model", {"meta": True}),
        "audio-h": "decoded-audio",
    }
    with mock.patch.object(worker, "_get_whisperx", return_value=fake_whisperx):
        result = _align(objects, raw_segments=[{"text": "hi"}], align_handle="align-h", audio_handle="audio-h", device="cpu")
    fake_whisperx.align.assert_called_once_with(
        [{"text": "hi"}], "align-model", {"meta": True}, "decoded-audio", device="cpu", return_char_alignments=False
    )
    assert result == {"aligned": True}


def test_load_diarization_pipeline_handler_stores_pipeline_handle():
    """_load_diarization_pipeline should build a DiarizationPipeline from whisperx.diarize
    (not a nonexistent "whisperx.diarization" attribute) using the `token` keyword WhisperX
    3.8.6's constructor actually expects (not the older `use_auth_token`), and store it."""
    fake_diarize_module = mock.MagicMock()
    fake_pipeline = object()
    fake_diarize_module.DiarizationPipeline.return_value = fake_pipeline
    objects = {}
    test_hf_token = "-".join(["fake", "test", "hf", "token"])
    with mock.patch("modules.inference.engines.whisperx_worker.importlib.import_module", return_value=fake_diarize_module) as mock_import:
        handle = _load_diarization_pipeline(objects, token=test_hf_token, device="cpu")
    mock_import.assert_called_once_with("whisperx.diarize")
    fake_diarize_module.DiarizationPipeline.assert_called_once_with(token=test_hf_token, device="cpu")
    assert objects[handle] is fake_pipeline


def test_run_diarization_handler_invokes_stored_pipeline_with_audio_and_speaker_bounds():
    """_run_diarization should call the stored pipeline with the resolved audio and bounds, and
    store the result server-side rather than returning it directly (keeps the diarization frame
    out of the parent<->worker pickle round trip)."""
    fake_pipeline = mock.MagicMock(return_value="diarize-segments")
    objects = {"pipeline-h": fake_pipeline, "audio-h": "decoded-audio"}
    handle = _run_diarization(objects, pipeline_handle="pipeline-h", audio_handle="audio-h", min_speakers=1, max_speakers=3)
    fake_pipeline.assert_called_once_with("decoded-audio", min_speakers=1, max_speakers=3)
    assert objects[handle] == "diarize-segments"


def test_assign_word_speakers_handler_delegates_to_whisperx():
    """_assign_word_speakers should resolve the diarize handle from the objects pool and
    delegate straight through to whisperx.assign_word_speakers."""
    fake_whisperx = mock.MagicMock()
    fake_whisperx.assign_word_speakers.return_value = {"labeled": True}
    objects = {"diarize-h": "diarize-segments"}
    with mock.patch.object(worker, "_get_whisperx", return_value=fake_whisperx):
        result = _assign_word_speakers(objects, diarize_handle="diarize-h", alignment_result={"aligned": True})
    fake_whisperx.assign_word_speakers.assert_called_once_with("diarize-segments", {"aligned": True})
    assert result == {"labeled": True}


def test_release_handler_pops_handle():
    """_release should remove the given handle from the objects pool."""
    objects = {"h": object()}
    _release(objects, "h")
    assert "h" not in objects


def test_release_handler_is_a_noop_for_missing_handle():
    """_release on a handle that was never stored must not raise."""
    objects = {}
    _release(objects, "missing")
    assert not objects


@pytest.mark.parametrize(
    "cmd_name",
    [
        "load_model",
        "transcribe",
        "detect_language",
        "unload_model",
        "load_audio",
        "load_align_model",
        "align",
        "load_diarization_pipeline",
        "run_diarization",
        "assign_word_speakers",
        "release",
    ],
)
def test_build_handlers_registers_all_object_scoped_commands(cmd_name: str):
    """Every object-scoped command name must be present in the dispatch table."""
    handlers = _build_handlers({})
    assert cmd_name in handlers
