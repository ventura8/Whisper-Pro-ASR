"""Tests for speaker diarization integration and WhisperX orchestration.

WhisperX now runs in an isolated subprocess (see
modules/inference/engines/whisperx_worker_client.py) instead of being
imported directly, so these tests fake out ``worker.call`` /
``worker.call_with_generation`` rather than injecting a mock ``whisperx``
module.
"""

import asyncio
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Optional
from unittest import mock

import pytest

from modules.api.routes.asr import get_request_params
from modules.core import utils
from modules.inference import scheduler
from modules.inference.runtime import model_manager

_DEFAULT_ASSIGN_RESULT = {"segments": [{"start": 0.0, "end": 1.0, "text": "hello", "speaker": "SPEAKER_00"}]}


def _make_worker_call(
    assign_result: Optional[dict[str, Any]] = None,
    align_result: Optional[dict[str, Any]] = None,
    diarize_result: str = "mock_diarize_segments",
) -> mock.MagicMock:
    """Build a fake ``worker.call(cmd, **kwargs)`` dispatcher mirroring the isolated worker's protocol."""
    responses: dict[str, Any] = {
        "load_audio": "dummy_audio_handle",
        "load_align_model": "align_handle",
        "align": align_result or {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}]},
        "load_diarization_pipeline": "diarize_pipeline_handle",
        "run_diarization": diarize_result,
        "assign_word_speakers": assign_result or _DEFAULT_ASSIGN_RESULT,
        "release": None,
    }

    def _call(cmd: str, **kwargs: Any) -> Any:
        return responses[cmd]

    return mock.MagicMock(side_effect=_call)


@contextmanager
def _patch_worker_rpc(call_impl: Any, generation: int = 0):
    """Patch both call and call_with_generation against the same dispatcher."""

    def _dispatch(cmd: str, **kwargs: Any) -> Any:
        return call_impl(cmd, **kwargs)

    def _with_generation(cmd: str, **kwargs: Any) -> tuple[Any, int]:
        return _dispatch(cmd, **kwargs), generation

    with (
        mock.patch("modules.inference.pipeline.diarization.worker.call", side_effect=_dispatch),
        mock.patch(
            "modules.inference.pipeline.diarization.worker.call_with_generation",
            side_effect=_with_generation,
        ),
        mock.patch("modules.inference.pipeline.diarization.worker.generation", return_value=generation),
    ):
        yield


@pytest.fixture(autouse=True)
def reset_state() -> Generator[None, None, None]:
    """Reset model_manager pools and scheduler states between tests."""
    model_manager.MODEL_POOL.clear()
    model_manager.PREPROCESSOR_POOL.clear()
    model_manager.DIARIZE_POOL.clear()
    model_manager.ALIGN_POOL.clear()

    with mock.patch("modules.core.config.HARDWARE_UNITS", [{"id": "CPU", "type": "CPU", "name": "CPU"}]):
        from modules.inference.scheduler import SchedulerState

        scheduler.STATE = SchedulerState()
        scheduler.STATE.engine_initialized = True

    # Reset thread context
    utils.THREAD_CONTEXT.reset()
    yield


def test_diarization_success() -> None:
    """Verify successful transcription, alignment, and diarization flow."""
    mock_model = mock.MagicMock()
    mock_info = mock.MagicMock(language="en", language_probability=0.95, duration=5.0)
    mock_segment = mock.MagicMock(start=0.0, end=1.0, text="hello")
    mock_model.transcribe.return_value = ([mock_segment], mock_info)

    model_manager.MODEL_POOL["CPU"] = mock_model

    mock_call = _make_worker_call()
    with _patch_worker_rpc(mock_call):
        result = model_manager.run_transcription(
            "test.wav",
            language="en",
            task="transcribe",
            diarize=True,
            min_speakers=1,
            max_speakers=2,
            hf_token="fake_token",
        )

    _assert_diarized_result(result)
    _assert_each_worker_step_called_once(mock_call)
    _assert_load_align_call_args(mock_call)
    _assert_load_pipeline_call_args(mock_call)
    _assert_run_diarize_call_args(mock_call)
    _assert_audio_handle_released_once(mock_call)


def _assert_diarized_result(result: dict) -> None:
    assert result["segments"][0]["speaker"] == "SPEAKER_00"
    assert result["segments"][0]["text"] == "hello"


def _assert_each_worker_step_called_once(mock_call) -> None:
    called_cmds = [c.args[0] for c in mock_call.call_args_list]
    expected_once = ["load_audio", "load_align_model", "align", "load_diarization_pipeline", "run_diarization", "assign_word_speakers"]
    assert {cmd: called_cmds.count(cmd) for cmd in expected_once} == {cmd: 1 for cmd in expected_once}


def _find_call(mock_call: mock.MagicMock, cmd: str):
    return next(c for c in mock_call.call_args_list if c.args[0] == cmd)


def _assert_load_align_call_args(mock_call) -> None:
    assert _find_call(mock_call, "load_align_model").kwargs == {"lang_code": "en", "device": "cpu"}


def _assert_load_pipeline_call_args(mock_call) -> None:
    assert _find_call(mock_call, "load_diarization_pipeline").kwargs == {"token": "fake_token", "device": "cpu"}


def _assert_run_diarize_call_args(mock_call) -> None:
    kwargs = _find_call(mock_call, "run_diarization").kwargs
    assert (kwargs["min_speakers"], kwargs["max_speakers"]) == (1, 2)


def _assert_audio_handle_released_once(mock_call) -> None:
    release_calls = [c for c in mock_call.call_args_list if c.args[0] == "release" and c.kwargs.get("handle") == "dummy_audio_handle"]
    assert len(release_calls) == 1


def test_diarization_caching_and_unloading() -> None:
    """Verify align and diarize model handles are cached, and the worker is torn down on unload."""
    mock_model = mock.MagicMock()
    mock_info = mock.MagicMock(language="en", language_probability=0.95, duration=5.0)
    mock_segment = mock.MagicMock(start=0.0, end=1.0, text="hello")
    mock_model.transcribe.return_value = ([mock_segment], mock_info)

    model_manager.MODEL_POOL["CPU"] = mock_model

    mock_call = _make_worker_call()
    with _patch_worker_rpc(mock_call):
        model_manager.run_transcription("test.wav", language="en", task="transcribe", diarize=True, hf_token="fake_token")
        model_manager.run_transcription("test.wav", language="en", task="transcribe", diarize=True, hf_token="fake_token")

    # Verify loading functions only called once (cached on the second call) -- pool
    # cardinality alone only proves one entry exists, not that the loader itself was
    # skipped on the second (cached) call, so also assert the actual load command
    # counts directly.
    assert (len(model_manager.DIARIZE_POOL), len(model_manager.ALIGN_POOL)) == (1, 1)
    called_cmds = [c.args[0] for c in mock_call.call_args_list]
    assert (called_cmds.count("load_align_model"), called_cmds.count("load_diarization_pipeline")) == (1, 1)

    # Unload models — this shuts down the isolated whisperx worker outright.
    with mock.patch("modules.inference.runtime.model_manager.utils.get_system_telemetry", return_value={}):
        with mock.patch("modules.inference.runtime.model_lifecycle.whisperx_worker_client.shutdown") as mock_shutdown:
            model_manager.unload_models()

    mock_shutdown.assert_called_once()
    # Verify pools are cleared
    assert (len(model_manager.DIARIZE_POOL), len(model_manager.ALIGN_POOL)) == (0, 0)


def test_diarization_missing_token_fallback() -> None:
    """Verify fallback to non-diarized output when HF token is missing."""
    mock_model = mock.MagicMock()
    mock_info = mock.MagicMock(language="en", language_probability=0.95, duration=5.0)
    mock_segment = mock.MagicMock(start=0.0, end=1.0, text="hello")
    mock_model.transcribe.return_value = ([mock_segment], mock_info)

    model_manager.MODEL_POOL["CPU"] = mock_model

    # Ensure config token is empty
    with mock.patch("modules.core.config.DIARIZATION_HF_TOKEN", ""):
        result = model_manager.run_transcription("test.wav", language="en", task="transcribe", diarize=True, hf_token=None)

    # Should fall back to standard results without speaker
    assert "speaker" not in result["segments"][0]
    assert result["segments"][0]["text"] == "hello"


def _make_align_failing_call(call_log: list[tuple[str, dict[str, Any]]]):
    delegate = _make_worker_call().side_effect

    def _failing_call(cmd: str, **kwargs: Any) -> Any:
        call_log.append((cmd, kwargs))
        if cmd == "align":
            raise RuntimeError("Align fail")
        return delegate(cmd, **kwargs)

    return _failing_call


def test_diarization_failure_fallback() -> None:
    """Verify fallback to non-diarized output when the isolated worker's alignment step fails,
    and that the load_audio handle allocated before the failure is still released -- a failed
    align call must not leak the audio handle in the worker's objects pool."""
    mock_model = mock.MagicMock()
    mock_info = mock.MagicMock(language="en", language_probability=0.95, duration=5.0)
    mock_segment = mock.MagicMock(start=0.0, end=1.0, text="hello")
    mock_model.transcribe.return_value = ([mock_segment], mock_info)

    model_manager.MODEL_POOL["CPU"] = mock_model

    call_log: list[tuple[str, dict[str, Any]]] = []

    with _patch_worker_rpc(_make_align_failing_call(call_log)):
        result = model_manager.run_transcription("test.wav", language="en", task="transcribe", diarize=True, hf_token="fake_token")

    # Should fall back to a standard (non-diarized) segment, and the load_audio handle
    # ("dummy_audio_handle", per _make_worker_call's responses) must still be released
    # even though align raised.
    segment = result["segments"][0]
    release_handles = {kwargs["handle"] for cmd, kwargs in call_log if cmd == "release"}
    assert ("speaker" not in segment, segment["text"], "dummy_audio_handle" in release_handles) == (True, "hello", True)


def test_routes_extract_diarize_params() -> None:
    """Verify that ASR endpoints parse and forward diarization params."""
    mock_req = mock.MagicMock()
    mock_req.headers = {"X-HF-Token": "test_tok"}
    mock_req.query_params = {"diarize": "true", "min_speakers": "2", "max_speakers": "4"}
    mock_req.url.path = "/asr"
    params = asyncio.run(get_request_params(mock_req, {}))
    assert (params["diarize"], params["min_speakers"], params["max_speakers"], params["hf_token"]) == (
        True,
        2,
        4,
        "test_tok",
    )

    mock_req_default = mock.MagicMock()
    mock_req_default.headers = {}
    mock_req_default.query_params = {}
    mock_req_default.url.path = "/asr"
    params_default = asyncio.run(get_request_params(mock_req_default, {}))

    mock_req_invalid = mock.MagicMock()
    mock_req_invalid.headers = {}
    mock_req_invalid.query_params = {"min_speakers": "invalid", "max_speakers": "invalid"}
    mock_req_invalid.url.path = "/asr"
    params_invalid = asyncio.run(get_request_params(mock_req_invalid, {}))

    mock_req_header = mock.MagicMock()
    mock_req_header.query_params = {}
    mock_req_header.headers = {"X-HF-Token": "header_tok"}
    mock_req_header.url.path = "/asr"
    params_header = asyncio.run(get_request_params(mock_req_header, {}))

    assert (params_default["diarize"], params_default["min_speakers"], params_default["max_speakers"], params_default["hf_token"]) == (
        False,
        None,
        None,
        None,
    )
    assert (params_invalid["min_speakers"], params_invalid["max_speakers"]) == (None, None)
    assert params_header["hf_token"] == "header_tok"


def test_utils_speaker_formatting() -> None:
    """Verify that subtitle/text writers properly format speaker labels."""
    result = {
        "segments": [
            {"start": 0.0, "end": 2.5, "text": "Hello world", "speaker": "SPEAKER_00"},
            {"start": 3.0, "end": 4.5, "text": "Goodbye", "speaker": "SPEAKER_01"},
        ]
    }

    assert all(
        [
            "[SPEAKER_00]: Hello world" in utils.generate_srt(result),
            "[SPEAKER_01]: Goodbye" in utils.generate_srt(result),
            "[SPEAKER_00]: Hello world" in utils.generate_vtt(result),
            "[SPEAKER_01]: Goodbye" in utils.generate_vtt(result),
            "[SPEAKER_00]: Hello world" in utils.generate_tsv(result),
            "[SPEAKER_01]: Goodbye" in utils.generate_tsv(result),
            "[SPEAKER_00]: Hello world" in utils.generate_txt(result),
            "[SPEAKER_01]: Goodbye" in utils.generate_txt(result),
        ]
    )


def test_run_diarization_duration_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that run_diarization skips when audio duration exceeds limit."""
    import modules.inference.pipeline.diarization as diarization_mod
    from modules.inference.pipeline.diarization import run_diarization

    # Set MAX_DIARIZE_DURATION_SEC
    monkeypatch.setattr(diarization_mod, "MAX_DIARIZE_DURATION_SEC", 10)

    info = mock.MagicMock(duration=15.0)
    raw_segments = [{"start": 0.0, "end": 1.0, "text": "hello"}]

    res = run_diarization(
        processed_path="dummy.wav",
        raw_segments=raw_segments,
        info=info,
        language="en",
        min_speakers=1,
        max_speakers=2,
        hf_token="token",
        unit_id="CPU",
    )
    assert len(res) == 1
    assert res[0]["text"] == "hello"
    assert "speaker" not in res[0]


def test_run_diarization_duration_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that run_diarization logs a warning for long files."""
    import modules.inference.pipeline.diarization as diarization_mod
    from modules.inference.pipeline.diarization import run_diarization

    monkeypatch.setattr(diarization_mod, "_DIARIZE_WARN_THRESHOLD_SEC", 10)

    info = mock.MagicMock(duration=15.0, language="en")
    raw_segments = [{"start": 0.0, "end": 1.0, "text": "hello"}]

    mock_call = _make_worker_call()
    with mock.patch("modules.inference.pipeline.diarization.logger") as mock_logger:
        with _patch_worker_rpc(mock_call):
            run_diarization(
                processed_path="dummy.wav",
                raw_segments=raw_segments,
                info=info,
                language="en",
                min_speakers=1,
                max_speakers=2,
                hf_token="token",
                unit_id="CPU",
            )
            mock_logger.warning.assert_called()


def test_run_diarization_words_key() -> None:
    """Verify that run_diarization preserves 'words' in result segments."""
    from modules.inference.pipeline.diarization import run_diarization

    info = mock.MagicMock(duration=5.0, language="en")
    raw_segments = [{"start": 0.0, "end": 1.0, "text": "hello"}]

    assign_result = {
        "segments": [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "hello",
                "speaker": "SPEAKER_00",
                "words": [{"word": "hello", "start": 0.0, "end": 0.5}],
            }
        ]
    }
    mock_call = _make_worker_call(assign_result=assign_result)
    with _patch_worker_rpc(mock_call):
        res = run_diarization(
            processed_path="dummy.wav",
            raw_segments=raw_segments,
            info=info,
            language="en",
            min_speakers=1,
            max_speakers=2,
            hf_token="token",
            unit_id="CPU",
        )
    assert res[0]["words"] == [{"word": "hello", "start": 0.0, "end": 0.5}]


def _worker_call_recording_to(call_log: list[str]):
    responses: dict[str, str] = {"load_align_model": "align_handle", "load_diarization_pipeline": "diarize_handle"}

    def _call(cmd: str, **_kwargs: Any) -> str:
        call_log.append(cmd)
        return responses[cmd]

    return _call


def _get_align_and_diarize_at_generation(diarization: Any, call_log: list[str], gen: int) -> tuple[str, str]:
    with _patch_worker_rpc(_worker_call_recording_to(call_log), generation=gen):
        align = diarization._get_align_model("en", "cpu", "CPU")
        diarize = diarization._get_diarize_pipeline("token", "cpu", "CPU")
    return align, diarize


def test_align_and_diarize_pools_reload_after_worker_generation_changes():
    """A worker crash+respawn bumps whisperx_worker_client.generation(); cached
    ALIGN_POOL/DIARIZE_POOL handles from the old generation must be treated as
    a miss and reloaded, not sent to the new (empty-objects-dict) worker."""
    from modules.inference.pipeline import diarization

    call_log: list[str] = []

    first_align, first_diarize = _get_align_and_diarize_at_generation(diarization, call_log, gen=1)
    # Second call at the same generation must hit the cache, not reload.
    cached_align, cached_diarize = _get_align_and_diarize_at_generation(diarization, call_log, gen=1)
    assert (first_align, first_diarize) == (cached_align, cached_diarize) == ("align_handle", "diarize_handle")
    assert (call_log.count("load_align_model"), call_log.count("load_diarization_pipeline")) == (1, 1)

    # Simulate a worker crash+respawn: generation bumps to 2. Both must reload
    # against the new generation, not reuse the stale cache entry.
    reloaded_align, reloaded_diarize = _get_align_and_diarize_at_generation(diarization, call_log, gen=2)
    assert (reloaded_align, reloaded_diarize) == ("align_handle", "diarize_handle")
    assert (call_log.count("load_align_model"), call_log.count("load_diarization_pipeline")) == (2, 2)
