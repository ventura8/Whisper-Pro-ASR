"""Unit tests for runtime segment processing helpers."""

from types import SimpleNamespace
from unittest import mock

from modules.inference.runtime import model_segment_processing


def test_consume_transcription_segments_updates_metadata_and_progress():
    """Consumption should emit SRT metadata, progress updates, and segment dicts."""
    segments = [
        SimpleNamespace(start=0.0, end=1.5, text=" hello ", words=[SimpleNamespace(start=0.0, end=0.5, word="he", probability=0.9)]),
        SimpleNamespace(start=1.5, end=3.0, text=" world ", words=None),
    ]
    info = SimpleNamespace(duration=10.0, language="en")

    with (
        mock.patch("modules.inference.runtime.model_segment_processing.utils.format_single_srt_block", side_effect=["A", "B"]),
        mock.patch("modules.inference.runtime.model_segment_processing.scheduler.update_task_metadata") as mock_metadata,
        mock.patch("modules.inference.runtime.model_segment_processing.scheduler.update_task_progress") as mock_progress,
        mock.patch("modules.inference.runtime.model_segment_processing.logger.info") as mock_log,
    ):
        results = model_segment_processing.consume_transcription_segments(
            segments,
            info,
            "translate",
            diarize=False,
            min_speakers=None,
            max_speakers=None,
            hf_token=None,
            unit_id="CPU",
            processed_path="audio.wav",
            preemption_check=lambda: None,
        )

    assert results == [
        {"start": 0.0, "end": 1.5, "text": "hello", "words": [{"start": 0.0, "end": 0.5, "word": "he", "probability": 0.9}]},
        {"start": 1.5, "end": 3.0, "text": "world"},
    ]
    assert mock_metadata.call_count == 2
    assert mock_progress.call_count == 2
    assert mock_log.call_count == 1


def test_consume_transcription_segments_logs_transcribing_progress():
    """Transcribe task should use the transcribing verb in progress updates."""
    segment = SimpleNamespace(start=0.0, end=2.0, text=" hello ", words=None)
    info = SimpleNamespace(duration=10.0, language="en")

    with (
        mock.patch("modules.inference.runtime.model_segment_processing.utils.format_single_srt_block", return_value="block"),
        mock.patch("modules.inference.runtime.model_segment_processing.scheduler.update_task_metadata"),
        mock.patch("modules.inference.runtime.model_segment_processing.scheduler.update_task_progress") as mock_progress,
    ):
        model_segment_processing.consume_transcription_segments(
            [segment],
            info,
            "transcribe",
            diarize=False,
            min_speakers=None,
            max_speakers=None,
            hf_token=None,
            unit_id="CPU",
            processed_path="audio.wav",
            preemption_check=lambda: None,
        )

    assert mock_progress.call_args.args[1].startswith("Transcribing")


def test_consume_transcription_segments_skips_progress_when_duration_is_zero():
    """A zero-duration info object should skip segment progress updates."""
    segments = [SimpleNamespace(start=0.0, end=1.0, text=" test ", words=None)]
    info = SimpleNamespace(duration=0.0, language="en")

    with (
        mock.patch("modules.inference.runtime.model_segment_processing.utils.format_single_srt_block", return_value="block"),
        mock.patch("modules.inference.runtime.model_segment_processing.scheduler.update_task_metadata") as mock_metadata,
        mock.patch("modules.inference.runtime.model_segment_processing.scheduler.update_task_progress") as mock_progress,
    ):
        results = model_segment_processing.consume_transcription_segments(
            segments,
            info,
            "transcribe",
            diarize=False,
            min_speakers=None,
            max_speakers=None,
            hf_token=None,
            unit_id="CPU",
            processed_path="audio.wav",
            preemption_check=lambda: None,
        )

    assert results == [{"start": 0.0, "end": 1.0, "text": "test"}]
    mock_metadata.assert_called_once()
    mock_progress.assert_not_called()


def test_run_diarization_safe_falls_back_to_raw_segments():
    """Diarization failures should preserve raw segment content and words."""
    info = SimpleNamespace(duration=10.0, language="en")

    with mock.patch(
        "modules.inference.runtime.model_segment_processing.diarization.run_diarization",
        side_effect=RuntimeError("boom"),
    ):
        results = model_segment_processing.consume_transcription_segments(
            [SimpleNamespace(start=0.0, end=1.0, text=" hello ", words=[SimpleNamespace(start=0.0, end=0.5, word="hello")])],
            info,
            "transcribe",
            diarize=True,
            min_speakers=None,
            max_speakers=None,
            hf_token=None,
            unit_id="CPU",
            processed_path="audio.wav",
            preemption_check=lambda: None,
        )

    assert results == [
        {
            "start": 0.0,
            "end": 1.0,
            "text": "hello",
            "words": [{"start": 0.0, "end": 0.5, "word": "hello", "probability": 1.0}],
        },
    ]


def test_run_diarization_safe_falls_back_with_words_preserved():
    """Diarization failures should preserve raw segment content and words."""
    segment = SimpleNamespace(start=0.0, end=1.0, text=" hello ", words=[SimpleNamespace(start=0.0, end=0.5, word="hello")])
    info = SimpleNamespace(duration=10.0, language="en")

    with (
        mock.patch("modules.inference.runtime.model_segment_processing.utils.format_single_srt_block", return_value="block"),
        mock.patch(
            "modules.inference.runtime.model_segment_processing.diarization.run_diarization",
            side_effect=RuntimeError("boom"),
        ),
    ):
        results = model_segment_processing.consume_transcription_segments(
            [segment],
            info,
            "transcribe",
            diarize=True,
            min_speakers=None,
            max_speakers=None,
            hf_token=None,
            unit_id="CPU",
            processed_path="audio.wav",
            preemption_check=lambda: None,
        )

    assert results == [
        {
            "start": 0.0,
            "end": 1.0,
            "text": "hello",
            "words": [{"start": 0.0, "end": 0.5, "word": "hello", "probability": 1.0}],
        },
    ]


def test_run_diarization_safe_returns_empty_list_without_segments():
    """Empty raw segment input should short-circuit before diarization runs."""
    info = SimpleNamespace(language="en")

    with mock.patch("modules.inference.runtime.model_segment_processing.diarization.run_diarization") as mock_run:
        results = model_segment_processing.consume_transcription_segments(
            [],
            info,
            "transcribe",
            diarize=True,
            min_speakers=None,
            max_speakers=None,
            hf_token=None,
            unit_id="CPU",
            processed_path="audio.wav",
            preemption_check=lambda: None,
        )

    assert results == []
    mock_run.assert_not_called()


def test_a_progress_window_confines_one_call_to_its_share_of_the_bar():
    """The second of two calls climbs from the middle of the bar, never back from zero."""
    segments = [SimpleNamespace(start=0.0, end=5.0, text="a", words=None), SimpleNamespace(start=5.0, end=10.0, text="b", words=None)]
    info = SimpleNamespace(duration=10.0, language="en")

    with (
        mock.patch("modules.inference.runtime.model_segment_processing.scheduler.update_task_metadata"),
        mock.patch("modules.inference.runtime.model_segment_processing.scheduler.update_task_progress") as mock_progress,
    ):
        model_segment_processing.consume_segments(
            segments, info, "transcribe", diarize=False, preemption_check=lambda: None, progress_window=(1, 2)
        )

    assert [call.args[0] for call in mock_progress.call_args_list] == [75, 95]


def test_without_a_window_progress_is_the_whole_bar():
    """The single-call path is unchanged: a segment at the end of the file is at the cap."""
    assert model_segment_processing._segment_progress_pct(10.0, 10.0, 95) == 100
    assert model_segment_processing._segment_progress_pct(2.5, 10.0, 95) == 25
    assert model_segment_processing._segment_progress_pct(2.5, 10.0, 80) == 20


def _quiet(monkeypatch):
    for name in ("_update_live_srt_metadata", "_update_segment_progress", "_maybe_log_segment_progress"):
        monkeypatch.setattr(model_segment_processing, name, lambda *a, **k: None)


def test_consume_until_pause_keeps_the_segment_in_hand_and_pulls_no_more(monkeypatch):
    """A pause is asked for after each segment is kept, so nothing the worker produced is
    lost, and the next segment is never requested: the stream is left for the caller to close."""
    _quiet(monkeypatch)
    pulled: list[int] = []

    def segments():
        for index in range(4):
            pulled.append(index)
            yield SimpleNamespace(start=float(index), end=index + 1.0, text=f"s{index}", words=None)

    answers = iter([False, True])
    results, paused = model_segment_processing.consume_until_pause(
        segments(), SimpleNamespace(duration=4.0, language="en"), "transcribe", diarize=False, pause_pending=lambda: next(answers, False)
    )
    assert paused is True
    assert [seg["text"] for seg in results] == ["s0", "s1"]
    assert pulled == [0, 1], "the third segment was never requested from the worker"


def test_consume_until_pause_returns_everything_when_nothing_asks(monkeypatch):
    """No pause: the same segments consume_segments would return, and paused is False."""
    _quiet(monkeypatch)
    segments = [SimpleNamespace(start=0.0, end=1.0, text="a", words=None), SimpleNamespace(start=1.0, end=2.0, text="b", words=None)]
    results, paused = model_segment_processing.consume_until_pause(
        segments, SimpleNamespace(duration=2.0, language="en"), "transcribe", diarize=False, pause_pending=lambda: False
    )
    assert paused is False
    assert [seg["text"] for seg in results] == ["a", "b"]
