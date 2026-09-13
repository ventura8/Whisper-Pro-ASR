"""One decoder call per language, each told which language it is decoding.

The runs carry the language the pre-decode detection settled on with hysteresis. These tests
pin that the decoder is *told* that language rather than left to re-detect it from a clip,
that a monolingual file is still exactly one call, that the caller's prompt reaches only the
file's own language, and that the segments of several calls come back as one transcript.
"""

from unittest import mock

import pytest

from modules.inference.runtime import run_decoding


def _run(start, end, language, confidence=0.9, regions=None):
    """A run over one region unless ``regions`` lists several; the clips are the regions."""
    regions = regions or [(start, end)]
    return {
        "start": start,
        "end": end,
        "language": language,
        "confidence": confidence,
        "regions": [{"start": a, "end": b} for a, b in regions],
    }


def _segment(start, end, text):
    return mock.MagicMock(start=start, end=end, text=text, words=None)


OPTIONS = {
    "language": "en",
    "task": "transcribe",
    "beam_size": 5,
    "initial_prompt": "Notes from the meeting.",
    "word_timestamps": False,
    "vad_parameters": {"min_silence_duration_ms": 500, "threshold": 0.5},
    "multilingual": True,
    "vad_filter": True,
}


@pytest.fixture(name="consume")
def consume_fixture():
    """The consumption kwargs model_manager passes through, diarization off."""
    return {
        "task": "transcribe",
        "diarize": False,
        "min_speakers": None,
        "max_speakers": None,
        "hf_token": None,
        "unit_id": "cuda:0",
        "processed_path": "clip.wav",
        "preemption_check": lambda: None,
    }


@pytest.fixture(autouse=True)
def _quiet_scheduler():
    with (
        mock.patch.object(run_decoding.model_segment_processing.scheduler, "update_task_progress"),
        mock.patch.object(run_decoding.model_segment_processing.scheduler, "update_task_metadata"),
    ):
        yield


@pytest.fixture(autouse=True)
def _pad_margin(monkeypatch):
    monkeypatch.setattr(run_decoding.speech_clips.config, "SEGMENT_SPLIT_PAD_MS", 200)


def _model(*calls):
    """A model whose transcribe returns the given (segments, info) pairs in order."""
    model = mock.MagicMock()
    model.transcribe.side_effect = list(calls)
    return model


def _info(language, duration=100.0):
    return mock.MagicMock(language=language, language_probability=0.9, duration=duration)


class TestOneLanguage:
    """A monolingual file is one call, told its language, with re-detection off."""

    def test_the_run_language_is_forced_and_redetection_is_off(self, consume):
        """Two runs of one language: one call, told the language, their regions as its clips."""
        model = _model(([_segment(0.0, 4.0, "hello")], _info("en")))
        runs = [_run(0.0, 4.0, "en"), _run(10.0, 14.0, "en")]

        results, info, clips = run_decoding.decode(model, "clip.wav", runs, options=OPTIONS, consume=consume)

        kwargs = model.transcribe.call_args.kwargs
        assert kwargs["language"] == "en"
        assert kwargs["multilingual"] is False
        assert kwargs["clip_timestamps"] == [0.0, 4.2, 9.8, 14.2]
        assert kwargs["vad_filter"] is False
        assert kwargs["initial_prompt"] == "Notes from the meeting."
        assert [seg["text"] for seg in results] == ["hello"]
        assert info.language == "en"
        assert clips == [{"start": 0.0, "end": 4.2}, {"start": 9.8, "end": 14.2}]

    def test_a_run_of_several_lines_is_decoded_line_by_line(self, consume):
        """The run decides the language; the decoder gets one clip per region, never the run.

        A window holding a run's worth of lines dropped some of them on the film fixture; the
        pause between two of its scenes must not be inside a window either.
        """
        model = _model(([_segment(0.0, 1.0, "a")], _info("ru")))
        runs = [_run(0.0, 14.0, "ru", regions=[(0.0, 1.2), (1.6, 2.4), (12.0, 14.0)])]

        _, _, clips = run_decoding.decode(model, "clip.wav", runs, options=OPTIONS, consume=consume)

        assert model.transcribe.call_args.kwargs["clip_timestamps"] == [0.0, 1.4, 1.4, 2.6, 11.8, 14.2]
        assert len(clips) == 3

    def test_a_language_other_than_the_files_still_gets_one_call_but_no_prompt(self, consume):
        """The whole file turned out to be Spanish though the montage said English."""
        model = _model(([_segment(0.0, 4.0, "hola")], _info("es")))

        run_decoding.decode(model, "clip.wav", [_run(0.0, 4.0, "es")], options=OPTIONS, consume=consume)

        kwargs = model.transcribe.call_args.kwargs
        assert kwargs["language"] == "es"
        assert kwargs["initial_prompt"] is None

    def test_diarization_runs_once_over_the_single_call(self, consume):
        """The one-call path is the path that always existed, diarization included -- run
        once over the complete result, after the decode, as the per-language path does."""
        model = _model(([_segment(0.0, 4.0, "hello")], _info("en")))
        with mock.patch.object(run_decoding.model_segment_processing, "diarize_segments", return_value=["diarized"]) as diarized:
            results, _, _ = run_decoding.decode(
                model, "clip.wav", [_run(0.0, 4.0, "en")], options=OPTIONS, consume={**consume, "diarize": True}
            )
        assert results == ["diarized"]
        assert [seg["text"] for seg in diarized.call_args.args[0]] == ["hello"]
        assert diarized.call_args.kwargs["processed_path"] == "clip.wav"


class TestSeveralLanguages:
    """Code-switched audio: a call per language, merged back into one transcript."""

    RUNS = [_run(0.0, 4.0, "en"), _run(5.0, 12.0, "es"), _run(13.0, 16.0, "en")]

    def test_each_language_is_decoded_in_its_own_call(self, consume):
        """English, Spanish, English: two calls, and the transcript back in time order."""
        model = _model(
            ([_segment(0.0, 4.0, "hello"), _segment(13.0, 16.0, "bye")], _info("en")), ([_segment(5.0, 12.0, "hola")], _info("es"))
        )

        results, info, clips = run_decoding.decode(model, "clip.wav", self.RUNS, options=OPTIONS, consume=consume)

        calls = [call.kwargs for call in model.transcribe.call_args_list]
        assert [(c["language"], c["multilingual"]) for c in calls] == [("en", False), ("es", False)]
        assert calls[0]["clip_timestamps"] == [0.0, 4.2, 12.8, 16.2]
        assert calls[1]["clip_timestamps"] == [4.8, 12.2]
        assert [seg["text"] for seg in results] == ["hello", "hola", "bye"]
        assert info.language == "en"
        assert len(clips) == 3

    def test_the_language_with_the_most_speech_goes_first_and_owns_the_info(self, consume):
        """Seven seconds of Spanish against seven of English in two pieces: English first."""
        runs = [_run(0.0, 8.0, "es"), _run(9.0, 12.0, "en")]
        model = _model(([_segment(0.0, 8.0, "hola")], _info("es")), ([_segment(9.0, 12.0, "hi")], _info("en")))

        _, info, _ = run_decoding.decode(model, "clip.wav", runs, options=OPTIONS, consume=consume)

        assert [call.kwargs["language"] for call in model.transcribe.call_args_list] == ["es", "en"]
        assert info.language == "es"

    def test_dominance_is_spoken_seconds_not_the_span_of_a_run(self, consume):
        """A run of two short lines fifty seconds apart spans 52 s but holds 4 s of speech;
        eight seconds of Spanish in one line outranks it and goes first."""
        runs = [_run(0.0, 52.0, "en", regions=[(0.0, 2.0), (50.0, 52.0)]), _run(60.0, 68.0, "es")]
        model = _model(([_segment(60.0, 68.0, "hola")], _info("es")), ([_segment(0.0, 2.0, "hi")], _info("en")))

        _, info, _ = run_decoding.decode(model, "clip.wav", runs, options=OPTIONS, consume=consume)

        assert [call.kwargs["language"] for call in model.transcribe.call_args_list] == ["es", "en"]
        assert info.language == "es"

    def test_the_prompt_reaches_only_the_files_own_language(self, consume):
        """A prompt in one language pulls the decoder toward it; the other run must not see it."""
        model = _model(([_segment(0.0, 4.0, "hello")], _info("en")), ([_segment(5.0, 12.0, "hola")], _info("es")))

        run_decoding.decode(model, "clip.wav", self.RUNS, options=OPTIONS, consume=consume)

        prompts = {call.kwargs["language"]: call.kwargs["initial_prompt"] for call in model.transcribe.call_args_list}
        assert prompts == {"en": "Notes from the meeting.", "es": None}

    def test_progress_is_windowed_per_call_and_diarization_runs_once_at_the_end(self, consume):
        """Speakers are numbered per fingerprinting, so the file is fingerprinted once, merged."""
        model = _model(([_segment(0.0, 4.0, "hello")], _info("en")), ([_segment(5.0, 12.0, "hola")], _info("es")))
        with (
            mock.patch.object(
                run_decoding.resumable_decode.model_segment_processing,
                "consume_until_pause",
                side_effect=[
                    ([{"start": 0.0, "end": 4.0, "text": "hello"}], False),
                    ([{"start": 5.0, "end": 12.0, "text": "hola"}], False),
                ],
            ) as consumed,
            mock.patch.object(run_decoding.model_segment_processing, "diarize_segments", return_value=["diarized"]) as diarized,
        ):
            results, _, _ = run_decoding.decode(model, "clip.wav", self.RUNS, options=OPTIONS, consume={**consume, "diarize": True})

        assert [call.kwargs["progress_window"] for call in consumed.call_args_list] == [(0, 2), (1, 2)]
        # Diarization is pending, so each call reports against the 80% ceiling, not 95%.
        assert all(call.kwargs["diarize"] is True for call in consumed.call_args_list)
        assert diarized.call_count == 1
        assert [seg["text"] for seg in diarized.call_args.args[0]] == ["hello", "hola"]
        assert results == ["diarized"]


class TestWithoutRuns:
    """No runs, or runs without a language: the calls that always existed."""

    def test_no_runs_is_the_plain_whole_file_call(self, consume):
        """An explicit language, another engine, or one region: the options go through as they are."""
        model = _model(([_segment(0.0, 4.0, "hello")], _info("en")))

        results, info, clips = run_decoding.decode(model, "clip.wav", [], options=OPTIONS, consume=consume)

        model.transcribe.assert_called_once_with("clip.wav", **OPTIONS)
        assert [seg["text"] for seg in results] == ["hello"]
        assert info.language == "en"
        assert clips == []

    def test_unlabelled_runs_keep_redetection_on_with_clips(self, consume):
        """The detector failed outright: clips are still worth having, and the decoder picks."""
        model = _model(([_segment(0.0, 4.0, "hello")], _info("en")))
        runs = [_run(0.0, 4.0, None, 0.0), _run(5.0, 9.0, None, 0.0)]

        run_decoding.decode(model, "clip.wav", runs, options=OPTIONS, consume=consume)

        kwargs = model.transcribe.call_args.kwargs
        assert kwargs["language"] == "en"
        assert kwargs["multilingual"] is True
        assert kwargs["clip_timestamps"] == [0.0, 4.2, 4.8, 9.2]
        assert kwargs["initial_prompt"] == "Notes from the meeting."
