"""Re-transcribing speech the first pass left uncovered, one gap at a time.

Forcing a single language across a whole file makes the decoder stop emitting segments once
it meets audio that does not match, which is the recorded dropped-code-switched-legs defect.
Gap-fill is the repair: VAD says where speech is, the segments say what was covered, and the
difference is re-detected and re-transcribed on its own slice.

Everything below is mocked -- no model, no ffmpeg, no VAD. The behaviour under test is the
orchestration: what gets skipped, what gets merged, and what the timestamps mean afterwards.
"""

# The unit under test is the module's internals; reaching them by name is the point.

from unittest import mock

import pytest

from modules.inference.runtime import gap_filling


@pytest.fixture(name="options")
def _options():
    return {"task": "transcribe", "initial_prompt": None, "vad_filter": True, "word_timestamps": False}


def _model(*, language=("es", 0.87, []), segments=None):
    """A model whose detection and transcription results the test dictates."""
    model = mock.MagicMock()
    model.detect_language.return_value = language
    info = mock.MagicMock(duration=5.0, language=language[0], language_probability=language[1])
    model.transcribe.return_value = (iter(segments or []), info)
    return model


def _fill(model, segments, segment_languages, options, *, gaps, consumed=None, slice_path="/tmp/slice.wav"):
    """Run fill_language_gaps with VAD, slicing and segment consumption stubbed out."""
    with (
        mock.patch.object(gap_filling, "_scan_speech", return_value=[{"start": 0, "end": 1}]),
        mock.patch.object(gap_filling.language_detection_core, "find_uncovered_speech_gaps", return_value=gaps),
        mock.patch.object(gap_filling.vad, "extract_slice_to_file", return_value=slice_path),
        mock.patch.object(gap_filling, "consume_transcription_segments", return_value=list(consumed or [])),
        mock.patch.object(gap_filling.os, "remove") as removed,
    ):
        result = gap_filling.fill_language_gaps(
            model,
            "/tmp/clip.wav",
            segments,
            segment_languages,
            options=options,
            duration_sec=60.0,
            unit_id="cuda:0",
            preemption_check=lambda: None,
        )
    return result, removed


class TestNothingToDo:
    """The common case: a well-behaved single-language transcript costs one VAD pass."""

    def test_a_failed_vad_scan_leaves_the_transcript_untouched(self, options):
        """ "The scan failed" is not "there is no speech", and must not be read as either.

        Returning the inputs unchanged is the only honest answer: without VAD there is no
        evidence about what is uncovered, so inventing gaps would re-transcribe arbitrary audio.
        """
        segments = [{"start": 0.0, "end": 5.0, "text": "hello"}]
        languages = [{"start": 0.0, "end": 5.0, "language": "en", "confidence": 0.9}]
        model = _model()

        with mock.patch.object(gap_filling, "_scan_speech", return_value=None):
            out_segments, out_languages = gap_filling.fill_language_gaps(
                model,
                "/tmp/clip.wav",
                segments,
                languages,
                options=options,
                duration_sec=5.0,
                unit_id="cpu",
                preemption_check=lambda: None,
            )

        assert out_segments is segments
        assert out_languages is languages
        model.detect_language.assert_not_called()

    def test_no_gaps_means_no_further_work(self, options):
        model = _model()
        segments = [{"start": 0.0, "end": 5.0, "text": "hello"}]

        _result, _removed = _fill(model, segments, [], options, gaps=[])

        model.detect_language.assert_not_called()
        model.transcribe.assert_not_called()


class TestFillingOneGap:
    """The repair itself: detect, transcribe, offset, merge."""

    def test_a_gap_is_detected_and_transcribed_on_its_own_slice(self, options):
        """The slice is passed as a path, never as decoded samples.

        IsolatedEngine runs in a worker subprocess and can only be handed a path across that
        boundary -- an array raises TypeError there, which is how this was found on hardware.
        """
        model = _model(language=("es", 0.87, []))
        consumed = [{"start": 0.5, "end": 1.5, "text": "hola"}]

        (segments, languages), _removed = _fill(
            model,
            [{"start": 0.0, "end": 4.0, "text": "hello"}],
            [],
            options,
            gaps=[{"start": 4.0, "end": 6.0}],
            consumed=consumed,
        )

        model.detect_language.assert_called_once_with("/tmp/slice.wav")
        assert model.transcribe.call_args.args[0] == "/tmp/slice.wav"
        assert model.transcribe.call_args.kwargs["language"] == "es"
        # Slice-relative timestamps are shifted to file-relative, or every filled gap would
        # claim to start near zero and collide with the first pass's segments.
        assert [(s["start"], s["end"]) for s in segments] == [(0.0, 4.0), (4.5, 5.5)]
        assert languages == [{"start": 4.0, "end": 6.0, "language": "es", "confidence": 0.87}]

    def test_merged_segments_are_re_sorted_into_the_transcript(self, options):
        model = _model()
        (segments, _languages), _removed = _fill(
            model,
            [{"start": 10.0, "end": 12.0, "text": "later"}],
            [],
            options,
            gaps=[{"start": 2.0, "end": 4.0}],
            consumed=[{"start": 0.0, "end": 1.0, "text": "earlier"}],
        )

        assert [s["text"] for s in segments] == ["earlier", "later"]

    def test_the_slice_file_is_always_removed(self, options):
        _result, removed = _fill(
            _model(),
            [],
            [],
            options,
            gaps=[{"start": 1.0, "end": 2.0}],
            consumed=[{"start": 0.0, "end": 0.5, "text": "x"}],
        )
        removed.assert_called_once_with("/tmp/slice.wav")


class TestGapsThatAreSkipped:
    """A gap that cannot be filled must not cost the gaps that can."""

    def test_a_failed_detection_skips_the_gap_without_transcribing(self, options):
        model = _model()
        model.detect_language.side_effect = RuntimeError("detector exploded")

        (segments, languages), removed = _fill(model, [], [], options, gaps=[{"start": 1.0, "end": 2.0}])

        model.transcribe.assert_not_called()
        assert (segments, languages) == ([], [])
        assert removed.call_count == 1, "the slice is still cleaned up"

    def test_an_empty_detected_language_skips_the_gap(self, options):
        model = _model(language=("", 0.0, []))

        _result, _removed = _fill(model, [], [], options, gaps=[{"start": 1.0, "end": 2.0}])

        model.transcribe.assert_not_called()

    def test_a_failed_transcription_skips_the_gap(self, options):
        model = _model()
        model.transcribe.side_effect = RuntimeError("decoder exploded")

        (segments, languages), _removed = _fill(model, [], [], options, gaps=[{"start": 1.0, "end": 2.0}])

        assert (segments, languages) == ([], [])

    def test_a_gap_that_cannot_be_sliced_is_skipped(self, options):
        model = _model()
        with (
            mock.patch.object(gap_filling, "_scan_speech", return_value=[{"start": 0, "end": 1}]),
            mock.patch.object(gap_filling.language_detection_core, "find_uncovered_speech_gaps", return_value=[{"start": 1.0, "end": 2.0}]),
            mock.patch.object(gap_filling.vad, "extract_slice_to_file", side_effect=OSError("no space")),
        ):
            segments, languages = gap_filling.fill_language_gaps(
                model,
                "/tmp/clip.wav",
                [],
                [],
                options=options,
                duration_sec=5.0,
                unit_id="cpu",
                preemption_check=lambda: None,
            )

        assert (segments, languages) == ([], [])
        model.detect_language.assert_not_called()

    def test_a_non_numeric_confidence_does_not_abort_the_remaining_gaps(self, options):
        """An isolated worker can hand back None here, and round(None, 4) is a TypeError.

        Raised after the gap's segments were already merged, it aborted every remaining gap
        over a value used for nothing but reporting.
        """
        model = _model(language=("fr", None, []))

        (segments, languages), _removed = _fill(
            model,
            [],
            [],
            options,
            gaps=[{"start": 1.0, "end": 2.0}],
            consumed=[{"start": 0.0, "end": 0.5, "text": "bonjour"}],
        )

        assert len(segments) == 1
        assert languages == [{"start": 1.0, "end": 2.0, "language": "fr", "confidence": 0.0}]


# _sanitize_gap_segments itself is covered in tests/inference/runtime/test_model_manager_gap_fill.py
# (TestSanitizeGapSegments), which holds the fuller set: empty and whitespace text, the end
# clamp, an already-in-range segment, the zero-length drop, and a mixed list. This module had
# three of those verbatim; two copies of one contract meant a behaviour change had to be
# found in both places, and nothing said which was authoritative.


def test_word_timestamps_are_offset_with_their_segment():
    """Word timings are file-relative too, or highlighting lands on the wrong words."""
    segments = [{"start": 1.0, "end": 2.0, "words": [{"start": 1.0, "end": 1.5, "word": "hi"}]}]
    gap_filling._offset_segment_times(segments, 10.0)

    assert (segments[0]["start"], segments[0]["end"]) == (11.0, 12.0)
    assert (segments[0]["words"][0]["start"], segments[0]["words"][0]["end"]) == (11.0, 11.5)


class TestClampingWordTimesToTheirSlice:
    """A word may not claim to end after the audio it was decoded from.

    The decoder occasionally emits a timestamp past the end of its slice; left alone those
    times are offset into the parent timeline and produce words that overlap the following
    segment or run past the end of the media.
    """

    def test_times_beyond_the_slice_are_pulled_back_to_its_end(self):
        seg = {"words": [{"start": 1.0, "end": 9.9}, {"start": 9.5, "end": 12.0}]}
        gap_filling._clamp_word_times(seg, 5.0)
        assert seg["words"] == [{"start": 1.0, "end": 5.0}, {"start": 5.0, "end": 5.0}]

    def test_times_inside_the_slice_are_left_alone(self):
        seg = {"words": [{"start": 0.5, "end": 2.0}]}
        gap_filling._clamp_word_times(seg, 5.0)
        assert seg["words"] == [{"start": 0.5, "end": 2.0}]

    def test_a_segment_without_words_is_not_an_error(self):
        for seg in ({}, {"words": None}, {"words": []}):
            gap_filling._clamp_word_times(seg, 5.0)
