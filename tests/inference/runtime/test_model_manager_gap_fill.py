"""Tests for gap-fill: coverage-gap re-transcription for code-switched audio."""

from types import SimpleNamespace
from unittest import mock

from modules.inference.runtime import gap_filling


class TestSanitizeGapSegments:
    """_sanitize_gap_segments discards padding artifacts and clamps overrun timestamps."""

    def test_empty_text_segment_is_dropped(self):
        # The exact shape found on hardware: mix_en_es.wav's 0.64s gap slice produced a
        # segment with end=~30s (Whisper's internal processing window) and no text.
        segments = [{"start": 0.0, "end": 29.98, "text": ""}]
        assert gap_filling._sanitize_gap_segments(segments, slice_duration=0.64) == []

    def test_whitespace_only_text_segment_is_dropped(self):
        segments = [{"start": 0.0, "end": 29.98, "text": "   "}]
        assert gap_filling._sanitize_gap_segments(segments, slice_duration=0.64) == []

    def test_real_segment_end_clamped_to_slice_duration(self):
        segments = [{"start": 0.0, "end": 29.98, "text": "hello"}]
        result = gap_filling._sanitize_gap_segments(segments, slice_duration=0.64)
        assert result == [{"start": 0.0, "end": 0.64, "text": "hello"}]

    def test_segment_within_slice_duration_is_unchanged(self):
        segments = [{"start": 0.1, "end": 0.5, "text": "hi"}]
        result = gap_filling._sanitize_gap_segments(segments, slice_duration=0.64)
        assert result == [{"start": 0.1, "end": 0.5, "text": "hi"}]

    def test_segment_clamped_to_zero_length_is_dropped(self):
        # start already at or past slice_duration: clamping end down to slice_duration
        # would make end <= start, which is not a real segment.
        segments = [{"start": 0.64, "end": 29.98, "text": "artifact"}]
        assert gap_filling._sanitize_gap_segments(segments, slice_duration=0.64) == []

    def test_multiple_segments_filtered_and_clamped_independently(self):
        segments = [
            {"start": 0.0, "end": 0.3, "text": "first"},
            {"start": 0.3, "end": 29.98, "text": ""},
            {"start": 0.3, "end": 29.98, "text": "second"},
        ]
        result = gap_filling._sanitize_gap_segments(segments, slice_duration=0.64)
        assert result == [
            {"start": 0.0, "end": 0.3, "text": "first"},
            {"start": 0.3, "end": 0.64, "text": "second"},
        ]

    def test_empty_input_returns_empty(self):
        assert gap_filling._sanitize_gap_segments([], slice_duration=1.0) == []


class TestFillLanguageGapsWiring:
    """The entry point must reach the engine with the caller's settings intact.

    The decode settings travel as one ``options`` mapping and the preemption hook is passed
    in rather than imported, so this pins both: a gap re-transcribed with the wrong task,
    or one that ignores preemption, would still produce plausible-looking output.
    """

    def _run(self, *, gaps, options=None):
        model = mock.MagicMock()
        model.detect_language.return_value = ("es", 0.91, [])
        model.transcribe.return_value = (
            iter([]),
            SimpleNamespace(language="es", language_probability=0.91, duration=2.0),
        )
        preempted = {"n": 0}

        def preemption_check():
            preempted["n"] += 1

        segments, segment_languages = [], []
        with (
            mock.patch.object(gap_filling.vad, "decode_audio", return_value=[0.0] * 16000),
            mock.patch.object(gap_filling.vad, "get_speech_timestamps", return_value=[{"start": 0.0, "end": 2.0}]),
            mock.patch.object(gap_filling.vad, "extract_slice_to_file", return_value="/tmp/slice.wav"),
            mock.patch.object(gap_filling.language_detection_core, "find_uncovered_speech_gaps", return_value=gaps),
            mock.patch.object(gap_filling, "consume_transcription_segments", return_value=[]),
            mock.patch.object(gap_filling.os, "remove"),
        ):
            result = gap_filling.fill_language_gaps(
                model,
                "/tmp/clip.wav",
                segments,
                segment_languages,
                options=options or {"task": "transcribe", "initial_prompt": None, "vad_filter": True, "word_timestamps": False},
                duration_sec=10.0,
                unit_id="cuda:0",
                preemption_check=preemption_check,
            )
        return model, preempted, result

    def test_no_gaps_means_no_engine_calls(self):
        model, preempted, (segments, languages) = self._run(gaps=[])

        assert model.transcribe.call_count == 0
        assert (segments, languages) == ([], [])
        assert preempted["n"] == 0

    def test_the_callers_decode_settings_reach_the_engine(self):
        options = {"task": "translate", "initial_prompt": "context", "vad_filter": False, "word_timestamps": True}
        model, _preempted, _result = self._run(gaps=[{"start": 1.0, "end": 3.0}], options=options)

        kwargs = model.transcribe.call_args.kwargs
        assert kwargs["task"] == "translate"
        assert kwargs["initial_prompt"] == "context"
        assert kwargs["vad_filter"] is False
        assert kwargs["word_timestamps"] is True
        assert kwargs["language"] == "es", "the gap must be decoded in its own detected language"

    def test_preemption_is_checked_per_gap(self):
        _model, preempted, _result = self._run(gaps=[{"start": 1.0, "end": 3.0}, {"start": 5.0, "end": 6.0}])

        assert preempted["n"] >= 2, "each gap must offer a preemption point before decoding"

    def test_a_failed_vad_scan_leaves_the_transcript_untouched(self):
        model = mock.MagicMock()
        segments, languages = [{"start": 0.0, "end": 1.0, "text": "hola"}], []
        with mock.patch.object(gap_filling.vad, "decode_audio", side_effect=RuntimeError("no audio")):
            result = gap_filling.fill_language_gaps(
                model,
                "/tmp/clip.wav",
                segments,
                languages,
                options={"task": "transcribe", "initial_prompt": None, "vad_filter": True, "word_timestamps": False},
                duration_sec=10.0,
                unit_id="cuda:0",
                preemption_check=lambda: None,
            )

        assert result == (segments, languages)
        assert model.transcribe.call_count == 0
