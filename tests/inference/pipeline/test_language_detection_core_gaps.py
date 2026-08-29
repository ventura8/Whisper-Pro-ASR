"""Tests for language_detection_core.find_uncovered_speech_gaps (code-switch gap-fill)."""

from modules.inference.pipeline import language_detection_core as ldc


class TestFindUncoveredSpeechGaps:
    def test_fully_covered_audio_has_no_gaps(self):
        segments = [{"start": 0.0, "end": 4.3}]
        speech_ts = [{"start": 0.0, "end": 4.3}]
        assert ldc.find_uncovered_speech_gaps(segments, speech_ts, duration_sec=4.3) == []

    def test_second_leg_dropped_entirely_is_one_gap(self):
        # Exactly the shape found on hardware: mix_de_en.wav, 4.3s, German segment
        # produced for 0-1.22s, the English leg (speech per VAD out to 4.3s) missing.
        segments = [{"start": 0.0, "end": 1.22}]
        speech_ts = [{"start": 0.0, "end": 4.2}]
        gaps = ldc.find_uncovered_speech_gaps(segments, speech_ts, duration_sec=4.3, min_gap_sec=0.5)
        assert gaps == [{"start": 1.22, "end": 4.2}]

    def test_gap_shorter_than_minimum_is_ignored(self):
        segments = [{"start": 0.0, "end": 1.0}]
        speech_ts = [{"start": 0.0, "end": 1.2}]
        assert ldc.find_uncovered_speech_gaps(segments, speech_ts, duration_sec=1.2, min_gap_sec=0.5) == []

    def test_gap_between_two_covered_segments(self):
        segments = [{"start": 0.0, "end": 1.0}, {"start": 4.0, "end": 5.0}]
        speech_ts = [{"start": 0.0, "end": 5.0}]
        gaps = ldc.find_uncovered_speech_gaps(segments, speech_ts, duration_sec=5.0)
        assert gaps == [{"start": 1.0, "end": 4.0}]

    def test_speech_region_beyond_last_segment_is_a_gap(self):
        # The "segments stop short of the clip end" defect: a coverage hole at the tail.
        segments = [{"start": 0.0, "end": 10.0}]
        speech_ts = [{"start": 0.0, "end": 10.0}, {"start": 12.0, "end": 20.0}]
        gaps = ldc.find_uncovered_speech_gaps(segments, speech_ts, duration_sec=20.0)
        assert gaps == [{"start": 12.0, "end": 20.0}]

    def test_no_segments_at_all_reports_the_whole_speech_region(self):
        speech_ts = [{"start": 1.0, "end": 3.0}]
        gaps = ldc.find_uncovered_speech_gaps([], speech_ts, duration_sec=3.0)
        assert gaps == [{"start": 1.0, "end": 3.0}]

    def test_no_speech_regions_means_no_gaps_regardless_of_segments(self):
        assert ldc.find_uncovered_speech_gaps([], [], duration_sec=10.0) == []

    def test_a_speech_region_overrunning_the_file_is_clamped_not_discarded(self):
        """A VAD region routinely ends slightly past the decoded duration.

        The old contract dropped the whole gap when that happened, so a clip whose single
        speech region overran by any amount got no gap-fill at all -- silently disabling
        the recovery on exactly the untranscribed-tail case it exists for. The in-range
        portion is kept and the overrun is trimmed to the file's own end.
        """
        segments = []
        speech_ts = [{"start": 0.0, "end": 100.0}]
        assert ldc.find_uncovered_speech_gaps(segments, speech_ts, duration_sec=4.3) == [{"start": 0.0, "end": 4.3}]

    def test_a_region_starting_past_the_end_yields_nothing(self):
        """Clamping must not invent a zero-length or inverted gap past the file's end."""
        speech_ts = [{"start": 9.0, "end": 12.0}]
        assert ldc.find_uncovered_speech_gaps([], speech_ts, duration_sec=4.3) == []

    def test_clamping_still_respects_the_minimum_gap(self):
        """A region trimmed below min_gap_sec is not worth a re-detection pass."""
        speech_ts = [{"start": 4.0, "end": 90.0}]
        assert ldc.find_uncovered_speech_gaps([], speech_ts, duration_sec=4.3, min_gap_sec=0.5) == []

    def test_segment_fully_inside_a_speech_region_splits_it_into_two_gaps(self):
        segments = [{"start": 2.0, "end": 3.0}]
        speech_ts = [{"start": 0.0, "end": 5.0}]
        gaps = ldc.find_uncovered_speech_gaps(segments, speech_ts, duration_sec=5.0)
        assert gaps == [{"start": 0.0, "end": 2.0}, {"start": 3.0, "end": 5.0}]

    def test_out_of_order_segments_still_subtract_correctly(self):
        # _gaps_within_region assumes covered is sorted; find_uncovered_speech_gaps sorts.
        segments = [{"start": 4.0, "end": 5.0}, {"start": 0.0, "end": 1.0}]
        speech_ts = [{"start": 0.0, "end": 5.0}]
        gaps = ldc.find_uncovered_speech_gaps(segments, speech_ts, duration_sec=5.0)
        assert gaps == [{"start": 1.0, "end": 4.0}]
