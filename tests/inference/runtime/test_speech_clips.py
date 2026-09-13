"""Which spans of audio get handed to the decoder as clips, and when none are.

A decode window is the unit of language commitment, and left to itself it is 30 seconds of
whatever VAD compacted together -- so on audio that switches faster than that, per-window
re-detection has nothing useful to choose between. That is the recorded `windows` defect, 67
of 118 windows below the overlap bar; clips capping each window at one speech region measured
0 of 118 (RTX 3080, three identical runs).

Both halves of the pairing are pinned here, because clips *without* per-window re-detection
measured 82 of 118 -- worse than shipping today. Nothing should ever apply one without the
other.
"""

from unittest import mock

import pytest

from modules.inference.runtime import speech_clips


def _regions(*spans):
    return [{"start": s, "end": e} for s, e in spans]


#: The settings the exact clip edges asserted below are computed from, pinned so an operator's
#: environment cannot change what these tests measure.
_PINNED = {
    "ASR_SEGMENT_FIRST": True,
    "ASR_ENGINE": "FASTER-WHISPER",
    "SEGMENT_FIRST_MIN_REGIONS": 2,
    "SEGMENT_SPLIT_PAD_MS": 200,
    "SEGMENT_CLIP_MERGE_GAP_SEC": 0.3,
}


@pytest.fixture(name="vad")
def _vad(monkeypatch):
    """Stub the VAD scan, returning a long two-region file by default, under pinned settings."""
    for name, value in _PINNED.items():
        monkeypatch.setattr(speech_clips.config, name, value)
    with mock.patch.object(speech_clips, "vad") as module:
        module.get_speech_timestamps_from_path.return_value = _regions((1.0, 20.0), (25.0, 40.0))
        yield module


class TestWhenRegionsAreChosen:
    """Clips ride on per-window re-detection, and on an engine that will honour them."""

    def test_an_auto_detected_request_is_decoded_by_speech_region(self, vad):
        """Regions come back with their real edges; padding is applied to the runs later."""
        assert speech_clips.regions_for("clip.wav", True) == _regions((1.0, 20.0), (25.0, 40.0))

    def test_padding_widens_each_clip_without_letting_two_touch_a_word(self, vad):
        """Applied to the decode runs, so a clip boundary never cuts a word."""
        assert speech_clips.pad(_regions((1.0, 20.0), (25.0, 40.0))) == _regions((0.8, 20.2), (24.8, 40.2))

    def test_an_explicitly_requested_language_is_left_alone(self, vad):
        """No re-detection means no reason to reshape the windows, and the caller asked.

        The flag is whether the request may follow a switch, which the orchestration derives:
        false for a transcription in a named language, true for auto-detect and for every
        translation, whose output is English whatever was spoken."""
        assert not speech_clips.regions_for("clip.wav", False)
        vad.get_speech_timestamps_from_path.assert_not_called()

    def test_the_feature_can_be_switched_off(self, vad, monkeypatch):
        """ASR_SEGMENT_FIRST=0 restores the single whole-file decode."""
        monkeypatch.setattr(speech_clips.config, "ASR_SEGMENT_FIRST", False)
        assert not speech_clips.regions_for("clip.wav", True)
        vad.get_speech_timestamps_from_path.assert_not_called()

    def test_an_engine_that_would_ignore_clips_never_receives_them(self, vad, monkeypatch):
        """WhisperX accepts arbitrary kwargs and forwards none, so clips there are a silent
        no-op: the whole file decoded as one span while the pipeline believes otherwise."""
        monkeypatch.setattr(speech_clips.config, "ASR_ENGINE", "WHISPERX")
        assert not speech_clips.regions_for("clip.wav", True)
        vad.get_speech_timestamps_from_path.assert_not_called()

    def test_the_split_uses_its_own_settings_not_the_decode_vad(self, vad):
        """A clip boundary decides where the language may change, not what is transcribed."""
        speech_clips.regions_for("clip.wav", True)
        kwargs = vad.get_speech_timestamps_from_path.call_args.kwargs
        assert kwargs["min_silence_duration_ms"] == speech_clips.config.SEGMENT_SPLIT_MIN_SILENCE_MS

    def test_the_scan_is_asked_for_unpadded_regions(self, vad):
        """The load-bearing argument, and the one that was wrong.

        Silero pads each side by ``speech_pad_ms`` and splits any silence shorter than twice
        that down the middle, so a padded gap is the real silence minus up to 2*pad -- and
        every silence below 2*pad arrives as ~0, indistinguishable from a breath. Merging on
        those gaps read 0.3s and merged at 0.7s of real silence, which fused 7 language
        boundaries on the long-form fixture. Padding is applied after the merge instead.
        """
        speech_clips.regions_for("clip.wav", True)
        assert vad.get_speech_timestamps_from_path.call_args.kwargs["speech_pad_ms"] == 0


class TestRejoiningSplitUtterances:
    """Breath-length pauses are rejoined; turn boundaries are not.

    Silero splits inside an utterance, and the sub-second fragment that produces is where
    per-window language detection is least reliable -- across nine real film excerpts the
    wrongly-rendered segments had a 1.36s median against 2.40s for correct ones.

    The gaps compared here are real silence. Comparing padded ones instead is what broke it:
    the threshold read 0.3s but merged at 0.7s, because the VAD had already taken up to 0.4s
    out of every gap. On the long-form fixture, where every adjacent utterance is a different
    language and the closest are 0.403s apart, that fused 7 language boundaries and put 5 of
    118 windows at an overlap of exactly 0.00. The 0.403s spacing is asserted below as the
    constraint it is.
    """

    def test_a_breath_length_pause_is_rejoined(self, vad):
        """0.2s of real silence is a breath, so the two fragments become one clip."""
        vad.get_speech_timestamps_from_path.return_value = _regions((0.0, 3.0), (3.2, 9.0), (20.0, 40.0))
        assert speech_clips.regions_for("clip.wav", True) == _regions((0.0, 9.0), (20.0, 40.0))

    def test_a_turn_boundary_is_left_alone(self, vad):
        """0.403s is the tightest gap in the long-form fixture, and it changes language."""
        vad.get_speech_timestamps_from_path.return_value = _regions((0.0, 6.0), (6.403, 12.0), (20.0, 40.0))
        assert len(speech_clips.regions_for("clip.wav", True)) == 3

    def test_a_run_of_fragments_collapses_into_one_clip(self, vad):
        """Breath-length splits inside one utterance are one clip again."""
        vad.get_speech_timestamps_from_path.return_value = _regions((0.0, 1.0), (1.1, 2.0), (2.2, 3.0), (3.1, 4.0), (30.0, 45.0))
        assert speech_clips.regions_for("clip.wav", True) == _regions((0.0, 4.0), (30.0, 45.0))

    def test_merging_cannot_lose_audio(self, vad):
        """Every clip must still contain the speech it came from; padding may only widen it."""
        raw = _regions((1.0, 3.0), (3.1, 5.0), (9.0, 40.0))
        vad.get_speech_timestamps_from_path.return_value = raw
        merged = speech_clips.pad(speech_clips.regions_for("clip.wav", True))
        for region in raw:
            assert any(m["start"] <= region["start"] and m["end"] >= region["end"] for m in merged), f"{region} is not inside any clip"
        assert all(m["end"] > m["start"] for m in merged)

    def test_clips_stay_ordered_and_never_overlap(self, vad):
        """``clip_timestamps`` is a flat seek-point list; an overlapping pair makes it
        non-monotonic, and the library then decodes from a point it has already passed. The
        pad is clamped to the midpoint of each gap, which is the rule Silero itself uses."""
        clips = speech_clips.pad(_regions((1.0, 2.0), (2.3, 3.0), (3.05, 9.0)))
        assert all(a["end"] <= b["start"] for a, b in zip(clips, clips[1:]))


class TestWhenTheScanDeclines:
    """Every decline falls through to the ordinary whole-file decode."""

    def test_a_file_with_no_detected_speech(self, vad):
        """Nothing to clip means no clips, not an empty seek list."""
        vad.get_speech_timestamps_from_path.return_value = []
        assert not speech_clips.regions_for("clip.wav", True)

    def test_a_file_with_a_single_region(self, vad):
        """One region is one window already; clips cannot change what it commits to."""
        vad.get_speech_timestamps_from_path.return_value = _regions((0.0, 90.0))
        assert not speech_clips.regions_for("clip.wav", True)

    def test_a_single_long_region_is_still_declined(self, vad):
        """Length is irrelevant; one region commits to one language whatever is done to it."""
        vad.get_speech_timestamps_from_path.return_value = _regions((0.0, 600.0))
        assert not speech_clips.regions_for("clip.wav", True)


class TestShortFilesAreNotExcluded:
    """A short file with two regions is exactly what clips fix, not what they skip.

    There used to be a 30-second floor here, justified by "below one decode window the file is
    already a single window". That describes the file rather than the decode: clips are what
    turn one file into several windows. Both of the recorded code-switched defects are under
    five seconds, and both are decoded as a single window without clips -- `mix_en_es` came
    back as one Spanish segment with the English leg missing entirely, and `mix_zh_en` as one
    Chinese segment with the English half missing. With clips each returns both legs.
    """

    def test_a_five_second_two_language_clip_is_clipped(self, vad):
        """The recorded code-switched defects are under five seconds; a duration floor would skip them."""
        vad.get_speech_timestamps_from_path.return_value = _regions((0.1, 2.3), (2.6, 4.9))
        assert speech_clips.regions_for("clip.wav", True) == _regions((0.1, 2.3), (2.6, 4.9))
        assert speech_clips.pad(_regions((0.1, 2.3), (2.6, 4.9))) == _regions((0.0, 2.45), (2.45, 5.1))

    def test_the_region_count_is_the_only_bar(self, vad, monkeypatch):
        """One region is one window already; two is where clipping starts to matter."""
        monkeypatch.setattr(speech_clips.config, "SEGMENT_FIRST_MIN_REGIONS", 3)
        vad.get_speech_timestamps_from_path.return_value = _regions((0.1, 2.3), (2.6, 4.9))
        assert not speech_clips.regions_for("clip.wav", True)


class TestTheDecodeOptions:
    """The library reads a flat, ordered, even-length [start, end, ...] list."""

    def test_regions_are_flattened_in_order(self):
        """faster-whisper takes a flat, monotonic [start, end, start, end, ...] list."""
        options = speech_clips.decode_options(_regions((0.0, 10.0), (12.5, 22.0), (30.0, 45.0)))
        clips = options["clip_timestamps"]
        assert clips == [0.0, 10.0, 12.5, 22.0, 30.0, 45.0]
        assert clips == sorted(clips), "an out-of-order list makes the decoder seek backwards"
        assert len(clips) % 2 == 0, "an odd-length list decodes the whole file as one clip"

    def test_the_internal_vad_is_turned_off(self):
        """The clips are the VAD output; leaving the filter on runs Silero twice."""
        assert speech_clips.decode_options(_regions((0.0, 10.0), (12.0, 20.0)))["vad_filter"] is False

    def test_no_regions_means_no_options_at_all(self):
        """An empty clip list reaches faster-whisper as an odd seek list and decodes the whole
        file as one clip -- success-shaped, and exactly the behaviour clips exist to replace."""
        assert not speech_clips.decode_options([])
