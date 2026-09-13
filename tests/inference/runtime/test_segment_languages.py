"""Where each segment's reported language comes from, and when it is worth measuring.

The recorded `language` defect: a code-switched response could say which languages a file
contained but not where each was spoken, because no shipped engine reports a per-segment
language -- faster-whisper picks one per decode window and never says which. Decoding by
speech region gives clips that map onto segments, so the language can be measured per clip
instead of read back from a transcription that does not carry it.

It costs an encoder pass per region, paid before the decode: the labels are what group
regions into runs of one language, which is what keeps a single-language film from being
decoded one line at a time and fracturing.
"""

import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from modules.inference.runtime import segment_languages


def _info(language="es", duration=120.0, probability=0.9):
    return SimpleNamespace(language=language, duration=duration, language_probability=probability)


def _segments(*spans):
    return [{"start": s, "end": e, "text": "x"} for s, e in spans]


class _RegionDetector:
    """An isolated-style engine, as a real class.

    Not a MagicMock: the dispatch deliberately reads the method off ``type(model)`` because a
    MagicMock answers yes to any attribute asked of an *instance*, which would route every
    mocked engine down the worker path. Its class does not -- so a mock here would silently
    test the in-process branch instead.
    """

    EVENTS = (
        {"start": 0.0, "end": 10.0, "result": {"detected_language": "es", "confidence": 0.91}},
        {"start": 12.0, "end": 20.0, "result": {"detected_language": "fr", "confidence": 0.88}},
    )

    def __init__(self):
        self.calls = 0
        self.fail_with = None

    def detect_language_regions(self, audio_path, regions):
        """Yield the scripted detection for each region that was actually asked for.

        Filtered by the requested regions, as the real worker answers only what it is
        handed: a test that asked for one region and accepted two labels was passing on an
        event nobody requested.
        """
        self.calls += 1
        if self.fail_with:
            raise self.fail_with
        asked = {(float(r["start"]), float(r["end"])) for r in regions}
        return iter([event for event in self.EVENTS if (event["start"], event["end"]) in asked])


@pytest.fixture(name="detector")
def _detector():
    return _RegionDetector()


BOTH_REGIONS = [{"start": 0.0, "end": 10.0}, {"start": 12.0, "end": 20.0}]


class TestWhenDetectionIsPaidFor:
    """Labelling runs wherever clips do, and is not gated on a whole-file language signal.

    Every whole-file signal measured reads a genuinely code-switching film as monolingual: the
    montage samples a handful of windows across two hours and the localised foreign passages
    fall between them, so the top language holds 0.92-1.00 on real film either way. Gating on
    one would label the synthetic fixture and miss every real film it exists for.
    """

    def test_clips_are_measured(self, detector):
        """One detection per clip asked for, and none for a clip that was not."""
        spans = segment_languages.for_clips(detector, "clip.wav", BOTH_REGIONS)
        assert [s["language"] for s in spans] == ["es", "fr"]
        spans = segment_languages.for_clips(detector, "clip.wav", [{"start": 0.0, "end": 10.0}])
        assert [s["language"] for s in spans] == ["es"]

    def test_a_clip_the_detector_cannot_name_is_left_unlabelled(self, detector):
        """Music, noise or a half-second of breath: better no span than an invented one."""
        detector.EVENTS = (
            {"start": 0.0, "end": 10.0, "result": {"detected_language": None, "confidence": 0.0}},
            {"start": 12.0, "end": 20.0, "result": {"detected_language": "fr", "confidence": 0.88}},
        )
        spans = segment_languages.for_clips(detector, "clip.wav", BOTH_REGIONS)
        assert [s["language"] for s in spans] == ["fr"]

    def test_no_clips_means_nothing_to_label(self, detector):
        """An empty clip list yields no spans and no detector calls."""
        assert not segment_languages.for_clips(detector, "clip.wav", [])

    def test_the_switch_no_longer_stops_the_measurement(self, detector, monkeypatch):
        """The per-region languages are what group regions into decode runs; without them
        decoding by region fractures monolingual film. The switch now governs reporting only."""
        monkeypatch.setattr(segment_languages.config, "ASR_SEGMENT_LANGUAGES", False)
        spans = segment_languages.for_clips(detector, "clip.wav", BOTH_REGIONS)
        assert [s["language"] for s in spans] == ["es", "fr"]
        assert detector.calls == 1

    def test_a_detection_failure_costs_the_label_and_nothing_else(self, detector):
        """A missing label is a lost detail; losing the transcription over it would not be."""
        detector.fail_with = RuntimeError("worker gone")
        assert not segment_languages.for_clips(detector, "clip.wav", [{"start": 0.0, "end": 10.0}])

    def test_regions_go_to_the_worker_in_chunks_and_the_check_runs_between_them(self, detector):
        """The deadlock measured on the NUC: a preemption check that blocks *inside* a stream
        keeps the worker channel from the priority task that asked for the pause. Regions are
        sent in chunks of DETECTION_CHUNK, each consumed to the end, and the check runs only
        once no stream is open."""
        regions = [{"start": float(i), "end": float(i) + 0.5} for i in range(0, 40)]
        events = tuple({"start": r["start"], "end": r["end"], "result": {"detected_language": "es", "confidence": 0.9}} for r in regions)
        open_streams: list[bool] = []

        class _Detector(_RegionDetector):
            """Tracks whether a stream is open when the check runs."""

            EVENTS = events
            streaming = False

            def detect_language_regions(self, audio_path, regions):
                self.streaming = True
                yield from super().detect_language_regions(audio_path, regions)
                self.streaming = False

        tracked = _Detector()
        spans = segment_languages.for_clips(tracked, "clip.wav", regions, preemption_check=lambda: open_streams.append(tracked.streaming))
        assert len(spans) == 40
        assert tracked.calls == 3, "40 regions in chunks of 16"
        assert open_streams == [False, False, False], "one check per chunk, none with a stream open"

    def test_what_the_preemption_callback_raises_is_not_swallowed(self, detector):
        """The callback is the scheduler's, and runs between detections. Its exception used to
        fall into the detection fallback, which logged it and kept labelling audio the task
        had been told to abandon."""

        def abandon():
            raise RuntimeError("task cancelled")

        with pytest.raises(RuntimeError, match="task cancelled"):
            segment_languages.for_clips(detector, "clip.wav", BOTH_REGIONS, preemption_check=abandon)


class TestStampingSegments:
    """A segment is labelled by where it sits, and never by a guess."""

    SPANS = [
        {"start": 0.0, "end": 10.0, "language": "es", "confidence": 0.9},
        {"start": 12.0, "end": 20.0, "language": "fr", "confidence": 0.9},
    ]

    def test_each_segment_takes_the_language_of_the_clip_it_sits_in(self):
        """Stamping is by the clip a segment's midpoint falls in."""
        segs = _segments((1.0, 3.0), (13.0, 15.0))
        segment_languages.stamp(segs, self.SPANS, "en")
        assert [s["language"] for s in segs] == ["es", "fr"]

    def test_a_segment_straddling_two_clips_is_placed_by_its_midpoint(self):
        """Midpoint is the rule the long-form test uses, so labels and text agree on placement.

        A segment spanning 9-13s has its midpoint at 11.0, inside the silence between clips, so
        no clip contains it and it keeps the file language rather than being assigned to
        whichever clip it happens to touch first.
        """
        segs = _segments((9.0, 13.0))
        segment_languages.stamp(segs, self.SPANS, "en")
        assert segs[0]["language"] == "en"

    def test_a_segment_outside_every_clip_keeps_the_file_language(self):
        """A coarse label that is true beats a precise one that is invented."""
        segs = _segments((40.0, 45.0))
        segment_languages.stamp(segs, self.SPANS, "en")
        assert segs[0]["language"] == "en"

    def test_every_segment_is_labelled_even_with_nothing_measured(self):
        """With no runs, every segment still carries the file-level language."""
        segs = _segments((1.0, 3.0), (5.0, 7.0))
        segment_languages.stamp(segs, [], "de")
        assert [s["language"] for s in segs] == ["de", "de"]


class TestWhatGetsReported:
    """`segment_languages` in the response: per run when measured, file-level otherwise."""

    RUNS = [
        {"start": 0.0, "end": 10.0, "language": "es", "confidence": 0.91},
        {"start": 12.0, "end": 20.0, "language": "fr", "confidence": 0.88},
    ]

    def test_measured_runs_are_reported_and_stamped(self):
        """The runs are what is reported; segments inside them take their language."""
        segs = _segments((1.0, 3.0), (13.0, 15.0))
        spans = segment_languages.spans_for(self.RUNS, segs, _info())
        assert [s["language"] for s in spans] == ["es", "fr"]
        assert [s["language"] for s in segs] == ["es", "fr"]

    def test_the_file_level_span_is_the_fallback(self):
        """What the API reported before any of this -- still true, just coarse."""
        segs = _segments((1.0, 3.0))
        spans = segment_languages.spans_for([], segs, _info(language="it", duration=99.5))
        assert spans == [{"start": 0.0, "end": 99.5, "language": "it", "confidence": 0.9}]
        assert segs[0]["language"] == "it"

    def test_a_run_without_a_language_is_not_reported(self):
        """The detector failed and every region became a languageless run: report the file."""
        runs = [{"start": 0.0, "end": 5.0, "language": None, "confidence": 0.0}]
        spans = segment_languages.spans_for(runs, _segments((1.0, 2.0)), _info(language="de", duration=5.0))
        assert [s["language"] for s in spans] == ["de"]

    def test_switching_reporting_off_keeps_the_file_span_only(self, monkeypatch):
        """The runs still shaped the decode; the response just does not itemise them, and the
        segments come back in the shape they had before this release -- no language key."""
        monkeypatch.setattr(segment_languages.config, "ASR_SEGMENT_LANGUAGES", False)
        segs = _segments((1.0, 3.0), (13.0, 15.0))
        spans = segment_languages.spans_for(self.RUNS, segs, _info(language="es"))
        assert [s["language"] for s in spans] == ["es"]
        assert all("language" not in s for s in segs)


class _InProcessEngine:
    """An engine that is not behind a worker, so detection happens in this process."""

    def detect_language(self, audio):
        """Answer from the sample count, so a test can tell the slices apart."""
        return ("es" if len(audio) > 5 else "fr", 0.9, [])


class TestTheInProcessPath:
    """With no worker to stream from, the audio is decoded and sliced here.

    Only reachable for an engine whose *class* lacks the streaming method -- the same
    ``type(model)`` check the isolated dispatch uses, so a mock cannot fake its way in.
    """

    def _detect(self, regions, audio_len=32000):
        fake_vad = mock.MagicMock()
        fake_vad.decode_audio.return_value = list(range(audio_len))
        fake_core = mock.MagicMock()
        fake_core.run_language_detection_core.side_effect = lambda model, chunk, skip_vad: {
            "detected_language": "es" if len(chunk) > 8000 else "fr",
            "confidence": 0.8,
        }
        with mock.patch.dict(
            sys.modules,
            {
                "modules.inference.pipeline.vad": fake_vad,
                "modules.inference.pipeline.language_detection_core": fake_core,
            },
        ):
            return segment_languages.for_clips(_InProcessEngine(), "clip.wav", regions), fake_core

    def test_each_region_is_sliced_and_detected(self):
        """The worker slices the decoded file once per region, no montage -- and each slice
        starts at its own region's offset: the second one at sample 16000, not at zero."""
        spans, core = self._detect([{"start": 0.0, "end": 1.0}, {"start": 1.0, "end": 1.2}])
        assert [s["language"] for s in spans] == ["es", "fr"]
        chunks = [list(call.args[1]) for call in core.run_language_detection_core.call_args_list]
        assert chunks[0] == list(range(0, 16000))
        assert chunks[1] == list(range(16000, 19200))

    def test_the_region_is_not_vad_scanned_again(self):
        """The region already *is* VAD output; re-scanning a short slice can report no speech
        for audio the decoder is about to transcribe."""
        _, core = self._detect([{"start": 0.0, "end": 1.0}])
        assert core.run_language_detection_core.call_args.kwargs["skip_vad"] is True

    def test_a_region_past_the_end_of_the_audio_is_skipped(self):
        """A clip can outrun the decoded length by a rounding error; that is not an error."""
        spans, _ = self._detect([{"start": 10.0, "end": 11.0}], audio_len=1000)
        assert not spans

    def test_a_region_overrunning_the_audio_is_reported_where_it_was_cut(self):
        """VAD pads its last region past the file; the span says where the audio ends."""
        spans, _ = self._detect([{"start": 1.0, "end": 3.0}], audio_len=32000)
        assert (spans[0]["start"], spans[0]["end"]) == (1.0, 2.0)

    def test_a_region_starting_before_the_audio_is_clamped_to_it(self):
        """A negative start would slice from the end of the array; it is clamped to zero."""
        spans, core = self._detect([{"start": -0.5, "end": 3.0}], audio_len=32000)
        assert (spans[0]["start"], spans[0]["end"]) == (0.0, 2.0)
        assert list(core.run_language_detection_core.call_args.args[1]) == list(range(0, 32000))
