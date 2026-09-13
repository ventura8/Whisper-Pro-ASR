"""A paused decode must not hold the worker; it stops, waits, and picks up where it left off.

The deadlock this pins was measured on the Intel NUC (2026-09-12): a priority language
detection asked the single CPU unit to pause, the transcription paused *inside* its segment
stream with the worker channel held, and the priority task then waited for that channel --
forever. The invariant is that the blocking preemption wait runs only when no stream is
open, and that whatever was consumed before the pause is kept.
"""

from types import SimpleNamespace
from unittest import mock

import pytest

from modules.inference.runtime import resumable_decode


class _Stream:
    """A segment stream that knows whether it is open, like the channel's generator does."""

    def __init__(self, segments):
        self._segments = list(segments)
        self.open = True
        self.closed_early = False

    def __iter__(self):
        return self

    def __next__(self):
        if not self._segments:
            self.open = False
            raise StopIteration
        return self._segments.pop(0)

    def close(self):
        """What the channel's generator does on an early close: the lock is released."""
        self.closed_early = self.open
        self.open = False


def _segment(start, end, text="x"):
    return SimpleNamespace(start=start, end=end, text=text, words=None)


class _Model:
    """Answers each transcribe call with the next scripted stream; records the calls."""

    def __init__(self, *streams):
        self.streams = [_Stream(s) for s in streams]
        self.calls = []
        self.info = SimpleNamespace(language="en", language_probability=0.9, duration=60.0)

    def transcribe(self, path, **kwargs):
        """The engine contract: ``(segments, info)``, one stream per call."""
        self.calls.append(kwargs)
        return self.streams.pop(0), self.info


def _consume(pause_pending, preemption_check):
    return {"task": "transcribe", "diarize": False, "pause_pending": pause_pending, "preemption_check": preemption_check}


@pytest.fixture(autouse=True)
def _quiet_scheduler(monkeypatch):
    monkeypatch.setattr(resumable_decode.model_segment_processing, "_update_live_srt_metadata", lambda *a, **k: None)
    monkeypatch.setattr(resumable_decode.model_segment_processing, "_update_segment_progress", lambda *a, **k: None)
    monkeypatch.setattr(resumable_decode.model_segment_processing, "_maybe_log_segment_progress", lambda *a, **k: None)


class TestNoPause:
    """The path every request takes when nothing asks for the unit."""

    def test_a_decode_nobody_interrupts_is_one_call(self):
        """The common case is unchanged: one call, every segment, the call's own info."""
        model = _Model([_segment(0.0, 2.0, "a"), _segment(2.0, 4.0, "b")])
        results, info = resumable_decode.decode(model, "f.wav", {"language": "en"}, None, consume=_consume(lambda: False, mock.Mock()))
        assert [seg["text"] for seg in results] == ["a", "b"]
        assert info is model.info
        assert len(model.calls) == 1


class TestAPause:
    """What a pause costs: the stream, and nothing that was already consumed."""

    def test_the_stream_is_closed_before_the_blocking_wait(self):
        """The invariant: when the preemption flow blocks, the worker channel is free."""
        model = _Model([_segment(0.0, 2.0, "a"), _segment(2.0, 4.0, "b"), _segment(4.0, 6.0, "never")], [_segment(4.0, 6.0, "c")])
        first = model.streams[0]
        pending = iter([False, True])
        open_while_waiting = []
        clips = [{"start": 0.0, "end": 2.0}, {"start": 2.0, "end": 4.0}, {"start": 4.0, "end": 6.0}]
        results, _ = resumable_decode.decode(
            model,
            "f.wav",
            {"language": "en"},
            clips,
            consume=_consume(lambda: next(pending, False), lambda: open_while_waiting.append(first.open)),
        )
        assert open_while_waiting == [False], "the blocking wait ran once, with the stream already closed"
        assert first.closed_early, "the stream was abandoned, which is what releases the channel"
        assert [seg["text"] for seg in results] == ["a", "b", "c"]

    def test_the_resume_starts_where_the_last_consumed_segment_ended(self):
        """Two segments in hand end at 4.0; the second call asks for the clips past 4.0 only."""
        model = _Model([_segment(0.0, 2.0, "a"), _segment(2.0, 4.0, "b"), _segment(4.0, 6.0)], [_segment(4.0, 6.0, "c")])
        pending = iter([False, True])
        clips = [{"start": 0.0, "end": 2.0}, {"start": 2.0, "end": 4.0}, {"start": 4.0, "end": 6.0}]
        resumable_decode.decode(
            model, "f.wav", {"language": "en", "vad_filter": True}, clips, consume=_consume(lambda: next(pending, False), mock.Mock())
        )
        assert len(model.calls) == 2
        assert model.calls[1]["clip_timestamps"] == [4.0, 6.0]
        assert model.calls[1]["vad_filter"] is False, "the remainder is decoded as clips, so the internal VAD stays off"
        assert model.calls[1]["language"] == "en", "the call keeps everything but its span"

    def test_a_clip_straddling_the_pause_point_is_trimmed_not_repeated(self):
        """A segment consumed up to 3.0 inside a 2.0-6.0 clip: the resume decodes 3.0-6.0."""
        model = _Model([_segment(0.0, 3.0, "a"), _segment(3.0, 6.0)], [_segment(3.0, 6.0, "b")])
        pending = iter([True])
        clips = [{"start": 0.0, "end": 2.0}, {"start": 2.0, "end": 6.0}]
        resumable_decode.decode(model, "f.wav", {"language": "en"}, clips, consume=_consume(lambda: next(pending, False), mock.Mock()))
        assert model.calls[1]["clip_timestamps"] == [3.0, 6.0]

    def test_a_whole_file_decode_resumes_on_the_speech_the_vad_finds_after_the_pause(self, monkeypatch):
        """No clips to trim, so the remainder is scanned with the decode VAD's own settings."""
        model = _Model([_segment(0.0, 10.0, "a"), _segment(10.0, 12.0)], [_segment(11.0, 15.0, "b")])
        pending = iter([True])
        scan = mock.Mock(return_value=[{"start": 11.0, "end": 15.0}, {"start": 20.0, "end": 30.0}])
        monkeypatch.setattr(resumable_decode.vad, "get_speech_timestamps_from_path", scan)
        resumable_decode.decode(
            model, "f.wav", {"language": "en", "vad_filter": True}, None, consume=_consume(lambda: next(pending, False), mock.Mock())
        )
        assert scan.call_args.kwargs["start_offset"] == 10.0
        assert scan.call_args.args[1] == resumable_decode.config.VAD_THRESHOLD
        assert model.calls[1]["clip_timestamps"] == [11.0, 15.0, 20.0, 30.0]

    def test_a_pause_with_nothing_left_decodes_to_the_end_of_the_audio(self):
        """The last clip was consumed and then a pause came: no empty seek list reaches the
        decoder (it would read as the whole file); the remainder runs to the file's end."""
        model = _Model([_segment(0.0, 4.0, "a"), _segment(4.0, 5.0)], [])
        pending = iter([True])
        clips = [{"start": 0.0, "end": 4.0}]
        resumable_decode.decode(model, "f.wav", {"language": "en"}, clips, consume=_consume(lambda: next(pending, False), mock.Mock()))
        assert model.calls[1]["clip_timestamps"] == [4.0, 60.0]

    def test_a_pause_right_after_the_first_segment_resumes_on_the_rest(self):
        """The segment in hand is kept; the span after it is asked for again."""
        model = _Model([_segment(0.0, 1.0, "a"), _segment(1.0, 2.0)], [_segment(1.0, 2.0, "b")])
        pending = iter([True])
        clips = [{"start": 0.0, "end": 2.0}]
        results, _ = resumable_decode.decode(
            model, "f.wav", {"language": "en"}, clips, consume=_consume(lambda: next(pending, False), mock.Mock())
        )
        assert model.calls[1]["clip_timestamps"] == [1.0, 2.0]
        assert [seg["text"] for seg in results] == ["a", "b"]


class TestAskingWithoutWaiting:
    """``preemption_pending`` reads the scheduler's flag; it never blocks."""

    def _info(self, unit_id, is_priority):
        return ("t1", 1, unit_id, "active", is_priority, {})

    def test_a_priority_task_is_never_asked_to_pause(self, monkeypatch):
        """Priority tasks are the ones doing the asking."""
        monkeypatch.setattr(resumable_decode, "_get_current_task_info", lambda: self._info("CPU", True))
        monkeypatch.setattr(resumable_decode, "_determine_preemption_needed", mock.Mock(return_value=(True, None, None, None)))
        assert resumable_decode.preemption_pending() is False

    def test_a_task_without_a_unit_has_nothing_to_give_up(self, monkeypatch):
        """Before a unit is assigned there is nothing a priority task could borrow."""
        monkeypatch.setattr(resumable_decode, "_get_current_task_info", lambda: self._info(None, False))
        assert resumable_decode.preemption_pending() is False

    def test_the_scheduler_flag_is_what_is_reported(self, monkeypatch):
        """The same flag the blocking flow reads, read without the flow."""
        monkeypatch.setattr(resumable_decode, "_get_current_task_info", lambda: self._info("CPU", False))
        asked = mock.Mock(return_value=(True, None, None, None))
        monkeypatch.setattr(resumable_decode, "_determine_preemption_needed", asked)
        assert resumable_decode.preemption_pending() is True
        asked.assert_called_once_with("CPU")


class _IsolatedPrep:
    """An isolated preprocessing manager: a raising yield cancels its separation."""

    yield_may_block = False

    def __init__(self, ticks=3):
        self.ticks = ticks
        self.attempts = 0
        self.cancelled = 0

    def preprocess_audio(self, audio_path, force=False, yield_cb=None, stage="Vocal Separation"):
        """Tick ``ticks`` times, asking the callback each time, then answer with a path."""
        self.attempts += 1
        for _ in range(self.ticks):
            try:
                yield_cb()
            except BaseException:
                self.cancelled += 1
                raise
        return audio_path + ".vocals.wav"


class TestSeparationReleasesTheWorker:
    """The second NUC deadlock: a pause taken inside the preprocessing stream."""

    def test_a_pending_pause_abandons_the_separation_waits_and_starts_it_again(self, monkeypatch):
        """The stream is cancelled (the worker's channel freed) before the blocking wait, and the
        separation is run again afterwards -- from the start, on a free worker."""
        prep = _IsolatedPrep()
        pending = iter([False, True])
        monkeypatch.setattr(resumable_decode, "preemption_pending", lambda: next(pending, False))
        waited = []
        result = resumable_decode.separate(
            prep, "/tmp/clip.wav", force=False, stage="LD Prep", preemption_check=lambda: waited.append(prep.cancelled)
        )
        assert result == "/tmp/clip.wav.vocals.wav"
        assert waited == [1], "the wait ran once, after the first separation had been cancelled"
        assert prep.attempts == 2

    def test_no_pause_means_one_separation_and_no_wait(self, monkeypatch):
        """The common case is one worker call, exactly as before."""
        prep = _IsolatedPrep()
        monkeypatch.setattr(resumable_decode, "preemption_pending", lambda: False)
        wait = mock.Mock()
        assert resumable_decode.separate(prep, "/tmp/clip.wav", force=False, stage="LD Prep", preemption_check=wait).endswith(".vocals.wav")
        assert prep.attempts == 1
        assert not wait.called

    def test_an_in_process_manager_keeps_its_blocking_yield(self):
        """No channel to hold, so the blocking check inside the chunk loop is still right."""
        prep = mock.Mock(spec=["preprocess_audio"])
        prep.preprocess_audio.return_value = "/tmp/clip.vocals.wav"
        wait = mock.Mock()
        resumable_decode.separate(prep, "/tmp/clip.wav", force=True, stage="Vocal Separation", preemption_check=wait)
        assert prep.preprocess_audio.call_args.kwargs["yield_cb"] is wait
        assert prep.preprocess_audio.call_args.kwargs["force"] is True
