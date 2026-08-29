"""Language detection across the process boundary, and the paths around it.

The isolated route is the interesting one: an out-of-process engine is handed a *path* and
decodes on the far side, so no audio array ever crosses the pipe. Getting that decision
wrong is not a crash -- it is a silent copy of the whole clip through a pipe, once per
detection -- so the branch that chooses it is worth pinning explicitly.
"""

from __future__ import annotations

from unittest import mock

import pytest

from modules.inference.pipeline import language_detection_core as ldc


class FakeIsolated:
    """Duck-types the proxy, including the class attribute callers detect it by."""

    IS_ISOLATED_ENGINE = True

    def __init__(self, events=None, detection=("es", 0.8, [("es", 0.8)])):
        """Record the construction arguments."""
        self._events = events or []
        self._detection = detection
        self.detect_calls: list = []

    def detect_language_batch(self, audio_path, segment_count):
        """Yield the scripted per-window detection events."""
        self.batch_call = (audio_path, segment_count)
        yield from self._events

    def detect_language(self, audio):
        """Record the input and return the scripted detection."""
        self.detect_calls.append(audio)
        return self._detection


class FakeInProcess:
    """A test double for in process."""

    def __init__(self, detection=("en", 0.9, [("en", 0.9)])):
        """Record the construction arguments."""
        self._detection = detection
        self.detect_calls: list = []

    def detect_language(self, audio):
        """Record the input and return the scripted detection."""
        self.detect_calls.append(audio)
        return self._detection


class TestChoosingTheIsolatedRoute:
    """choosing the isolated route."""

    def test_an_isolated_engine_given_a_path_detects_over_the_pipe(self):
        """An isolated engine given a path detects over the pipe."""
        assert ldc._should_detect_over_the_pipe(FakeIsolated(), "/clip.wav") is True

    def test_an_in_process_engine_never_takes_that_route(self):
        """An in process engine never takes that route."""
        assert ldc._should_detect_over_the_pipe(FakeInProcess(), "/clip.wav") is False

    def test_an_isolated_engine_given_samples_warns_and_falls_back(self, caplog):
        """Silently falling through would hide that the caller decoded for nothing."""
        with caplog.at_level("WARNING"):
            assert ldc._should_detect_over_the_pipe(FakeIsolated(), [0.0, 1.0]) is False
        assert "needs an audio path" in caplog.text


class TestBatchDetection:
    """One small result per window, with progress republished in this process."""

    def _events(self, count):
        """The event kinds the fake channel recorded."""
        return [{"event": "detection", "index": i, "result": {"language": "es"}} for i in range(count)]

    def test_isolated_batch_detection_returns_one_result_per_window(self, monkeypatch):
        """Isolated batch detection returns one result per window."""
        monkeypatch.setattr(ldc.scheduler, "update_task_progress", lambda *a, **k: None)
        model = FakeIsolated(events=self._events(3))

        results = ldc._detect_segments_isolated(model, "/clip.wav", 3)

        assert results == [{"language": "es"}] * 3
        assert model.batch_call == ("/clip.wav", 3)

    def test_progress_is_republished_here_because_the_worker_has_no_scheduler(self, monkeypatch):
        """Progress is republished here because the worker has no scheduler."""
        reported: list[tuple[int, str]] = []
        monkeypatch.setattr(ldc.scheduler, "update_task_progress", lambda pct, stage: reported.append((pct, stage)))

        ldc._detect_segments_isolated(FakeIsolated(events=self._events(2)), "/clip.wav", 2)

        assert [stage for _pct, stage in reported] == ["Inference (1/2 segments)", "Inference (2/2 segments)"]
        assert reported[-1][0] == 95, "the last window should land at the top of this stage's range"

    def test_a_zero_segment_count_does_not_divide_by_zero(self, monkeypatch):
        """A zero segment count does not divide by zero."""
        monkeypatch.setattr(ldc.scheduler, "update_task_progress", lambda *a, **k: None)
        assert ldc._detect_segments_isolated(FakeIsolated(events=[]), "/clip.wav", 0) == []

    def test_the_direct_entry_point_routes_an_isolated_engine_to_the_worker(self, monkeypatch):
        """The direct entry point routes an isolated engine to the worker."""
        monkeypatch.setattr(ldc.scheduler, "update_task_progress", lambda *a, **k: None)
        decoded = []
        monkeypatch.setattr(ldc.vad, "decode_audio", lambda p: decoded.append(p) or [0.0])

        results = ldc.run_batch_language_detection_direct(FakeIsolated(events=self._events(1)), "/clip.wav", 1)

        assert results == [{"language": "es"}]
        assert decoded == [], "decoding in the parent is exactly what the isolated route avoids"

    def test_a_detection_failure_returns_an_empty_list_rather_than_raising(self, monkeypatch, caplog):
        """Detection is best-effort; the caller falls back to the configured language."""

        class Exploding(FakeIsolated):
            """exploding."""

            def detect_language_batch(self, audio_path, segment_count):
                """Yield the scripted per-window detection events."""
                raise RuntimeError("worker died")
                yield  # keeps this a generator

        monkeypatch.setattr(ldc.scheduler, "update_task_progress", lambda *a, **k: None)
        with caplog.at_level("ERROR"):
            assert ldc.run_batch_language_detection_direct(Exploding(), "/clip.wav", 1) == []
        assert "Batch detection failed" in caplog.text


class TestSpeechDurationResolution:
    """speech duration resolution."""

    def test_skip_vad_assumes_a_full_window_without_scanning(self):
        """Skip vad assumes a full window without scanning."""
        assert ldc._resolve_speech_duration("/clip.wav", skip_vad=True) == (30.0, None)

    def test_no_speech_returns_the_neutral_english_result(self, monkeypatch):
        """A silent clip must not be reported as confidently English."""
        monkeypatch.setattr(ldc, "_get_ld_speech_ts", lambda _a: [])

        speech_sec, result = ldc._resolve_speech_duration("/clip.wav", skip_vad=False)

        assert speech_sec == 0.0
        assert result["confidence"] == 0.0
        assert result["speech_duration"] == 0.0

    def test_speech_regions_are_summed(self, monkeypatch):
        """Speech regions are summed."""
        monkeypatch.setattr(ldc, "_get_ld_speech_ts", lambda _a: [{"start": 0.0, "end": 2.0}, {"start": 5.0, "end": 6.5}])
        speech_sec, result = ldc._resolve_speech_duration("/clip.wav", skip_vad=False)
        assert speech_sec == pytest.approx(3.5)
        assert result is None

    def test_a_path_is_scanned_from_disk_and_samples_in_memory(self, monkeypatch):
        """A path is scanned from disk and samples in memory."""
        calls = {}
        monkeypatch.setattr(ldc.vad, "get_speech_timestamps_from_path", lambda p, threshold: calls.setdefault("path", p) or [])
        monkeypatch.setattr(ldc.vad, "get_speech_timestamps", lambda a, threshold: calls.setdefault("samples", True) or [])

        ldc._get_ld_speech_ts("/clip.wav")
        ldc._get_ld_speech_ts([0.0, 1.0])

        assert calls == {"path": "/clip.wav", "samples": True}


class TestPrimaryDetection:
    """primary detection."""

    def test_the_result_carries_the_language_confidence_and_probabilities(self):
        """The result carries the language confidence and probabilities."""
        result = ldc._detect_language_primary(FakeInProcess(("fr", 0.83, [("fr", 0.83), ("en", 0.1)])), [0.0], 4.0)

        assert result["detected_language"] == "fr"
        assert result["language"] == "fr"
        assert result["confidence"] == 0.83
        assert result["all_probabilities"] == {"fr": 0.83, "en": 0.1}
        assert result["speech_duration"] == 4.0

    def test_negligible_probabilities_are_dropped_from_the_report(self):
        """The full table is 99 languages; the dashboard only needs the plausible ones."""
        result = ldc._detect_language_primary(FakeInProcess(("fr", 0.9, [("fr", 0.9), ("zz", 0.0001)])), [0.0], 1.0)
        assert result["all_probabilities"] == {"fr": 0.9}

    def test_an_engine_reporting_no_table_falls_back_to_the_single_answer(self):
        """An engine reporting no table falls back to the single answer."""
        result = ldc._detect_language_primary(FakeInProcess(("de", 0.5, None)), [0.0], 1.0)
        assert result["all_probabilities"] == {"de": 0.5}


class TestRunLanguageDetectionCore:
    """The entry point every caller uses."""

    def test_a_silent_clip_short_circuits_before_the_engine_is_touched(self, monkeypatch):
        """A silent clip short circuits before the engine is touched."""
        model = FakeInProcess()
        monkeypatch.setattr(ldc, "_get_ld_speech_ts", lambda _a: [])

        result = ldc.run_language_detection_core(model, [0.0], skip_vad=False)

        assert result["confidence"] == 0.0
        assert model.detect_calls == [], "no speech means nothing worth asking the model about"

    def test_an_isolated_engine_is_handed_the_path_unmodified(self):
        """An isolated engine is handed the path unmodified."""
        model = FakeIsolated()
        result = ldc.run_language_detection_core(model, "/clip.wav", skip_vad=True)

        assert model.detect_calls == ["/clip.wav"], "the path must survive; sanitising would decode it here"
        assert result["detected_language"] == "es"

    def test_an_in_process_engine_gets_sanitised_samples(self, monkeypatch):
        """An in process engine gets sanitised samples."""
        model = FakeInProcess()
        monkeypatch.setattr(ldc, "_sanitized_or_original", lambda a: ["sanitised"])

        ldc.run_language_detection_core(model, [0.0], skip_vad=True)
        assert model.detect_calls == [["sanitised"]]

    def test_a_primary_failure_falls_back_rather_than_propagating(self, monkeypatch):
        """Detection failing must degrade to a usable answer, not fail the whole request."""
        model = mock.MagicMock()
        model.detect_language.side_effect = RuntimeError("engine exploded")
        monkeypatch.setattr(ldc, "_is_isolated_engine", lambda _m: False)
        monkeypatch.setattr(ldc, "_sanitized_or_original", lambda a: a)
        monkeypatch.setattr(ldc, "_detect_language_fallback", lambda *a: {"detected_language": "en", "fallback": True})

        assert ldc.run_language_detection_core(model, [0.0], skip_vad=True)["fallback"] is True
