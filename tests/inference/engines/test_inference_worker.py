"""The child-side engine host: model pool, payload flattening, and the streaming commands.

Everything here runs in the worker process, so the existing suites reached it only through
a real subprocess and could not assert on any of it. The payload converters matter most:
they are the boundary where an engine's own object graph becomes the plain dicts the parent
reconstructs, and a field silently dropped here surfaces much later as missing word
timings or a wrong language on the dashboard.

No subprocess is spawned. The functions take a handle and a fake engine, which is all they
need -- `worker_main` itself is the only part that touches a pipe.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

from modules.inference.engines import inference_worker


@pytest.fixture(autouse=True)
def clean_engine_pool():
    """The pool is module-global; a leaked handle would leak into the next test."""
    inference_worker._ENGINES.clear()
    yield
    inference_worker._ENGINES.clear()


class FakeEngine:
    """A test double for engine."""

    def __init__(self, segments=None, info=None, detection=None):
        """Record the construction arguments."""
        self._segments = segments or []
        self._info = info or SimpleNamespace(language="en", language_probability=0.9, duration=8.3)
        self._detection = detection or ("es", 0.77, [("es", 0.77), ("en", 0.2)])
        self.unloaded = False
        self.transcribe_kwargs: dict[str, Any] = {}

    def transcribe(self, audio_path, **kwargs):
        """Record the decode parameters and return the scripted result."""
        self.transcribe_kwargs = {"audio_path": audio_path, **kwargs}
        return iter(self._segments), self._info

    def detect_language(self, audio):
        """Record the input and return the scripted detection."""
        self.detected_from = audio
        return self._detection

    def unload(self):
        """Record that the engine was released."""
        self.unloaded = True


class TestTheModelPool:
    """One worker per engine type, holding every unit's model for that type."""

    def test_loading_registers_the_engine_under_the_unit_id(self, monkeypatch):
        """Loading registers the engine under the unit id."""
        created = {}

        def create_engine(engine_type, model_id, unit):
            """Record the engine request and return a fake engine."""
            created.update(engine_type=engine_type, model_id=model_id, unit=unit)
            return FakeEngine()

        monkeypatch.setattr(inference_worker.importlib, "import_module", lambda name: SimpleNamespace(create_engine=create_engine))
        handle = inference_worker._load_model("FASTER-WHISPER", "/models/ct2", {"id": "cuda:0", "name": "GPU 0"})

        assert handle == "cuda:0"
        assert created["engine_type"] == "FASTER-WHISPER"
        assert inference_worker._loaded_handles() == ["cuda:0"]

    def test_env_overrides_are_applied_before_the_engine_stack_is_imported(self, monkeypatch):
        """This ordering is the whole reason the import is deferred: an Intel worker must set
        CUDA_VISIBLE_DEVICES="" before anything can create a CUDA context.
        """
        order = []
        # delenv on the real mapping, not setattr(os, "environ", {}). Replacing the whole
        # object leaves every other reader -- subprocess, tempfile, any library imported
        # during the test -- looking at an empty environment for the duration, which is a
        # far larger blast radius than the one variable this test is about.
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

        def import_module(_name):
            """Return the fake module the test scripted."""
            order.append(("import", inference_worker.os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")))
            return SimpleNamespace(create_engine=lambda *a, **k: FakeEngine())

        monkeypatch.setattr(inference_worker.importlib, "import_module", import_module)
        # Restored explicitly: _load_model sets the isolation env on the real os.environ, and
        # delenv above records nothing when the variable was already unset, so the "" would
        # outlive this test and tell later tests the process is CUDA-blinded.
        with mock.patch.dict(os.environ, {}, clear=False):
            inference_worker._load_model("INTEL-WHISPER", "/ir", {"id": "NPU"}, env={"CUDA_VISIBLE_DEVICES": ""})

        assert order[0][1] == "", "the engine stack was imported before the isolation env landed"

    def test_a_second_load_for_the_same_unit_reuses_the_resident_engine(self, monkeypatch):
        """A second load for the same unit reuses the resident engine."""
        calls = []
        monkeypatch.setattr(
            inference_worker.importlib,
            "import_module",
            lambda _n: SimpleNamespace(create_engine=lambda *a, **k: calls.append(1) or FakeEngine()),
        )
        inference_worker._load_model("E", "m", {"id": "cuda:0"})
        inference_worker._load_model("E", "m", {"id": "cuda:0"})
        assert len(calls) == 1, "isolation relocates resident models, it must not duplicate them"

    def test_asking_for_an_unloaded_handle_is_a_named_error(self):
        """Asking for an unloaded handle is a named error."""
        with pytest.raises(KeyError, match="No engine loaded for handle 'ghost'"):
            inference_worker._get_engine("ghost")

    def test_unloading_drops_the_engine_and_calls_its_own_unload(self):
        """Unloading drops the engine and calls its own unload."""
        engine = FakeEngine()
        inference_worker._ENGINES["cuda:0"] = engine

        assert inference_worker._unload_model("cuda:0") is True
        assert engine.unloaded is True
        assert inference_worker._loaded_handles() == []

    def test_unloading_an_absent_handle_reports_false_rather_than_raising(self):
        """Unloading an absent handle reports false rather than raising."""
        assert inference_worker._unload_model("ghost") is False

    def test_an_engine_without_unload_is_still_dropped(self):
        """An engine without unload is still dropped."""
        inference_worker._ENGINES["cpu"] = object()
        assert inference_worker._unload_model("cpu") is True
        assert inference_worker._loaded_handles() == []

    def test_unload_all_returns_how_many_it_released(self):
        """Unload all returns how many it released."""
        inference_worker._ENGINES.update({"a": FakeEngine(), "b": FakeEngine()})
        assert inference_worker._unload_all() == 2
        assert inference_worker._loaded_handles() == []


class TestFlatteningToPicklablePayloads:
    """Engines yield dataclasses, Namespaces and library objects; the pipe takes dicts."""

    def test_a_full_segment_round_trips_every_field(self):
        """A full segment round trips every field."""
        word = SimpleNamespace(start=0.1, end=0.4, word=" The", probability=0.82)
        segment = SimpleNamespace(start=0.0, end=2.7, text="  The quick brown fox  ", words=[word])

        assert inference_worker._segment_to_dict(segment) == {
            "start": 0.0,
            "end": 2.7,
            "text": "  The quick brown fox  ",
            "words": [{"start": 0.1, "end": 0.4, "word": " The", "probability": 0.82}],
        }

    def test_missing_and_none_fields_both_become_zero_rather_than_raising(self):
        """Engines disagree about absent fields: some omit the attribute, some set None."""
        assert inference_worker._segment_to_dict(SimpleNamespace(start=None, end=None, text=None, words=None)) == {
            "start": 0.0,
            "end": 0.0,
            "text": "",
            "words": None,
        }
        assert inference_worker._segment_to_dict(SimpleNamespace()) == {"start": 0.0, "end": 0.0, "text": "", "words": None}

    def test_a_word_spelled_text_instead_of_word_is_still_carried(self):
        """WhisperX and faster-whisper disagree on the attribute name for the same value."""
        assert inference_worker._word_to_dict(SimpleNamespace(start=1.0, end=1.2, text="hi"))["word"] == "hi"

    def test_a_word_with_neither_spelling_becomes_empty_rather_than_none(self):
        """A word with neither spelling becomes empty rather than none."""
        assert inference_worker._word_to_dict(SimpleNamespace(start=1.0, end=1.2))["word"] == ""

    def test_an_empty_word_list_is_reported_as_absent(self):
        """None, not [], so the parent can tell "no word timestamps requested" from "none found"."""
        assert inference_worker._words_to_list([]) is None
        assert inference_worker._words_to_list(None) is None

    def test_info_carries_the_language_duration_and_probability_table(self):
        """Info carries the language duration and probability table."""
        info = SimpleNamespace(language="fr", language_probability=0.91, duration=12.5, all_language_probs=[("fr", 0.91), ("en", 0.05)])
        assert inference_worker._info_to_dict(info) == {
            "language": "fr",
            "language_probability": 0.91,
            "duration": 12.5,
            "all_language_probs": [("fr", 0.91), ("en", 0.05)],
        }

    def test_info_without_a_language_defaults_to_english_not_none(self):
        """The parent builds an InferenceInfo from this; None there is a downstream crash."""
        assert inference_worker._info_to_dict(SimpleNamespace(language=None))["language"] == "en"

    def test_an_absent_probability_table_is_none_rather_than_an_empty_list(self):
        """An absent probability table is none rather than an empty list."""
        assert inference_worker._info_to_dict(SimpleNamespace())["all_language_probs"] is None

    def test_probability_entries_are_coerced_to_str_and_float(self):
        """They cross a pickle boundary; a numpy scalar here is a needless dependency there."""
        probs = inference_worker._info_to_dict(SimpleNamespace(all_language_probs=[(b"en".decode(), 1)]))["all_language_probs"]
        assert probs == [("en", 1.0)]
        assert isinstance(probs[0][1], float)


class TestTranscribeStream:
    """`info` leads, then one event per segment -- matching the in-process contract."""

    def test_info_is_the_first_event_and_segments_follow(self):
        """Info is the first event and segments follow."""
        segments = [
            SimpleNamespace(start=0.0, end=1.0, text="one", words=None),
            SimpleNamespace(start=1.0, end=2.0, text="two", words=None),
        ]
        inference_worker._ENGINES["u"] = FakeEngine(segments=segments)

        events = list(inference_worker._transcribe("u", "/clip.wav"))

        assert [e["event"] for e in events] == ["info", "segment", "segment"]
        assert events[0]["info"]["language"] == "en"
        assert [e["segment"]["text"] for e in events[1:]] == ["one", "two"]

    def test_decode_params_are_forwarded_to_the_engine_untouched(self):
        """Decode params are forwarded to the engine untouched."""
        engine = FakeEngine()
        inference_worker._ENGINES["u"] = engine
        list(inference_worker._transcribe("u", "/clip.wav", {"language": "es", "word_timestamps": True}))

        assert engine.transcribe_kwargs == {"audio_path": "/clip.wav", "language": "es", "word_timestamps": True}

    def test_no_params_is_not_the_same_as_none_params(self):
        """No params is not the same as none params."""
        engine = FakeEngine()
        inference_worker._ENGINES["u"] = engine
        list(inference_worker._transcribe("u", "/clip.wav"))
        assert engine.transcribe_kwargs == {"audio_path": "/clip.wav"}


class TestDetectLanguage:
    """Paths in, plain payloads out -- audio never crosses the pipe."""

    def test_a_path_is_decoded_here_and_the_result_is_flattened(self, monkeypatch):
        """A path is decoded here and the result is flattened."""
        engine = FakeEngine(detection=("de", 0.66, [("de", 0.66)]))
        inference_worker._ENGINES["u"] = engine
        monkeypatch.setattr(inference_worker.importlib, "import_module", lambda _n: SimpleNamespace(decode_audio=lambda p: [0.0, 1.0]))

        assert inference_worker._detect_language("u", "/clip.wav") == {
            "language": "de",
            "probability": 0.66,
            "all_probs": [("de", 0.66)],
        }
        assert engine.detected_from == [0.0, 1.0], "the worker must decode, so no array crosses the pipe"

    def test_a_none_probability_becomes_zero_rather_than_crossing_the_pipe(self):
        """An engine that reports no confidence must not make the parent's round() raise."""
        inference_worker._ENGINES["u"] = FakeEngine(detection=("en", None, None))
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(inference_worker.importlib, "import_module", lambda _n: SimpleNamespace(decode_audio=lambda p: []))
            result = inference_worker._detect_language("u", "/clip.wav")
        assert result == {"language": "en", "probability": 0.0, "all_probs": []}


class TestDetectLanguageBatch:
    """Windowed detection, decoded once in this process."""

    def _patch(self, monkeypatch, audio, calls):
        """Point the worker's lazy imports at fakes."""

        def import_module(name):
            """Return the fake module the test scripted."""
            if name.endswith("vad"):
                return SimpleNamespace(decode_audio=lambda _p: audio)
            return SimpleNamespace(
                run_language_detection_core=lambda engine, chunk, skip_vad: calls.append(len(chunk)) or {"language": "en"}
            )

        monkeypatch.setattr(inference_worker.importlib, "import_module", import_module)

    def test_one_event_per_thirty_second_window(self, monkeypatch):
        """One event per thirty second window."""
        calls: list[int] = []
        inference_worker._ENGINES["u"] = FakeEngine()
        self._patch(monkeypatch, [0.0] * (30 * 16000 * 2), calls)

        events = list(inference_worker._detect_language_batch("u", "/clip.wav", segment_count=2))

        assert [e["event"] for e in events] == ["detection", "detection"]
        assert [e["index"] for e in events] == [0, 1]
        assert calls == [30 * 16000, 30 * 16000]

    def test_the_scan_stops_at_the_end_of_the_audio_not_at_segment_count(self, monkeypatch):
        """A 40s clip asked for 10 windows must yield 2, not 10 empty ones."""
        calls: list[int] = []
        inference_worker._ENGINES["u"] = FakeEngine()
        self._patch(monkeypatch, [0.0] * (40 * 16000), calls)

        events = list(inference_worker._detect_language_batch("u", "/clip.wav", segment_count=10))

        assert len(events) == 2
        assert calls == [30 * 16000, 10 * 16000], "the final window is the remainder, not a padded full one"
