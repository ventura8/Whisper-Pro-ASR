"""How the resolved language runs reach the decoder from the orchestration.

Split from test_model_manager.py, which is at the module line cap. These exercise the whole
route -- VAD regions, the per-region labels, the runs, the decoder call -- with the engine
and the detector mocked, so the option each call carries can be asserted exactly.
"""

from unittest import mock

import pytest

from modules.inference.runtime import model_manager

#: The settings the clip timestamps and run grouping asserted below are computed from. Pinned
#: so an operator's environment (a different pad, a longer switch bar, the feature off) cannot
#: change what these tests measure.
_PINNED = {
    "ASR_SEGMENT_FIRST": True,
    "ASR_ENGINE": "FASTER-WHISPER",
    "ASR_MULTILINGUAL_SEGMENTATION": True,
    "SEGMENT_FIRST_MIN_REGIONS": 2,
    "SEGMENT_SPLIT_PAD_MS": 200,
    "SEGMENT_CLIP_MERGE_GAP_SEC": 0.3,
    "SEGMENT_RUN_MIN_SWITCH_SEC": 3.0,
    "SEGMENT_RUN_MIN_SWITCH_SHARE": 0.2,
    "LD_MIN_CONFIDENCE": 0.5,
}


@pytest.fixture(autouse=True)
def _cpu_pool(monkeypatch):
    """An empty engine pool and the pinned configuration, both restored afterwards."""
    for name, value in _PINNED.items():
        monkeypatch.setattr(model_manager.config, name, value)
    model_manager.MODEL_POOL.clear()
    yield
    model_manager.MODEL_POOL.clear()


def test_clips_reach_the_decoder_and_turn_its_own_vad_off():
    """The clips-applied path, which no mocked test previously exercised.

    The regions are labelled before the decode and grouped into runs, and each run's
    language is what the decoder is told: clips cap each decode window at a run, and the
    run's language is forced rather than re-detected from the clip. The internal VAD goes off
    because the clips already are its output -- and passing vad_filter twice is a TypeError
    only reachable when clips are applied, which is how it survived a green suite until a
    real engine was called.
    """
    mock_model = mock.MagicMock()
    mock_info = mock.MagicMock(language="es", language_probability=0.9, duration=120.0)
    mock_model.transcribe.return_value = ([mock.MagicMock(start=0.0, end=1.0, text="hola")], mock_info)
    mock_model.detect_language.return_value = ("es", 0.9, [("es", 0.9)])
    model_manager.MODEL_POOL["CPU"] = mock_model

    regions = [{"start": 0.0, "end": 20.0}, {"start": 25.0, "end": 60.0}]
    labels = [
        {"start": 0.0, "end": 20.0, "language": "es", "confidence": 0.9},
        {"start": 25.0, "end": 60.0, "language": "es", "confidence": 0.9},
    ]
    with (
        mock.patch.object(model_manager.speech_clips, "vad") as vad_module,
        mock.patch.object(model_manager.segment_languages, "for_clips", return_value=labels),
    ):
        vad_module.get_speech_timestamps_from_path.return_value = regions
        model_manager.run_transcription("test.wav", language=None, task="transcribe", vad_filter=True)

    kwargs = mock_model.transcribe.call_args.kwargs
    # One Spanish run in two pieces -- the 5 s pause is longer than a scene's -- padded by
    # SEGMENT_SPLIT_PAD_MS so a clip edge never cuts a word, and clamped to the midpoint of
    # the gap so the flat seek-point list stays monotonic.
    assert kwargs["clip_timestamps"] == [0.0, 20.2, 24.8, 60.2]
    assert kwargs["vad_filter"] is False
    assert kwargs["language"] == "es"
    assert kwargs["multilingual"] is False


def test_unlabelled_clips_still_let_the_decoder_pick_the_language():
    """The detector failed on every region: clips stay, and per-window re-detection stays on."""
    mock_model = mock.MagicMock()
    mock_info = mock.MagicMock(language="es", language_probability=0.9, duration=120.0)
    mock_model.transcribe.return_value = ([mock.MagicMock(start=0.0, end=1.0, text="hola")], mock_info)
    model_manager.MODEL_POOL["CPU"] = mock_model

    regions = [{"start": 0.0, "end": 20.0}, {"start": 25.0, "end": 60.0}]
    with (
        mock.patch.object(model_manager.speech_clips, "vad") as vad_module,
        mock.patch.object(model_manager.segment_languages, "for_clips", return_value=[]),
    ):
        vad_module.get_speech_timestamps_from_path.return_value = regions
        model_manager.run_transcription("test.wav", language=None, task="transcribe", vad_filter=True)

    kwargs = mock_model.transcribe.call_args.kwargs
    assert kwargs["clip_timestamps"] == [0.0, 20.2, 24.8, 60.2]
    assert kwargs["multilingual"] is True


def _mixed_film(mock_model):
    """A Romanian film with a Turkish scene, as the detector labels it."""
    mock_info = mock.MagicMock(language="ro", language_probability=0.9, duration=120.0)
    mock_model.transcribe.return_value = ([mock.MagicMock(start=0.0, end=1.0, text="x")], mock_info)
    model_manager.MODEL_POOL["CPU"] = mock_model
    regions = [{"start": 0.0, "end": 20.0}, {"start": 25.0, "end": 60.0}, {"start": 65.0, "end": 90.0}]
    labels = [
        {"start": 0.0, "end": 20.0, "language": "ro", "confidence": 0.9},
        {"start": 25.0, "end": 60.0, "language": "tr", "confidence": 0.9},
        {"start": 65.0, "end": 90.0, "language": "ro", "confidence": 0.9},
    ]
    return regions, labels


@pytest.mark.parametrize("task", ["translate", "Translate"])
def test_a_translation_follows_the_switches_even_when_a_language_was_named(task):
    """The case a subtitle client presents: it names the audio track's language and asks for
    English. The output is English whatever was spoken, so the named language is the main
    audio, not a constraint -- the Turkish scene is translated *from Turkish*, in its own
    call, and not decoded under the Romanian token into noise."""
    mock_model = mock.MagicMock()
    regions, labels = _mixed_film(mock_model)
    unchanged = mock.Mock(side_effect=lambda _m, _p, segs, spans, **_k: (segs, spans))
    with (
        mock.patch.object(model_manager.speech_clips, "vad") as vad_module,
        mock.patch.object(model_manager.segment_languages, "for_clips", return_value=labels),
        mock.patch.object(model_manager.gap_filling, "fill_language_gaps", unchanged) as gaps,
    ):
        vad_module.get_speech_timestamps_from_path.return_value = regions
        model_manager.run_transcription("film.wav", language="ro", task=task, vad_filter=True, auto_detected=False)

    calls = [(c.kwargs["language"], c.kwargs["task"], c.kwargs["clip_timestamps"]) for c in mock_model.transcribe.call_args_list]
    assert calls == [("ro", task, [0.0, 20.2, 64.8, 90.2]), ("tr", task, [24.8, 60.2])]
    assert all(c.kwargs["multilingual"] is False for c in mock_model.transcribe.call_args_list)
    assert gaps.called, "a translation's gaps are filled in the language of the run they sit in"


def test_a_transcription_in_a_named_language_is_still_decoded_as_one():
    """The other half of the rule: a caller who asked to *read* Romanian gets Romanian, one
    call, no clips -- the request is honoured as given."""
    mock_model = mock.MagicMock()
    regions, labels = _mixed_film(mock_model)
    with (
        mock.patch.object(model_manager.speech_clips, "vad") as vad_module,
        mock.patch.object(model_manager.segment_languages, "for_clips", return_value=labels),
        mock.patch.object(model_manager.gap_filling, "fill_language_gaps") as gaps,
    ):
        vad_module.get_speech_timestamps_from_path.return_value = regions
        model_manager.run_transcription("film.wav", language="ro", task="transcribe", vad_filter=True, auto_detected=False)

    assert mock_model.transcribe.call_count == 1
    kwargs = mock_model.transcribe.call_args.kwargs
    assert kwargs["language"] == "ro"
    assert "clip_timestamps" not in kwargs
    assert not gaps.called
