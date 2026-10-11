"""Which model serves a call on a NeMo engine (modules/inference/engines/nemo_routing.py)."""

import pytest

from modules.core import nemo_catalog
from modules.inference.engines import nemo_routing

PARAKEET = nemo_catalog.CATALOG["nemo-parakeet-tdt-0.6b-v3"]
PARAKEET_EN = nemo_catalog.CATALOG["nemo-parakeet-tdt-0.6b-v2"]
CANARY = nemo_catalog.CATALOG["nemo-canary-1b-v2"]


@pytest.mark.parametrize(
    ("spec", "language", "task", "target", "target_language"),
    [
        # Covered languages are transcribed by the NeMo model, in their own language.
        (PARAKEET, "de", "transcribe", "nemo", "de"),
        (CANARY, "fr", "transcribe", "nemo", "fr"),
        # Outside the 25 -- these models would transcribe Japanese into the nearest European
        # language they know, so Whisper must take it.
        (PARAKEET, "ja", "transcribe", "whisper", None),
        (CANARY, "zh", "translate", "whisper", None),
        # The English-only Parakeet covers English and nothing else.
        (PARAKEET_EN, "en", "transcribe", "nemo", "en"),
        (PARAKEET_EN, "de", "transcribe", "whisper", None),
        # Translation: Canary does X->en natively; Parakeet has none.
        (CANARY, "de", "translate", "nemo", "en"),
        (PARAKEET, "de", "translate", "whisper", None),
        # Translating English is transcribing it, whichever model.
        (PARAKEET, "en", "translate", "nemo", "en"),
        (CANARY, "en", "TRANSLATE", "nemo", "en"),
    ],
)
def test_route_table(spec, language, task, target, target_language):
    """Every row of the routing table in nemo_routing's docstring."""
    route = nemo_routing.choose(spec, language, task)
    assert route.target == target
    assert route.target_language == target_language
    assert route.reason


@pytest.mark.parametrize("language", [None, "", "  "])
def test_no_language_goes_to_whisper_because_only_it_can_identify_one(language):
    """Detection failed upstream, or this is the pipeline's own detection fallback."""
    route = nemo_routing.choose(PARAKEET, language, "transcribe")
    assert route.target == nemo_routing.TARGET_WHISPER
    assert route.language is None


@pytest.mark.parametrize(("tag", "code"), [("pt-BR", "pt"), ("de_AT", "de"), (" EN ", "en")])
def test_regional_tags_collapse_to_their_language(tag, code):
    """pt-BR is Portuguese for routing purposes."""
    route = nemo_routing.choose(PARAKEET, tag, "transcribe")
    assert route.target == nemo_routing.TARGET_NEMO
    assert route.language == code


def test_parakeet_is_given_no_language_keywords():
    """onnx-asr forwards whatever it is given to the decoder, and Parakeet's takes none."""
    route = nemo_routing.choose(PARAKEET, "de", "transcribe")
    assert not route.recognize_kwargs(PARAKEET)


def test_canary_is_always_told_its_source_language():
    """Without one Canary's prompt defaults to English and translates German audio (measured)."""
    route = nemo_routing.choose(CANARY, "de", "transcribe")
    assert route.recognize_kwargs(CANARY) == {"language": "de", "target_language": "de", "pnc": True}


def test_canary_translation_targets_english():
    """A Canary translation is X -> en."""
    route = nemo_routing.choose(CANARY, "uk", "translate")
    assert route.recognize_kwargs(CANARY)["target_language"] == "en"
    assert route.recognize_kwargs(CANARY)["language"] == "uk"
