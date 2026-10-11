"""Which model serves one ``transcribe`` call on a NeMo engine: the NeMo model or Whisper.

The NeMo models cover 25 European languages and have no language-identification head. They
do not fail on audio outside their set; they transcribe it into the nearest language they
know, and nothing in the output says so. So the decision is made here, per call, from the
language the pipeline has already settled on:

==============================  ======================================================
request                         served by
==============================  ======================================================
no language                     Whisper -- detection failed upstream, or this is the
                                pipeline's own detection fallback; either way only a
                                model that can identify the language may answer
language outside the model's    Whisper
translate, language ``en``      NeMo, as a transcription (it is English already)
translate, Canary               NeMo, ``target_language="en"`` (native X->en)
translate, Parakeet             Whisper -- Parakeet has no translation
anything else                   NeMo
==============================  ======================================================

The pipeline makes one call per language run (run_decoding), so a Japanese run inside a
French film reaches Whisper on its own, exactly as it would on FASTER-WHISPER.
"""

from __future__ import annotations

from dataclasses import dataclass

from modules.core.nemo_catalog import NemoModelSpec

TARGET_NEMO = "nemo"
TARGET_WHISPER = "whisper"


@dataclass(frozen=True)
class Route:
    """Where a call goes, and in which language the NeMo model is told to work.

    ``reason`` is for the log line: the routing is invisible in a transcript, and a
    transcript that came from the wrong model is otherwise indistinguishable from a bad one.
    """

    target: str
    language: str | None
    target_language: str | None
    reason: str

    def recognize_kwargs(self, spec: NemoModelSpec) -> dict:
        """The language keywords onnx-asr's ``recognize`` takes for this model.

        Only Canary takes any. onnx-asr forwards whatever it is given to the model's
        decoder, and Parakeet's has no language parameter to receive them.
        """
        if not spec.needs_language:
            return {}
        return {"language": self.language, "target_language": self.target_language, "pnc": True}


def choose(spec: NemoModelSpec, language: str | None, task: str | None) -> Route:
    """Route one call; see the module docstring for the table."""
    code = _normalise(language)
    if code is None:
        return Route(TARGET_WHISPER, None, None, "no language to route by")
    if code not in spec.languages:
        return Route(TARGET_WHISPER, code, None, f"{code} is outside {spec.name}")
    if _is_translation(task):
        return _route_translation(spec, code)
    return Route(TARGET_NEMO, code, code, f"{code} transcription")


def _route_translation(spec: NemoModelSpec, code: str) -> Route:
    if code == "en":
        return Route(TARGET_NEMO, code, code, "translation of English is a transcription")
    if spec.translates_to_english:
        return Route(TARGET_NEMO, code, "en", f"{code}->en translation")
    return Route(TARGET_WHISPER, code, None, f"{spec.name} cannot translate")


def _normalise(language: str | None) -> str | None:
    """A two-letter code, or None. Regional tags such as ``pt-BR`` collapse to their language."""
    code = str(language or "").strip().lower()
    if not code:
        return None
    return code.replace("_", "-").split("-", 1)[0]


def _is_translation(task: str | None) -> bool:
    return str(task or "").strip().lower() == "translate"
