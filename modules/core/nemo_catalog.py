"""The NVIDIA NeMo models the PARAKEET and CANARY engines can load, as plain data.

Every entry is an ONNX export served by onnx-asr, pinned to one Hugging Face revision. The
pin is the supply-chain guard: weights are fetched by this project's provisioning, never by
onnx-asr (which would follow the repository head), so what runs is exactly what was
validated. ``NEMO_MODEL`` may only name an entry here (see config_nemo).

No environment reads and no heavy imports: config_nemo, provisioning and the engine all
read this table.
"""

from __future__ import annotations

from dataclasses import dataclass

from modules.core import engine_registry

#: The 25 European languages both Parakeet-TDT-0.6B-v3 and Canary-1B-v2 were trained on, as
#: the ISO 639-1 codes the rest of the pipeline uses. Anything outside this set is routed to
#: the Whisper companion: these models do not fail on an unknown language, they transcribe
#: it into the nearest one they know, which no caller could tell from a real transcript.
EU25 = frozenset(
    {
        "bg",
        "cs",
        "da",
        "de",
        "el",
        "en",
        "es",
        "et",
        "fi",
        "fr",
        "hr",
        "hu",
        "it",
        "lt",
        "lv",
        "mt",
        "nl",
        "pl",
        "pt",
        "ro",
        "ru",
        "sk",
        "sl",
        "sv",
        "uk",
    }
)

#: Model families by ONNX graph layout, which decides the decoder file onnx-asr opens.
ARCH_TDT = "tdt"  # Parakeet: FastConformer encoder + token-and-duration transducer
ARCH_AED = "aed"  # Canary: FastConformer encoder + attention encoder-decoder


@dataclass(frozen=True)
class NemoModelSpec:
    """One loadable model and what it can do.

    ``needs_language`` is the Canary contract: its decoder is prompted with a source
    language and defaults to English, so German audio sent without one comes back as an
    English *translation* (measured, 2026-10-09). Parakeet is the opposite -- it takes no
    language at all and decides internally, without reporting what it decided.

    ``word_timestamps`` is whether onnx-asr returns token timings for this architecture.
    The transducer does; the encoder-decoder export returns None, so Canary segments are
    timed from the chunk they came from.
    """

    name: str
    repo_id: str
    revision: str
    architecture: str
    languages: frozenset
    translates_to_english: bool
    word_timestamps: bool
    needs_language: bool


_PARAKEET_V3 = NemoModelSpec(
    name="nemo-parakeet-tdt-0.6b-v3",
    repo_id="istupakov/parakeet-tdt-0.6b-v3-onnx",
    revision="8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce",
    architecture=ARCH_TDT,
    languages=EU25,
    translates_to_english=False,
    word_timestamps=True,
    needs_language=False,
)

_PARAKEET_V2 = NemoModelSpec(
    name="nemo-parakeet-tdt-0.6b-v2",
    repo_id="istupakov/parakeet-tdt-0.6b-v2-onnx",
    revision="0bbb45a3365852604aef28b538a8f066f4ccaa85",
    architecture=ARCH_TDT,
    languages=frozenset({"en"}),
    translates_to_english=False,
    word_timestamps=True,
    needs_language=False,
)

_CANARY_V2 = NemoModelSpec(
    name="nemo-canary-1b-v2",
    repo_id="istupakov/canary-1b-v2-onnx",
    revision="5ebc1520cef7b6b318b3526ad17adbfe00bc1bfc",
    architecture=ARCH_AED,
    languages=EU25,
    translates_to_english=True,
    word_timestamps=False,
    needs_language=True,
)

#: Every model, by its onnx-asr name.
CATALOG: dict[str, NemoModelSpec] = {spec.name: spec for spec in (_PARAKEET_V3, _PARAKEET_V2, _CANARY_V2)}

#: Which catalog entries each engine may load. A Canary name under ASR_ENGINE=PARAKEET would
#: load an encoder-decoder into the transducer path; the families are kept apart here.
FAMILY_MODELS: dict[str, tuple[str, ...]] = {
    engine_registry.ENGINE_PARAKEET: (_PARAKEET_V3.name, _PARAKEET_V2.name),
    engine_registry.ENGINE_CANARY: (_CANARY_V2.name,),
}

#: The model an engine loads when NEMO_MODEL is unset.
FAMILY_DEFAULT: dict[str, str] = {
    engine_registry.ENGINE_PARAKEET: _PARAKEET_V3.name,
    engine_registry.ENGINE_CANARY: _CANARY_V2.name,
}

_DECODER_STEM = {ARCH_TDT: "decoder_joint-model", ARCH_AED: "decoder-model"}


def _suffix(quantization: str | None) -> str:
    return f".{quantization}" if quantization else ""


def required_files(spec: NemoModelSpec, quantization: str | None) -> list[str]:
    """The files onnx-asr opens for ``spec`` at ``quantization`` (None meaning fp32).

    The fp32 encoders of both families exceed protobuf's 2 GB limit and keep their weights
    in an external ``encoder-model.onnx.data`` file beside the graph; the int8 ones do not.
    """
    suffix = _suffix(quantization)
    files = ["config.json", "vocab.txt", f"encoder-model{suffix}.onnx", f"{_DECODER_STEM[spec.architecture]}{suffix}.onnx"]
    if not quantization:
        files.append("encoder-model.onnx.data")
    return files


def required_files_for(spec: NemoModelSpec, quantizations) -> list[str]:
    """The union of :func:`required_files` over every quantization a deployment needs.

    One directory holds every variant of a model, and provisioning validates it as a whole:
    validating per variant would purge the directory -- and the variant already in it --
    whenever the other one was still missing.
    """
    files: list[str] = []
    for quantization in quantizations:
        files.extend(name for name in required_files(spec, quantization) if name not in files)
    return files
