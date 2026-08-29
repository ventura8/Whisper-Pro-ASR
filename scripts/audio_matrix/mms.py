"""Meta MMS-TTS synthesis backend, used for languages Piper has no voice for.

Piper's catalogue tops out at ~51 language families, which left 18 Whisper-supported
languages in the matrix with ``"voice": null`` and an ``unsupported_reason``. MMS-TTS
covers nine of them (Amharic, Gujarati, Khmer, Burmese, Punjabi, Tamil, Thai, Tagalog,
Yoruba) at comparable quality: like Piper it is a VITS model, so this is a second voice
source rather than a drop to a lower-fidelity synthesizer such as espeak.

**Determinism.** The committed fixtures are content-addressed, so a rebuild must be
byte-identical or every fixture churns. VITS is stochastic by default in two places, and
both are pinned here from the manifest exactly as Piper's ``noise_scale`` /
``noise_w_scale`` are:

* ``noise_scale``          -- flow sampling noise      -> pinned to 0
* ``noise_scale_duration`` -- duration predictor noise -> pinned to 0

``speaking_rate`` carries the manifest's ``length_scale`` (VITS expresses it as a rate,
so it is the reciprocal). The seed is set as well, which is belt-and-braces once both
noise terms are zero but costs nothing and documents the intent.
"""

from __future__ import annotations

import importlib
import struct
import wave
from pathlib import Path

#: Prefix for every MMS voice id, so the manifest can carry a short ISO 639-3 code.
MODEL_PREFIX = "facebook/mms-tts-"


def mms_available() -> bool:
    """Whether the optional transformers/torch tooling is installed."""
    try:
        importlib.import_module("transformers")
        importlib.import_module("torch")
    except ImportError:
        return False
    return True


def mms_version() -> str:
    """Report the transformers version, or 'missing' when the tooling is absent."""
    try:
        return importlib.import_module("transformers").__version__
    except ImportError:
        return "missing"


def model_id(voice: str) -> str:
    """Expand a manifest voice into a full model id.

    Accepts either a bare ISO 639-3 code ("tha") or an explicit repo id, so a future
    entry can point at a fine-tune without changing this module.
    """
    return voice if "/" in voice else f"{MODEL_PREFIX}{voice}"


def _prepare_text(text: str, tokenizer) -> str:
    """Romanize the input when the model was trained on romanized text.

    A handful of MMS models (Amharic among the ones used here) set ``is_uroman`` on their
    tokenizer. Feeding them native script does not raise -- the tokenizer yields nothing
    usable and synthesis fails deep inside the model with "narrow(): length must be
    non-negative", or in other cases emits an empty waveform. Romanizing up front keeps
    the failure impossible rather than merely diagnosable.
    """
    if not getattr(tokenizer, "is_uroman", False):
        return text
    try:
        uroman = importlib.import_module("uroman")
    except ImportError as exc:  # pragma: no cover - depends on optional tooling
        raise RuntimeError(
            "This MMS voice needs romanized input; install the tools group (`poetry install --with tools`) for uroman."
        ) from exc
    return uroman.Uroman().romanize_string(text)


def _write_wav(samples, sample_rate: int, dest: Path) -> None:
    """Write float samples as 16-bit PCM mono.

    Written by hand rather than via soundfile/scipy to keep the tools group small; the
    output is immediately re-encoded by ffmpeg like every other synthesized clip.
    """
    clipped = [max(-1.0, min(1.0, float(value))) for value in samples]
    frames = b"".join(struct.pack("<h", int(value * 32767.0)) for value in clipped)
    with wave.open(str(dest), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(frames)


def synth(text: str, voice: str, dest: Path, pins: dict) -> None:
    """Render ``text`` to ``dest`` with the MMS voice, using the determinism pins.

    ``pins`` comes from the manifest's ``defaults``, the same source Piper reads, so the
    settings that make output reproducible stay reviewable data rather than constants.
    """
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")

    repo = model_id(voice)
    tokenizer = transformers.AutoTokenizer.from_pretrained(repo)
    model = transformers.VitsModel.from_pretrained(repo)
    model.eval()

    # The two stochastic terms, pinned to the manifest's values (0.0) so a re-render is
    # byte-identical. Without this the flow and duration predictors sample fresh noise
    # per run and every committed fixture churns on rebuild.
    model.noise_scale = float(pins["noise_scale"])
    model.noise_scale_duration = float(pins["noise_w_scale"])
    length_scale = float(pins.get("length_scale", 1.0)) or 1.0
    model.speaking_rate = 1.0 / length_scale

    torch.manual_seed(0)
    inputs = tokenizer(_prepare_text(text, tokenizer), return_tensors="pt")
    with torch.no_grad():
        waveform = model(**inputs).waveform[0]

    _write_wav(waveform.tolist(), int(model.config.sampling_rate), dest)
