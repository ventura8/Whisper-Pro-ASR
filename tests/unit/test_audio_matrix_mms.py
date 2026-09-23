"""Tests for scripts/audio_matrix/mms.py, the MMS-TTS backend.

MMS covers the nine languages Piper has no voice for. transformers and torch are optional
tooling that the test image does not carry, so they are injected as stubs -- which is also
what lets the determinism pins be asserted at all.
"""

import contextlib
import struct
import types
import wave
from unittest import mock

import pytest

from scripts.audio_matrix import mms


def test_model_id_expands_a_bare_language_code():
    """The manifest carries a short ISO 639-3 code."""
    assert mms.model_id("tha") == "facebook/mms-tts-tha"


def test_model_id_passes_an_explicit_repo_through():
    """A future entry can point at a fine-tune without changing this module."""
    assert mms.model_id("someone/mms-tts-tha-finetuned") == "someone/mms-tts-tha-finetuned"


def test_mms_available_is_false_when_the_tooling_is_absent():
    """The tools group is optional; absence is reported, not raised."""
    with mock.patch.object(mms.importlib, "import_module", side_effect=ImportError):
        assert mms.mms_available() is False


def test_mms_available_is_true_when_both_imports_succeed():
    """Both transformers and torch are needed."""
    with mock.patch.object(mms.importlib, "import_module", return_value=object()):
        assert mms.mms_available() is True


def test_mms_version_reports_transformers_version():
    """The version feeds the cache digest alongside Piper's."""
    stub = types.SimpleNamespace(__version__="5.17.0")
    with mock.patch.object(mms.importlib, "import_module", return_value=stub):
        assert mms.mms_version() == "5.17.0"


def test_mms_version_reports_missing_without_the_tooling():
    """A distinct sentinel, so a digest computed without MMS is distinguishable."""
    with mock.patch.object(mms.importlib, "import_module", side_effect=ImportError):
        assert mms.mms_version() == "missing"


def test_prepare_text_passes_native_script_through_for_a_normal_tokenizer():
    """Only models trained on romanized text need romanizing."""
    tokenizer = types.SimpleNamespace(is_uroman=False)

    assert mms._prepare_text("ไทย", tokenizer) == "ไทย"


def test_prepare_text_romanizes_when_the_tokenizer_demands_it():
    """Amharic's MMS model sets is_uroman; native script yields nothing usable.

    Synthesis then fails deep inside the model with "narrow(): length must be
    non-negative", or emits an empty waveform -- romanizing up front keeps that
    impossible rather than merely diagnosable.
    """
    tokenizer = types.SimpleNamespace(is_uroman=True)
    uroman_stub = types.ModuleType("uroman")

    class _Uroman:
        """Stand-in romanizer."""

        def romanize_string(self, text):
            """Return a recognisable romanization."""
            return f"romanized::{text}"

    uroman_stub.Uroman = _Uroman

    with mock.patch.object(mms.importlib, "import_module", return_value=uroman_stub):
        assert mms._prepare_text("አማርኛ", tokenizer) == "romanized::አማርኛ"


def test_prepare_text_names_the_missing_tools_group():
    """The error has to say how to fix it: uroman lives in the optional tools group."""
    tokenizer = types.SimpleNamespace(is_uroman=True)

    with mock.patch.object(mms.importlib, "import_module", side_effect=ImportError):
        with pytest.raises(RuntimeError, match="poetry install --with tools"):
            mms._prepare_text("አማርኛ", tokenizer)


def test_write_wav_produces_16_bit_mono_pcm(tmp_path):
    """Written by hand to keep the tools group small; ffmpeg re-encodes it afterwards."""
    dest = tmp_path / "out.wav"

    mms._write_wav([0.0, 0.5, -0.5], 16000, dest)

    with wave.open(str(dest), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 16000
        frames = handle.readframes(handle.getnframes())

    assert struct.unpack("<3h", frames) == (0, 16383, -16383)


def test_write_wav_clips_rather_than_wrapping(tmp_path):
    """A sample past full scale must saturate; wrapping would invert the waveform."""
    dest = tmp_path / "out.wav"

    mms._write_wav([3.0, -3.0], 16000, dest)

    with wave.open(str(dest), "rb") as handle:
        frames = handle.readframes(handle.getnframes())

    assert struct.unpack("<2h", frames) == (32767, -32767)


class _Waveform:
    """Stands in for the tensor MMS returns."""

    def __init__(self, values):
        self._values = values

    def tolist(self):
        """Return plain floats, as the real waveform tensor does."""
        return self._values


def _torch_stub(recorder):
    """A torch stub recording the seed and providing no_grad."""
    torch = types.ModuleType("torch")

    def _manual_seed(value):
        recorder["seed"] = value

    torch.manual_seed = _manual_seed
    # contextlib.nullcontext rather than a MagicMock: pylint reads the ModuleType as a real
    # module and rejects attribute assignment of a mock on it, and a null context is what
    # torch.no_grad() means here anyway.
    torch.no_grad = contextlib.nullcontext
    return torch


def _transformers_stub(model_holder):
    """A transformers stub whose VitsModel records the pins it was given."""
    transformers = types.ModuleType("transformers")

    class _Model:
        """Records pin assignments and returns a fixed waveform."""

        def __init__(self):
            self.config = types.SimpleNamespace(sampling_rate=16000)
            model_holder["model"] = self

        def eval(self):
            """Match the real model's interface."""
            model_holder["eval"] = True

        def __call__(self, **kwargs):
            """Return an object exposing `.waveform`."""
            model_holder["inputs"] = kwargs
            return types.SimpleNamespace(waveform=[_Waveform([0.0, 0.25])])

    transformers.VitsModel = types.SimpleNamespace(from_pretrained=lambda repo: _Model())
    tokenizer = mock.MagicMock()
    tokenizer.is_uroman = False
    tokenizer.return_value = {"input_ids": [[1, 2]]}
    transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda repo: tokenizer)
    model_holder["tokenizer"] = tokenizer
    return transformers


def test_synth_pins_both_stochastic_terms_and_the_rate(tmp_path):
    """VITS samples noise in two places; both are pinned or every fixture churns.

    speaking_rate carries the manifest's length_scale, which VITS expresses as a rate --
    so it is the reciprocal, and getting that backwards would speed up every clip.
    """
    recorder, holder = {}, {}
    modules = {"torch": _torch_stub(recorder), "transformers": _transformers_stub(holder)}

    with mock.patch.object(mms.importlib, "import_module", side_effect=lambda name: modules[name]):
        mms.synth("hello", "tha", tmp_path / "out.wav", {"noise_scale": 0, "noise_w_scale": 0, "length_scale": 2.0})

    model = holder["model"]
    assert model.noise_scale == 0.0
    assert model.noise_scale_duration == 0.0
    assert model.speaking_rate == pytest.approx(0.5)
    assert recorder["seed"] == 0
    assert holder["eval"] is True
    assert (tmp_path / "out.wav").exists()


def test_synth_treats_a_zero_length_scale_as_unity(tmp_path):
    """A zero would divide by zero; the module falls back to 1.0 rather than raising."""
    recorder, holder = {}, {}
    modules = {"torch": _torch_stub(recorder), "transformers": _transformers_stub(holder)}

    with mock.patch.object(mms.importlib, "import_module", side_effect=lambda name: modules[name]):
        mms.synth("hello", "tha", tmp_path / "out.wav", {"noise_scale": 0, "noise_w_scale": 0, "length_scale": 0})

    assert holder["model"].speaking_rate == pytest.approx(1.0)
