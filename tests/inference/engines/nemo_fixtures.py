"""Stand-ins for onnx-asr and ONNX Runtime, for the NeMo engine tests.

Neither library is installed in the test image (the ml group is not), and these tests must
not depend on one being there. They are injected per test with ``monkeypatch.setitem`` on
``sys.modules`` rather than stubbed globally in tests/mock_setup.py, so an accidental
module-level import of either one fails loudly instead of being silently satisfied.

The fakes reproduce the shapes measured from onnx-asr 0.12.0 on 2026-10-09: tokens carry a
leading space at a word start, timestamps are token starts in 0.08 s steps, and the
recogniser keeps its encoder session at ``model.asr._encoder``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import numpy as np


def timed_result(text: str, tokens: list[str], timestamps: list[float], logprobs: list[float] | None = None):
    """An onnx-asr TimestampedResult."""
    return SimpleNamespace(text=text, tokens=tokens, timestamps=timestamps, logprobs=logprobs)


#: "The quick brown fox." as Parakeet tokenises and times it.
FOX = timed_result(
    "The quick brown fox.",
    [" The", " qu", "ick", " bro", "wn", " fox", "."],
    [0.0, 0.24, 0.4, 0.64, 0.8, 0.96, 1.2],
    [-0.01, -0.1, -0.01, -0.01, -0.01, -0.02, -0.01],
)


class FakeSession:
    """An InferenceSession that reports the provider it ended up on."""

    def __init__(self, provider: str):
        self._provider = provider

    def get_providers(self):
        """The providers this session runs on, first one first."""
        return [self._provider, "CPUExecutionProvider"]


class FakeModel:
    """An onnx-asr adapter: ``recognize`` returns one canned result per input chunk."""

    def __init__(self, provider: str = "CPUExecutionProvider", result=FOX, fail_on: int | None = None):
        self.asr = SimpleNamespace(_encoder=FakeSession(provider))
        self.result = result
        self.calls: list[dict] = []
        self._fail_on = fail_on

    def with_timestamps(self):
        """onnx-asr's timestamped adapter; the fake already returns timed results."""
        return self

    def recognize(self, waveform, **kwargs):
        """Record the call and return one canned result per chunk, failing on request."""
        self.calls.append({"waveform": waveform, **kwargs})
        if self._fail_on is not None and len(self.calls) == self._fail_on:
            raise RuntimeError("provider rejected the input shape")
        if isinstance(waveform, list):
            return [self.result for _ in waveform]
        return self.result


def fake_onnx_asr(model: FakeModel | None = None) -> SimpleNamespace:
    """An ``onnx_asr`` module whose ``load_model`` records its arguments and returns ``model``."""
    built = model or FakeModel()
    return SimpleNamespace(load_model=mock.MagicMock(return_value=built), model=built)


def fake_onnxruntime(providers: list[str]) -> SimpleNamespace:
    """An ``onnxruntime`` module offering ``providers``."""
    return SimpleNamespace(
        __version__="1.29.0",
        __file__="/app/libs/test/onnxruntime/__init__.py",
        get_available_providers=lambda: list(providers),
        SessionOptions=SimpleNamespace,
    )


def companion() -> mock.MagicMock:
    """A Whisper companion: a FasterWhisperEngine-shaped mock."""
    fake = mock.MagicMock()
    fake.device = "cuda"
    fake.transcribe.return_value = (iter(["whisper-segment"]), SimpleNamespace(language="ja"))
    fake.detect_language.return_value = ("de", 0.97, [("de", 0.97), ("nl", 0.02)])
    return fake


def speech(seconds: float, *, quiet_at: float | None = None) -> np.ndarray:
    """``seconds`` of loud noise at 16 kHz, with one silent 0.1 s frame at ``quiet_at`` when given."""
    rng = np.random.default_rng(0)
    audio = rng.uniform(-0.5, 0.5, int(seconds * 16000)).astype(np.float32)
    if quiet_at is not None:
        start = int(quiet_at * 16000)
        stop = start + 1600
        audio[start:stop] = 0.0
    return audio
