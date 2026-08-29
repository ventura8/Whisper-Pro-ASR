"""Regression coverage for language detection being handed a path instead of samples.

Callers pass a path, because the isolated proxy can only accept one. The in-process engine
handed it straight to faster-whisper, whose detect_language skips decoding and reads
``waveform.dtype`` -- so every gap-fill returned HTTP 500, but only under
ASR_ISOLATE_ENGINES=0, which is why the isolated default never surfaced it.
"""

from modules.inference.engines.faster_whisper_engine import FasterWhisperEngine


def _must_not_decode():
    raise AssertionError("decode_audio must not be called for decoded samples")


class _Model:
    def __init__(self):
        self.seen = None

    def detect_language(self, audio):
        self.seen = audio
        return ("en", 0.99)


class TestWhatLanguageDetectionIsHandedTheModel:
    def _engine(self, model):
        engine = FasterWhisperEngine.__new__(FasterWhisperEngine)
        engine.model = model
        return engine

    def test_a_path_is_decoded_before_it_reaches_the_model(self, monkeypatch):
        samples = [0.0, 0.5, 1.0]
        decoded = []
        monkeypatch.setattr(
            "modules.inference.pipeline.vad.decode_audio",
            lambda path: decoded.append(path) or samples,
        )
        model = _Model()
        assert self._engine(model).detect_language("/tmp/chunk.wav") == ("en", 0.99)
        assert decoded == ["/tmp/chunk.wav"]
        assert model.seen is samples

    def test_already_decoded_samples_are_passed_through_untouched(self, monkeypatch):
        monkeypatch.setattr(
            "modules.inference.pipeline.vad.decode_audio",
            lambda path: _must_not_decode(),
        )
        samples = [0.1, 0.2]
        model = _Model()
        assert self._engine(model).detect_language(samples) == ("en", 0.99)
        assert model.seen is samples
