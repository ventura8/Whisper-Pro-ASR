"""How the engines treat language: per-window re-detection, and detection input shape.

Split out of test_engine_factory.py, which had grown past the project's module-length
limit. Both classes here pin behaviour found on real hardware rather than in review, and
both fail invisibly in production -- a misreported language, or a detection call that
throws on every short slice while the request still returns 200.
"""


class TestMultilingualWindowDetection:
    """Per-window language re-detection is on for auto-detect, off for explicit requests.

    multilingual=True overrides a requested language: forcing "fr" on Spanish audio
    returns Spanish text while still reporting language="fr" (verified on an RTX 5090).
    Better transcription, but it ignores the caller and misreports the result, so an
    explicit request must keep the single-language path.
    """

    def _engine(self):
        from unittest import mock

        from modules.inference.engines import faster_whisper_engine

        engine = faster_whisper_engine.FasterWhisperEngine.__new__(faster_whisper_engine.FasterWhisperEngine)
        engine.model = mock.MagicMock()
        return engine

    def test_auto_detect_enables_per_window_detection(self):
        engine = self._engine()
        engine.transcribe("clip.wav", language=None)
        assert engine.model.transcribe.call_args.kwargs["multilingual"] is True

    def test_explicit_language_keeps_it_off(self):
        engine = self._engine()
        engine.transcribe("clip.wav", language="fr")
        assert engine.model.transcribe.call_args.kwargs["multilingual"] is False

    def test_caller_can_override_either_way(self):
        engine = self._engine()
        engine.transcribe("clip.wav", language="fr", multilingual=True)
        assert engine.model.transcribe.call_args.kwargs["multilingual"] is True


class TestOpenAIWhisperDetectLanguageShape:
    """Short audio must be padded to the encoder's 30s window before the mel is built.

    Whisper's encoder takes exactly one 30-second window. Without padding,
    detect_language raises "incorrect audio shape". Found on an Intel Arc iGPU: gap-fill
    detects the language of sub-second slices, so every one of those calls failed and
    gap-fill silently recovered nothing on this engine while the request still returned
    200 -- a failure that is invisible from the outside.
    """

    def _engine(self):
        from unittest import mock

        from modules.inference.engines import openai_whisper_engine

        engine = openai_whisper_engine.OpenaiWhisperEngine.__new__(openai_whisper_engine.OpenaiWhisperEngine)
        engine.whisper = mock.MagicMock()
        engine.model = mock.MagicMock()
        engine.model.dims.n_mels = 128
        engine.model.detect_language.return_value = (None, {"fr": 0.1, "en": 0.9})
        return engine

    def test_audio_is_padded_before_the_mel_is_computed(self):
        engine = self._engine()
        engine.detect_language("clip.wav")

        engine.whisper.pad_or_trim.assert_called_once()
        # The mel must be built from the padded audio, not the raw clip.
        assert engine.whisper.log_mel_spectrogram.call_args.args[0] is engine.whisper.pad_or_trim.return_value

    def test_model_n_mels_is_used_not_the_library_default(self):
        engine = self._engine()
        engine.detect_language("clip.wav")

        assert engine.whisper.log_mel_spectrogram.call_args.kwargs["n_mels"] == 128

    def test_returns_the_most_probable_language_first(self):
        engine = self._engine()
        lang, prob, ordered = engine.detect_language("clip.wav")

        assert lang == "en"
        assert prob == 0.9
        assert ordered[0][0] == "en"
