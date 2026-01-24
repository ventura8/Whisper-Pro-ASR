"""Tests for modules/language_detection.py."""
from types import SimpleNamespace
from unittest import mock
import numpy as np
import pytest
from modules import language_detection

# pylint: disable=protected-access


def _make_ld_model():
    """Return a mock Whisper model for language detection tests."""
    model = mock.MagicMock()
    mock_output = mock.MagicMock()
    logits = np.zeros(10)
    logits[1] = 10.0  # 'en'
    mock_scores = [mock.MagicMock()]
    mock_scores[0].__getitem__.return_value = logits
    mock_output.scores = mock_scores
    model.generate.return_value = mock_output
    return model


def _make_ld_processor():
    """Return a mock Whisper processor for language detection tests."""
    processor = mock.MagicMock()
    processor.tokenizer.get_vocab.return_value = {
        "<|en|>": 1,
        "<|fr|>": 2,
        "<|de|>": 3,
        "<|startoftranscript|>": 0
    }
    processor.tokenizer.convert_tokens_to_ids.side_effect = (
        lambda x: 0 if "start" in x else 1
    )
    processor.return_value.input_features = "mock_features"
    if hasattr(processor.tokenizer, "get_lang_to_id"):
        processor.tokenizer.get_lang_to_id.return_value = None
    return processor


class TestDetectionPipeline:
    """Tests for run_detection_pipeline."""

    def test_run_detection_pipeline_success(self):
        """Detection pipeline returns language and chunk count on success."""
        model = _make_ld_model()
        processor = _make_ld_processor()
        with mock.patch(
            "modules.language_detection._collect_active_speech_chunks"
        ) as mock_collect:
            mock_collect.return_value = [np.zeros(16000)]
            with mock.patch("modules.language_detection.torch") as mock_torch:
                mock_torch.long = "long"
                mock_torch.tensor.return_value = "tensor"
                mock_softmax_res = mock.MagicMock()
                mock_softmax_res.tolist.return_value = [0.9]
                mock_torch.softmax.return_value = mock_softmax_res

                result = language_detection.run_detection_pipeline(
                    model, processor, "dummy.wav"
                )
                assert result["detected_language"] == "en"
                assert result["chunks_processed"] == 1

    def test_run_detection_pipeline_no_chunks(self):
        """Detection pipeline handles no speech chunks."""
        model = _make_ld_model()
        processor = _make_ld_processor()
        with mock.patch(
            "modules.language_detection._collect_active_speech_chunks"
        ) as mock_collect:
            mock_collect.return_value = []
            result = language_detection.run_detection_pipeline(
                model, processor, "dummy.wav"
            )
            assert result["detected_language"] == "en"
            assert result["chunks_processed"] == 0

    def test_run_detection_pipeline_err_fallback(self):
        """Detection pipeline falls back on collect error."""
        model = _make_ld_model()
        processor = _make_ld_processor()
        mock_path = "modules.language_detection._collect_active_speech_chunks"
        with mock.patch(mock_path, side_effect=ValueError("fail")):
            with mock.patch(
                "modules.language_detection._sample_audio_chunks"
            ) as mock_sample:
                mock_sample.return_value = [np.zeros(16000)]
                mock_detect = (
                    "modules.language_detection._detect_languages_from_chunks"
                )
                with mock.patch(mock_detect, return_value=[{"en": 1.0}]):
                    result = language_detection.run_detection_pipeline(
                        model, processor, "dummy.wav"
                    )
                    assert result["chunks_processed"] == 1
                    mock_sample.assert_called_once()

    def test_run_detection_pipeline_unhandled_err(self):
        """Detection pipeline propagates unhandled errors."""
        model = _make_ld_model()
        processor = _make_ld_processor()
        mock_path = "modules.language_detection._collect_active_speech_chunks"
        with mock.patch(mock_path, side_effect=IndexError("critical")):
            with pytest.raises(IndexError):
                language_detection.run_detection_pipeline(
                    model, processor, "dummy.wav"
                )


class TestLanguageDetectionHelpers:
    """Tests for aggregate, winner, mapping, fallback, sampling target."""

    def test_aggregate_language_probs_empty(self):
        """Aggregate probs returns empty dict for empty input."""
        result = language_detection._aggregate_language_probs([])
        assert not result

    def test_find_language_winner_empty(self):
        """Find winner returns en/0.0 when no probs."""
        winner, conf = language_detection._find_language_winner({})
        assert winner == "en"
        assert conf == 0.0

    def test_get_language_token_mapping_standard(self):
        """Token mapping uses get_lang_to_id when present."""
        processor = _make_ld_processor()
        processor.tokenizer.get_lang_to_id.return_value = {"<|en|>": 1}
        mapping = language_detection._get_language_token_mapping(processor)
        assert mapping == {"<|en|>": 1}

    def test_get_language_token_mapping_fallback(self):
        """Token mapping fallback when get_lang_to_id missing."""
        processor = _make_ld_processor()
        del processor.tokenizer.get_lang_to_id
        mapping = language_detection._get_language_token_mapping(processor)
        assert "<|en|>" in mapping

    def test_detect_languages_fallback(self):
        """Fallback detection returns per-chunk language probs."""
        model = _make_ld_model()
        processor = _make_ld_processor()
        model.generate.return_value = [[1, 2, 3]]
        processor.decode.return_value = "<|fr|>"
        chunks = [np.zeros(16000)]
        results = language_detection._detect_languages_fallback(
            model, processor, chunks
        )
        assert results[0] == {"fr": 1.0}

    def test_get_sampling_target(self):
        """Sampling target scales with duration."""
        assert language_detection._get_sampling_target(30) == 3
        assert language_detection._get_sampling_target(300) == 5
        assert language_detection._get_sampling_target(3000) == 12
        assert language_detection._get_sampling_target(10000) == 20
        assert language_detection._get_sampling_target(20000) == 25


class TestLanguageDetectionChunks:
    """Tests for read_chunk, collect, search, sample, detect_from_chunks."""

    def test_read_chunk_mono_resample_pad(self):
        """Read chunk resamples and pads to target."""
        with mock.patch("modules.language_detection.sf.info") as mock_info:
            mock_info.return_value.samplerate = 44100
            with mock.patch(
                "modules.language_detection.sf.read"
            ) as mock_read:
                stereo_audio = np.ones((44100, 2))
                mock_read.return_value = (stereo_audio, 44100)
                chunk = language_detection._read_chunk(
                    "dummy.wav", 0, 1, 44100
                )
                assert chunk.shape == (480000,)

    def test_collect_active_speech_chunks(self):
        """Collect chunks returns expected count."""
        with mock.patch("modules.language_detection.sf.info") as mock_info:
            mock_info.return_value.duration = 100
            mock_info.return_value.samplerate = 16000
            with mock.patch(
                "modules.language_detection._read_chunk"
            ) as mock_read:
                mock_read.return_value = np.zeros(480000)
                chunks = language_detection._collect_active_speech_chunks(
                    "dummy.wav"
                )
                assert len(chunks) == 3

    def test_search_or_fallback_zone_speech_found(self):
        """Search zone returns audio when speech found."""
        with mock.patch(
            "modules.language_detection.config"
        ) as mock_config:
            mock_config.SMART_SAMPLING_SEARCH = True
            with mock.patch(
                "modules.language_detection._read_chunk"
            ) as mock_read:
                mock_read.return_value = np.ones(480000) * 0.1
                info = SimpleNamespace(samplerate=16000, duration=1000)
                res = language_detection._search_or_fallback_zone(
                    "path", None, None, (0, 100),
                    (1, 3, info),
                )

                assert np.all(res == 0.1)

    def test_sample_audio_chunks(self):
        """Sample audio chunks respects duration and rate."""
        with mock.patch("modules.language_detection.sf.info") as mock_info:
            mock_info.return_value.duration = 600
            mock_info.return_value.samplerate = 16000
            with mock.patch(
                "modules.language_detection._read_chunk"
            ) as mock_read:
                mock_read.return_value = np.zeros(480000)
                chunks = language_detection._sample_audio_chunks("dummy.wav")
                assert len(chunks) == 5

    def test_detect_languages_from_chunks_no_mapping(self):
        """Detect from chunks uses fallback when no token mapping."""
        model = _make_ld_model()
        processor = _make_ld_processor()
        mapping_patch = mock.patch(
            "modules.language_detection._get_language_token_mapping",
            return_value={},
        )
        with mapping_patch:
            with mock.patch(
                "modules.language_detection._detect_languages_fallback",
            ) as mock_fallback:
                mock_fallback.return_value = [{"en": 1.0}]
                chunks = [np.zeros(16000)]
                res = language_detection._detect_languages_from_chunks(
                    model, processor, chunks,
                )
                assert res == [{"en": 1.0}]


class TestRunVotingDetection:
    """Tests for run_voting_detection."""

    def test_run_voting_detection_success(self):
        """Voting detection returns language from model manager."""
        mock_mm = mock.MagicMock()
        mock_mm.run_language_detection.return_value = {
            "detected_language": "en",
            "confidence": 0.9,
        }
        with mock.patch(
            "modules.language_detection.utils.get_audio_duration",
            return_value=60,
        ):
            with mock.patch("modules.language_detection.subprocess.run"):
                with mock.patch("modules.language_detection.os.remove"):
                    with mock.patch(
                        "modules.language_detection.config"
                    ) as mock_config:
                        mock_config.ENABLE_LD_PREPROCESSING = False
                        mock_config.ASR_THREADS = 1
                        mock_config.SMART_SAMPLING_SEARCH = False
                        result = language_detection.run_voting_detection(
                            "dummy.wav", mock_mm
                        )
                        assert result["detected_language"] == "en"

    def test_run_voting_detection_empty_votes(self):
        """Voting detection handles empty vote list."""
        mock_mm = mock.MagicMock()
        mock_mm.run_language_detection.return_value = {
            "detected_language": "en",
            "confidence": 0.9,
        }
        with mock.patch(
            "modules.language_detection.utils.get_audio_duration",
            return_value=60,
        ):
            with mock.patch(
                "modules.language_detection.ThreadPoolExecutor"
            ) as mock_exec:
                mock_exec.return_value.__enter__.return_value.map.return_value = []
                result = language_detection.run_voting_detection(
                    "dummy.wav", mock_mm
                )
                assert result["detected_language"] == "en"

    def test_run_voting_detection_complex(self):
        """Voting detection with preprocessing and many scans."""
        mock_mm = mock.MagicMock()
        mock_mm.run_language_detection.return_value = {
            "detected_language": "fr",
            "confidence": 0.8,
            "all_probabilities": {"fr": 0.8, "en": 0.1},
            "chunks_processed": 25
        }

        dur = mock.patch(
            "modules.language_detection.utils.get_audio_duration",
            return_value=15000,
        )
        with dur:
            with mock.patch("modules.language_detection.subprocess.run"):
                with mock.patch(
                    "modules.language_detection.os.path.exists",
                    return_value=True,
                ):
                    with mock.patch(
                        "modules.language_detection.os.remove"
                    ):
                        with mock.patch(
                            "modules.language_detection.config",
                        ) as mock_config:
                            mock_config.ENABLE_LD_PREPROCESSING = True
                            mock_config.SMART_SAMPLING_SEARCH = True
                            mock_config.ASR_THREADS = 4
                            zone = mock.patch(
                                "modules.language_detection"
                                "._find_best_offset_in_zone",
                                return_value=10.0,
                            )
                            with zone:
                                pm = mock.patch(
                                    "modules.preprocessing.get_manager",
                                )
                                with pm as mock_pm:
                                    pa = (
                                        mock_pm.return_value
                                        .process_audio_file
                                    )
                                    pa.return_value = "proc.wav"
                                    result = (
                                        language_detection
                                        .run_voting_detection(
                                            "dummy.wav", mock_mm
                                        )
                                    )
                                    assert (
                                        result["detected_language"]
                                        == "fr"
                                    )
                                    assert (
                                        result["chunks_processed"] == 25
                                    )


class TestFindBestOffset:
    """Tests for _find_best_offset_in_zone."""

    def test_find_best_offset_in_zone_failure(self):
        """Find best offset returns fallback on sf.info error."""
        with mock.patch(
            "modules.language_detection.sf.info",
            side_effect=Exception("fail"),
        ):
            res = language_detection._find_best_offset_in_zone(
                "path", 10, 50, 200
            )
            assert res == 35

    def test_find_best_offset_in_zone_success(self):
        """Find best offset returns base + retry step on success."""
        with mock.patch(
            "modules.language_detection.sf.info"
        ) as mock_info:
            mock_info.return_value.samplerate = 16000
            with mock.patch(
                "modules.language_detection.sf.read",
            ) as mock_read:
                mock_read.side_effect = [
                    (np.zeros(480000), 16000),
                    (np.ones(480000) * 0.1, 16000),
                ]
                res = language_detection._find_best_offset_in_zone(
                    "path.wav", 100, 50, 500,
                )
                assert res == 110


class TestDetectSingleChunkProbs:
    """Tests for _detect_single_chunk_probs."""

    def test_detect_single_chunk_probs_cpu_friendly(self):
        """Single-chunk probs return aggregated language dict."""
        model = _make_ld_model()
        processor = _make_ld_processor()
        lang_ids = [1, 2]
        id_to_lang = {1: "en", 2: "fr"}
        with mock.patch(
            "modules.language_detection.torch"
        ) as mock_torch:
            mock_torch.long = "long"
            mock_torch.tensor.return_value = "tensor"
            mock_softmax_res = mock.MagicMock()
            mock_softmax_res.tolist.return_value = [0.99, 0.01]
            mock_torch.softmax.return_value = mock_softmax_res
            chunk = np.zeros(16000)
            res = language_detection._detect_single_chunk_probs(
                model, processor, chunk, lang_ids, id_to_lang
            )
            assert res == {"en": 0.99, "fr": 0.01}

    def test_detect_single_chunk_probs_no_valid_ids(self):
        """Single-chunk probs handle invalid lang ids."""
        model = _make_ld_model()
        processor = _make_ld_processor()
        lang_ids = [100, 200]
        id_to_lang = {100: "xx", 200: "yy"}
        chunk = np.zeros(16000)
        model.generate.return_value.scores[0][0] = np.zeros(10)
        res = language_detection._detect_single_chunk_probs(
            model, processor, chunk, lang_ids, id_to_lang
        )
        assert res == {"en": 0.0}

    def test_detect_single_chunk_probs_non_list_softmax(self):
        """Single-chunk probs handle scalar softmax result."""
        model = _make_ld_model()
        processor = _make_ld_processor()
        lang_ids = [1]
        id_to_lang = {1: "en"}
        with mock.patch(
            "modules.language_detection.torch"
        ) as mock_torch:
            mock_torch.softmax.return_value.tolist.return_value = 0.95
            chunk = np.zeros(16000)
            res = language_detection._detect_single_chunk_probs(
                model, processor, chunk, lang_ids, id_to_lang
            )
            assert res == {"en": 0.95}
