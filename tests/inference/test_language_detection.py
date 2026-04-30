"""Tests for modules/inference/language_detection.py."""
# pylint: disable=protected-access, too-few-public-methods
from unittest import mock
import numpy as np
import pytest
from modules.inference import language_detection


class TestDetectionPipeline:
    """Tests for run_voting_detection and aggregation."""

    def test_run_voting_detection_success(self):
        """Test successful voting detection across multiple segments."""
        with mock.patch("modules.inference.language_detection.utils.get_audio_duration", return_value=120):
            with mock.patch("modules.inference.language_detection._generate_sampling_tasks", return_value=[0, 30, 60]):
                mock_mm = mock.MagicMock()
                # Mock internal worker result
                mock_res = {
                    'detected_language': 'fr',
                    'all_probabilities': {'fr': 1.0, 'en': 0.0}
                }
                mock_mm.model_lock_ctx.return_value.__enter__.return_value = (
                    mock.MagicMock(), "CPU")
                mock_mm.run_vocal_isolation_direct.return_value = "isolated.wav"
                mock_mm.run_batch_language_detection_direct.return_value = [
                    mock_res, mock_res, mock_res
                ]
                with mock.patch("modules.inference.language_detection._prepare_montage", return_value="montage.wav"):
                    result = language_detection.run_voting_detection("dummy.wav", mock_mm)
                    assert result["detected_language"] == "fr"
                    assert result["confidence"] == 1.0

    def test_run_voting_detection_fallback(self):
        """Test fallback to single detection if voting yields no results."""
        with mock.patch("modules.inference.language_detection.utils.get_audio_duration", return_value=120):
            with mock.patch("modules.inference.language_detection._generate_sampling_tasks", return_value=[0, 30, 60]):
                mock_mm = mock.MagicMock()
                mock_mm.model_lock_ctx.return_value.__enter__.return_value = (
                    mock.MagicMock(), "CPU")
                mock_mm.run_vocal_isolation_direct.return_value = "isolated.wav"
                mock_mm.run_batch_language_detection_direct.return_value = [
                    None, None, None
                ]
                mock_mm.run_language_detection.return_value = {
                    "detected_language": "en", "confidence": 0.5
                }
                with mock.patch("modules.inference.language_detection._prepare_montage", return_value="montage.wav"):
                    result = language_detection.run_voting_detection("dummy.wav", mock_mm)
                    assert result["detected_language"] == "en"
                    mock_mm.run_language_detection.assert_called_with("dummy.wav")

    def test_run_voting_detection_failure(self):
        """Test absolute fallback when batch scan fails with exception."""
        mock_mm = mock.MagicMock()
        mock_mm.run_language_detection.return_value = {"detected_language": "en"}
        with mock.patch("modules.inference.language_detection._prepare_montage", side_effect=Exception("forced failure")):
            result = language_detection.run_voting_detection("dummy.wav", mock_mm)
            assert result["detected_language"] == "en"
            mock_mm.run_language_detection.assert_called_once_with("dummy.wav")

    def test_run_voting_detection_empty_results(self):
        """Test fallback when batch scan returns no probabilities."""
        mock_mm = mock.MagicMock()
        mock_mm.run_language_detection.return_value = {"detected_language": "fr"}
        mock_mm.model_lock_ctx.return_value.__enter__.return_value = (mock.MagicMock(), "CPU")
        mock_mm.run_batch_language_detection_direct.return_value = []
        mock_mm.run_vocal_isolation_direct.return_value = "i.wav"
        with mock.patch("modules.inference.language_detection._prepare_montage", return_value="m.wav"):
            result = language_detection.run_voting_detection("dummy.wav", mock_mm)
            assert result["detected_language"] == "fr"

    def test_cleanup_batch_assets(self):
        """Test asset cleanup utility."""
        with mock.patch("os.path.exists", return_value=True):
            with mock.patch("os.remove") as mock_remove:
                language_detection._cleanup_batch_assets("m.wav", "i.wav")
                assert mock_remove.call_count == 2


class TestLanguageDetectionHelpers:
    """Tests for essential LD helpers."""

    def test_get_sampling_target(self):
        """Sampling target scales with duration."""
        assert language_detection._get_sampling_target(30) >= 1
        assert language_detection._get_sampling_target(
            300) > language_detection._get_sampling_target(30)


class TestFindBestOffset:
    """Tests for _find_best_offset_in_zone."""

    def test_find_best_offset_in_zone_failure(self):
        """Find best offset returns fallback on sf.info error."""
        with mock.patch("modules.inference.language_detection._get_sf") as mock_gsf:
            mock_gsf.return_value.info.side_effect = Exception("fail")
            res = language_detection._find_best_offset_in_zone("path", 10, 50, 200)
            assert res == 35

    def test_find_best_offset_in_zone_success(self):
        """Find best offset returns base + retry step on success."""
        with mock.patch("modules.inference.language_detection._get_sf") as mock_gsf:
            mock_gsf.return_value.info.return_value.samplerate = 16000
            mock_gsf.return_value.read.side_effect = [
                (np.zeros(480000), 16000),
                (np.ones(480000) * 0.1, 16000),
            ]
            res = language_detection._find_best_offset_in_zone("path.wav", 100, 50, 500)
            assert res == 110


def test_ld_sampling_targets():
    """Cover all duration brackets in _get_sampling_target."""
    assert language_detection._get_sampling_target(300) == 3
    assert language_detection._get_sampling_target(700) == 5
    assert language_detection._get_sampling_target(1500) == 9
    assert language_detection._get_sampling_target(4000) == 13
    assert language_detection._get_sampling_target(20000) == 15


def test_ld_aggregate_edge_cases():
    """Cover zero score branch in _aggregate_language_probs."""
    assert language_detection._aggregate_language_probs([{"en": 0.0}]) == {}


def test_ld_step_isolate_disabled():
    """Cover disabled preprocessing step."""
    with mock.patch("modules.config.ENABLE_LD_PREPROCESSING", False):
        res = language_detection._step_isolate_vocals("test.wav", None, "CPU", {})
        assert res == "test.wav"


def test_ld_smart_sampling_logic():
    """Cover smart sampling search loop."""
    with mock.patch("modules.config.SMART_SAMPLING_SEARCH", True), \
            mock.patch("modules.inference.language_detection._find_best_offset_in_zone", return_value=10.0):
        tasks = language_detection._generate_sampling_tasks("test.wav", 100, 2)
        assert tasks == [10.0, 10.0]


def test_ld_find_best_offset_fallback():
    """Cover fallbacks in _find_best_offset_in_zone."""
    # Test non-wav optimization
    res = language_detection._find_best_offset_in_zone("test.mp3", 10, 20, 100)
    assert res == 10

    # Test stereo mean and threshold fallback
    mock_sf = mock.MagicMock()
    mock_sf.info.return_value = mock.MagicMock(samplerate=16000)
    mock_sf.read.return_value = (np.zeros((16000, 2)), 16000)  # Silent stereo

    with mock.patch("modules.inference.language_detection._get_sf", return_value=mock_sf):
        res = language_detection._find_best_offset_in_zone("test.wav", 10, 20, 100)
        assert res == 10 + (20 / 2)  # Fallback to center


def test_ld_format_empty():
    """Cover empty voting details in _format_detection_result."""
    res = language_detection._format_detection_result({}, 5)
    assert res["confidence"] == 0.0


def test_ld_lazy_load_sf():
    """Cover lazy loading of soundfile."""
    with mock.patch("importlib.import_module", return_value="mock_sf"):
        language_detection._LIBS["sf"] = None
        res = language_detection._get_sf()
        assert res == "mock_sf"
