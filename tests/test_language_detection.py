"""Tests for modules/language_detection.py."""
from unittest import mock
import numpy as np
from modules import language_detection

# pylint: disable=protected-access, too-few-public-methods


class TestDetectionPipeline:
    """Tests for run_voting_detection and aggregation."""

    def test_run_voting_detection_success(self):
        """Test successful voting detection across multiple segments."""
        with mock.patch("modules.language_detection.utils.get_audio_duration",
                        return_value=120):
            with mock.patch("modules.language_detection._generate_sampling_tasks",
                            return_value=[0, 30, 60]):
                mock_mm = mock.MagicMock()
                # Mock worker result
                mock_res = {
                    'detected_language': 'fr',
                    'all_probabilities': {'fr': 1.0, 'en': 0.0}
                }
                mock_mm.run_vocal_isolation.return_value = "isolated.wav"
                mock_mm.run_batch_language_detection.return_value = [
                    mock_res, mock_res, mock_res
                ]
                with mock.patch("subprocess.run") as mock_run:
                    mock_run.return_value.returncode = 0
                    result = language_detection.run_voting_detection("dummy.wav", mock_mm)
                    assert result["detected_language"] == "fr"
                    assert result["confidence"] == 1.0

    def test_run_voting_detection_fallback(self):
        """Test fallback to single detection if voting yields no results."""
        with mock.patch("modules.language_detection.utils.get_audio_duration",
                        return_value=120):
            with mock.patch("modules.language_detection._generate_sampling_tasks",
                            return_value=[0, 30, 60]):
                mock_mm = mock.MagicMock()
                mock_mm.run_vocal_isolation.return_value = "isolated.wav"
                mock_mm.run_batch_language_detection.return_value = [
                    None, None, None
                ]
                mock_mm.run_language_detection.return_value = {
                    "detected_language": "en", "confidence": 0.5
                }
                with mock.patch("subprocess.run"):
                    result = language_detection.run_voting_detection("dummy.wav", mock_mm)
                    assert result["detected_language"] == "en"
                    mock_mm.run_language_detection.assert_called_with("dummy.wav")

    def test_run_voting_detection_failure(self):
        """Test absolute fallback when batch scan fails with exception."""
        mock_mm = mock.MagicMock()
        mock_mm.run_language_detection.return_value = {"detected_language": "en"}
        with mock.patch("modules.language_detection._prepare_montage",
                        side_effect=Exception("forced failure")):
            result = language_detection.run_voting_detection("dummy.wav", mock_mm)
            assert result["detected_language"] == "en"
            mock_mm.run_language_detection.assert_called_once_with("dummy.wav")

    def test_run_voting_detection_empty_results(self):
        """Test fallback when batch scan returns no probabilities."""
        mock_mm = mock.MagicMock()
        mock_mm.run_language_detection.return_value = {"detected_language": "fr"}
        mock_mm.run_batch_language_detection.return_value = []
        mock_mm.run_vocal_isolation.return_value = "i.wav"
        with mock.patch("modules.language_detection._prepare_montage", return_value="m.wav"):
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
        assert language_detection._get_sampling_target(30) == 3
        assert language_detection._get_sampling_target(300) == 5
        assert language_detection._get_sampling_target(3000) == 9
        assert language_detection._get_sampling_target(10000) == 13
        assert language_detection._get_sampling_target(20000) == 15


class TestRunVotingDetection:
    """Tests for run_voting_detection infrastructure."""

    def test_run_voting_detection_success_simple(self):
        """Voting detection returns language from model manager."""
        mock_mm = mock.MagicMock()
        mock_mm.run_language_detection.return_value = {
            "detected_language": "en",
            "confidence": 0.9,
        }
        with mock.patch("modules.language_detection.utils.get_audio_duration",
                        return_value=60):
            with mock.patch("modules.language_detection.config") as mock_config:
                mock_config.ENABLE_LD_PREPROCESSING = False
                mock_config.ASR_THREADS = 1
                mock_config.SMART_SAMPLING_SEARCH = False
                result = language_detection.run_voting_detection("dummy.wav", mock_mm)
                assert result["detected_language"] == "en"


class TestFindBestOffset:
    """Tests for _find_best_offset_in_zone."""

    def test_find_best_offset_in_zone_failure(self):
        """Find best offset returns fallback on sf.info error."""
        with mock.patch("modules.language_detection.sf.info",
                        side_effect=Exception("fail")):
            res = language_detection._find_best_offset_in_zone("path", 10, 50, 200)
            assert res == 35

    def test_find_best_offset_in_zone_success(self):
        """Find best offset returns base + retry step on success."""
        with mock.patch("modules.language_detection.sf.info") as mock_info:
            mock_info.return_value.samplerate = 16000
            with mock.patch("modules.language_detection.sf.read") as mock_read:
                mock_read.side_effect = [
                    (np.zeros(480000), 16000),
                    (np.ones(480000) * 0.1, 16000),
                ]
                res = language_detection._find_best_offset_in_zone("path.wav", 100, 50, 500)
                assert res == 110
