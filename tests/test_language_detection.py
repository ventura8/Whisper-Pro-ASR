"""Tests for modules/language_detection.py."""
from unittest import mock
from modules import language_detection

# pylint: disable=protected-access

def test_run_voting_detection_success():
    """Verify that run_voting_detection correctly orchestrates extraction and detection."""
    mock_mm = mock.MagicMock()
    mock_mm.run_language_detection.return_value = {
        "detected_language": "en",
        "confidence": 0.9,
        "all_probabilities": {"en": 0.9, "fr": 0.1}
    }

    with mock.patch("modules.language_detection.utils.get_audio_duration", return_value=600):
        with mock.patch("modules.language_detection.subprocess.run") as mock_run:
            with mock.patch("modules.language_detection.config") as mock_config:
                mock_config.get_temp_dir.return_value = "/tmp"
                mock_config.ENABLE_LD_PREPROCESSING = False

                # Mock tempfile to avoid actual file creation issues in tests
                mock_tmp_path = "modules.language_detection.tempfile.NamedTemporaryFile"
                with mock.patch(mock_tmp_path) as mock_temp:
                    mock_temp.return_value.__enter__.return_value.name = "mock_chunk.wav"

                    result = language_detection.run_voting_detection("dummy.wav", mock_mm)

                    # Verify dynamic size logic: 5% of 600s is 30s, but min is 300s
                    # Verify FFmpeg call
                    mock_run.assert_called_once()
                    cmd = mock_run.call_args[0][0]
                    assert "-t" in cmd
                    # For 600s, chunk_duration = max(300, 600 * 0.05) = 300
                    idx = cmd.index("-t")
                    assert cmd[idx+1] == "300"

                    assert result["detected_language"] == "en"
                    assert result["segments_processed"] == 1
                    assert "en" in result["voting_details"]

def test_run_voting_detection_long_movie():
    """Verify dynamic chunk scaling for long movies."""
    mock_mm = mock.MagicMock()
    mock_mm.run_language_detection.return_value = {
        "detected_language": "fr",
        "confidence": 0.85,
        "all_probabilities": {"fr": 0.85}
    }

    # 4 hour movie = 14400 seconds
    duration = 14400
    with mock.patch("modules.language_detection.utils.get_audio_duration", return_value=duration):
        with mock.patch("modules.language_detection.subprocess.run") as mock_run:
            with mock.patch("modules.language_detection.config") as mock_config:
                mock_config.get_temp_dir.return_value = "/tmp"
                mock_config.ENABLE_LD_PREPROCESSING = False
                mock_tmp_path = "modules.language_detection.tempfile.NamedTemporaryFile"
                with mock.patch(mock_tmp_path) as mock_temp:
                    mock_temp.return_value.__enter__.return_value.name = "movie_chunk.wav"

                    language_detection.run_voting_detection("movie.wav", mock_mm)

                    cmd = mock_run.call_args[0][0]
                    # chunk_duration = max(300, 14400 * 0.05) = 720.0
                    idx = cmd.index("-t")
                    assert cmd[idx+1] == "720.0"

                    # start_offset is now always 0
                    idx_ss = cmd.index("-ss")
                    assert cmd[idx_ss+1] == "0"

def test_run_voting_detection_fallback():
    """Verify fallback when extraction fails."""
    mock_mm = mock.MagicMock()
    mock_mm.run_language_detection.return_value = {"detected_language": "de"}

    with mock.patch("modules.language_detection.utils.get_audio_duration", return_value=100):
        mock_sub = "modules.language_detection.subprocess.run"
        with mock.patch(mock_sub, side_effect=Exception("FFmpeg failed")):
            result = language_detection.run_voting_detection("fail.wav", mock_mm)
            assert result["detected_language"] == "de"

def test_format_detection_result():
    """Verify result schema formatting."""
    details = {"en": 0.95, "fr": 0.005, "de": 0.04}
    res = language_detection._format_detection_result(details, 1)

    assert res["detected_language"] == "en"
    assert res["confidence"] == 0.95
    assert "fr" not in res["voting_details"] # Below 1% threshold
    assert "de" in res["voting_details"]

def test_format_detection_result_empty():
    """Verify result schema formatting with empty details."""
    res = language_detection._format_detection_result({}, 1)
    assert res["detected_language"] == "en"
    assert res["segments_processed"] == 1

def test_run_voting_detection_preprocessing():
    """Verify preprocessing path."""
    mock_mm = mock.MagicMock()
    mock_mm.run_language_detection.return_value = {"all_probabilities": {"en": 1.0}}

    with mock.patch("modules.language_detection.utils.get_audio_duration", return_value=100):
        with mock.patch("modules.language_detection.subprocess.run"):
            with mock.patch("modules.language_detection.config") as mock_config:
                mock_config.ENABLE_LD_PREPROCESSING = True
                with mock.patch("modules.language_detection.preprocessing.get_manager") as mock_pm:
                    mock_pm.return_value.process_audio_file.return_value = "processed.wav"
                    with mock.patch("modules.language_detection.os.path.exists", return_value=True):
                        mock_tmp = "modules.language_detection.tempfile.NamedTemporaryFile"
                        with mock.patch(mock_tmp) as mock_temp:
                            mock_temp.return_value.__enter__.return_value.name = "chunk.wav"
                            language_detection.run_voting_detection("dummy.wav", mock_mm)
                            mock_pm.return_value.process_audio_file.assert_called_once()

def test_run_voting_detection_cleanup_fail():
    """Verify cleanup handles exceptions (coverage only)."""
    mock_mm = mock.MagicMock()
    mock_mm.run_language_detection.return_value = {}

    with mock.patch("modules.language_detection.utils.get_audio_duration", return_value=100):
        with mock.patch("modules.language_detection.subprocess.run"):
            with mock.patch("modules.language_detection.os.path.exists", return_value=True):
                with mock.patch("modules.language_detection.os.remove",
                                side_effect=Exception("perm error")):
                    # This should not raise
                    language_detection.run_voting_detection("dummy.wav", mock_mm)
