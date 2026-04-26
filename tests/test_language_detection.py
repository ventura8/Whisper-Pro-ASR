"""Tests for modules/language_detection.py."""
from unittest import mock
from modules import language_detection

# pylint: disable=protected-access, too-many-locals


def test_calculate_chunk_duration():
    """Verify linear scaling of chunk duration."""
    # Under 1h
    assert language_detection._calculate_chunk_duration(0) == 300
    assert language_detection._calculate_chunk_duration(1800) == 390
    assert language_detection._calculate_chunk_duration(3600) == 480

    # 1h to 4h
    duration_2h = 7200
    expected_2h = 480 + (7200 - 3600) * (720 / 10800)
    assert language_detection._calculate_chunk_duration(duration_2h) == expected_2h
    assert language_detection._calculate_chunk_duration(14400) == 1200

    # Over 4h
    assert language_detection._calculate_chunk_duration(20000) == 1200


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
                v_path = "modules.language_detection.vad.get_speech_timestamps_from_path"
                with mock.patch(v_path) as mock_vad:
                    mock_config.get_temp_dir.return_value = "/tmp"
                    mock_config.ENABLE_LD_PREPROCESSING = False
                    mock_vad.return_value = [{'start': 0, 'end': 10}]  # Speech found

                    # Mock tempfile to avoid actual file creation issues in tests
                    mock_tmp_path = "modules.language_detection.tempfile.NamedTemporaryFile"
                    with mock.patch(mock_tmp_path) as mock_temp:
                        mock_temp.return_value.__enter__.return_value.name = "mock_chunk.wav"

                        result = language_detection.run_voting_detection("dummy.wav", mock_mm)

                        # Verify dynamic size logic: 300 + 600 * 0.05 = 330
                        mock_run.assert_called_once()
                        cmd = mock_run.call_args[0][0]
                        assert "-t" in cmd
                        idx = cmd.index("-t")
                        assert cmd[idx+1] == "330.0"

                        assert result["detected_language"] == "en"
                        assert result["segments_processed"] == 1


def test_run_voting_detection_with_retry():
    """Verify that run_voting_detection retries on silent chunks."""
    mock_mm = mock.MagicMock()
    mock_mm.run_language_detection.return_value = {
        "detected_language": "en",
        "confidence": 0.9,
        "all_probabilities": {"en": 0.9}
    }

    with mock.patch("modules.language_detection.utils.get_audio_duration", return_value=3600):
        with mock.patch("modules.language_detection.subprocess.run"):
            with mock.patch("modules.language_detection.config") as mock_config:
                v_path = "modules.language_detection.vad.get_speech_timestamps_from_path"
                with mock.patch(v_path) as mock_vad:
                    mock_config.get_temp_dir.return_value = "/tmp"
                    mock_config.ENABLE_LD_PREPROCESSING = False

                    # First attempt: silent, Second attempt: speech
                    mock_vad.side_effect = [[], [{'start': 0, 'end': 1}]]

                    mock_tmp_path = "modules.language_detection.tempfile.NamedTemporaryFile"
                    with mock.patch(mock_tmp_path) as mock_temp:
                        mock_temp.return_value.__enter__.return_value.name = "mock_chunk.wav"

                        result = language_detection.run_voting_detection("dummy.wav", mock_mm)

                        # Verify we returned a result and performed 2 attempts
                        assert result["detected_language"] == "en"
                        assert result["segments_processed"] == 2
                        assert mock_vad.call_count == 2


def test_run_voting_detection_all_silent():
    """Verify behavior when all chunks are silent."""
    mock_mm = mock.MagicMock()

    with mock.patch("modules.language_detection.utils.get_audio_duration", return_value=3600):
        with mock.patch("modules.language_detection.subprocess.run"):
            with mock.patch("modules.language_detection.config") as mock_config:
                v_path = "modules.language_detection.vad.get_speech_timestamps_from_path"
                with mock.patch(v_path) as mock_vad:
                    mock_config.get_temp_dir.return_value = "/tmp"
                    mock_config.ENABLE_LD_PREPROCESSING = False

                    # All attempts return no speech
                    mock_vad.return_value = []

                    mock_tmp_path = "modules.language_detection.tempfile.NamedTemporaryFile"
                    with mock.patch(mock_tmp_path) as mock_temp:
                        mock_temp.return_value.__enter__.return_value.name = "mock_chunk.wav"

                        result = language_detection.run_voting_detection("dummy.wav", mock_mm)

                        # Should return fallback (en)
                        assert result["detected_language"] == "en"
                        assert result["confidence"] == 0.0
                        assert mock_mm.run_language_detection.called is False


def test_run_voting_detection_fallback():
    """Verify fallback when extraction fails repeatedly."""
    mock_mm = mock.MagicMock()
    mock_mm.run_language_detection.return_value = {"detected_language": "de"}

    with mock.patch("modules.language_detection.utils.get_audio_duration", return_value=100):
        mock_sub = "modules.language_detection.subprocess.run"
        # Always fail FFmpeg
        with mock.patch(mock_sub, side_effect=Exception("FFmpeg failed")):
            # It will retry 5 times and then return the last_result or fallback
            result = language_detection.run_voting_detection("fail.wav", mock_mm)
            # Since last_result is None, it returns default en
            assert result["detected_language"] == "en"


def test_format_detection_result():
    """Verify result schema formatting."""
    details = {"en": 0.95, "fr": 0.005, "de": 0.04}
    res = language_detection._format_detection_result(details, 1)

    assert res["detected_language"] == "en"
    assert res["confidence"] == 0.95
    assert "fr" not in res["voting_details"]  # Below 1% threshold
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
                v_path = "modules.language_detection.vad.get_speech_timestamps_from_path"
                with mock.patch(v_path) as mock_vad:
                    mock_vad.return_value = [{'start': 0, 'end': 1}]
                    mock_config.ENABLE_LD_PREPROCESSING = True
                    mock_prep = "modules.language_detection.preprocessing.get_manager"
                    with mock.patch(mock_prep) as mock_gm:
                        mock_pm = mock.MagicMock()
                        mock_pm.process_audio_file.return_value = "processed.wav"
                        mock_gm.return_value = mock_pm
                        with mock.patch("modules.language_detection.os.path.exists",
                                        return_value=True):
                            mock_tmp = "modules.language_detection.tempfile.NamedTemporaryFile"
                            with mock.patch(mock_tmp) as mock_temp:
                                mock_temp.return_value.__enter__.return_value.name = "chunk.wav"
                                language_detection.run_voting_detection("dummy.wav", mock_mm)
                                mock_pm.process_audio_file.assert_called_once()


def test_run_voting_detection_backup_vad():
    """Verify that backup VAD logic triggers when isolation is too aggressive."""
    mock_mm = mock.MagicMock()
    mock_mm.run_language_detection.return_value = {"all_probabilities": {"en": 1.0}}

    with mock.patch("modules.language_detection.utils.get_audio_duration", return_value=100):
        with mock.patch("modules.language_detection.subprocess.run"):
            with mock.patch("modules.language_detection.config") as mock_config:
                v_path = "modules.language_detection.vad.get_speech_timestamps_from_path"
                with mock.patch(v_path) as mock_vad:
                    mock_config.ENABLE_LD_PREPROCESSING = True
                    mock_config.get_temp_dir.return_value = "/tmp"

                    # First call (isolated): silent
                    # Second call (original): speech found
                    mock_vad.side_effect = [[], [{'start': 0, 'end': 1}]]

                    mock_prep = "modules.language_detection.preprocessing.get_manager"
                    with mock.patch(mock_prep) as mock_gm:
                        mock_pm = mock.MagicMock()
                        mock_pm.process_audio_file.return_value = "processed.wav"
                        mock_gm.return_value = mock_pm
                        with mock.patch("modules.language_detection.os.path.exists",
                                        return_value=True):
                            mock_tmp = "modules.language_detection.tempfile.NamedTemporaryFile"
                            with mock.patch(mock_tmp) as mock_temp:
                                mock_temp.return_value.__enter__.return_value.name = "chunk.wav"

                                language_detection.run_voting_detection("dummy.wav", mock_mm)

                                # Verify VAD was called twice (isolated then original)
                                assert mock_vad.call_count == 2
                                # Verify we still ran detection on something
                                mock_mm.run_language_detection.assert_called_once()
