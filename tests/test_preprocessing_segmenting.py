"""Tests for segmented vocal separation in PreprocessingManager."""
import unittest
from unittest import mock
import numpy as np
from modules.preprocessing import PreprocessingManager


class TestPreprocessingSegmenting(unittest.TestCase):
    """Test suite for segmented processing in PreprocessingManager."""

    def setUp(self):
        self.mock_logger = mock.patch('modules.preprocessing.logger').start()

    def tearDown(self):
        mock.patch.stopall()

    def test_segmenting_triggered(self):
        """Test that large files trigger segmenting logic."""
        # pylint: disable=protected-access, unused-variable
        with mock.patch('modules.preprocessing.config') as mock_config:
            mock_config.VOCAL_SEPARATION_SEGMENT_DURATION = 10
            mock_config.ENABLE_VOCAL_SEPARATION = True
            mock_config.PREPROCESS_DEVICE = "CPU"
            mock_config.PREPROCESSING_CACHE_DIR = "/tmp"

            manager = PreprocessingManager()
            manager.separator = mock.MagicMock()
            manager._init_separator = mock.MagicMock()

            # Mock soundfile
            with mock.patch('modules.preprocessing.sf') as mock_sf:
                # Mock file info (long file)
                mock_info = mock.MagicMock()
                mock_info.samplerate = 16000
                mock_info.frames = 16000 * 25  # 25 seconds
                mock_sf.info.return_value = mock_info

                # Mock SoundFile context manager
                mock_writer = mock.MagicMock()
                mock_sf.SoundFile.return_value.__enter__.return_value = mock_writer

                # Mock read (return dummy data)
                mock_sf.read.return_value = (np.zeros((16000, 2)), 16000)

                # Mock separate
                manager.separator.separate.return_value = ["segment_vocals.wav"]

                # Mock identify vocals
                manager._identify_vocals = mock.MagicMock(
                    return_value="segment_vocals.wav")
                manager._cleanup_stems = mock.MagicMock()

                # Mock file ops
                with mock.patch('os.path.exists', return_value=True):
                    with mock.patch('os.remove'):
                        with mock.patch(
                                'tempfile.mkstemp',
                                return_value=(1, "temp.wav")):
                            with mock.patch('os.close'):
                                with mock.patch(
                                        'modules.preprocessing.utils.get_audio_duration',
                                        return_value=25.0):
                                    # Also patch format_duration
                                    with mock.patch(
                                            'modules.preprocessing.utils.format_duration',
                                            return_value="25s"):
                                        result = manager.process_audio_file(
                                            "long_audio.wav")

        # Assertions
        # Should have called separate 3 times (10s, 10s, 5s)
        self.assertEqual(manager.separator.separate.call_count, 3)
        self.assertTrue(mock_writer.write.called)
