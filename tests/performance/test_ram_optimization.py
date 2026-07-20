"""Test suite for RAM usage optimization features."""

from unittest import mock

import pytest

from modules.api.support import request_utils as routes


class TestRAMOptimization:
    """Test suite for RAM usage optimization features."""

    def test_handle_upload_streamed_saving(self):
        """Test that handle_upload uses chunked reading for streamed ingestion."""
        mock_file = mock.MagicMock()
        mock_file.filename = "test.wav"

        # Mock UploadFile structure
        mock_file.file = mock.MagicMock()
        mock_file.file.read.side_effect = [b"a" * 2048, b""]

        with mock.patch("modules.api.support.request_utils.config") as cfg:
            cfg.get_temp_dir.return_value = "/tmp/whisper"
            with mock.patch("os.path.getsize", return_value=2048):
                with mock.patch("builtins.open", mock.mock_open()):
                    with mock.patch("os.path.exists", return_value=True):
                        res_path, _, _ = routes.handle_upload(mock_file)

                        # Verify that we read from the file stream in chunks
                        assert mock_file.file.read.call_count >= 2
                        assert res_path.startswith("/tmp/whisper")
                        assert "upload_" in res_path

    def test_handle_upload_empty_stream_detection(self):
        """Test that handle_upload detects empty streams in streamed mode."""
        mock_file = mock.MagicMock()
        mock_file.filename = "test.wav"
        mock_file.file = mock.MagicMock()
        mock_file.file.read.side_effect = [b"", b""]

        with mock.patch("modules.api.support.request_utils.config") as cfg:
            cfg.get_temp_dir.return_value = "/tmp/whisper"
            with mock.patch("os.path.getsize", return_value=0):
                with mock.patch("builtins.open", mock.mock_open(read_data=b"")):
                    with mock.patch("os.path.exists", return_value=True):
                        with mock.patch("os.remove") as mock_remove:
                            with pytest.raises(ValueError, match="empty"):
                                routes.handle_upload(mock_file)
                            mock_remove.assert_called()
