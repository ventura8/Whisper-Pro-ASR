"""Test suite for RAM usage optimization features."""

from unittest import mock

import pytest

from modules.api import routes_utils as routes


class TestRAMOptimization:
    """Test suite for RAM usage optimization features."""

    def test_handle_upload_streamed_saving(self):
        """Test that handle_upload uses chunked reading for streamed ingestion."""
        mock_file = mock.MagicMock()
        mock_file.filename = "test.wav"

        # Mock UploadFile structure
        mock_file.file = mock.MagicMock()
        mock_file.file.read.side_effect = [b"a" * 2048, b""]

        with mock.patch("modules.api.routes_utils.config") as cfg:
            cfg.get_temp_dir.return_value = "/tmp/whisper"
            with mock.patch("os.path.getsize", return_value=2048):
                with mock.patch("builtins.open", mock.mock_open()):
                    with mock.patch("os.path.exists", return_value=True):
                        res_path, _, _ = routes.handle_upload(mock_file)

                        # Verify that we read from the file stream in chunks
                        mock_file.file.read.assert_called()
                        assert res_path.startswith("/tmp/whisper")
                        assert "upload_" in res_path

    def test_handle_upload_null_byte_verification(self):
        """Test that handle_upload still detects corrupted files (null bytes) in streamed mode."""
        mock_file = mock.MagicMock()
        mock_file.filename = "test.wav"
        mock_file.file = mock.MagicMock()
        mock_file.file.read.side_effect = [b"\x00" * 2048, b""]

        with mock.patch("modules.api.routes_utils.config") as cfg:
            cfg.get_temp_dir.return_value = "/tmp/whisper"
            with mock.patch("os.path.getsize", return_value=2048):
                # Mock reading back the header (all zeros)
                with mock.patch("builtins.open", mock.mock_open(read_data=b"\x00" * 1024)):
                    with mock.patch("os.path.exists", return_value=True):
                        with mock.patch("os.remove") as mock_remove:
                            with pytest.raises(ValueError, match="corrupted"):
                                routes.handle_upload(mock_file)
                            # Verify cleanup happened
                            mock_remove.assert_called()
