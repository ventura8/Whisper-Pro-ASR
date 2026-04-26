"""Test suite for RAM usage optimization features."""
from unittest import mock
import pytest
from flask import Flask
from modules import routes

# pylint: disable=protected-access


class TestRAMOptimization:
    """Test suite for RAM usage optimization features."""

    def test_handle_upload_streamed_saving(self):
        """Test that _handle_upload uses .save() for streamed ingestion."""
        app = Flask(__name__)
        mock_file = mock.MagicMock()
        mock_file.filename = "test.wav"
        mock_file.save = mock.MagicMock()

        with app.test_request_context(method='POST'):
            with mock.patch("modules.routes.request") as mock_req:
                mock_req.files = {"audio_file": mock_file}
                mock_req.content_length = 2048
                with mock.patch("modules.routes.config") as cfg:
                    cfg.get_temp_dir.return_value = "/tmp/whisper"
                    with mock.patch("os.path.getsize", return_value=2048):
                        with mock.patch("builtins.open", mock.mock_open(read_data=b"a" * 1024)):
                            with mock.patch("os.path.exists", return_value=True):
                                res_path, _, _ = routes._handle_upload()

                                # Verify .save() was called (streamed)
                                mock_file.save.assert_called_once()
                                assert res_path.startswith("/tmp/whisper")
                                assert "upload_" in res_path

    def test_handle_upload_null_byte_verification(self):
        """Test that _handle_upload still detects corrupted files (null bytes) in streamed mode."""
        app = Flask(__name__)
        mock_file = mock.MagicMock()
        mock_file.filename = "test.wav"

        with app.test_request_context(method='POST'):
            with mock.patch("modules.routes.request") as mock_req:
                mock_req.files = {"audio_file": mock_file}
                mock_req.content_length = 2048
                with mock.patch("modules.routes.config") as cfg:
                    cfg.get_temp_dir.return_value = "/tmp/whisper"
                    with mock.patch("os.path.getsize", return_value=2048):
                        # Mock reading back the header (all zeros)
                        with mock.patch("builtins.open", mock.mock_open(read_data=b"\x00" * 1024)):
                            with mock.patch("os.path.exists", return_value=True):
                                with mock.patch("os.remove") as mock_remove:
                                    with pytest.raises(ValueError, match="corrupted"):
                                        routes._handle_upload()
                                    # Verify cleanup happened
                                    mock_remove.assert_called()
