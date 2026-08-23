"""Shared fixtures for integration tests."""

from __future__ import annotations

import os
from typing import Any, Optional
from unittest import mock

import pytest
from fastapi import FastAPI

from modules.api.routes import asr as routes_asr
from modules.api.routes import detect as routes_detect
from modules.api.routes import system as routes_system
from tests.conftest import FlaskCompatibleClient
from whisper_pro_asr import create_app

_BAZARR_TRANSCRIPTION_RESULT = {
    "text": "Hello world from Bazarr",
    "segments": [
        {"timestamp": (0.0, 2.5), "text": "Hello world"},
        {"timestamp": (2.5, 5.0), "text": "from Bazarr"},
    ],
}

_BAZARR_DETECTION_RESULT = {
    "confidence": 0.97,
    "detected_language": "en",
    "language": "en",
    "language_code": "en",
}


@pytest.fixture()
def bazarr_wav(tmp_path):
    """Create a realistic WAV file simulating a Bazarr volume-mapped media path."""
    media_dir = tmp_path / "media" / "movies"
    media_dir.mkdir(parents=True)
    wav_file = media_dir / "Movie (2024).mkv"
    wav_file.write_bytes(b"RIFF" + b"\x00" * 40)
    return str(wav_file)


def _configure_bazarr_mocks(mock_mm_asr, mock_mm_det, mock_ld):
    """Wire up Bazarr-specific mock return values on pre-patched managers."""
    mock_mm_asr.is_engine_initialized.return_value = True
    mock_mm_det.is_engine_initialized.return_value = True
    mock_mm_asr.increment_active_session.return_value = None
    mock_mm_asr.decrement_active_session.return_value = None
    mock_mm_det.increment_active_session.return_value = None
    mock_mm_det.decrement_active_session.return_value = None
    mock_ld.return_value = _BAZARR_DETECTION_RESULT
    mock_mm_asr.run_transcription.return_value = _BAZARR_TRANSCRIPTION_RESULT


def mock_asr_manager_for_transcription(
    mock_mm: mock.MagicMock,
    result: Optional[dict[str, Any]] = None,
) -> None:
    """Wire up a mocked ``modules.api.routes.asr.model_manager`` for a single
    successful transcription call, without also patching the detect-language
    manager/voting function (unlike ``_configure_bazarr_mocks``) — for tests
    that only exercise the transcription routes directly."""
    mock_mm.is_engine_initialized.return_value = True
    mock_mm.increment_active_session.return_value = None
    mock_mm.decrement_active_session.return_value = None
    mock_mm.run_transcription.return_value = result if result is not None else _BAZARR_TRANSCRIPTION_RESULT


@pytest.fixture()
def bazarr_client():
    """Create test client with mocked inference layer for Bazarr integration."""
    with (
        mock.patch("modules.api.routes.asr.model_manager") as mock_mm_asr,
        mock.patch("modules.api.routes.detect.model_manager") as mock_mm_det,
        mock.patch("modules.inference.pipeline.language_detection.run_voting_detection") as mock_ld,
    ):
        _configure_bazarr_mocks(mock_mm_asr, mock_mm_det, mock_ld)
        app = FastAPI(title="Bazarr Integration Test")
        app.include_router(routes_system.router)
        app.include_router(routes_asr.router)
        app.include_router(routes_detect.router)
        yield FlaskCompatibleClient(app)


@pytest.fixture()
def bazarr_secured_client():
    """Create a full app client with API key middleware enabled for Bazarr endpoint coverage."""
    with (
        mock.patch.dict(os.environ, {"API_KEY": "integration-secret"}, clear=False),
        mock.patch("modules.api.routes.asr.model_manager") as mock_mm_asr,
        mock.patch("modules.api.routes.detect.model_manager") as mock_mm_det,
        mock.patch("modules.inference.pipeline.language_detection.run_voting_detection") as mock_ld,
    ):
        _configure_bazarr_mocks(mock_mm_asr, mock_mm_det, mock_ld)
        app = create_app(testing=True)
        yield FlaskCompatibleClient(app)
