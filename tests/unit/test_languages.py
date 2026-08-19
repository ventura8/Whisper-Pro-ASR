"""Tests for static language mapping helpers."""

from modules.core import languages


def test_languages_contains_common_codes():
    """Language map should include common Whisper ISO codes."""
    assert languages.LANGUAGES["en"] == "English"
    assert languages.LANGUAGES["de"] == "German"
    assert "zh" in languages.LANGUAGES
