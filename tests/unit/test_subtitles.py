"""Tests for subtitle promo generation utilities"""

from unittest import mock

from modules.core import utils


def test_subtitle_promo_generation():
    """Verify that generate_srt and generate_vtt correctly prepend the promo card when enabled."""
    res = {"segments": [{"start": 1.0, "end": 2.5, "text": "Dialogue text"}]}

    # 1. Enabled (default)
    with (
        mock.patch("modules.core.config.SUBTITLE_PROMO_ENABLED", True),
        mock.patch("modules.core.config.SUBTITLE_PROMO_TEXT", "Made with Whisper Pro ASR"),
        mock.patch("modules.core.config.SUBTITLE_PROMO_DURATION", 3.0),
    ):
        srt = utils.generate_srt(res)
        assert "1\n00:00:00,000 --> 00:00:03,000\nMade with Whisper Pro ASR" in srt
        assert "2\n00:00:01,000 --> 00:00:02,500\nDialogue text" in srt

        vtt = utils.generate_vtt(res)
        assert "1\n00:00:00.000 --> 00:00:03.000\nMade with Whisper Pro ASR" in vtt
        assert "2\n00:00:01.000 --> 00:00:02.500\nDialogue text" in vtt

    # 2. Disabled
    with mock.patch("modules.core.config.SUBTITLE_PROMO_ENABLED", False):
        srt = utils.generate_srt(res)
        assert "Made with Whisper Pro ASR" not in srt
        assert "1\n00:00:01,000 --> 00:00:02,500\nDialogue text" in srt

        vtt = utils.generate_vtt(res)
        assert "Made with Whisper Pro ASR" not in vtt
        assert "1\n00:00:01.000 --> 00:00:02.500\nDialogue text" in vtt

    # 3. Fallback segment timings are shifted when promo is enabled
    empty_res = {"segments": []}
    with (
        mock.patch("modules.core.config.SUBTITLE_PROMO_ENABLED", True),
        mock.patch("modules.core.config.SUBTITLE_PROMO_TEXT", "Made with Whisper Pro ASR"),
        mock.patch("modules.core.config.SUBTITLE_PROMO_DURATION", 3.0),
    ):
        srt = utils.generate_srt(empty_res)
        assert "1\n00:00:00,000 --> 00:00:03,000\nMade with Whisper Pro ASR" in srt
        assert "2\n00:00:03,000 --> 00:00:08,000\n[No dialogue detected]" in srt
