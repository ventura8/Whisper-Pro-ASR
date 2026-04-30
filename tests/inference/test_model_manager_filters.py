"""Tests for model_manager post-processing filters."""
# pylint: disable=protected-access
from unittest import mock
from modules.inference import model_manager
from modules import config


def test_post_process_silence_threshold():
    """Test that low probability segments are dropped."""
    # Mock config
    with mock.patch("modules.config.HALLUCINATION_SILENCE_THRESHOLD", 0.85):
        with mock.patch("modules.config.HALLUCINATION_PHRASES", []):
            segments = [
                {"text": "Normal speech", "probability": 0.90},
                {"text": "Ghost noise", "probability": 0.80},
                {"text": "Clear speech", "probability": 0.99}
            ]
            result = {"segments": segments}

            processed = model_manager._post_process_results(result, "dummy")

            p_segments = processed["segments"]
            assert p_segments[0]["text"] == "Normal speech"
            assert p_segments[1]["text"] == ""  # Dropped
            assert p_segments[2]["text"] == "Clear speech"


def test_post_process_repetition_threshold():
    """Test that repetitive segments are dropped after threshold."""
    with mock.patch("modules.config.HALLUCINATION_REPETITION_THRESHOLD", 2):
        with mock.patch("modules.config.HALLUCINATION_PHRASES", []):
            segments = [
                {"text": "Loop", "probability": 0.9},
                {"text": "Loop", "probability": 0.9},  # 1 repetition (allowed)
                {"text": "Loop", "probability": 0.9},  # 2 repetitions (dropped)
                {"text": "Break", "probability": 0.9},
                {"text": "Loop", "probability": 0.9}  # Reset
            ]
            result = {"segments": segments}

            # _post_process_results logs info, so we can let it run
            processed = model_manager._post_process_results(result, "dummy")

            p_segments = processed["segments"]
            assert p_segments[0]["text"] == "Loop"
            assert p_segments[1]["text"] == "Loop"
            assert p_segments[2]["text"] == ""  # Dropped
            assert p_segments[3]["text"] == "Break"
            assert p_segments[4]["text"] == "Loop"


def test_post_process_phrases():
    """Test that known hallucination phrases are dropped."""
    with mock.patch("modules.config.HALLUCINATION_PHRASES", ["bad phrase"]):
        segments = [
            {"text": "Good text", "probability": 0.9},
            {"text": "Bad phrase", "probability": 0.9},
            # Keep if too long diff, logic: len(clean) < len(phrase) + 10
            {"text": "Bad phrase here", "probability": 0.9}
        ]
        result = {"segments": segments}

        processed = model_manager._post_process_results(result, "dummy")

        p_segments = processed["segments"]
        assert p_segments[0]["text"] == "Good text"
        assert p_segments[1]["text"] == ""
        # "Bad phrase here" length 15. "Bad phrase" len 10. 15 < 20. Should drop.
        assert p_segments[2]["text"] == ""
