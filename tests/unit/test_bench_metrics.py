"""Scoring for the engine benchmark (tests/real_audio/bench_metrics.py)."""

import json

import pytest

from modules.core import nemo_catalog
from tests.real_audio import bench_metrics


def test_benchmark_buckets_match_the_engines_language_set():
    """Mirrored rather than imported in the benchmark; this keeps the two from drifting."""
    assert bench_metrics.NEMO_LANGUAGES == nemo_catalog.EU25


@pytest.mark.parametrize(
    ("reference", "hypothesis", "distance"),
    [([], [], 0), (["a"], [], 1), ([], ["a"], 1), (["a", "b", "c"], ["a", "x", "c"], 1), (["a", "b"], ["b", "a"], 2)],
)
def test_edit_distance(reference, hypothesis, distance):
    """Levenshtein distance over token sequences."""
    assert bench_metrics.edit_distance(reference, hypothesis) == distance


def test_word_error_rate_uses_the_matrix_normalization():
    """Punctuation and case do not count as errors."""
    assert bench_metrics.error_rate("The quick, brown fox!", ["the", "quick", "brown", "fox"]) == 0.0
    assert bench_metrics.error_rate("The quick brown fox", ["the", "slow", "brown", "fox"]) == 0.25


def test_character_error_rate_for_scripts_scored_by_characters():
    """CJK is scored by characters, as the matrix scores it."""
    assert bench_metrics.error_rate("你好世界", ["你好", "世界"], tokenizer="chars") == 0.0
    assert bench_metrics.error_rate("你好世界", ["你好", "世"], tokenizer="chars") == 0.25


def test_folded_characters_ignore_combining_marks():
    """chars_folded drops tone and vowel marks."""
    assert bench_metrics.units(["kọ̀lọ̀"], "chars_folded") == list("kolo")


def test_error_rate_is_capped_so_one_hallucination_cannot_swamp_a_mean():
    """An invented paragraph scores 1.0, not 400%."""
    assert bench_metrics.error_rate("yes", ["a", "whole", "invented", "paragraph"]) == 1.0


def test_empty_reference():
    """An empty reference is perfect only when nothing was said."""
    assert bench_metrics.error_rate("", []) == 0.0
    assert bench_metrics.error_rate("", ["noise"]) == 1.0


def test_bucket():
    """Covered languages versus those routed to Whisper."""
    assert bench_metrics.bucket("de") == "nemo-covered"
    assert bench_metrics.bucket("ja") == "whisper-routed"


def _row(mode, bucket, **values):
    """A minimal benchmark record."""
    return {"mode": mode, "bucket": bucket, **values}


def test_summary_groups_by_mode_and_bucket():
    """One summary row per mode and bucket, with the right aggregates."""
    records = [
        _row("transcribe+lang", "nemo-covered", error_rate=0.1, rtf=0.02, wall_sec=1.0),
        _row("transcribe+lang", "nemo-covered", error_rate=0.3, rtf=0.04, wall_sec=3.0, hallucinated=True),
        _row("detect", "whisper-routed", error_rate=None, detect_correct=True, wall_sec=2.0),
        _row("detect", "whisper-routed", error_rate=None, detect_correct=False, wall_sec=4.0),
    ]
    detect, transcribe = bench_metrics.summarize(records)
    assert transcribe["mean_error_rate"] == 0.2
    assert transcribe["hallucinated"] == 1
    assert transcribe["median_rtf"] == 0.03
    assert detect["detect_correct"] == 0.5
    assert detect["mean_error_rate"] is None
    assert detect["median_wall_sec"] == 3.0


def test_report_files(tmp_path):
    """Both report files are written and the JSON keeps every record."""
    records = [_row("transcribe+lang", "nemo-covered", error_rate=0.0, rtf=0.01, wall_sec=1.0)]
    markdown = bench_metrics.write_report(tmp_path / "out", "PARAKEET-local", records, {"engine": "PARAKEET", "host": "local"})
    assert "# Engine benchmark: PARAKEET on local" in markdown.read_text(encoding="utf-8")
    assert "| transcribe+lang | nemo-covered | 1 | 0.0 |" in markdown.read_text(encoding="utf-8")
    saved = json.loads((tmp_path / "out" / "PARAKEET-local.json").read_text(encoding="utf-8"))
    assert saved["records"] == records
    assert saved["summary"][0]["clips"] == 1
