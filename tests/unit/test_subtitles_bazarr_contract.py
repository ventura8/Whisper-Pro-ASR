"""Bazarr-compatible output format contract/edge-case tests (pure formatter level).

Closes coverage gaps identified in docs/E2E_TEST_PLAN_ORCHESTRATION.md section 6:

- 6.6 Non-ASCII / multi-byte text correctness (accents, CJK, RTL) across SRT/VTT/JSON.
- 6.8 Overlapping / out-of-order segment timestamps.
- 6.9 Real structural parseability of generated SRT/VTT via a strict, independent validator.

6.3 (JSON schema/contract stability) and the end-to-end / HTTP-level parts of
6.6 and 6.9 live in tests/integration/test_bazarr_integration.py, which has
the fixtures (`bazarr_client`, `bazarr_wav`) needed to exercise the real API
routes.
"""

from __future__ import annotations

import json
import re
from typing import Any, TypedDict
from unittest import mock

import pytest

from modules.core import utils


class SubtitleEntry(TypedDict):
    """One parsed SRT block / VTT cue, as returned by `_validate_one_block`/`_validate_blocks`."""

    index: int
    start: float
    end: float
    text: str


# ---------------------------------------------------------------------------
# 6.9: strict, independent SRT/VTT structural validators.
#
# These are intentionally NOT reusing any internal formatting helpers from
# modules/core/subtitles.py -- they parse the *output text* the way a real
# third-party consumer (Bazarr / pysrt / webvtt-py) would, so a regression in
# the internal string-building code is caught even if the unit tests that
# assert against the same code paths are wrong or too permissive.
# ---------------------------------------------------------------------------

_SRT_TIMESTAMP = r"\d{2}:[0-5]\d:[0-5]\d,\d{3}"
_VTT_TIMESTAMP = r"\d{2,}:[0-5]\d:[0-5]\d\.\d{3}"


def _timestamp_to_seconds(ts: str) -> float:
    hms, ms = re.split(r"[,.]", ts)
    h, m, s = (int(x) for x in hms.split(":"))
    return h * 3600 + m * 60 + s + int(ms) / 1000.0


def _split_block_header_and_text(block: str, block_name: str, expected_idx: int) -> tuple[str, str, list[str]]:
    """Split one block into (index_line, timestamp_line, text_lines), asserting
    it has at least the minimum 3 lines a valid block/cue requires."""
    lines = block.split("\n")
    assert len(lines) >= 3, f"{block_name} {expected_idx} malformed (fewer than 3 lines): {block!r}"
    idx_line, ts_line, *text_lines = lines
    return idx_line, ts_line, text_lines


def _assert_sequential_index(idx_line: str, block_name: str, expected_idx: int) -> None:
    """Assert the block/cue's index line is numeric and matches the expected sequence position."""
    assert idx_line.strip().isdigit(), f"{block_name} {expected_idx} has non-numeric index: {idx_line!r}"
    assert int(idx_line.strip()) == expected_idx, f"{block_name} index not sequential: expected {expected_idx}, got {idx_line.strip()}"


def _parse_timestamp_line(ts_line: str, timestamp_re: str, block_name: str, expected_idx: int) -> tuple[float, float]:
    """Match `ts_line` against `timestamp_re` via `re.fullmatch` (both formats
    use fullmatch: the real generator never emits trailing VTT cue settings,
    so a fullmatch is appropriate for both and catches any unexpected trailing
    text) and return (start_seconds, end_seconds)."""
    m = re.fullmatch(rf"({timestamp_re}) --> ({timestamp_re})", ts_line.strip())
    assert m, f"{block_name} {expected_idx} has invalid timestamp line: {ts_line!r}"
    return _timestamp_to_seconds(m.group(1)), _timestamp_to_seconds(m.group(2))


def _assert_monotonic_start(start_ts: float, prev_start: float, block_name: str, expected_idx: int) -> None:
    """Assert `start_ts` is not earlier than the previous block/cue's start time."""
    is_monotonic = start_ts >= prev_start
    monotonic_msg = f"{block_name} {expected_idx} start {start_ts} < previous start {prev_start} (non-monotonic)"
    assert is_monotonic, monotonic_msg


def _validate_one_block(block: str, timestamp_re: str, block_name: str, expected_idx: int, prev_start: float) -> SubtitleEntry:
    """Validate a single SRT block / VTT cue against the shared structural rules
    (sequential index, valid timestamp line, non-empty text, monotonic start),
    parameterized by format-specific timestamp regex and block-name label."""
    idx_line, ts_line, text_lines = _split_block_header_and_text(block, block_name, expected_idx)
    _assert_sequential_index(idx_line, block_name, expected_idx)

    start_ts, end_ts = _parse_timestamp_line(ts_line, timestamp_re, block_name, expected_idx)
    assert end_ts > start_ts, f"{block_name} {expected_idx} end {end_ts} is not after start {start_ts}"

    text = "\n".join(text_lines).strip()
    assert text, f"{block_name} {expected_idx} has empty text"

    _assert_monotonic_start(start_ts, prev_start, block_name, expected_idx)
    return {"index": expected_idx, "start": start_ts, "end": end_ts, "text": text}


def _validate_blocks(blocks: list[str], timestamp_re: str, block_name: str) -> list[SubtitleEntry]:
    """Validate a sequence of SRT blocks / VTT cues, tracking monotonicity across them."""
    parsed: list[SubtitleEntry] = []
    prev_start = -1.0
    for expected_idx, block in enumerate(blocks, start=1):
        entry = _validate_one_block(block, timestamp_re, block_name, expected_idx, prev_start)
        prev_start = entry["start"]
        parsed.append(entry)
    return parsed


def validate_srt_structure(content: str) -> list[SubtitleEntry]:
    """Strictly validate SRT structure like a real downstream parser would.

    Enforces: sequential block numbers starting at 1, valid HH:MM:SS,mmm -->
    HH:MM:SS,mmm timestamp lines, non-empty text, and non-decreasing start
    times across blocks. Raises AssertionError with a precise reason on any
    violation.
    """
    blocks = [b for b in content.strip().split("\n\n") if b.strip()]
    assert blocks, "SRT content has no blocks"
    return _validate_blocks(blocks, _SRT_TIMESTAMP, "Block")


def validate_vtt_structure(content: str) -> list[SubtitleEntry]:
    """Strictly validate WebVTT structure like a real downstream parser would."""
    assert content.startswith("WEBVTT"), "VTT content missing WEBVTT header"
    body = content.removeprefix("WEBVTT").lstrip("\n")
    blocks = [b for b in body.strip().split("\n\n") if b.strip()]
    assert blocks, "VTT content has no cue blocks"
    return _validate_blocks(blocks, _VTT_TIMESTAMP, "Cue")


def _no_promo() -> Any:
    """Existing subtitle tests explicitly toggle the promo card; do the same here
    so these formatter-level tests aren't coupled to whatever the environment's
    SUBTITLE_PROMO_ENABLED default happens to be.
    """
    return mock.patch("modules.core.config.SUBTITLE_PROMO_ENABLED", False)


class TestSrtVttThirdPartyParseability:
    """6.9: run generated SRT/VTT through the strict structural validator above."""

    def test_srt_output_is_structurally_valid(self):
        """Verify generated SRT passes the strict structural validator with correct text/order."""
        result = {
            "segments": [
                {"start": 0.0, "end": 2.0, "text": "First line"},
                {"start": 2.0, "end": 4.5, "text": "Second line"},
                {"start": 4.5, "end": 7.25, "text": "Third line"},
            ]
        }
        with _no_promo():
            srt = utils.generate_srt(result)
        parsed = validate_srt_structure(srt)
        assert [b["text"] for b in parsed] == ["First line", "Second line", "Third line"]
        assert [b["index"] for b in parsed] == [1, 2, 3]

    def test_vtt_output_is_structurally_valid(self):
        """Verify generated VTT passes the strict structural validator with correct text order."""
        result = {
            "segments": [
                {"start": 0.0, "end": 2.0, "text": "First line"},
                {"start": 2.0, "end": 4.5, "text": "Second line"},
            ]
        }
        with _no_promo():
            vtt = utils.generate_vtt(result)
        parsed = validate_vtt_structure(vtt)
        assert [b["text"] for b in parsed] == ["First line", "Second line"]

    def test_bazarr_fixture_srt_and_vtt_pass_strict_validator(self):
        """Use the same shape of fixture the Bazarr integration tests exercise via the real API."""
        result = {
            "segments": [
                {"timestamp": ts, "text": text}
                for (ts, text) in [
                    ((0.0, 2.5), "Hello world"),
                    ((2.5, 5.0), "from Bazarr"),
                ]
            ]
        }
        with _no_promo():
            srt = utils.generate_srt(result)
            vtt = utils.generate_vtt(result)
        validate_srt_structure(srt)
        validate_vtt_structure(vtt)

    def test_srt_rejects_out_of_range_minutes_seconds_timestamp(self):
        """Verify the validator rejects an out-of-range timestamp like 00:99:99,000
        (minutes/seconds must be 00-59) instead of matching it as well-formed."""
        malformed_srt = "1\n00:99:99,000 --> 00:00:01,000\nBad timestamp\n"
        with pytest.raises(AssertionError, match="invalid timestamp"):
            validate_srt_structure(malformed_srt)

    def test_vtt_rejects_out_of_range_minutes_seconds_timestamp(self):
        """Verify the validator rejects an out-of-range timestamp like 00:99:99.000
        (minutes/seconds must be 00-59) instead of matching it as well-formed."""
        malformed_vtt = "WEBVTT\n\n1\n00:99:99.000 --> 00:00:01.000\nBad timestamp\n"
        with pytest.raises(AssertionError, match="invalid timestamp"):
            validate_vtt_structure(malformed_vtt)

    def test_vtt_accepts_three_digit_hour_for_long_cues(self):
        """Verify a >=100-hour cue (3-digit hour field) passes the validator: WebVTT's
        spec requires two-or-more hour digits (unlike SRT's fixed 2), and
        `format_timestamp()`'s `{hours:02d}` zero-pads without truncating, so a
        real >99h59m59s video naturally produces a 3+ digit hour like "100:00:00.000"."""
        long_cue_vtt = "WEBVTT\n\n1\n100:00:00.000 --> 100:00:05.000\nLong movie cue\n"
        parsed = validate_vtt_structure(long_cue_vtt)
        assert parsed[0]["start"] == 100 * 3600


class TestNonAsciiEncoding:
    """6.6: non-ASCII/multi-byte text survives SRT/VTT/JSON generation without mangling."""

    ACCENTED = "Café à côté, déjà vu — naïve résumé"
    CJK = "你好，世界。日本語のテスト。한국어 테스트입니다。"
    RTL_ARABIC = "مرحبا بالعالم"
    RTL_HEBREW = "שלום עולם"

    def _segments(self) -> dict:
        return {
            "segments": [
                {"start": 0.0, "end": 1.0, "text": self.ACCENTED},
                {"start": 1.0, "end": 2.0, "text": self.CJK},
                {"start": 2.0, "end": 3.0, "text": self.RTL_ARABIC},
                {"start": 3.0, "end": 4.0, "text": self.RTL_HEBREW},
            ]
        }

    def test_srt_preserves_non_ascii_text_exactly(self):
        """Verify accented, CJK, and RTL text survives SRT generation exactly."""
        result = self._segments()
        with _no_promo():
            srt = utils.generate_srt(result)
        parsed = validate_srt_structure(srt)
        texts = [b["text"] for b in parsed]
        assert texts == [self.ACCENTED, self.CJK, self.RTL_ARABIC, self.RTL_HEBREW]

    def test_vtt_preserves_non_ascii_text_exactly(self):
        """Verify accented, CJK, and RTL text survives VTT generation exactly."""
        result = self._segments()
        with _no_promo():
            vtt = utils.generate_vtt(result)
        parsed = validate_vtt_structure(vtt)
        texts = [b["text"] for b in parsed]
        assert texts == [self.ACCENTED, self.CJK, self.RTL_ARABIC, self.RTL_HEBREW]

    def test_srt_utf8_round_trip_no_mangling_no_bom(self):
        """Encoding to UTF-8 bytes and decoding back must be lossless, with no BOM prefix."""
        result = self._segments()
        with _no_promo():
            srt = utils.generate_srt(result)
        encoded = srt.encode("utf-8")
        assert not encoded.startswith(b"\xef\xbb\xbf"), "SRT output must not have a UTF-8 BOM"
        assert encoded.decode("utf-8") == srt

    def test_vtt_utf8_round_trip_no_mangling_no_bom(self):
        """Encoding to UTF-8 bytes and decoding back must be lossless, with no BOM prefix."""
        result = self._segments()
        with _no_promo():
            vtt = utils.generate_vtt(result)
        encoded = vtt.encode("utf-8")
        assert not encoded.startswith(b"\xef\xbb\xbf"), "VTT output must not have a UTF-8 BOM"
        assert encoded.decode("utf-8") == vtt

    def test_json_result_round_trips_non_ascii_text(self):
        """JSON serialization (as returned by the API) must not mangle multi-byte text."""
        result = self._segments()
        dumped = json.dumps(result, ensure_ascii=False)
        reloaded = json.loads(dumped)
        assert reloaded == result
        assert reloaded["segments"][1]["text"] == self.CJK


class TestOutOfOrderTimestamps:
    """Document current (as-implemented) behavior for malformed upstream ASR timing.

    generate_srt / generate_vtt do NOT sort or validate segment ordering --
    they emit blocks strictly in the input segment order, using whatever
    start/end each segment carries. This means an out-of-order or
    overlapping segments list from an upstream ASR/diarization glitch
    produces a *structurally well-formed* but *non-monotonic* SRT/VTT file
    (block numbering and timestamp syntax are still valid; only the timing
    order is off). These tests pin that exact behavior down so any future
    change (e.g. adding sort-by-start-time) is a deliberate, visible change
    to this test, not a silent behavior shift.

    NOTE (finding, not fixed): this means a formatter consuming out-of-order
    upstream segments silently produces a subtitle file with non-monotonic
    cue timestamps rather than sorting or raising. Bazarr / most players
    tolerate this, but it is a latent correctness gap. Left unfixed here
    since a "should we sort or reject" behavior change is a product decision,
    not a safe, minimal, drive-by fix.
    """

    def test_srt_out_of_order_segments_are_emitted_in_input_order_not_sorted(self):
        """Verify SRT generation preserves input order verbatim and the validator flags non-monotonic timing."""
        result = {
            "segments": [
                {"start": 5.0, "end": 6.0, "text": "Second in time, first in list"},
                {"start": 1.0, "end": 2.0, "text": "First in time, second in list"},
            ]
        }
        with _no_promo():
            srt = utils.generate_srt(result)
        # Structurally valid syntax-wise (numbering, timestamp format), but a strict
        # monotonic validator must flag the non-decreasing-start-time violation.
        with pytest.raises(AssertionError, match="non-monotonic"):
            validate_srt_structure(srt)

        # Confirm the formatter itself preserves input order verbatim (no silent sort).
        assert srt.index("Second in time, first in list") < srt.index("First in time, second in list")

        # Confirm each block's own timestamp line is preserved unaltered (not just the
        # text) in input order -- no reordering, clamping, or resequencing of timing.
        assert "00:00:05,000 --> 00:00:06,000" in srt
        assert "00:00:01,000 --> 00:00:02,000" in srt
        assert srt.index("00:00:05,000 --> 00:00:06,000") < srt.index("00:00:01,000 --> 00:00:02,000")

    def _assert_two_blocks_present(self, srt: str) -> None:
        """Assert the output contains exactly two blocks (one per segment)."""
        blocks = [b for b in srt.strip().split("\n\n") if b.strip()]
        assert len(blocks) == 2

    def _assert_both_overlapping_blocks_present(self, srt: str) -> None:
        """Assert both overlapping-segment blocks made it into the output."""
        self._assert_two_blocks_present(srt)
        assert "Overlaps with next" in srt
        assert "Overlaps with previous" in srt

    def _assert_overlapping_timestamps_unaltered(self, srt: str) -> None:
        """Assert both blocks retain their own (overlapping) timestamps --
        no clamping/adjustment applied by the formatter."""
        assert "00:00:00,000 --> 00:00:05,000" in srt
        assert "00:00:02,000 --> 00:00:07,000" in srt

    def _assert_overlapping_blocks_emitted_unaltered(self, srt: str) -> None:
        """Assert both overlapping blocks are present with their own untouched
        timestamps -- no clamping/adjustment applied by the formatter."""
        self._assert_both_overlapping_blocks_present(srt)
        self._assert_overlapping_timestamps_unaltered(srt)

    def test_srt_overlapping_segments_both_emitted_without_error(self):
        """Overlapping start/end ranges must not raise -- formatter emits both blocks as-is."""
        result = {
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "Overlaps with next"},
                {"start": 2.0, "end": 7.0, "text": "Overlaps with previous"},
            ]
        }
        with _no_promo():
            srt = utils.generate_srt(result)
        self._assert_overlapping_blocks_emitted_unaltered(srt)

    def test_vtt_out_of_order_segments_are_emitted_in_input_order_not_sorted(self):
        """Verify VTT generation preserves input order verbatim and the validator flags non-monotonic timing."""
        result = {
            "segments": [
                {"start": 10.0, "end": 11.0, "text": "Later timestamp, first block"},
                {"start": 1.0, "end": 2.0, "text": "Earlier timestamp, second block"},
            ]
        }
        with _no_promo():
            vtt = utils.generate_vtt(result)
        with pytest.raises(AssertionError, match="non-monotonic"):
            validate_vtt_structure(vtt)
        assert vtt.index("Later timestamp, first block") < vtt.index("Earlier timestamp, second block")

        # Confirm each cue's own timestamp line is preserved unaltered (not just the
        # text) in input order -- no reordering, clamping, or resequencing of timing.
        assert "00:00:10.000 --> 00:00:11.000" in vtt
        assert "00:00:01.000 --> 00:00:02.000" in vtt
        assert vtt.index("00:00:10.000 --> 00:00:11.000") < vtt.index("00:00:01.000 --> 00:00:02.000")

    def _assert_vtt_overlapping_cues_present_and_unaltered(self, vtt: str) -> None:
        """Assert both overlapping VTT cue texts are present with their original,
        unclamped timestamp ranges intact."""
        assert "VTT overlaps with next" in vtt
        assert "VTT overlaps with previous" in vtt
        # VTT timestamps are formatted as HH:MM:SS.mmm.
        assert "00:00:00.000 --> 00:00:05.000" in vtt
        assert "00:00:02.000 --> 00:00:07.000" in vtt

    def test_vtt_overlapping_segments_both_emitted_without_alteration(self):
        """Overlapping-timestamp VTT segments must both be emitted with their original
        cue texts and start/end timestamp ranges intact -- no clamping, merging, or
        reordering applied by the formatter.

        This is the VTT-specific counterpart to
        test_srt_overlapping_segments_both_emitted_without_error (in the SRT suite).
        """
        result = {
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "VTT overlaps with next"},
                {"start": 2.0, "end": 7.0, "text": "VTT overlaps with previous"},
            ]
        }
        with _no_promo():
            vtt = utils.generate_vtt(result)

        self._assert_vtt_overlapping_cues_present_and_unaltered(vtt)
        # Input order must be preserved: first segment's cue text appears before second's.
        assert vtt.index("VTT overlaps with next") < vtt.index("VTT overlaps with previous")
