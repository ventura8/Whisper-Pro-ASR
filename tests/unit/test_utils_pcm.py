"""Tests for PCM duration helpers in modules/core/utils.py."""

from unittest import mock

import pytest

from modules.core import utils


def test_get_audio_duration_explicit_input_flags_passed_to_ffprobe():
    """Explicit input_flags are forwarded into the ffprobe command."""
    captured_cmd = []

    def _capture(cmd, **_kw):
        captured_cmd.extend(cmd)
        return "5.0"

    flags = ["-f", "s16le", "-ar", "16000", "-ac", "1"]
    with mock.patch("modules.core.process_exec.check_output_text", side_effect=_capture):
        result = utils.get_audio_duration("test.raw", input_flags=flags)
    assert result == 5.0
    for flag in flags:
        assert flag in captured_cmd


def test_get_audio_duration_thread_context_input_flags_used_when_none_passed():
    """When input_flags=None, THREAD_CONTEXT.input_flags is used as the source."""
    ctx_flags = ["-f", "s16le", "-ar", "16000", "-ac", "1"]
    token = utils.INPUT_FLAGS_VAR.set(ctx_flags)
    try:
        with mock.patch("modules.core.process_exec.check_output_text", return_value="3.0\n"):
            result = utils.get_audio_duration("test.raw")
        assert result == 3.0
    finally:
        utils.INPUT_FLAGS_VAR.reset(token)


def test_get_audio_duration_ffprobe_failure_with_input_flags_uses_size_fallback():
    """ffprobe failure with input_flags set falls back to size-based PCM duration (not 0.0)."""
    flags = ["-f", "s16le", "-ar", "16000", "-ac", "1"]
    file_size = 32000 * 10  # 10 seconds of 16-bit mono 16kHz = 320000 bytes
    with (
        mock.patch("modules.core.process_exec.check_output_text", side_effect=Exception("fail")),
        mock.patch("os.path.exists", return_value=True),
        mock.patch("os.path.getsize", return_value=file_size),
    ):
        result = utils.get_audio_duration("test.raw", input_flags=flags)
    assert result == pytest.approx(10.0)


def test_calculate_pcm_fallback_duration_branches():
    """calculate_pcm_fallback_duration: size-based path uses dynamic rate; no-flags falls back to 0.0."""
    fn = utils.calculate_pcm_fallback_duration
    # With input_flags [-f s16le] but no -ar/-ac: defaults (16kHz mono) → 32000 B/s → 64000/32000 = 2.0 s
    with (
        mock.patch("os.path.exists", return_value=True),
        mock.patch("os.path.getsize", return_value=64000),
    ):
        assert fn("test.raw", ["-f", "s16le"]) == pytest.approx(2.0)
    # With explicit -ar 44100 -ac 2 (HQ stereo) → 176400 B/s → 176400/176400 = 1.0 s
    with (
        mock.patch("os.path.exists", return_value=True),
        mock.patch("os.path.getsize", return_value=176400),
    ):
        assert fn("test.raw", ["-f", "s16le", "-ar", "44100", "-ac", "2"]) == pytest.approx(1.0)
    # Without input_flags: returns 0.0 immediately
    assert fn("test.raw", []) == 0.0
    assert fn("test.raw", None) == 0.0


@pytest.mark.parametrize(
    "flags, expected",
    [
        (None, 32000.0),
        ([], 32000.0),
        (["-f", "s16le", "-ar", "16000", "-ac", "1"], 32000.0),
        (["-ar", "44100", "-ac", "2"], 176400.0),
        (["-ar", "badval", "-ac", "1"], 32000.0),
        (["-ac", "1", "-ar"], 32000.0),
    ],
)
def test_pcm_bytes_per_second(flags, expected):
    """pcm_bytes_per_second derives rate from flags; defaults match STANDARD_AUDIO_FLAGS (16kHz mono s16le)."""
    assert utils.pcm_bytes_per_second(flags) == pytest.approx(expected)
