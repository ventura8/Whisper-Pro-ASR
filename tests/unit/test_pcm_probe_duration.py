"""Duration probing: which helper answers, and what happens when ffprobe cannot.

`probe_audio_duration` decides between three sources -- an explicit-flag ffprobe, a native
container probe, and a size-based raw-PCM calculation -- and the precedence matters.
Getting it wrong inflates a duration rather than failing: raw-PCM input flags applied to a
`.mkv` or a 48 kHz stereo WAV make ffprobe report a length several times the real one, and
that number reaches UVR chunk counts and the dashboard's progress denominator, where it
looks like a stall rather than a bad measurement.
"""

from __future__ import annotations

import pytest

from modules.core import pcm_helpers

RAW_PCM_FLAGS = ["-f", "s16le", "-ar", "16000", "-ac", "1"]
THREAD_FLAGS = ["-threads", "4"]


def _returns(*values):
    """A check_output stand-in yielding one response per call, recording the flags it saw."""
    calls: list[list[str]] = []
    remaining = list(values)

    def check_output(cmd, **_kwargs):
        """Match the ffprobe caller's signature; the timeout it passes is irrelevant here."""
        calls.append(list(cmd))
        outcome = remaining.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    check_output.calls = calls
    return check_output


class TestExplicitFlags:
    """input_flags is a caller saying "this file is raw PCM"; the native probe is skipped."""

    def test_a_successful_probe_is_used(self):
        """A caller-declared raw-PCM file is measured with the flags it supplied."""
        probe = _returns("12.5\n")

        assert pcm_helpers.probe_audio_duration("a.pcm", RAW_PCM_FLAGS, THREAD_FLAGS, probe) == 12.5
        assert len(probe.calls) == 1, "an explicit-flag probe must not also run the native probe"
        assert "s16le" in probe.calls[0]

    def test_a_failed_probe_falls_back_to_the_size_calculation(self, tmp_path):
        """When ffprobe cannot answer, byte count over byte rate is exact for raw PCM."""
        # 32000 bytes at 16 kHz mono s16le is exactly one second.
        clip = tmp_path / "a.pcm"
        clip.write_bytes(b"\0" * 32000)
        probe = _returns(RuntimeError("ffprobe exploded"))

        assert pcm_helpers.probe_audio_duration(str(clip), RAW_PCM_FLAGS, THREAD_FLAGS, probe) == pytest.approx(1.0)

    @pytest.mark.parametrize("bad", ["0.0\n", "-3\n", "not-a-number", ""], ids=["zero", "negative", "malformed", "empty"])
    def test_a_non_positive_or_unparseable_answer_is_not_trusted(self, tmp_path, bad):
        """0 and garbage both mean "ffprobe could not tell", so neither may be returned."""
        clip = tmp_path / "a.pcm"
        clip.write_bytes(b"\0" * 32000)

        assert pcm_helpers.probe_audio_duration(str(clip), RAW_PCM_FLAGS, THREAD_FLAGS, _returns(bad)) == pytest.approx(1.0)


class TestNativePrecedence:
    """With no explicit flags the container's own metadata wins, which is the recorded fix."""

    def test_the_native_probe_answers_without_any_input_flags(self):
        """A container's own metadata is authoritative and must be consulted first."""
        probe = _returns("2640.0\n")

        assert pcm_helpers.probe_audio_duration("movie.mkv", None, THREAD_FLAGS, probe) == 2640.0
        assert len(probe.calls) == 1
        # No -f/-ar/-ac: applying raw-PCM flags to a container is what inflated durations.
        assert "s16le" not in probe.calls[0]

    def test_a_failed_native_probe_retries_with_the_contextual_flags(self):
        """Only after the native probe fails may the caller's contextual flags be applied."""
        probe = _returns(RuntimeError("no native metadata"), "7.0\n")

        assert pcm_helpers.probe_audio_duration("odd.bin", None, THREAD_FLAGS, probe) == 7.0
        assert len(probe.calls) == 2
        assert "-threads" in probe.calls[1]

    def test_both_probes_failing_falls_through_to_the_size_calculation(self, tmp_path):
        """Last resort, and only correct because the pipeline standardises to 16 kHz mono."""
        clip = tmp_path / "a.pcm"
        clip.write_bytes(b"\0" * 32000)
        probe = _returns(RuntimeError("native failed"), RuntimeError("contextual failed"))

        # THREAD_FLAGS carry no sample format, so pcm_bytes_per_second uses its defaults --
        # 16 kHz mono s16le, the pipeline's standard.
        assert pcm_helpers.probe_audio_duration(str(clip), None, THREAD_FLAGS, probe) == pytest.approx(1.0)

    def test_with_no_contextual_flags_a_failed_native_probe_yields_zero(self, tmp_path):
        """Nothing left to measure with: 0.0 is the honest answer, not a guess."""
        clip = tmp_path / "a.pcm"
        clip.write_bytes(b"\0" * 32000)
        probe = _returns(RuntimeError("native failed"))

        assert pcm_helpers.probe_audio_duration(str(clip), None, None, probe) == 0.0
        assert len(probe.calls) == 1, "there is no contextual probe to run"


def test_a_missing_file_does_not_raise(tmp_path):
    """Duration probing is called on paths that may already have been cleaned up."""
    probe = _returns(RuntimeError("no such file"))

    assert pcm_helpers.probe_audio_duration(str(tmp_path / "gone.pcm"), RAW_PCM_FLAGS, None, probe) == 0.0
