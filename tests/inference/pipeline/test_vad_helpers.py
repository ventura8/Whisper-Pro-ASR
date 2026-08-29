"""VAD helpers around the faster-whisper boundary: patching, slicing, and cleanup.

These are the parts that run whether or not faster_whisper is importable, so they are
testable in the test image (which does not ship it). The slice helper matters most: it
writes a temporary file that the caller may or may not own depending on which entry point
was used, and getting that wrong leaks a wav per gap on every code-switched clip.
"""

from __future__ import annotations

import os
import sys
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest

from modules.inference.pipeline import vad


def _run_then_delete_output(cmd, timeout=None, *, check=False):
    """Stand in for a successful ffmpeg run whose output vanishes before cleanup."""
    os.remove(cmd[-1])


def _delete_output_then_fail(cmd, timeout=None, *, check=False):
    """Stand in for an ffmpeg run that fails after its output has already gone."""
    os.remove(cmd[-1])
    raise RuntimeError("ffmpeg failed")


def _noop(*_args, **_kwargs):
    """A stand-in get_speech_timestamps, so a patch can be observed replacing it."""
    return []


class TestPatchingLoadedFasterWhisperModules:
    """The wrapper has to reach modules that were imported before it existed."""

    def test_nothing_is_patched_when_there_is_no_wrapper(self):
        """Nothing is patched when there is no wrapper."""
        vad._patch_loaded_faster_whisper_modules(None)  # must not raise

    def test_a_loaded_faster_whisper_module_gets_the_wrapper(self):
        """A loaded faster whisper module gets the wrapper."""
        module = ModuleType("faster_whisper.vad")
        module.get_speech_timestamps = _noop

        def wrapper(*_a, **_k):
            """The patched replacement, distinguishable from the original."""
            return ["patched"]

        with mock.patch.dict(sys.modules, {"faster_whisper.vad": module}):
            vad._patch_loaded_faster_whisper_modules(wrapper)

        assert module.get_speech_timestamps is wrapper

    def test_unrelated_modules_are_left_alone(self):
        """Unrelated modules are left alone."""
        module = ModuleType("numpy_lookalike")
        original = _noop
        module.get_speech_timestamps = original

        with mock.patch.dict(sys.modules, {"numpy_lookalike": module}):
            vad._patch_loaded_faster_whisper_modules(lambda *a, **k: None)

        assert module.get_speech_timestamps is original

    def test_a_faster_whisper_module_without_the_symbol_is_skipped(self):
        """A faster whisper module without the symbol is skipped."""
        module = ModuleType("faster_whisper.tokenizer")
        with mock.patch.dict(sys.modules, {"faster_whisper.tokenizer": module}):
            vad._patch_loaded_faster_whisper_modules(lambda *a, **k: None)
        assert not hasattr(module, "get_speech_timestamps")

    def test_a_module_registry_that_changes_underneath_is_tolerated(self):
        """sys.modules is mutated by any import on another thread; this runs at engine load."""
        with mock.patch.object(vad, "_is_patchable_faster_whisper_module", side_effect=RuntimeError("dict changed")):
            vad._patch_loaded_faster_whisper_modules(lambda *a, **k: None)  # must not raise

    @pytest.mark.parametrize(
        ("name", "module", "expected"),
        [
            ("faster_whisper.vad", SimpleNamespace(get_speech_timestamps=_noop), True),
            ("faster_whisper.vad", None, False),
            ("faster_whisper.vad", SimpleNamespace(), False),
            ("other.vad", SimpleNamespace(get_speech_timestamps=_noop), False),
        ],
    )
    def test_the_patchability_predicate(self, name, module, expected):
        """The patchability predicate."""
        assert vad._is_patchable_faster_whisper_module(name, module) is expected


class TestDirectDecodeDecision:
    """Whether a slice is needed at all."""

    def test_no_offset_and_no_duration_decodes_the_whole_file_directly(self):
        """No offset and no duration decodes the whole file directly."""
        assert vad._should_decode_directly() is True

    @pytest.mark.parametrize(("offset", "duration"), [(1.0, None), (None, 2.0), (1.0, 2.0), (0.0, None)])
    def test_any_bound_requires_a_slice(self, offset, duration):
        """Any bound requires a slice."""
        assert vad._should_decode_directly(offset, duration) is False


class TestFfmpegDecodeCommand:
    """ffmpeg decode command."""

    def test_the_whole_file_command_carries_no_bounds(self):
        """The whole file command carries no bounds."""
        cmd = vad._build_ffmpeg_decode_cmd("/in.mkv", "/out.wav")
        assert "-ss" not in cmd and "-t" not in cmd
        assert cmd[-1] == "/out.wav"

    def test_an_offset_and_duration_are_both_passed(self):
        """An offset and duration are both passed."""
        cmd = vad._build_ffmpeg_decode_cmd("/in.mkv", "/out.wav", start_offset=12.5, duration=3.0)
        assert cmd[cmd.index("-ss") + 1] == "12.5"
        assert cmd[cmd.index("-t") + 1] == "3.0"

    def test_the_output_is_sixteen_kilohertz_mono(self):
        """Everything downstream -- VAD windows, the engines' own front ends -- assumes it."""
        cmd = vad._build_ffmpeg_decode_cmd("/in.mkv", "/out.wav")
        assert cmd[cmd.index("-ar") + 1] == "16000"
        assert cmd[cmd.index("-ac") + 1] == "1"


class TestDecodingASliceThroughFfmpeg:
    """The temp file is this helper's own, and must not survive it."""

    def test_the_decoded_samples_are_returned_and_the_temp_file_removed(self, monkeypatch, tmp_path):
        """The decoded samples are returned and the temp file removed."""
        monkeypatch.setattr(vad.config, "get_temp_dir", lambda: str(tmp_path))
        monkeypatch.setattr(vad.process_exec, "run_capture", lambda cmd, timeout=None, *, check=False: None)
        seen = {}

        def decode(path, sampling_rate):
            """Record the path the decoder was handed."""
            seen["path"] = path
            seen["existed"] = os.path.exists(path)
            return [0.1, 0.2]

        assert vad._decode_audio_slice_with_ffmpeg(decode, "/in.mkv", 1.0, 2.0) == [0.1, 0.2]
        assert seen["existed"] is True, "the decoder must see the file ffmpeg wrote"
        assert not os.path.exists(seen["path"]), "the slice is this helper's own and must not leak"

    def test_a_failed_ffmpeg_run_still_removes_the_temp_file(self, monkeypatch, tmp_path):
        """One leaked wav per failed gap extraction, on every code-switched clip."""
        monkeypatch.setattr(vad.config, "get_temp_dir", lambda: str(tmp_path))

        def explode(cmd, timeout=None, *, check=False):
            """Raise, so the failure path is exercised."""
            raise RuntimeError("ffmpeg failed")

        monkeypatch.setattr(vad.process_exec, "run_capture", explode)

        with pytest.raises(RuntimeError):
            vad._decode_audio_slice_with_ffmpeg(lambda p, sampling_rate: None, "/in.mkv", 1.0, 2.0)

        assert list(tmp_path.iterdir()) == []

    def test_a_temp_file_already_gone_is_not_an_error(self, monkeypatch, tmp_path):
        """tmpfs can be remounted underneath a long job; the cleanup must not mask the result.

        The file is really removed rather than os.remove being stubbed to raise. `vad.os` is
        the actual os module, so patching its `remove` replaces it for the whole process --
        every other test, and pytest's own machinery, share that attribute for the duration.
        Deleting the file exercises the same cleanup branch through a real FileNotFoundError.
        """
        monkeypatch.setattr(vad.config, "get_temp_dir", lambda: str(tmp_path))
        monkeypatch.setattr(vad.process_exec, "run_capture", _run_then_delete_output)

        assert vad._decode_audio_slice_with_ffmpeg(lambda p, sampling_rate: ["ok"], "/in.mkv", 1.0, 2.0) == ["ok"]


class TestVadStatisticsLogging:
    """vad statistics logging."""

    def test_the_summary_reports_how_much_silence_was_removed(self, caplog):
        """The summary reports how much silence was removed."""
        audio = [0.0] * (16000 * 10)
        with caplog.at_level("INFO"):
            vad._log_vad_statistics(audio, [{"start": 0, "end": 16000 * 4}])
        assert "60.0% silence" in caplog.text

    def test_empty_audio_does_not_divide_by_zero(self, caplog):
        """Empty audio does not divide by zero."""
        with caplog.at_level("INFO"):
            vad._log_vad_statistics([], [])
        assert "0.0% silence" in caplog.text


class TestExtractSliceToFile:
    """The one decode whose output the CALLER owns -- and must be handed a path, not samples.

    gap_filling uses this for every uncovered speech gap, because an IsolatedEngine runs in
    a worker subprocess and can only be given a path; passing an array raises TypeError
    there. So the file has to survive the call on success, and must not survive it on
    failure -- nothing downstream ever learns the path of a slice that was never returned.
    """

    def test_the_path_is_returned_and_the_file_still_exists(self, monkeypatch, tmp_path):
        """The path is returned and the file still exists."""
        monkeypatch.setattr(vad.config, "get_temp_dir", lambda: str(tmp_path))
        monkeypatch.setattr(vad.process_exec, "run_capture", lambda cmd, timeout=None, *, check=False: None)

        path = vad.extract_slice_to_file("/in.mkv", 4.0, 1.5)

        assert os.path.exists(path), "the caller owns this file and is about to hand it to a worker"
        assert path.endswith(".wav")

    def test_the_requested_window_reaches_ffmpeg(self, monkeypatch, tmp_path):
        """The requested window reaches ffmpeg."""
        monkeypatch.setattr(vad.config, "get_temp_dir", lambda: str(tmp_path))
        captured = {}
        monkeypatch.setattr(vad.process_exec, "run_capture", lambda cmd, timeout=None, *, check=False: captured.setdefault("cmd", cmd))

        vad.extract_slice_to_file("/in.mkv", 4.0, 1.5)

        cmd = captured["cmd"]
        assert cmd[cmd.index("-ss") + 1] == "4.0"
        assert cmd[cmd.index("-t") + 1] == "1.5"

    def test_a_failed_extraction_removes_the_file_and_re_raises(self, monkeypatch, tmp_path):
        """A failed extraction removes the file and re-raises."""
        monkeypatch.setattr(vad.config, "get_temp_dir", lambda: str(tmp_path))

        def explode(cmd, timeout=None, *, check=False):
            raise RuntimeError("ffmpeg failed")

        monkeypatch.setattr(vad.process_exec, "run_capture", explode)

        with pytest.raises(RuntimeError, match="ffmpeg failed"):
            vad.extract_slice_to_file("/in.mkv", 4.0, 1.5)

        assert not list(tmp_path.iterdir()), "a slice nobody received is a slice nobody can clean up"

    def test_cleanup_also_runs_when_the_extraction_is_cancelled(self, monkeypatch, tmp_path):
        """Cleanup also runs when the extraction is cancelled.

        `except BaseException`, not `except Exception`: cooperative preemption cancels a
        long transcription mid-gap-fill, and a KeyboardInterrupt or GeneratorExit here would
        otherwise walk past the cleanup and leak the slice.
        """
        monkeypatch.setattr(vad.config, "get_temp_dir", lambda: str(tmp_path))

        def cancel(cmd, timeout=None, *, check=False):
            raise KeyboardInterrupt()

        monkeypatch.setattr(vad.process_exec, "run_capture", cancel)

        with pytest.raises(KeyboardInterrupt):
            vad.extract_slice_to_file("/in.mkv", 4.0, 1.5)

        assert not list(tmp_path.iterdir())

    def test_a_file_already_gone_during_cleanup_does_not_mask_the_real_error(self, monkeypatch, tmp_path):
        """The ffmpeg failure must survive a cleanup that finds nothing to clean.

        The slice is really deleted before the failure, rather than os.remove being stubbed
        to raise -- `vad.os` is the process-wide os module, and patching it there reaches
        every other test running at the same time.
        """
        monkeypatch.setattr(vad.config, "get_temp_dir", lambda: str(tmp_path))
        monkeypatch.setattr(vad.process_exec, "run_capture", _delete_output_then_fail)

        with pytest.raises(RuntimeError, match="ffmpeg failed"):
            vad.extract_slice_to_file("/in.mkv", 4.0, 1.5)
