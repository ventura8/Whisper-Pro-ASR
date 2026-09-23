"""Tests for scripts/audio_matrix/render.py, the ffmpeg wrappers.

Every fixture's encoding is defined exactly once, here, so these tests assert the argument
vectors rather than running ffmpeg: the encoding parameters are the contract, and a silent
change to them would alter committed audio that is supposed to be bit-reproducible.
"""

import json
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from scripts.audio_matrix import render


@pytest.fixture(name="ffmpeg_run")
def _ffmpeg_run():
    """Capture the argument vector run_ffmpeg would hand to subprocess."""
    with mock.patch.object(render.subprocess, "run") as run:
        yield run


def _args(run):
    """Return the ffmpeg argv from a captured subprocess.run call."""
    return run.call_args[0][0]


def test_ffmpeg_available_requires_both_binaries():
    """ffprobe is needed for durations, so ffmpeg alone is not enough."""
    with mock.patch.object(render.shutil, "which", side_effect=lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None):
        assert render.ffmpeg_available() is False

    with mock.patch.object(render.shutil, "which", return_value="/usr/bin/x"):
        assert render.ffmpeg_available() is True


def test_ffmpeg_version_returns_the_first_line():
    """The version string feeds the cache digest, so only the stable first line is used."""
    completed = subprocess.CompletedProcess([], 0, stdout="ffmpeg version 6.1\nbuilt with gcc\n", stderr="")
    with mock.patch.object(render.shutil, "which", return_value="/usr/bin/ffmpeg"):
        with mock.patch.object(render.subprocess, "run", return_value=completed):
            assert render.ffmpeg_version() == "ffmpeg version 6.1"


def test_ffmpeg_version_is_empty_without_ffmpeg():
    """A missing binary yields an empty version rather than raising."""
    with mock.patch.object(render.shutil, "which", return_value=None):
        assert render.ffmpeg_version() == ""


def test_ffmpeg_version_is_empty_when_ffmpeg_says_nothing():
    """Empty stdout is not an index error."""
    completed = subprocess.CompletedProcess([], 0, stdout="", stderr="")
    with mock.patch.object(render.shutil, "which", return_value="/usr/bin/ffmpeg"):
        with mock.patch.object(render.subprocess, "run", return_value=completed):
            assert render.ffmpeg_version() == ""


def test_run_ffmpeg_is_quiet_checked_and_bounded(ffmpeg_run):
    """Every invocation overwrites, stays quiet, is checked, and cannot hang forever."""
    render.run_ffmpeg(["-i", "in.wav", "out.wav"])

    assert _args(ffmpeg_run)[:5] == ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    assert ffmpeg_run.call_args.kwargs["check"] is True
    assert ffmpeg_run.call_args.kwargs["timeout"] == render.FFMPEG_TIMEOUT_SEC


def test_to_pcm16_mono_encodes_the_format_whisper_reads(ffmpeg_run):
    """16 kHz mono signed 16-bit PCM is what the engines are fed."""
    render.to_pcm16_mono(Path("in.wav"), Path("out.wav"), 16000)

    args = _args(ffmpeg_run)
    assert args[-7:] == ["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "out.wav"]
    assert args[args.index("-i") + 1] == "in.wav"


def test_to_flac_is_lossless_and_mono(ffmpeg_run):
    """Committed fixtures are FLAC so they decode to exactly the cached PCM."""
    render.to_flac(Path("in.wav"), Path("out.flac"), 16000)

    args = _args(ffmpeg_run)
    assert "flac" in args
    assert args[args.index("-ac") + 1] == "1"


def test_apply_gain_passes_a_linear_factor(ffmpeg_run):
    """The gain is a linear multiplier, not decibels."""
    render.apply_gain(Path("in.wav"), Path("out.wav"), 0.005, 16000)

    assert "volume=0.005" in _args(ffmpeg_run)


def test_lavfi_bounds_the_generator_on_the_output_side(ffmpeg_run):
    """`-t` must be an output cap: generators run forever and a duration attached to the
    wrong filter element silently produces an unbounded render."""
    render.lavfi("sine=f=440", Path("out.wav"), 16000, seconds=1.5)

    args = _args(ffmpeg_run)
    assert args[args.index("-i") + 1] == "sine=f=440"
    assert args[args.index("-t") + 1] == "1.500"
    assert args.index("-t") > args.index("-i")


def test_lavfi_without_a_duration_passes_no_cap(ffmpeg_run):
    """A source that ends on its own needs no `-t`."""
    render.lavfi("anullsrc", Path("out.wav"), 16000)

    assert "-t" not in _args(ffmpeg_run)


def test_mix_declares_one_input_per_source_without_normalising(ffmpeg_run):
    """amix must not normalise: the per-source levels are set deliberately upstream."""
    render.mix(["sine=f=697", "sine=f=1209"], Path("out.wav"), 16000)

    args = _args(ffmpeg_run)
    assert args.count("-f") == 2
    assert args.count("lavfi") == 2
    assert "amix=inputs=2:duration=first:normalize=0" in args[args.index("-filter_complex") + 1]


def test_mix_appends_a_tail_chain_and_duration(ffmpeg_run):
    """The bed builder appends tremolo/volume/limiter after the mix."""
    render.mix(["sine=f=440"], Path("out.wav"), 16000, tail="volume=0.5", seconds=2.0)

    args = _args(ffmpeg_run)
    assert args[args.index("-filter_complex") + 1].endswith(",volume=0.5")
    assert args[args.index("-t") + 1] == "2.000"


def test_filtered_applies_the_chain_and_channel_count(ffmpeg_run):
    """Used for the degraded variants -- telephone band, fake stereo, resampling."""
    render.filtered(Path("in.wav"), Path("out.wav"), "highpass=f=300", 8000, channels=2)

    args = _args(ffmpeg_run)
    assert args[args.index("-af") + 1] == "highpass=f=300"
    assert args[args.index("-ac") + 1] == "2"


def test_to_mp3_forces_the_mp3_container(ffmpeg_run):
    """The adversarial clip needs contents that contradict a .wav name."""
    render.to_mp3(Path("in.wav"), Path("out.wav"), 16000)

    args = _args(ffmpeg_run)
    assert "libmp3lame" in args
    assert args[args.index("-f") + 1] == "mp3"


def test_concat_quote_escapes_an_apostrophe(tmp_path):
    """A checkout under a path with an apostrophe would otherwise truncate the filename.

    The concat demuxer wraps each path in single quotes, so a literal quote has to close,
    escape, and reopen.
    """
    quoted = render._concat_quote(tmp_path / "Sergiu's Projects" / "clip.wav")

    assert "'\\''" in quoted
    assert quoted.count("'") > 1


def test_concat_writes_a_listing_and_removes_it(ffmpeg_run, tmp_path):
    """The listing is a temporary of this call and must not survive it."""
    sources = [tmp_path / "a.wav", tmp_path / "b.wav"]
    for source in sources:
        source.write_bytes(b"RIFF")
    dest = tmp_path / "joined.wav"

    render.concat(sources, dest, 16000)

    args = _args(ffmpeg_run)
    assert args[args.index("-f") + 1] == "concat"
    assert args[args.index("-safe") + 1] == "0"
    assert not dest.with_suffix(".concat.txt").exists()


def test_concat_removes_the_listing_even_when_ffmpeg_fails(tmp_path):
    """Left behind, the stray .concat.txt makes a regeneration run report a dirty tree
    that has nothing to do with the audio."""
    sources = [tmp_path / "a.wav"]
    sources[0].write_bytes(b"RIFF")
    dest = tmp_path / "joined.wav"

    with mock.patch.object(render.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "ffmpeg")):
        with pytest.raises(subprocess.CalledProcessError):
            render.concat(sources, dest, 16000)

    assert not dest.with_suffix(".concat.txt").exists()


def test_probe_duration_reads_the_format_duration(tmp_path):
    """Ground-truth offsets in the long-form fixture depend on this being exact."""
    payload = json.dumps({"format": {"duration": "1204.567"}})
    completed = subprocess.CompletedProcess([], 0, stdout=payload, stderr="")

    with mock.patch.object(render.subprocess, "run", return_value=completed) as run:
        assert render.probe_duration(tmp_path / "clip.wav") == pytest.approx(1204.567)

    assert run.call_args[0][0][0] == render.FFPROBE
