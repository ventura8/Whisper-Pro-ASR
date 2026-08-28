"""Real (non-mocked) end-to-end validation of Bazarr stream-selection/delay-correction
replication: real ffmpeg/ffprobe subprocesses against synthesized multi-track and
offset media, proving the feature actually selects the right audio track and shifts
audio timing correctly -- not just that our own mocks were satisfied."""

from __future__ import annotations

import os
import re
import shutil
import struct
import subprocess
import wave

import pytest

from modules.core import utils, utils_helpers

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="real ffmpeg/ffprobe binaries not available",
)


def _run_ffmpeg(args: list[str]) -> None:
    result = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], capture_output=True, check=False)
    assert result.returncode == 0, result.stderr.decode()


def _build_multi_track_media(tmp_path) -> str:
    """A real 2-audio-track container: stream 0 = silence/eng, stream 1 = 440Hz tone/fre."""
    path = str(tmp_path / "multi_track.mkv")
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=16000:cl=mono",
            "-t",
            "1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000",
            "-t",
            "1",
            "-map",
            "0:a",
            "-map",
            "1:a",
            "-c:a",
            "pcm_s16le",
            "-metadata:s:a:0",
            "language=eng",
            "-metadata:s:a:1",
            "language=fre",
            path,
        ]
    )
    return path


def _build_delayed_tone(tmp_path, offset_sec: float) -> str:
    """A real single-track file whose audio content only starts `offset_sec` in,
    giving ffprobe a genuine nonzero first-packet PTS to detect (mirrors a real
    audio/video sync offset baked into a source file)."""
    path = str(tmp_path / "delayed.mkv")
    _run_ffmpeg(
        [
            "-itsoffset",
            str(offset_sec),
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000",
            "-t",
            "1",
            "-c:a",
            "pcm_s16le",
            "-fflags",
            "+genpts",
            path,
        ]
    )
    return path


def _wav_rms(path: str, *, start_sec: float = 0.0, duration_sec: float | None = None) -> float:
    """RMS amplitude of a WAV file's samples in the given time window (16-bit PCM)."""
    with wave.open(path, "rb") as wav_file:
        rate = wav_file.getframerate()
        wav_file.setpos(int(start_sec * rate))
        n_frames = wav_file.getnframes() - wav_file.tell() if duration_sec is None else int(duration_sec * rate)
        n_frames = max(0, min(n_frames, wav_file.getnframes() - wav_file.tell()))
        raw = wav_file.readframes(n_frames)
    if not raw:
        return 0.0
    samples = struct.unpack(f"<{len(raw) // 2}h", raw)
    return (sum(s * s for s in samples) / len(samples)) ** 0.5


_SILENCE_RMS_THRESHOLD = 100.0  # well below a real 440Hz tone's RMS at any reasonable volume


def test_real_probe_streams_and_packets_reads_actual_language_tags(tmp_path):
    """Real ffprobe (no mocking) must report both tracks' language tags correctly."""
    media_path = _build_multi_track_media(tmp_path)
    result = getattr(utils_helpers, "_probe_streams_and_packets")(media_path)

    streams = getattr(utils_helpers, "_extract_audio_streams")(result)
    assert {"index": 0, "language": "eng"} in streams
    assert {"index": 1, "language": "fre"} in streams


def test_real_select_audio_stream_index_picks_correct_track_for_target_language(tmp_path):
    """End-to-end (real ffprobe): selecting language 'fr' must resolve to the French
    track (index 1), not the default-first English/silent track (index 0)."""
    media_path = _build_multi_track_media(tmp_path)

    stream_index, _delay_filter = utils.get_stream_alignment_directives(media_path, "fr")

    assert stream_index == 1


def test_real_select_audio_stream_index_english_target_picks_first_track(tmp_path):
    """Selecting 'en' must resolve to the English track (index 0)."""
    media_path = _build_multi_track_media(tmp_path)

    stream_index, _delay_filter = utils.get_stream_alignment_directives(media_path, "en")

    assert stream_index == 0


def test_real_select_audio_stream_index_no_target_language_returns_none(tmp_path):
    """With no target language, real probing must still run cleanly but select nothing
    explicit (ffmpeg's own default stream selection applies)."""
    media_path = _build_multi_track_media(tmp_path)

    stream_index, _delay_filter = utils.get_stream_alignment_directives(media_path, None)

    assert stream_index is None


def test_real_build_stream_alignment_directives_detects_delay_from_itsoffset(tmp_path):
    """Real ffprobe must detect the genuine first-packet PTS offset baked into the
    file by -itsoffset, and build the correct adelay filter for it."""
    media_path = _build_delayed_tone(tmp_path, offset_sec=0.5)

    stream_index, delay_filter = utils.get_stream_alignment_directives(media_path, None)

    assert stream_index is None  # single track, no language tag to match against
    # ffprobe PTS rounding may produce 499 or 501 ms depending on codec granularity;
    # accept ±1 ms while still requiring the exact adelay=<value>:all=1 filter form.
    assert delay_filter is not None
    m = re.fullmatch(r"adelay=(\d+):all=1", delay_filter)
    assert m is not None, f"Unexpected delay_filter format: {delay_filter!r}"
    assert 499 <= int(m.group(1)) <= 501, f"Delay out of tolerance: {delay_filter!r}"


def test_real_build_stream_alignment_directives_no_delay_for_zero_offset_media(tmp_path):
    """A normal file with no PTS offset must not trigger any delay correction."""
    media_path = _build_delayed_tone(tmp_path, offset_sec=0.0)

    _stream_index, delay_filter = utils.get_stream_alignment_directives(media_path, None)

    assert delay_filter is None


def test_real_convert_to_wav_stream_index_selects_the_correct_audio_track(tmp_path):
    """End-to-end (real ffmpeg execution): converting with stream_index=1 must produce
    a WAV containing the actual tone (French track), not the silent English track that
    ffmpeg would pick by default with no -map at all."""
    media_path = _build_multi_track_media(tmp_path)

    default_output = utils.convert_to_wav(media_path)
    try:
        selected_output = utils.convert_to_wav(media_path, stream_index=1)
        try:
            assert _wav_rms(default_output) < _SILENCE_RMS_THRESHOLD  # default (no -map) picks the silent track
            assert _wav_rms(selected_output) > _SILENCE_RMS_THRESHOLD  # explicit -map 0:1 picks the real tone
        finally:
            os.remove(selected_output)
    finally:
        os.remove(default_output)


def test_real_convert_to_wav_delay_filter_inserts_leading_silence(tmp_path):
    """End-to-end (real ffmpeg execution): applying the detected delay_filter must
    shift the tone later in the output, inserting real silence at the front that
    matches the original file's true (video-relative) offset -- exactly the correction
    Bazarr's own client applies before uploading."""
    media_path = _build_delayed_tone(tmp_path, offset_sec=0.5)
    _stream_index, delay_filter = utils.get_stream_alignment_directives(media_path, None)
    assert delay_filter is not None
    assert re.fullmatch(r"adelay=(?:499|500|501):all=1", delay_filter)

    uncorrected = utils.convert_to_wav(media_path)
    try:
        corrected = utils.convert_to_wav(media_path, delay_filter=delay_filter)
        try:
            # Without correction, ffmpeg normalizes the demuxed audio to start immediately.
            assert _wav_rms(uncorrected, start_sec=0.0, duration_sec=0.1) > _SILENCE_RMS_THRESHOLD
            # With correction, the first ~0.4s must be silent (the reinserted offset), and the
            # tone must still be present later in the file.
            assert _wav_rms(corrected, start_sec=0.0, duration_sec=0.4) < _SILENCE_RMS_THRESHOLD
            assert _wav_rms(corrected, start_sec=0.6, duration_sec=0.3) > _SILENCE_RMS_THRESHOLD
        finally:
            os.remove(corrected)
    finally:
        os.remove(uncorrected)
