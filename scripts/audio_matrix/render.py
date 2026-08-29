"""Thin ffmpeg wrappers used to post-process synthesized speech.

Piper emits 22.05 kHz WAV; Whisper wants 16 kHz mono. Everything that touches audio after
synthesis goes through here so the encoding parameters have exactly one definition.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"

#: Ceiling for one ffmpeg invocation. The longest thing built here is the 20-minute
#: long-form concat, which takes seconds; a call still running after this is wedged (a
#: generator filter left unbounded, say) and would otherwise hang the whole run forever.
FFMPEG_TIMEOUT_SEC = 600


def ffmpeg_available() -> bool:
    """Return whether both ffmpeg and ffprobe are on PATH."""
    return bool(shutil.which(FFMPEG)) and bool(shutil.which(FFPROBE))


def ffmpeg_version() -> str:
    """Return the first line of ``ffmpeg -version``, or an empty string."""
    if not shutil.which(FFMPEG):
        return ""
    result = subprocess.run([FFMPEG, "-version"], capture_output=True, text=True, check=False, timeout=FFMPEG_TIMEOUT_SEC)
    return result.stdout.splitlines()[0] if result.stdout else ""


def run_ffmpeg(args: list[str]) -> None:
    """Run ffmpeg with ``args``, raising CalledProcessError with its stderr on failure.

    Raises ``subprocess.TimeoutExpired`` once FFMPEG_TIMEOUT_SEC elapses.
    """
    subprocess.run(
        [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", *args],
        check=True,
        capture_output=True,
        timeout=FFMPEG_TIMEOUT_SEC,
    )


def to_pcm16_mono(src: Path, dest: Path, rate: int) -> None:
    """Transcode ``src`` to signed 16-bit little-endian mono PCM WAV at ``rate``."""
    run_ffmpeg(["-i", str(src), "-ac", "1", "-ar", str(rate), "-c:a", "pcm_s16le", str(dest)])


def to_flac(src: Path, dest: Path, rate: int) -> None:
    """Transcode ``src`` to mono FLAC at ``rate`` -- the committed-fixture format.

    FLAC is lossless, so a committed fixture decodes to exactly the PCM the cache holds,
    at roughly half the bytes in the repository.
    """
    run_ffmpeg(["-i", str(src), "-ac", "1", "-ar", str(rate), "-c:a", "flac", str(dest)])


def apply_gain(src: Path, dest: Path, factor: float, rate: int) -> None:
    """Scale ``src`` by a linear ``factor``, clipping on write when it exceeds full scale."""
    run_ffmpeg(["-i", str(src), "-af", f"volume={factor}", "-ac", "1", "-ar", str(rate), "-c:a", "pcm_s16le", str(dest)])


def lavfi(source: str, dest: Path, rate: int, seconds: float | None = None) -> None:
    """Render an ffmpeg lavfi audio source (sine, anoisesrc, anullsrc, ...) to ``dest``.

    ``seconds`` becomes an output-side ``-t`` cap rather than a filter argument. Generators
    like ``sine`` and ``anoisesrc`` run forever unless bounded, and a duration appended to
    the wrong element of a filter chain silently produces an unbounded render -- so the
    bound is applied where it cannot be attached to the wrong filter.
    """
    duration = ["-t", f"{seconds:.3f}"] if seconds is not None else []
    run_ffmpeg(["-f", "lavfi", "-i", source, *duration, "-ac", "1", "-ar", str(rate), "-c:a", "pcm_s16le", str(dest)])


def mix(sources: list[str], dest: Path, rate: int, tail: str = "", seconds: float | None = None) -> None:
    """Mix several lavfi sources into one track, optionally appending a filter chain."""
    inputs: list[str] = []
    for source in sources:
        inputs.extend(["-f", "lavfi", "-i", source])
    chain = f"amix=inputs={len(sources)}:duration=first:normalize=0" + (f",{tail}" if tail else "")
    duration = ["-t", f"{seconds:.3f}"] if seconds is not None else []
    run_ffmpeg([*inputs, "-filter_complex", chain, *duration, "-ac", "1", "-ar", str(rate), "-c:a", "pcm_s16le", str(dest)])


def filtered(src: Path, dest: Path, chain: str, rate: int, channels: int = 1) -> None:
    """Apply an ffmpeg filter ``chain`` to ``src``."""
    run_ffmpeg(["-i", str(src), "-af", chain, "-ac", str(channels), "-ar", str(rate), "-c:a", "pcm_s16le", str(dest)])


def to_mp3(src: Path, dest: Path, rate: int) -> None:
    """Encode ``src`` as MP3 -- used to build a file whose contents contradict its name."""
    run_ffmpeg(["-i", str(src), "-ac", "1", "-ar", str(rate), "-c:a", "libmp3lame", "-f", "mp3", str(dest)])


def _concat_quote(path: Path) -> str:
    """Return ``path`` escaped for the single-quoted form the concat demuxer expects.

    ffconcat wraps each path in single quotes, so an apostrophe in the path -- a checkout
    under "Sergiu's Projects", say -- closes the quote early and hands ffmpeg a truncated
    filename. Its escape for a literal quote inside a quoted string is to close the quote,
    emit a backslash-escaped quote, and reopen: ``'`` becomes ``'\\''``.
    """
    return str(path.resolve()).replace("'", "'\\''")


def concat(sources: list[Path], dest: Path, rate: int) -> None:
    """Concatenate PCM sources end to end via the concat demuxer.

    Byte-exact boundaries matter here: the long-form clip's ground-truth offsets are only
    trustworthy because concatenation does not resample or re-time anything.
    """
    listing = dest.with_suffix(".concat.txt")
    entries = "".join(f"file '{_concat_quote(path)}'\n" for path in sources)
    listing.write_text(entries, encoding="utf-8")
    # The listing is a temporary of this call, so it goes whether ffmpeg succeeded or not.
    # Left behind on failure it sat next to the fixtures as an untracked .concat.txt, and a
    # regeneration run then reported a dirty tree that had nothing to do with the audio.
    try:
        run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(listing), "-ac", "1", "-ar", str(rate), "-c:a", "pcm_s16le", str(dest)])
    finally:
        listing.unlink(missing_ok=True)


def probe_duration(path: Path) -> float:
    """Return the duration of ``path`` in seconds."""
    args = [FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)]
    result = subprocess.run(args, capture_output=True, text=True, check=True, timeout=FFMPEG_TIMEOUT_SEC)
    return float(json.loads(result.stdout)["format"]["duration"])
