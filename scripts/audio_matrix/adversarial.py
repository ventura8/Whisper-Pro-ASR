"""Builders for the degraded and malformed audio cases.

Each builder produces one artifact the service should survive: audio that is technically
valid but hostile (clipped, near-silent, telephone-band, wrong sample rate), and files that
are not really audio at all (truncated headers, zero bytes, an MP3 wearing a .wav name).

The point is not to prove the service transcribes these -- most of them have nothing to
transcribe. It is to prove it fails *gracefully and promptly* rather than returning a
500, hallucinating text into silence, or pinning a worker forever.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from scripts.audio_matrix import render

# A valid RIFF/WAVE header is 44 bytes; keeping fewer leaves a file that announces itself
# as audio and then stops mid-declaration.
TRUNCATED_HEADER_BYTES = 20


def _source(params: dict, context: dict) -> Path:
    """Return the rendered clip an entry derives from."""
    return context["root"] / f"{params['source']}.wav"


def build_silence(dest: Path, params: dict, context: dict) -> None:
    """Digital silence: nothing to transcribe, and nothing to hallucinate about."""
    render.lavfi(f"anullsrc=r={context['rate']}:cl=mono", dest, context["rate"], seconds=params.get("seconds", 10))


def build_noise(dest: Path, params: dict, context: dict) -> None:
    """Broadband noise at speech-like level, seeded so the file is reproducible."""
    source = f"anoisesrc=color=white:amplitude={params.get('amplitude', 0.5)}:seed={params.get('seed', 7)}"
    render.lavfi(source, dest, context["rate"], seconds=params.get("seconds", 10))


def build_tones(dest: Path, params: dict, context: dict) -> None:
    """A DTMF-style dual tone: periodic, loud, and completely speechless."""
    render.mix(["sine=f=697", "sine=f=1209"], dest, context["rate"], seconds=params.get("seconds", 10))


def build_clipped(dest: Path, params: dict, context: dict) -> None:
    """Real speech driven far past full scale, so it hard-clips on write."""
    render.apply_gain(_source(params, context), dest, params.get("gain", 12.0), context["rate"])


def build_quiet(dest: Path, params: dict, context: dict) -> None:
    """Real speech attenuated to near the noise floor."""
    render.apply_gain(_source(params, context), dest, params.get("gain", 0.005), context["rate"])


def build_telephone(dest: Path, params: dict, context: dict) -> None:
    """Speech band-limited and resampled the way a phone line would deliver it."""
    rate = params.get("rate", 8000)
    render.filtered(_source(params, context), dest, f"aresample={rate},highpass=f=300,lowpass=f=3400", rate)


def build_stereo(dest: Path, params: dict, context: dict) -> None:
    """Two-channel audio with different content per channel."""
    render.filtered(_source(params, context), dest, "pan=stereo|c0=c0|c1=0.5*c0", context["rate"], channels=2)


def build_resampled(dest: Path, params: dict, context: dict) -> None:
    """Valid speech at a sample rate the pipeline must resample from."""
    rate = int(params["rate"])
    render.filtered(_source(params, context), dest, f"aresample={rate}", rate)


def build_speech_after_silence(dest: Path, params: dict, context: dict) -> None:
    """A long lead-in of silence before any speech, to catch early-stopping VAD."""
    lead = context["root"] / "_lead_silence.wav"
    render.lavfi(f"anullsrc=r={context['rate']}:cl=mono", lead, context["rate"], seconds=params.get("lead_seconds", 30))
    render.concat([lead, _source(params, context)], dest, context["rate"])
    lead.unlink(missing_ok=True)


def build_tiny(dest: Path, params: dict, context: dict) -> None:
    """A clip far shorter than any analysis window."""
    render.lavfi(f"anullsrc=r={context['rate']}:cl=mono", dest, context["rate"], seconds=params.get("seconds", 0.05))


def build_truncated_header(dest: Path, params: dict, context: dict) -> None:
    """A file that begins announcing itself as a WAV and then stops."""
    dest.write_bytes(_source(params, context).read_bytes()[:TRUNCATED_HEADER_BYTES])


def build_zero_byte(dest: Path, _params: dict, _context: dict) -> None:
    """An empty upload."""
    dest.write_bytes(b"")


def build_mp3_named_wav(dest: Path, params: dict, context: dict) -> None:
    """Real MP3 content behind a .wav filename, to test sniffing over trusting the name."""
    render.to_mp3(_source(params, context), dest, context["rate"])


BUILDERS: dict[str, Callable[[Path, dict, dict], None]] = {
    "silence": build_silence,
    "noise": build_noise,
    "tones": build_tones,
    "clipped": build_clipped,
    "quiet": build_quiet,
    "telephone": build_telephone,
    "stereo": build_stereo,
    "resampled": build_resampled,
    "speech_after_silence": build_speech_after_silence,
    "tiny": build_tiny,
    "truncated_header": build_truncated_header,
    "zero_byte": build_zero_byte,
    "mp3_named_wav": build_mp3_named_wav,
}


def build(entry: dict[str, Any], dest: Path, context: dict) -> None:
    """Dispatch one adversarial entry to its builder."""
    builder = BUILDERS.get(entry["builder"])
    if builder is None:
        raise KeyError(f"unknown adversarial builder {entry['builder']!r}")
    builder(dest, entry.get("params") or {}, context)
