"""Tests for scripts/audio_matrix/adversarial.py, the degraded and malformed builders.

These fixtures exist to prove the service fails gracefully and promptly rather than
returning a 500, hallucinating text into silence, or pinning a worker. The builders
themselves are thin, so what is pinned here is the traversal guard on `source`, the
builder registry staying in step with its dispatch table, and the two builders that touch
the filesystem directly rather than going through ffmpeg.
"""

from pathlib import Path
from unittest import mock

import pytest

from scripts.audio_matrix import adversarial


@pytest.fixture(name="context")
def _context(tmp_path):
    """A builder context rooted at a temporary audio-matrix directory."""
    (tmp_path / "en_core.wav").write_bytes(b"RIFF" + b"\x00" * 60)
    return {"root": tmp_path, "rate": 16000}


@pytest.fixture(name="calls")
def _calls():
    """Capture every render call the builders make."""
    with mock.patch.multiple(
        adversarial.render,
        lavfi=mock.DEFAULT,
        mix=mock.DEFAULT,
        apply_gain=mock.DEFAULT,
        filtered=mock.DEFAULT,
        concat=mock.DEFAULT,
        to_mp3=mock.DEFAULT,
    ) as patched:
        yield patched


def test_source_resolves_a_flat_sibling_clip(context):
    """The ordinary case: a manifest entry naming another clip in the same root."""
    resolved = adversarial._source({"source": "en_core"}, context)

    assert resolved == (context["root"] / "en_core.wav").resolve()


@pytest.mark.parametrize(
    "hostile",
    ["../../etc/passwd", "../escape", "nested/clip", "/etc/passwd"],
)
def test_source_rejects_anything_leaving_the_matrix_root(context, hostile):
    """`source` is manifest data and reaches nine builders, several of which read the file
    or hand it to ffmpeg -- so a path escaping the fixture tree is refused at the one point
    they all share rather than at each call site."""
    with pytest.raises(ValueError, match="outside the audio matrix root"):
        adversarial._source({"source": hostile}, context)


def test_builders_registry_matches_the_module_functions():
    """Every registered builder resolves to a real function, and none is missing.

    The manifest selects builders by name, so a typo here surfaces as an unbuildable entry
    rather than an import error.
    """
    for name, builder in adversarial.BUILDERS.items():
        assert callable(builder), name
        assert builder.__name__ == f"build_{name}"


def test_silence_noise_and_tones_need_no_source_clip(context, calls):
    """The synthetic cases are generated outright, so they never touch `source`."""
    adversarial.build_silence(context["root"] / "out.wav", {"seconds": 3}, context)
    adversarial.build_noise(context["root"] / "out.wav", {"seconds": 4}, context)
    adversarial.build_tones(context["root"] / "out.wav", {"seconds": 5}, context)

    assert calls["lavfi"].call_count == 2
    assert calls["mix"].call_count == 1
    assert "anullsrc" in calls["lavfi"].call_args_list[0][0][0]
    assert "anoisesrc" in calls["lavfi"].call_args_list[1][0][0]


def test_noise_is_seeded_so_the_clip_is_reproducible(context, calls):
    """An unseeded generator would change the committed fixture on every rebuild."""
    adversarial.build_noise(context["root"] / "out.wav", {}, context)

    assert "seed=7" in calls["lavfi"].call_args[0][0]


def test_clipped_and_quiet_scale_in_opposite_directions(context, calls):
    """Clipped drives past full scale; quiet drops to near the noise floor."""
    adversarial.build_clipped(context["root"] / "a.wav", {"source": "en_core"}, context)
    adversarial.build_quiet(context["root"] / "b.wav", {"source": "en_core"}, context)

    gains = [call[0][2] for call in calls["apply_gain"].call_args_list]
    assert gains[0] > 1.0 > gains[1]


def test_telephone_band_limits_and_resamples(context, calls):
    """A phone line delivers 8 kHz speech through a 300-3400 Hz band."""
    adversarial.build_telephone(context["root"] / "out.wav", {"source": "en_core"}, context)

    chain = calls["filtered"].call_args[0][2]
    assert "aresample=8000" in chain
    assert "highpass=f=300" in chain
    assert "lowpass=f=3400" in chain


def test_stereo_requests_two_channels(context, calls):
    """The channel count is what makes this case different from the source clip."""
    adversarial.build_stereo(context["root"] / "out.wav", {"source": "en_core"}, context)

    assert calls["filtered"].call_args.kwargs["channels"] == 2


def test_resampled_uses_the_requested_rate(context, calls):
    """The pipeline must resample from whatever rate the upload arrives at."""
    adversarial.build_resampled(context["root"] / "out.wav", {"source": "en_core", "rate": "44100"}, context)

    assert "aresample=44100" in calls["filtered"].call_args[0][2]
    assert calls["filtered"].call_args[0][3] == 44100


def test_speech_after_silence_removes_its_lead_file(context, calls):
    """The lead file's name is fixed, so one left behind is picked up by the next entry --
    and it is 30 seconds of silence, so the artifact looks plausible rather than wrong."""
    lead = context["root"] / "_lead_silence.wav"

    def _make_lead(*args, **kwargs):
        lead.write_bytes(b"RIFF")

    calls["lavfi"].side_effect = _make_lead

    adversarial.build_speech_after_silence(context["root"] / "out.wav", {"source": "en_core"}, context)

    assert not lead.exists()
    assert calls["concat"].call_count == 1


def test_speech_after_silence_removes_the_lead_even_when_concat_fails(context, calls):
    """Failure is exactly when a stale lead file would survive to poison the next entry."""
    lead = context["root"] / "_lead_silence.wav"
    calls["lavfi"].side_effect = lambda *a, **k: lead.write_bytes(b"RIFF")
    calls["concat"].side_effect = RuntimeError("ffmpeg exploded")

    with pytest.raises(RuntimeError):
        adversarial.build_speech_after_silence(context["root"] / "out.wav", {"source": "en_core"}, context)

    assert not lead.exists()


def test_tiny_is_shorter_than_any_analysis_window(context, calls):
    """The default has to stay far below a frame, or the case proves nothing."""
    adversarial.build_tiny(context["root"] / "out.wav", {}, context)

    assert calls["lavfi"].call_args.kwargs["seconds"] < 0.1


def test_truncated_header_keeps_less_than_a_riff_header(context):
    """A valid RIFF/WAVE header is 44 bytes; this stops mid-declaration."""
    dest = context["root"] / "out.wav"

    adversarial.build_truncated_header(dest, {"source": "en_core"}, context)

    assert len(dest.read_bytes()) == adversarial.TRUNCATED_HEADER_BYTES
    assert adversarial.TRUNCATED_HEADER_BYTES < 44


def test_truncated_header_refuses_a_traversing_source(context):
    """The guard covers the builders that read the file directly, not only ffmpeg ones."""
    with pytest.raises(ValueError):
        adversarial.build_truncated_header(context["root"] / "out.wav", {"source": "../../etc/passwd"}, context)


def test_zero_byte_writes_an_empty_file(context):
    """An empty upload is its own failure mode."""
    dest = context["root"] / "out.wav"

    adversarial.build_zero_byte(dest, {}, {})

    assert dest.read_bytes() == b""


def test_mp3_named_wav_encodes_mp3_behind_a_wav_name(context, calls):
    """The service must sniff content rather than trust the filename."""
    dest = context["root"] / "out.wav"

    adversarial.build_mp3_named_wav(dest, {"source": "en_core"}, context)

    assert calls["to_mp3"].call_count == 1
    assert Path(calls["to_mp3"].call_args[0][1]).suffix == ".wav"


def test_build_dispatches_to_the_named_builder(context, calls):
    """The manifest names a builder; dispatch is where that name becomes a call."""
    adversarial.build({"builder": "silence", "params": {"seconds": 2}}, context["root"] / "out.wav", context)

    assert calls["lavfi"].call_args.kwargs["seconds"] == 2


def test_build_defaults_missing_params_to_an_empty_mapping(context, calls):
    """An entry with no params is valid; the builder's own defaults apply."""
    adversarial.build({"builder": "zero_byte"}, context["root"] / "out.wav", context)

    assert (context["root"] / "out.wav").read_bytes() == b""


def test_build_rejects_an_unknown_builder(context):
    """A typo in the manifest must name itself rather than failing somewhere downstream."""
    with pytest.raises(KeyError, match="unknown adversarial builder"):
        adversarial.build({"builder": "not_a_builder"}, context["root"] / "out.wav", context)
