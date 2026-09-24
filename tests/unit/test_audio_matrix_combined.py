"""Tests for scripts/audio_matrix/combined.py, the code-switched clip builder.

The sidecar these clips carry is what lets the real-audio suite assert *which* language
appeared *where*, so the leg bounds have to describe the file that was actually written.
Several of the cases below cover failures the module's comments record as having happened.
"""

import json
from unittest import mock

import pytest

from scripts.audio_matrix import combined


@pytest.fixture(name="context")
def _context(tmp_path):
    """A builder context rooted at a temporary audio-matrix directory."""
    return {"root": tmp_path, "rate": 16000, "pins": {"length_scale": 1.0}}


@pytest.fixture(name="deps")
def _deps():
    """Stub piper synthesis and the ffmpeg wrappers combined.py drives."""
    with mock.patch.object(combined.piper, "ensure_voice", return_value="voice.onnx") as ensure_voice:
        with mock.patch.object(combined.piper, "synth") as synth:
            with mock.patch.multiple(
                combined.render,
                to_pcm16_mono=mock.DEFAULT,
                lavfi=mock.DEFAULT,
                concat=mock.DEFAULT,
                probe_duration=mock.DEFAULT,
            ) as render_calls:
                render_calls["probe_duration"].return_value = 2.0
                render_calls["to_pcm16_mono"].side_effect = lambda raw, dest, rate: dest.write_bytes(b"RIFF")
                render_calls["lavfi"].side_effect = lambda *a, **k: a[1].write_bytes(b"RIFF")
                yield {"ensure_voice": ensure_voice, "synth": synth, **render_calls}


def _entry(leg_count=2):
    """A code-switched entry with `leg_count` legs."""
    languages = ["en", "fr", "de"]
    return {
        "id": "mix_en_fr",
        "legs": [{"language": languages[i], "text": f"text-{i}", "voice": f"voice-{i}"} for i in range(leg_count)],
    }


def test_build_concatenates_legs_and_writes_the_sidecar(context, deps, tmp_path):
    """The ordinary path: two legs, one joiner, one sidecar describing both."""
    dest = tmp_path / "mix_en_fr.partial.wav"

    combined.build(_entry(), dest, context)

    assert deps["concat"].call_count == 1
    sidecar = dest.with_name(dest.name + ".legs.json")
    legs = json.loads(sidecar.read_text(encoding="utf-8"))["legs"]
    assert [leg["language"] for leg in legs] == ["en", "fr"]


def test_leg_bounds_account_for_the_joining_silence(context, deps, tmp_path):
    """The second leg starts after the first leg plus the joiner, not immediately."""
    dest = tmp_path / "mix_en_fr.partial.wav"

    combined.build(_entry(), dest, context)

    legs = json.loads(dest.with_name(dest.name + ".legs.json").read_text(encoding="utf-8"))["legs"]
    assert legs[0] == {"start": 0.0, "end": 2.0, "language": "en", "text": "text-0"}
    assert legs[1]["start"] == pytest.approx(2.0 + combined.JOIN_SILENCE_SECONDS)


def test_no_joiner_is_appended_after_the_last_leg(context, deps, tmp_path):
    """Appending after the last leg padded every clip with 0.6s the sidecar never
    described, so a test asserting a leg reached the end measured against a duration the
    manifest did not account for."""
    dest = tmp_path / "mix_en_fr.partial.wav"

    combined.build(_entry(leg_count=3), dest, context)

    parts = deps["concat"].call_args[0][0]
    assert len(parts) == 5
    assert "_join_" not in parts[-1].name


def test_the_sidecar_is_named_from_the_staging_path(context, deps, tmp_path):
    """`dest` is the caller's "<id>.partial.wav" staging path.

    with_suffix would have produced the FINAL "<id>.legs.json" -- publishing the sidecar
    before the audio it describes exists, and leaving it behind describing the previous
    render if the publish then failed. cli._render_atomically moves it into place after.
    """
    dest = tmp_path / "mix_en_fr.partial.wav"

    combined.build(_entry(), dest, context)

    assert (tmp_path / "mix_en_fr.partial.wav.legs.json").exists()
    assert not (tmp_path / "mix_en_fr.legs.json").exists()


def test_build_removes_every_temporary_it_made(context, deps, tmp_path):
    """Legs and joiner are this call's temporaries and must not survive it."""
    dest = tmp_path / "mix_en_fr.partial.wav"

    combined.build(_entry(), dest, context)

    leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith("_")]
    assert leftovers == []


def test_build_cleans_up_when_concat_fails(context, deps, tmp_path):
    """Failure is when leftovers would otherwise accumulate in the cache directory."""
    deps["concat"].side_effect = RuntimeError("ffmpeg exploded")
    dest = tmp_path / "mix_en_fr.partial.wav"
    entry = _entry()

    with pytest.raises(RuntimeError):
        combined.build(entry, dest, context)

    assert [p.name for p in tmp_path.iterdir() if p.name.startswith("_")] == []


def test_a_single_leg_entry_still_removes_its_joiner(context, deps, tmp_path):
    """A one-leg entry never appends the joiner to `parts`, so it is removed separately --
    otherwise it is left behind in the cache directory on every run."""
    dest = tmp_path / "solo.partial.wav"

    combined.build(_entry(leg_count=1), dest, context)

    assert [p.name for p in tmp_path.iterdir() if p.name.startswith("_join_")] == []


def test_render_leg_removes_a_partial_leg_when_synthesis_fails(context, deps, tmp_path):
    """A half-written leg is worse than none.

    build()'s cleanup only removes legs it was handed, and an incomplete one never gets
    returned -- so a partial would survive and be concatenated by the next attempt.
    """
    deps["synth"].side_effect = RuntimeError("piper exploded")

    with pytest.raises(RuntimeError):
        combined._render_leg({"text": "x", "voice": "v", "language": "en"}, 0, {**context, "entry_id": "mix"})

    assert [p.name for p in tmp_path.iterdir()] == []


def test_render_leg_removes_the_raw_file_on_success(context, deps, tmp_path):
    """The raw Piper output is an intermediate; only the PCM leg is kept."""
    leg = combined._render_leg({"text": "x", "voice": "v", "language": "en"}, 0, {**context, "entry_id": "mix"})

    assert leg.exists()
    assert not leg.with_name(leg.name + ".raw.wav").exists()


def test_each_leg_uses_its_own_voice(context, deps, tmp_path):
    """A code-switched clip is only meaningful if the legs sound like different speakers."""
    dest = tmp_path / "mix.partial.wav"

    combined.build(_entry(), dest, context)

    requested = [call[0][1] for call in deps["ensure_voice"].call_args_list]
    assert requested == ["voice-0", "voice-1"]
