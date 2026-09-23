"""Tests for the render, bed and publish stages of scripts/audio_matrix/longform.py.

A 20-minute build writes hundreds of numbered parts plus the beds, all with fixed names in
a shared cache directory -- so most of what matters here is what happens on failure. The
layout planners are covered by test_longform_layout.py; these cover everything downstream
of a plan.
"""

import json
from pathlib import Path
from unittest import mock

import pytest

from scripts.audio_matrix import longform


@pytest.fixture(name="context")
def _context(tmp_path):
    """A build context rooted at a temporary cache directory."""
    return {"root": tmp_path, "rate": 16000}


@pytest.fixture(name="render_calls")
def _render_calls():
    """Stub the ffmpeg wrappers, creating the files the real ones would."""
    with mock.patch.multiple(
        longform.render,
        lavfi=mock.DEFAULT,
        apply_gain=mock.DEFAULT,
        mix=mock.DEFAULT,
        concat=mock.DEFAULT,
        run_ffmpeg=mock.DEFAULT,
        probe_duration=mock.DEFAULT,
    ) as calls:
        calls["lavfi"].side_effect = lambda source, dest, rate, seconds=None: Path(dest).write_bytes(b"RIFF")
        calls["apply_gain"].side_effect = lambda src, dest, gain, rate: Path(dest).write_bytes(b"RIFF")
        calls["mix"].side_effect = lambda sources, dest, rate, **kw: Path(dest).write_bytes(b"RIFF")
        calls["concat"].side_effect = lambda sources, dest, rate: Path(dest).write_bytes(b"RIFF")
        calls["run_ffmpeg"].side_effect = lambda args: Path(args[-1]).write_bytes(b"RIFF")
        calls["probe_duration"].return_value = 10.0
        yield calls


def _speech_block(path, duration=3.0, language="en"):
    """One spoken block as the planners emit it."""
    return {
        "kind": "speech",
        "path": str(path),
        "duration": duration,
        "gain": 1.0,
        "language": language,
        "text": "hello",
        "expect_words": ["hello"],
    }


def test_render_blocks_registers_every_part_before_rendering_it(context, render_calls):
    """Registration before the render is what lets the caller clean up after a failure.

    Relying on the return value alone meant a failure part-way through returned nothing,
    so the hundreds of `_lf_NNNN.wav` files already written were never registered -- and
    they are the exact names the next attempt writes.
    """
    blocks = [_speech_block(context["root"] / "a.wav"), {"kind": "pause", "duration": 2.0}]
    temporaries = []
    render_calls["apply_gain"].side_effect = RuntimeError("ffmpeg exploded")

    with pytest.raises(RuntimeError):
        longform._render_blocks(blocks, context, temporaries)

    assert temporaries == [context["root"] / "_lf_0000.wav"]


def test_render_blocks_returns_speech_and_quiet_timelines(context, render_calls):
    """The sidecar's ground truth comes from here: what was said, and where it was quiet."""
    blocks = [
        _speech_block(context["root"] / "a.wav", duration=3.0),
        {"kind": "pause", "duration": longform.QUIET_WINDOW_MIN_SECONDS + 1},
        _speech_block(context["root"] / "b.wav", duration=2.0, language="fr"),
    ]

    parts, speech, quiet = longform._render_blocks(blocks, context, [])

    assert len(parts) == 3
    assert [s["language"] for s in speech] == ["en", "fr"]
    assert speech[0] == {"start": 0.0, "end": 3.0, "language": "en", "text": "hello", "expect_words": ["hello"], "gain": 1.0}
    assert quiet == [{"start": 3.0, "end": 3.0 + longform.QUIET_WINDOW_MIN_SECONDS + 1, "kind": "pause"}]


def test_a_short_pause_is_not_a_quiet_window(context, render_calls):
    """Quiet windows are the ones a decoder could hallucinate into; a brief gap is not."""
    blocks = [{"kind": "pause", "duration": longform.QUIET_WINDOW_MIN_SECONDS - 0.1}]

    _, _, quiet = longform._render_blocks(blocks, context, [])

    assert not quiet


def test_placed_bed_removes_its_head_and_tail(context, render_calls):
    """Both are fixed names in the shared cache directory.

    One left behind by a failed render is silently reused by the next build at whatever
    length it happened to have, shifting every ground-truth offset in the sidecar.
    """
    body = context["root"] / "body.wav"
    body.write_bytes(b"RIFF")
    dest = context["root"] / "_lf_music.wav"

    longform._placed_bed(body, 5.0, 30.0, dest, root=context["root"], rate=16000)

    assert not (context["root"] / "_lf_music_head.wav").exists()
    assert not (context["root"] / "_lf_music_tail.wav").exists()


def test_placed_bed_removes_head_and_tail_when_the_concat_fails(context, render_calls):
    """Failure is exactly when a stale head would survive into the next build."""
    body = context["root"] / "body.wav"
    body.write_bytes(b"RIFF")
    render_calls["concat"].side_effect = RuntimeError("ffmpeg exploded")

    with pytest.raises(RuntimeError):
        longform._placed_bed(body, 5.0, 30.0, context["root"] / "_lf_music.wav", root=context["root"], rate=16000)

    assert not (context["root"] / "_lf_music_head.wav").exists()
    assert not (context["root"] / "_lf_music_tail.wav").exists()


def test_placed_bed_never_asks_for_a_negative_tail(context, render_calls):
    """A body longer than the remaining time would otherwise request a negative duration."""
    body = context["root"] / "body.wav"
    body.write_bytes(b"RIFF")
    render_calls["probe_duration"].return_value = 100.0

    longform._placed_bed(body, 5.0, 30.0, context["root"] / "_lf_music.wav", root=context["root"], rate=16000)

    durations = [call.kwargs["seconds"] for call in render_calls["lavfi"].call_args_list]
    assert all(duration > 0 for duration in durations)


def test_build_beds_registers_every_path_and_drops_the_bodies(context, render_calls):
    """The two bodies are removed on success too: leaving them would keep ~105s alive."""
    temporaries = []

    beds = longform._build_beds(600.0, context, temporaries)

    assert len(beds) == 3
    assert context["root"] / "_lf_music_body.wav" in temporaries
    assert not (context["root"] / "_lf_music_body.wav").exists()
    assert not (context["root"] / "_lf_noise_body.wav").exists()


def test_mix_final_limits_the_result(context, render_calls):
    """amix does not normalise, so the limiter is what stops the mix clipping."""
    longform._mix_final(context["root"] / "speech.wav", [context["root"] / "a.wav"], context["root"] / "out.wav", 16000)

    args = render_calls["run_ffmpeg"].call_args[0][0]
    chain = args[args.index("-filter_complex") + 1]
    assert "amix=inputs=2" in chain
    assert "alimiter=limit=0.95" in chain


def test_publish_moves_the_clip_last(tmp_path):
    """The clip is the file a reader keys on, so it appears only once the sidecar exists.

    The clip used to be written straight to its final path with the sidecar written after
    cleanup, so an interruption left new audio beside the *previous* run's timeline --
    which looks entirely normal on disk and is worse than no sidecar at all.
    """
    staged = tmp_path / "lf.partial.wav"
    staged.write_bytes(b"RIFF")
    dest = tmp_path / "lf.wav"
    seen = []

    real_replace = longform.os.replace

    def _record(src, target):
        seen.append(Path(target).name)
        real_replace(src, target)

    with mock.patch.object(longform.os, "replace", side_effect=_record):
        longform._publish(staged, dest, {"duration": 10.0})

    assert seen == ["lf.timeline.json", "lf.wav"]
    assert json.loads((tmp_path / "lf.timeline.json").read_text(encoding="utf-8")) == {"duration": 10.0}


def test_publish_removes_its_staged_sidecar_on_failure(tmp_path):
    """A staged sidecar left behind would be published by the next run's replace."""
    staged = tmp_path / "lf.partial.wav"
    staged.write_bytes(b"RIFF")

    with mock.patch.object(longform.os, "replace", side_effect=OSError("cross-device link")):
        with pytest.raises(OSError):
            longform._publish(staged, tmp_path / "lf.wav", {"duration": 10.0})

    assert not (tmp_path / "lf.timeline.json.partial").exists()


@pytest.mark.parametrize("profile", ["stress", "natural", "film"])
def test_planner_for_returns_a_planner_per_profile(profile):
    """All three layouts are kept: each catches a different class of decoder failure."""
    assert callable(longform._planner_for(profile, []))


def test_planner_for_rejects_an_unknown_profile():
    """A typo used to render the stress grid under the variant's name and stamp it as a
    valid cached artifact."""
    with pytest.raises(ValueError, match="unknown long-form profile"):
        longform._planner_for("nonsense", [])


def test_the_film_planner_records_its_scenes():
    """The film bed is built from the scenes its planner chose, so they have to come back."""
    scenes = []
    planner = longform._planner_for("film", scenes)

    with mock.patch.object(longform.longform_film, "plan", return_value=([], [{"bed": "music"}])):
        planner([], mock.Mock())

    assert scenes == [{"bed": "music"}]


def test_build_removes_every_temporary_on_failure(context, render_calls):
    """A failure part-way used to leave every numbered part in the cache directory, where
    they are the exact names the next attempt writes -- so a regeneration silently mixed
    the previous run's fragments into the new clip."""
    sources = [{"language": "en", "path": str(context["root"] / "a.wav"), "text": "hello", "expect_words": ["hello"]}]
    render_calls["run_ffmpeg"].side_effect = RuntimeError("ffmpeg exploded")

    with mock.patch.object(longform, "_plan", return_value=[_speech_block(context["root"] / "a.wav")]):
        with pytest.raises(RuntimeError):
            longform.build(sources, context["root"] / "lf.wav", context)

    assert [p.name for p in context["root"].iterdir() if p.name.startswith("_lf")] == []


def test_build_writes_the_clip_and_its_timeline(context, render_calls):
    """The success path: audio and a sidecar describing it, and no leftovers."""
    sources = [{"language": "en", "path": str(context["root"] / "a.wav"), "text": "hello", "expect_words": ["hello"]}]
    dest = context["root"] / "lf.wav"

    with mock.patch.object(longform, "_plan", return_value=[_speech_block(context["root"] / "a.wav")]):
        timeline = longform.build(sources, dest, context)

    assert dest.exists()
    assert (context["root"] / "lf.timeline.json").exists()
    assert timeline["duration"] == 10.0
    assert [p.name for p in context["root"].iterdir() if p.name.startswith("_lf")] == []


def test_the_staged_name_keeps_an_extension_ffmpeg_recognises(context, render_calls):
    """ffmpeg picks its muxer from the extension, and "<name>.wav.partial" has none."""
    sources = [{"language": "en", "path": str(context["root"] / "a.wav"), "text": "hello", "expect_words": ["hello"]}]
    staged_names = []
    render_calls["run_ffmpeg"].side_effect = lambda args: (staged_names.append(Path(args[-1]).name), Path(args[-1]).write_bytes(b"RIFF"))

    with mock.patch.object(longform, "_plan", return_value=[_speech_block(context["root"] / "a.wav")]):
        longform.build(sources, context["root"] / "lf.wav", context)

    assert staged_names[-1] == "lf.partial.wav"


def test_the_natural_planner_returns_nothing_without_sources():
    """A spec naming no languages produces no blocks rather than failing on languages[0].

    Validation rejects such a spec, but the planner is also reached directly by the
    fixture generator, so it degrades rather than raising.
    """
    assert not longform._plan_natural([], mock.Mock())
