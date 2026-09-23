"""Tests for scripts/audio_matrix/cache.py.

The cache is what makes fixture generation idempotent: a clip is rebuilt only when
something that changes its audio changes. The properties worth pinning are that the digest
actually moves when the toolchain does, and that a damaged sidecar degrades into "stale"
rather than aborting a generation run.
"""

import json
import os
from pathlib import Path

from scripts.audio_matrix import cache


def test_cache_root_prefers_an_explicit_path(tmp_path):
    """An explicit path wins over the environment and the default."""
    assert cache.cache_root(str(tmp_path)) == tmp_path.resolve()


def test_cache_root_falls_back_to_the_environment(monkeypatch, tmp_path):
    """ASR_AUDIO_MATRIX_DIR is how the suites point at a prepared cache."""
    monkeypatch.setenv("ASR_AUDIO_MATRIX_DIR", str(tmp_path))

    assert cache.cache_root() == tmp_path.resolve()


def test_cache_root_defaults_under_gitignored_test_data(monkeypatch):
    """The default lives under test_data/, so generated audio cannot be committed."""
    monkeypatch.delenv("ASR_AUDIO_MATRIX_DIR", raising=False)

    root = cache.cache_root()

    assert root.parts[-2:] == ("test_data", "audio_matrix")


def test_cache_root_expands_a_user_path(monkeypatch):
    """A ~-relative override is expanded rather than taken literally."""
    monkeypatch.setenv("ASR_AUDIO_MATRIX_DIR", "~/matrix-cache")

    assert "~" not in str(cache.cache_root())


def test_spec_digest_is_stable_across_separately_built_inputs():
    """Re-running the generator with nothing changed must produce no work.

    The two inputs are constructed independently rather than reusing one object: a
    generator run builds its spec afresh from the manifest each time, so equal-by-value is
    the property that makes the cache idempotent. Passing the same object twice would hold
    for any function at all.
    """
    first = cache.spec_digest({"id": "en_core", "text": "hello"}, {"ffmpeg": "6.1", "piper": "1.8.0"})
    second = cache.spec_digest({"id": "en_core", "text": "hello"}, {"ffmpeg": "6.1", "piper": "1.8.0"})

    assert first == second


def test_spec_digest_ignores_tool_ordering():
    """Tool versions are sorted, so dict ordering cannot invalidate a whole cache."""
    spec = {"id": "en_core"}

    first = cache.spec_digest(spec, {"ffmpeg": "6.1", "piper": "1.8.0"})
    second = cache.spec_digest(spec, {"piper": "1.8.0", "ffmpeg": "6.1"})

    assert first == second


def test_spec_digest_changes_when_the_spec_changes():
    """A reworded clip is a different clip."""
    tools = {"ffmpeg": "6.1"}

    assert cache.spec_digest({"text": "hello"}, tools) != cache.spec_digest({"text": "goodbye"}, tools)


def test_spec_digest_changes_when_a_tool_version_changes():
    """A toolchain bump invalidates the cache -- the reason the digest covers tools at all."""
    spec = {"id": "en_core"}

    assert cache.spec_digest(spec, {"ffmpeg": "6.1"}) != cache.spec_digest(spec, {"ffmpeg": "7.0"})


def test_stamp_path_sits_beside_the_clip(tmp_path):
    """The sidecar keeps the clip's full name so two formats cannot collide."""
    assert cache.stamp_path(tmp_path / "en_core.wav").name == "en_core.wav.stamp.json"
    assert cache.stamp_path(tmp_path / "en_core.flac").name == "en_core.flac.stamp.json"


def test_read_stamp_returns_empty_when_absent(tmp_path):
    """No sidecar means no digest, which makes the clip stale."""
    assert cache.read_stamp(tmp_path / "missing.wav") == ""


def test_write_then_read_round_trips_the_digest(tmp_path):
    """The recorded digest is what a later run compares against."""
    target = tmp_path / "en_core.wav"
    cache.write_stamp(target, "abc123", {"voice": "en_US-amy"})

    assert cache.read_stamp(target) == "abc123"
    payload = json.loads(cache.stamp_path(target).read_text(encoding="utf-8"))
    assert payload["voice"] == "en_US-amy"


def test_write_stamp_leaves_no_temporary_behind(tmp_path):
    """The sidecar is renamed into place, so nothing but the stamp survives the write."""
    target = tmp_path / "en_core.wav"
    cache.write_stamp(target, "abc123")

    assert [p.name for p in tmp_path.iterdir()] == ["en_core.wav.stamp.json"]


def test_read_stamp_treats_a_truncated_sidecar_as_stale(tmp_path):
    """A half-written stamp regenerates the clip instead of aborting the run.

    Propagating the decode error would kill a whole generation over one sidecar, and leave
    no recovery except deleting it by hand.
    """
    target = tmp_path / "en_core.wav"
    cache.stamp_path(target).write_text('{"digest": "abc', encoding="utf-8")

    assert cache.read_stamp(target) == ""


def test_read_stamp_treats_a_non_object_sidecar_as_stale(tmp_path):
    """Valid JSON that is not an object carries no digest."""
    target = tmp_path / "en_core.wav"
    cache.stamp_path(target).write_text("[1, 2, 3]", encoding="utf-8")

    assert cache.read_stamp(target) == ""


def test_read_stamp_treats_undecodable_bytes_as_stale(tmp_path):
    """A sidecar that is not UTF-8 is unreadable, not fatal."""
    target = tmp_path / "en_core.wav"
    cache.stamp_path(target).write_bytes(b"\xff\xfe\x00binary")

    assert cache.read_stamp(target) == ""


def test_is_fresh_requires_both_the_clip_and_a_matching_digest(tmp_path):
    """A stamp without its clip is not freshness -- the audio is what the suites read."""
    target = tmp_path / "en_core.wav"
    cache.write_stamp(target, "abc123")
    assert cache.is_fresh(target, "abc123") is False

    target.write_bytes(b"RIFF")
    assert cache.is_fresh(target, "abc123") is True
    assert cache.is_fresh(target, "different") is False


def test_write_stamp_replaces_an_earlier_digest(tmp_path):
    """Regeneration overwrites the sidecar rather than accumulating entries."""
    target = tmp_path / "en_core.wav"
    cache.write_stamp(target, "old")
    cache.write_stamp(target, "new")

    assert cache.read_stamp(target) == "new"


def test_cache_module_exposes_a_repo_relative_default():
    """DEFAULT_CACHE is derived from the module location, not the caller's cwd."""
    assert isinstance(cache.DEFAULT_CACHE, Path)
    assert cache.DEFAULT_CACHE.is_absolute()
    assert os.path.basename(str(cache.DEFAULT_CACHE)) == "audio_matrix"
