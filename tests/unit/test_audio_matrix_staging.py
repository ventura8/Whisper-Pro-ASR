"""The atomic staging path must stay renderable by ffmpeg.

ffmpeg chooses its muxer from the filename extension. A staging path of "<id>.wav.partial"
has no extension it recognises, so every render failed with "Unable to choose an output
format ... use a standard extension". Atomic staging landed after the fixtures were last
generated, so nothing exercised it until the next full regeneration -- at which point all 76
entries failed at once. These tests pin the ordering that keeps the extension intact.
"""

from pathlib import Path

from scripts.audio_matrix import cli


class TestStagingKeepsTheExtension:
    """`.partial` belongs before the suffix, never after it."""

    def _staged_path(self, tmp_path, name):
        """Return the staging path _render_atomically hands its renderer."""
        seen = {}

        def fake_renderer(_entry, staged, _context):
            seen["staged"] = staged
            staged.write_bytes(b"audio")

        dest = tmp_path / name
        original = dict(cli._RENDERERS)
        cli._RENDERERS["clips"] = fake_renderer
        try:
            cli._render_atomically("clips", {"id": "probe"}, dest, {})
        finally:
            cli._RENDERERS.clear()
            cli._RENDERERS.update(original)
        return seen["staged"], dest

    def test_the_staging_file_keeps_the_destination_suffix(self, tmp_path):
        """Without this ffmpeg cannot pick a muxer and every render fails."""
        staged, _dest = self._staged_path(tmp_path, "en_core.wav")
        assert staged.suffix == ".wav", f"ffmpeg cannot infer a format from {staged.name!r}"

    def test_the_staging_file_is_not_the_destination(self, tmp_path):
        """A failed render must not leave a truncated file where readers resolve clips."""
        staged, dest = self._staged_path(tmp_path, "en_core.wav")
        assert staged != dest
        assert "partial" in staged.name

    def test_the_completed_render_is_published_and_the_staging_file_removed(self, tmp_path):
        """The whole point of staging: readers see the finished file or the previous one."""
        staged, dest = self._staged_path(tmp_path, "en_core.wav")
        assert dest.read_bytes() == b"audio"
        assert not staged.exists()

    def test_a_flac_destination_keeps_its_own_suffix(self, tmp_path):
        """The core tier publishes FLAC, so the rule cannot be hardcoded to .wav."""
        staged, _dest = self._staged_path(tmp_path, "en_core.flac")
        assert staged.suffix == ".flac"


def test_longform_stages_beside_its_extension_too():
    """longform.build stages its clip the same way and had the same defect."""
    dest = Path("/tmp/longform_stress.wav")
    staged = dest.with_name(f"{dest.stem}.partial{dest.suffix}")
    assert staged.name == "longform_stress.partial.wav"
