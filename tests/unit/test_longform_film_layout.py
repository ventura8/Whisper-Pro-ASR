"""The ``film`` long-form profile, pinned to the acoustic structure it was measured from.

The natural profile reproduces real pause spacing; this one reproduces the rest of what
real screen dialogue is made of and the other fixtures are not: short lines in short
exchanges, long non-speech passages, and a music-and-room bed that sits almost at
dialogue level. The numbers in ``longform_film.FILM_STATS`` are medians across nine
excerpts of real film, and these tests hold the layout to them so the fixture cannot drift
back toward the conveniently silent, evenly paced grid it replaced.

Only the planner is exercised: no audio is rendered, and the sources are stand-ins with the
durations real lines and recordings have.
"""

import random
import statistics

import pytest

from scripts.audio_matrix import longform_film

SEED = 20240901


def _sources():
    """Four short lines and two long recordings per language, as the matrix now has."""
    langs = ["en", "es", "fr", "de"]
    lines = [
        {
            "language": lang,
            "path": f"/tmp/{lang}_line{i}.wav",
            "duration": 1.3,
            "role": "line",
            "text": f"line {i}",
            "expect_words": ["line"],
            "gain": 1.0,
        }
        for lang in langs
        for i in range(1, 5)
    ]
    long = [
        {
            "language": lang,
            "path": f"/tmp/{lang}_scene{i}.wav",
            "duration": 6.3,
            "text": f"scene {i}",
            "expect_words": ["scene"],
            "gain": 1.0,
        }
        for lang in langs
        for i in range(2, 4)
    ]
    return lines + long


def _layout(target=1200.0):
    return longform_film.plan(_sources(), random.Random(SEED), target)


def _speech(blocks):
    return [b for b in blocks if b["kind"] == "speech"]


def _gaps(blocks):
    return [b["duration"] for b in blocks if b["kind"] == "gap"]


def _timed_speech(blocks):
    """Speech blocks with the start time the renderer will give them."""
    out, cursor = [], 0.0
    for block in blocks:
        if block["kind"] == "speech":
            # Rounded as the planner rounds its scene edges, so a line that starts exactly
            # on a boundary is placed by the same arithmetic on both sides.
            out.append({**block, "start": round(cursor, 3)})
        cursor += block["duration"]
    return out


def _lines_per_scene(blocks, scenes):
    speech = _timed_speech(blocks)
    return [sum(scene["start"] <= s["start"] < scene["end"] for s in speech) for scene in scenes]


class TestWhatTheFixtureIsMadeOf:
    """Short exchanges over a loud bed, with the long tail real film has."""

    def test_lines_are_short_and_mostly_from_the_line_clips(self):
        """Real utterances have a 1.28s median; the ~6s recordings supply only the tail."""
        speech = _speech(_layout()[0])
        assert statistics.median(s["duration"] for s in speech) < 2.0
        share_short = sum(s.get("role") == "line" for s in speech) / len(speech)
        assert 0.8 <= share_short <= 0.95

    def test_scenes_are_short_exchanges_with_a_long_tail(self):
        """Median 3 lines per scene, measured -- the natural profile's 8 is a paragraph --
        but a few long dialogue scenes carry most of the speech, so the mean is well above
        the median. Both are the shape of real film; a fixture with only the median would
        have a quarter of the speech density."""
        per_scene = _lines_per_scene(*_layout())
        assert 2 <= statistics.median(per_scene) <= 4
        assert statistics.mean(per_scene) >= 1.5 * statistics.median(per_scene)
        assert max(per_scene) >= 15

    def test_speech_density_is_that_of_film(self):
        """0.42 measured, range 0.18-0.75; the other profiles sit at 0.55 and 0.68."""
        blocks, _ = _layout()
        speech = sum(s["duration"] for s in _speech(blocks))
        total = speech + sum(_gaps(blocks))
        assert 0.30 <= speech / total <= 0.55

    def test_there_are_passages_over_a_minute_long(self):
        """Credits, action, a montage: the longest non-speech stretch on real film ran 30-232s.
        Nothing else in the matrix has a silence over 17s."""
        assert max(_gaps(_layout()[0])) > 45.0

    def test_the_bed_never_stops(self):
        """Scenes tile the whole clip, each spanning its own silence, so the bed built from
        them is continuous: the room does not go quiet between lines."""
        blocks, scenes = _layout()
        total = sum(s["duration"] for s in _speech(blocks)) + sum(_gaps(blocks))
        assert scenes[0]["start"] == 0.0
        assert all(a["end"] == b["start"] for a, b in zip(scenes, scenes[1:]))
        assert abs(scenes[-1]["end"] - total) < 0.01

    def test_bed_levels_come_from_the_measured_range(self):
        """Median 2.6 dB below speech, and sometimes above it, as one film in nine was."""
        levels = [s["bed_db"] for s in _layout()[1]]
        assert all(-6.0 <= level <= 9.8 for level in levels)
        assert 0.0 <= statistics.median(levels) <= 7.0
        assert min(levels) < 0.0, "no scene has the bed above the dialogue"


class TestLanguageIsHeldForAScene:
    """Switches happen between scenes, never inside one."""

    def test_no_switch_inside_a_scene(self):
        """Every line inside a scene span shares that scene's language."""
        blocks, scenes = _layout()
        speech = _timed_speech(blocks)
        for scene in scenes:
            languages = {s["language"] for s in speech if scene["start"] <= s["start"] < scene["end"]}
            assert len(languages) <= 1, f"scene {scene['start']:.0f}s holds {sorted(languages)}"

    def test_the_clip_does_switch(self):
        """It is a code-switching fixture: several languages, changing between scenes."""
        _, scenes = _layout()
        speech = _timed_speech(_layout()[0])
        assert len({s["language"] for s in speech}) >= 3
        assert len(scenes) >= 20

    def test_one_language_dominates(self):
        """A film is mostly one language with passages in others, not a round-robin."""
        speech = _speech(_layout()[0])
        top = max({s["language"] for s in speech}, key=lambda lang: sum(x["language"] == lang for x in speech))
        assert sum(s["language"] == top for s in speech) / len(speech) > 0.5


class TestNothingBreaksWithoutLines:
    """A language with no short clips still gets scenes, from whatever it has."""

    def test_falls_back_to_the_long_recordings(self):
        """The tail pool becomes the only pool; scenes are still laid out."""
        sources = [s for s in _sources() if s.get("role") != "line"]
        blocks, scenes = longform_film.plan(sources, random.Random(SEED), 300.0)
        assert _speech(blocks) and scenes

    def test_no_sources_means_no_layout(self):
        """Nothing to lay out is an empty plan, not an error."""
        assert longform_film.plan([], random.Random(SEED), 300.0) == ([], [])


class TestShapes:
    """The passages a real title has around its dialogue, laid down where they are due."""

    OPENING = {"at": "start", "seconds": (40.0, 40.0), "bed_db": -3.0, "bed": "music", "kind": "opening"}
    TITLE = {"at": "after", "after_seconds": (300.0, 300.0), "seconds": (30.0, 30.0), "bed_db": -6.0, "bed": "music", "kind": "title"}
    CREDITS = {"at": "end", "seconds": (90.0, 90.0), "bed_db": 0.0, "bed": "music", "kind": "credits"}

    def _with_shape(self, monkeypatch, passages, target=1200.0):
        monkeypatch.setitem(longform_film.film_shapes.SHAPES, "probe", {"passages": tuple(passages)})
        return longform_film.plan(_sources(), random.Random(SEED), target, "probe")

    def test_the_plain_film_shape_is_the_measured_layout(self, monkeypatch):
        """No passages: byte-identical to the layout every film number is stated against.

        The expected layout comes from a shape registered here with no passages at all, so
        this fails the day the default shape grows one; and the default's scenes carry no
        passage kind, which is what "no passages" means in the ground truth.
        """
        monkeypatch.setitem(longform_film.film_shapes.SHAPES, "bare", {"passages": ()})
        expected = longform_film.plan(_sources(), random.Random(SEED), 1200.0, "bare")
        blocks, scenes = longform_film.plan(_sources(), random.Random(SEED), 1200.0)
        assert (blocks, scenes) == expected
        assert longform_film.plan(_sources(), random.Random(SEED), 1200.0, "film") == expected
        assert not any("kind" in scene for scene in scenes)

    def test_a_passage_due_after_the_clip_ends_is_due_at_the_end(self, monkeypatch):
        """A title sequence drawn at 300 s on a 200-s clip: without the clamp the planner
        would lay dialogue until 300 s to reach it, and the clip would be half again as long."""
        monkeypatch.setitem(longform_film.film_shapes.SHAPES, "probe", {"passages": (self.TITLE, self.CREDITS)})
        due = longform_film.film_shapes.schedule("probe", random.Random(SEED), 200.0)
        assert all(entry["at"] <= 200.0 for entry in due)
        _, scenes = self._with_shape(monkeypatch, [self.TITLE], target=200.0)
        assert scenes[-1]["kind"] == "title"
        assert all(scene["start"] < 200.0 for scene in scenes if "kind" not in scene), "no dialogue is laid past the target"

    def test_a_long_scene_cannot_carry_the_clip_past_the_credits(self, monkeypatch):
        """Every scene is 28 lines long here, so without a bound the last one would run
        past the target and the credits due before it would never be laid down."""
        monkeypatch.setattr(longform_film, "SCENE_LENGTH_WEIGHTS", ((28, 1),))
        blocks, scenes = self._with_shape(monkeypatch, [self.CREDITS], target=300.0)
        closing = scenes[-1]
        assert closing["kind"] == "credits"
        assert closing["end"] - closing["start"] == 90.0
        assert _timed_speech(blocks)[-1]["start"] < closing["start"]
        # The scene before the credits stopped taking lines once the credits were due.
        assert closing["start"] < 210.0 + 10.0

    def test_an_opening_comes_first_with_its_own_bed(self, monkeypatch):
        """Forty seconds of logo music before the first line, louder than the dialogue."""
        blocks, scenes = self._with_shape(monkeypatch, [self.OPENING])
        assert blocks[0] == {"kind": "gap", "duration": 40.0}
        assert scenes[0] == {"start": 0.0, "end": 40.0, "bed_db": -3.0, "bed": "music", "kind": "opening"}
        assert _timed_speech(blocks)[0]["start"] >= 40.0

    def test_a_title_sequence_is_placed_after_the_cold_open(self, monkeypatch):
        """The scene in progress finishes; the title follows at the first scene boundary after 300 s."""
        blocks, scenes = self._with_shape(monkeypatch, [self.TITLE])
        title = next(scene for scene in scenes if scene.get("kind") == "title")
        assert 300.0 <= title["start"] < 360.0
        assert title["end"] - title["start"] == 30.0
        assert not any(title["start"] <= s["start"] < title["end"] for s in _timed_speech(blocks))

    def test_credits_close_the_clip(self, monkeypatch):
        """The last ninety seconds are credits, and the clip still ends near the target."""
        blocks, scenes = self._with_shape(monkeypatch, [self.CREDITS])
        closing = scenes[-1]
        assert closing["kind"] == "credits"
        assert closing["end"] - closing["start"] == 90.0
        assert _timed_speech(blocks)[-1]["start"] < closing["start"]
        assert closing["end"] < 1200.0 + 60.0

    def test_passages_are_quiet_windows_in_the_ground_truth(self, monkeypatch):
        """Silence in the speech track over a bed: exactly what ``quiet`` must hold against."""
        blocks, _ = self._with_shape(monkeypatch, [self.OPENING, self.TITLE, self.CREDITS])
        gaps = [b["duration"] for b in blocks if b["kind"] == "gap"]
        assert all(seconds in gaps for seconds in (30.0, 40.0, 90.0))

    def test_a_shape_may_set_its_own_bed_levels(self, monkeypatch):
        """A dub track or a loud score: every scene's bed drawn from the shape's range."""
        monkeypatch.setitem(longform_film.film_shapes.SHAPES, "loud", {"passages": (), "bed_levels_db": ((-6.0, 1),)})
        _, scenes = longform_film.plan(_sources(), random.Random(SEED), 600.0, "loud")
        assert {scene["bed_db"] for scene in scenes} == {-6.0}


class TestBedsWithoutScenes:
    """The tail bed extends the last scene; with no scene at all there is nothing to extend."""

    def test_audio_to_cover_and_no_scene_is_an_error_not_an_index_error(self, tmp_path):
        """A plan that laid out nothing used to reach ``scenes[-1]`` and raise IndexError."""
        with pytest.raises(ValueError, match="no scenes to render a bed from"):
            longform_film.beds([], 12.0, {"root": tmp_path, "rate": 16000}, [])
