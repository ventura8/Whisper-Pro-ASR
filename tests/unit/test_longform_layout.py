"""The long-form fixture's two layout profiles, and what each is allowed to look like.

The "natural" profile exists because the original layout is not what real media looks
like, and reading it as representative produced two wrong conclusions: that utterance-level
VAD segmentation is straightforward (it is not -- most real turns are separated by less
than the split threshold), and that a whole-file "is this multilingual" gate is the right
abstraction (real code-switching is scene-shaped, not per-line).

The targets below are measured from subtitle tracks of two genuinely code-switching films
-- Life Is Beautiful (1549 Italian cues, 116 min) and Aferim! (1386 Romanian cues, 98 min).
These tests pin the fixture to those measurements so it cannot drift back into being
conveniently easy to segment.
"""

# pylint: disable=protected-access
# The unit under test is the module's internals. Reaching them by name is the point
# of these tests, not an accident: the public surface is a thin wrapper and testing
# only through it would leave the rules below unpinned.

import random
import statistics

import pytest

from scripts.audio_matrix import longform


def _sources():
    """Six languages of ~2s utterances, the real median cue length."""
    langs = ["it", "de", "ro", "tr", "en", "fr"]
    return [
        {
            "language": lang,
            "path": f"/tmp/{lang}_{i}.wav",
            "duration": 2.1,
            "text": f"utterance {i} in {lang}",
            "expect_words": ["utterance", lang],
            "gain": 1.0,
        }
        for lang in langs
        for i in range(4)
    ]


def _gaps(blocks):
    return [b["duration"] for b in blocks if b["kind"] == "gap"]


def _speech(blocks):
    return [b for b in blocks if b["kind"] == "speech"]


class TestNaturalGapDistribution:  # pylint: disable=attribute-defined-outside-init
    """Pauses must match real dialogue, where most turns are tighter than a VAD split."""

    def setup_method(self):
        """Plan one layout with the fixture's own seed, so every assertion reads the same blocks."""
        self.blocks = longform._plan_natural(_sources(), random.Random(longform.LAYOUT_SEED))
        self.gaps = _gaps(self.blocks)

    def test_median_gap_matches_real_dialogue(self):
        """The middle pause must sit where real subtitle tracks put it."""
        # Measured: 0.30s (Life Is Beautiful) and 0.37s (Aferim!).
        assert 0.20 <= statistics.median(self.gaps) <= 0.55

    def test_most_turns_are_tighter_than_a_500ms_split_threshold(self):
        """Most real turns would be merged by VAD, not split; the fixture must reproduce that."""
        # The finding that matters: ~60% of real turns would be MERGED, not split, by
        # VAD at min_silence_duration_ms=500. A fixture without this property makes
        # utterance-level segmentation look far easier than it is.
        tight = sum(1 for g in self.gaps if g < 0.5) / len(self.gaps)
        assert tight >= 0.5, f"only {tight:.0%} of gaps are under 500ms; real dialogue is ~60%"

    def test_long_pauses_stay_rare(self):
        """Long silences exist but must stay as rare as they are in real film."""
        # Measured: 5% of gaps are 8s or longer.
        long_gaps = sum(1 for g in self.gaps if g >= 8.0) / len(self.gaps)
        assert 0.01 <= long_gaps <= 0.12

    def test_some_pauses_are_long_enough_to_judge_hallucination(self):
        """At least one pause must be long enough to catch invented speech in silence."""
        assert any(g >= longform.QUIET_WINDOW_MIN_SECONDS for g in self.gaps)


class TestNaturalLanguageLayout:  # pylint: disable=attribute-defined-outside-init
    """Language must be held for a scene, the way real film does it."""

    def setup_method(self):
        """Plan one layout with the fixture's own seed, so every assertion reads the same blocks."""
        self.blocks = longform._plan_natural(_sources(), random.Random(longform.LAYOUT_SEED))
        self.speech = _speech(self.blocks)

    def test_switches_are_sparse_not_per_utterance(self):
        """Real code-switching is scene-shaped, not line-by-line."""
        switches = sum(1 for a, b in zip(self.speech, self.speech[1:]) if a["language"] != b["language"])
        # The stress layout switches on every utterance. Real film switches a handful of
        # times across a whole feature; a 20-minute clip should be well under one in four.
        assert switches < len(self.speech) / 4, f"{switches} switches over {len(self.speech)} utterances is not scene-shaped"

    def test_more_than_one_language_is_present(self):
        """A code-switching fixture that speaks one language tests nothing."""
        assert len({block["language"] for block in self.speech}) > 1

    def test_one_language_dominates(self):
        """A film is mostly one language with passages in others."""
        counts = {}
        for block in self.speech:
            counts[block["language"]] = counts.get(block["language"], 0) + 1
        share = max(counts.values()) / len(self.speech)
        # A film is mostly one language with passages in others.
        assert share >= 0.4, f"dominant language holds only {share:.0%} of utterances"

    def test_scenes_run_for_several_utterances(self):
        """A scene is a run of turns, not a single line."""
        runs, current = [], 1
        for a, b in zip(self.speech, self.speech[1:]):
            if a["language"] == b["language"]:
                current += 1
            else:
                runs.append(current)
                current = 1
        runs.append(current)
        assert statistics.median(runs) >= 5, "language runs are too short to be scenes"


class TestStressProfileIsUnchanged:
    """The original worst case must stay as it was; the natural profile is an addition."""

    def test_stress_layout_switches_every_utterance(self):
        """The original worst case must stay the worst case."""
        # _plan cycles the source list in order, so it changes language on every utterance
        # exactly when there is one clip per language -- which is how the real fixture is
        # built (ten languages, one voice line each). That is the worst case it exists for.
        one_per_language = [clip for clip in _sources() if clip["path"].endswith("_0.wav")]
        blocks = longform._plan(one_per_language, random.Random(longform.LAYOUT_SEED))
        speech = _speech(blocks)
        switches = sum(1 for a, b in zip(speech, speech[1:]) if a["language"] != b["language"])
        assert switches == len(speech) - 1

    def test_stress_layout_switches_far_more_often_than_natural(self):
        """The two profiles must remain distinguishable, not converge."""
        sources = _sources()
        rng_kwargs = random.Random(longform.LAYOUT_SEED)
        stress = _speech(longform._plan(sources, rng_kwargs))
        natural = _speech(longform._plan_natural(sources, random.Random(longform.LAYOUT_SEED)))

        def rate(blocks):
            switches = sum(1 for a, b in zip(blocks, blocks[1:]) if a["language"] != b["language"])
            return switches / max(1, len(blocks))

        assert rate(stress) > rate(natural) * 2

    def test_stress_layout_keeps_its_generous_pauses(self):
        """The stress profile's pauses stay wide enough for VAD to split on."""
        blocks = longform._plan(_sources(), random.Random(longform.LAYOUT_SEED))
        assert min(_gaps(blocks)) >= longform.SHORT_GAP[0]


class TestBuildSelectsTheProfile:  # pylint: disable=too-few-public-methods
    """Which layout build() plans for a given profile name."""

    def test_unknown_profile_falls_back_to_stress(self, monkeypatch):
        """An unrecognised profile must yield the stress plan, not merely *a* plan.

        Asserting only "switches > 0" passed for the natural profile too, so the fallback
        could have silently selected the wrong layout. The two plans are compared instead.
        """
        plans = {}

        def fake_render(blocks, context):  # pylint: disable=unused-argument
            plans[fake_render.profile] = [(block["kind"], block.get("language"), round(block["duration"], 4)) for block in blocks]
            raise RuntimeError("stop after planning")

        monkeypatch.setattr(longform, "_render_blocks", fake_render)

        for profile in ("stress", "anything-else"):
            fake_render.profile = profile
            with pytest.raises(RuntimeError, match="stop after planning"):
                longform.build(_sources(), None, {"root": None, "rate": 16000}, profile=profile)
            assert plans[profile], f"{profile} produced no blocks"

        assert plans["anything-else"] == plans["stress"], "an unknown profile did not fall back to the stress layout"
