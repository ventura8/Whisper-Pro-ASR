"""Grouping regions into language runs, and what may and may not start one.

Language detection on a single ~1 s line is wrong about one time in five on real film:
measured on 19 monolingual films, a median 20% of speech was labelled -- and therefore
decoded -- as a neighbouring language. A run of several seconds is not. These tests pin
the hysteresis that turns per-region labels into runs the decoder can be trusted with,
and the boundary in the other direction: the stress fixture, where every utterance really
is a different language and every one must keep its own run.
"""

import pytest

from modules.inference.runtime import language_runs


def _regions(*spans):
    return [{"start": s, "end": e} for s, e in spans]


def _labels(*items):
    """(start, end, language, confidence) -> the spans the detector reports."""
    return [{"start": s, "end": e, "language": lang, "confidence": conf} for s, e, lang, conf in items]


@pytest.fixture(autouse=True)
def _thresholds(monkeypatch):
    monkeypatch.setattr(language_runs.config, "SEGMENT_RUN_MIN_SWITCH_SEC", 3.0)
    monkeypatch.setattr(language_runs.config, "SEGMENT_RUN_MIN_SWITCH_SHARE", 0.2)
    monkeypatch.setattr(language_runs.config, "LD_MIN_CONFIDENCE", 0.5)


class TestAMonolingualFilm:
    """The common case: one language, and a detector that is occasionally wrong about a line.

    Film scale throughout -- over 15 s of labelled speech -- so the bar in seconds decides; the
    share rule for short clips has its own class below.
    """

    def test_same_language_regions_become_one_run(self):
        """Three lines of Russian are one decode window, not three."""
        regions = _regions((0, 1.2), (1.5, 2.4), (2.8, 4.0))
        labels = _labels((0, 1.2, "ru", 0.9), (1.5, 2.4, "ru", 0.8), (2.8, 4.0, "ru", 0.9))
        runs = language_runs.build(regions, labels)
        assert [(r["start"], r["end"], r["language"]) for r in runs] == [(0, 4.0, "ru")]

    def test_a_one_second_mislabel_is_absorbed(self):
        """The measured failure: a Russian line read as Ukrainian, between Russian lines."""
        regions = _regions((0, 10.0), (10.3, 11.4), (11.7, 20.0))
        labels = _labels((0, 10.0, "ru", 0.9), (10.3, 11.4, "uk", 0.95), (11.7, 20.0, "ru", 0.9))
        runs = language_runs.build(regions, labels)
        assert [r["language"] for r in runs] == ["ru"]
        assert runs[0]["end"] == 20.0

    def test_two_short_mislabels_in_a_row_are_still_absorbed(self):
        """A pair of 1 s dissenters is 2 s of evidence, under the 3 s a switch needs."""
        regions = _regions((0, 10.0), (10.3, 11.3), (11.5, 12.5), (12.8, 20.0))
        labels = _labels((0, 10.0, "it", 0.9), (10.3, 11.3, "es", 0.9), (11.5, 12.5, "es", 0.9), (12.8, 20.0, "it", 0.9))
        assert [r["language"] for r in language_runs.build(regions, labels)] == ["it"]

    def test_a_confident_block_of_enough_speech_becomes_its_own_run(self):
        """Three seconds of another language, confidently detected, is a scene, not a slip --
        and the Italian after it returns to the film's language for free. Over 15 s of
        Italian around it, so the 3 s bar is the one deciding, not the short-clip share."""
        regions = _regions((0, 10.0), (10.3, 12.0), (12.2, 13.8), (14.0, 24.0))
        labels = _labels((0, 10.0, "it", 0.9), (10.3, 12.0, "es", 0.9), (12.2, 13.8, "es", 0.9), (14.0, 24.0, "it", 0.9))
        assert [r["language"] for r in language_runs.build(regions, labels)] == ["it", "es", "it"]
        under_the_bar = _labels((0, 10.0, "it", 0.9), (10.3, 12.0, "es", 0.9), (12.2, 13.4, "es", 0.9), (14.0, 24.0, "it", 0.9))
        regions_under = _regions((0, 10.0), (10.3, 12.0), (12.2, 13.4), (14.0, 24.0))
        assert [r["language"] for r in language_runs.build(regions_under, under_the_bar)] == ["it"], "2.9 s is under the bar"

    def test_a_region_the_detector_could_not_name_does_not_make_a_scene_unsure(self):
        """A breath in the middle of a foreign scene carries no label. It used to count as
        a zero, which pulled the scene's mean confidence under the bar and absorbed it."""
        regions = _regions((0, 2.0), (2.3, 4.0), (4.2, 4.6), (4.8, 6.5), (7.0, 9.0))
        labels = _labels((0, 2.0, "it", 0.9), (2.3, 4.0, "es", 0.7), (4.8, 6.5, "es", 0.7), (7.0, 9.0, "it", 0.9))
        runs = language_runs.build(regions, labels)
        assert [r["language"] for r in runs] == ["it", "es", "it"]
        assert runs[1]["confidence"] == 0.7, "the mean is over the labels, not over the regions"
        assert [r["start"] for r in runs[1]["regions"]] == [2.3, 4.2, 4.8], "the unlabelled region stays in its scene"

    def test_returning_to_the_dominant_language_needs_no_evidence(self):
        """Leaving the film's language is the claim that must be earned; returning is free."""
        regions = _regions((0, 4.0), (4.3, 8.0), (8.2, 9.0))
        labels = _labels((0, 4.0, "ro", 0.9), (4.3, 8.0, "en", 0.9), (8.2, 9.0, "ro", 0.9))
        runs = language_runs.build(regions, labels)
        assert [r["language"] for r in runs] == ["ro", "en", "ro"]

    def test_the_anchor_is_the_labels_not_the_file_detection(self):
        """A music-heavy Polish excerpt was detected as English at file level. Four labels in
        five are right, so the labels outvote the montage and the anchor is Polish."""
        regions = _regions((0, 5.0), (5.2, 10.2), (10.4, 11.4), (11.6, 16.6), (16.8, 21.8))
        labels = _labels(
            (0, 5.0, "pl", 0.9), (5.2, 10.2, "pl", 0.9), (10.4, 11.4, "en", 0.9), (11.6, 16.6, "pl", 0.9), (16.8, 21.8, "pl", 0.9)
        )
        runs = language_runs.build(regions, labels, file_language="en")
        assert [r["language"] for r in runs] == ["pl"]

    def test_short_blocks_of_a_minority_language_are_absorbed_at_either_end(self):
        """Symmetric for anything that is not the anchor: a 2 s block is a 2 s block, whether
        it opens the file or closes it. Spanish holds 8 s of the 12 and is the anchor; the
        two seconds of Italian on each side of it are not."""
        regions = _regions((0, 2.0), (2.3, 10.3), (10.5, 12.5))
        labels = _labels((0, 2.0, "it", 0.9), (2.3, 10.3, "es", 0.9), (10.5, 12.5, "it", 0.9))
        runs = language_runs.build(regions, labels)
        assert [(r["start"], r["end"], r["language"]) for r in runs] == [(0, 12.5, "es")]

    def test_an_unsure_block_is_absorbed_however_long(self):
        """Duration without confidence is not evidence."""
        regions = _regions((0, 2.0), (2.3, 6.0), (6.2, 8.0))
        labels = _labels((0, 2.0, "no", 0.9), (2.3, 6.0, "sv", 0.3), (6.2, 8.0, "no", 0.9))
        assert [r["language"] for r in language_runs.build(regions, labels)] == ["no"]

    def test_a_short_mislabel_that_opens_the_film_is_absorbed_into_what_follows(self):
        """The first line has no run before it; a lone Ukrainian label there must not
        become a permanent Ukrainian run just because it came first."""
        regions = _regions((0, 1.1), (1.4, 12.0), (12.3, 20.0))
        labels = _labels((0, 1.1, "uk", 0.95), (1.4, 12.0, "ru", 0.9), (12.3, 20.0, "ru", 0.9))
        runs = language_runs.build(regions, labels, "ru")
        assert [(r["start"], r["end"], r["language"]) for r in runs] == [(0, 20.0, "ru")]
        assert runs[0]["regions"] == regions

    def test_a_believable_opening_keeps_its_run(self):
        """A four-second Spanish scene before the Italian starts is a scene, wherever it sits."""
        regions = _regions((0, 4.0), (4.3, 20.0))
        labels = _labels((0, 4.0, "es", 0.9), (4.3, 20.0, "it", 0.9))
        assert [r["language"] for r in language_runs.build(regions, labels, "it")] == ["es", "it"]

    def test_a_region_the_detector_could_not_name_stays_in_its_run(self):
        """No label is not a different label; the run continues through it."""
        regions = _regions((0, 2.0), (2.3, 3.0), (3.2, 5.0))
        labels = _labels((0, 2.0, "de", 0.9), (3.2, 5.0, "de", 0.9))
        runs = language_runs.build(regions, labels)
        assert [(r["start"], r["end"], r["language"]) for r in runs] == [(0, 5.0, "de")]


class TestTheStressGrid:
    """Every utterance a different language, every one long enough: nothing may merge."""

    def test_every_utterance_keeps_its_own_run(self):
        """Four languages in a row, each over 3 s: four runs, no merging."""
        regions = _regions((0, 3.2), (3.6, 6.9), (7.3, 10.5), (11.0, 14.2))
        labels = _labels((0, 3.2, "en", 0.9), (3.6, 6.9, "de", 0.9), (7.3, 10.5, "fr", 0.9), (11.0, 14.2, "es", 0.9))
        assert [r["language"] for r in language_runs.build(regions, labels)] == ["en", "de", "fr", "es"]

    def test_the_two_halves_of_one_utterance_are_one_run(self):
        """Silero splits a pangram-plus-sentence recording in two; both halves are one language."""
        regions = _regions((0, 3.0), (3.4, 6.3), (6.8, 9.9))
        labels = _labels((0, 3.0, "en", 0.9), (3.4, 6.3, "en", 0.9), (6.8, 9.9, "de", 0.9))
        runs = language_runs.build(regions, labels)
        assert [(r["start"], r["end"], r["language"]) for r in runs] == [(0, 6.3, "en"), (6.8, 9.9, "de")]


class TestAShortCodeSwitchedClip:
    """The six code-switched fixtures: two legs of about two seconds, no film around them.

    Measured on the RTX 5090 (W3, 2026-09-11): with the bar in seconds alone, the second leg of
    `mix_de_en`, `mix_hi_en` and `mix_zh_en` was absorbed into the first and dropped -- the
    exact failure the clips exist to catch. Half of a five-second file is not a slip.
    """

    def test_a_two_second_leg_that_is_half_the_file_is_its_own_run(self):
        """English then Spanish, 2.2 s each: two runs, both kept."""
        regions = _regions((0, 2.2), (2.6, 4.8))
        labels = _labels((0, 2.2, "en", 0.95), (2.6, 4.8, "es", 0.95))
        assert [r["language"] for r in language_runs.build(regions, labels, "en")] == ["en", "es"]

    def test_the_share_rule_does_not_reach_a_film(self):
        """On ten minutes of labelled speech a fifth is two minutes; the 3 s bar decides."""
        regions = _regions((0, 20.0), (20.5, 22.5), (23.0, 40.0))
        labels = _labels((0, 20.0, "ru", 0.9), (20.5, 22.5, "uk", 0.95), (23.0, 40.0, "ru", 0.9))
        assert {r["language"] for r in language_runs.build(regions, labels, "ru")} == {"ru"}

    def test_an_unsure_leg_is_still_absorbed(self):
        """The share rule lowers the seconds, never the confidence a switch needs."""
        regions = _regions((0, 2.2), (2.6, 4.8))
        labels = _labels((0, 2.2, "en", 0.95), (2.6, 4.8, "es", 0.3))
        assert [r["language"] for r in language_runs.build(regions, labels, "en")] == ["en"]


class TestRunsCarryTheirRegions:
    """A run decides the language and hands the decoder its regions, one clip each.

    The run was once the decode clip. Measured on the film fixture with the language forced,
    a window holding a run's worth of lines lost 5 of 46 and region-sized windows lost none;
    and a run reaching across the passage between two scenes put that passage inside a
    window, the invented-text failure decoding by region had closed.
    """

    def test_a_run_lists_every_region_it_covers(self):
        """43 s of English across a 10 s pause: one run, four regions, nothing merged."""
        regions = _regions((0, 10), (11, 21), (31, 41), (42, 43))
        labels = _labels((0, 10, "en", 0.9), (11, 21, "en", 0.9), (31, 41, "en", 0.9), (42, 43, "en", 0.9))
        runs = language_runs.build(regions, labels)
        assert [(r["start"], r["end"], r["language"]) for r in runs] == [(0, 43, "en")]
        assert runs[0]["regions"] == regions

    def test_an_absorbed_region_is_carried_by_the_run_that_absorbed_it(self):
        """The Ukrainian-labelled line is decoded as Russian, as its own clip."""
        regions = _regions((0, 10.0), (10.3, 11.4), (11.7, 20.0))
        labels = _labels((0, 10.0, "ru", 0.9), (10.3, 11.4, "uk", 0.95), (11.7, 20.0, "ru", 0.9))
        runs = language_runs.build(regions, labels)
        assert [r["regions"] for r in runs] == [regions]


class TestWithoutLabels:
    """The detector failed outright: fall back to the pre-hysteresis behaviour, never to nothing."""

    def test_every_region_is_its_own_languageless_run(self):
        """The pre-hysteresis behaviour, which the decoder handles with its own detection."""
        regions = _regions((0, 2.0), (2.5, 4.0))
        runs = language_runs.build(regions, [])
        assert [(r["start"], r["end"], r["language"]) for r in runs] == [(0, 2.0, None), (2.5, 4.0, None)]

    def test_no_regions_means_no_runs(self):
        """Nothing in, nothing out."""
        assert language_runs.build([], []) == []

    def test_an_unlabelled_opening_takes_the_first_language_that_arrives(self):
        """The first run is never languageless when any label exists."""
        regions = _regions((0, 1.0), (1.2, 3.0), (3.2, 7.0))
        labels = _labels((1.2, 3.0, "pl", 0.9), (3.2, 7.0, "pl", 0.9))
        runs = language_runs.build(regions, labels)
        assert [(r["start"], r["end"], r["language"]) for r in runs] == [(0, 7.0, "pl")]
