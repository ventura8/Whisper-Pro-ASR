"""Whether the montage's per-offset votes agree, and what that decision costs.

`_votes_disagree` gates the expensive path. A full per-chunk language scan is a pass over
the whole file; the montage has already detected each of its sampled offsets independently,
so this reads that existing evidence to decide whether paying for the scan is warranted.
Getting it wrong is expensive in one direction and wrong in the other: too eager and every
monolingual request pays for a full scan, too strict and code-switched audio silently loses
a leg.
"""

# The unit under test is the module's internals; reaching them by name is the point.

from unittest import mock

import pytest

from modules.inference.pipeline import language_detection


def _vote(language: str, confidence: float = 0.9, **others: float) -> dict:
    """One offset's detection result, shaped as the montage scan produces it."""
    probabilities = {language: confidence, **others}
    return {"detected_language": language, "confidence": confidence, "all_probabilities": probabilities}


@pytest.fixture(autouse=True)
def _known_confidence_bar():
    """Pin LD_MIN_CONFIDENCE, so the threshold is the test's and not the environment's."""
    with mock.patch.object(language_detection.config, "LD_MIN_CONFIDENCE", 0.5):
        yield


class TestVotesDisagree:
    """The whole-file decision: did the sampled offsets name more than one language?"""

    def test_a_single_language_across_every_offset_agrees(self):
        """The overwhelmingly common case, which must never pay for a full-file scan."""
        assert language_detection._votes_disagree([_vote("en") for _ in range(5)]) is False

    def test_an_even_two_way_split_disagrees(self):
        """Two of four is half the evidence each way; neither can be called a misfire."""
        votes = [_vote("en"), _vote("en"), _vote("es"), _vote("es")]
        assert language_detection._votes_disagree(votes) is True

    def test_one_dissenter_in_nine_is_treated_as_noise(self):
        """1/9 is below the one-third credibility share, so it does not trigger a full scan.

        This is the case the share test exists for: a single misdetected offset in a long
        monolingual file must not cost a whole-file per-chunk scan.
        """
        votes = [_vote("en") for _ in range(8)] + [_vote("de")]
        assert language_detection._votes_disagree(votes) is False

    def test_one_dissenter_in_two_is_credible(self):
        """On a short clip the montage samples one to three offsets, so 1/2 is real evidence.

        Requiring two votes per language outright would be stricter and useless here: a clip
        under ten minutes is never sampled often enough for a second language to reach a
        second vote, which would make code-switching undetectable by construction.
        """
        assert language_detection._votes_disagree([_vote("en"), _vote("fr")]) is True

    def test_a_lone_vote_cannot_disagree_with_itself(self):
        """One offset is one opinion; disagreement needs at least two."""
        assert language_detection._votes_disagree([_vote("en")]) is False

    def test_low_confidence_votes_are_not_counted_at_all(self):
        """The same bar the vote itself uses; a guess below it is not evidence of anything."""
        votes = [_vote("en"), _vote("en"), _vote("ja", confidence=0.2), _vote("ko", confidence=0.1)]
        assert language_detection._votes_disagree(votes) is False

    def test_results_without_probabilities_are_skipped(self):
        """A failed offset yields no probabilities and must not be read as a language."""
        assert language_detection._votes_disagree([_vote("en"), {}, None, {"confidence": 0.9}]) is False


class TestTopLanguageOfVote:
    """Which language one offset actually named."""

    def test_the_highest_scoring_language_wins(self):
        """The offset's answer is its top probability, as the vote itself reports it."""
        assert language_detection._top_language_of_vote({"en": 0.7, "es": 0.2, "de": 0.1}) == "en"

    def test_bookkeeping_keys_cannot_be_returned_as_a_language(self):
        """The vote dict carries metadata alongside the probabilities.

        `_speech_duration` is a duration in seconds, so it dwarfs every probability and would
        win a naive max() outright -- being returned as the detected "language" for the
        offset. Any of the metadata keys would do the same; the filter is on the whole class
        of them rather than on that one name.
        """
        vote = {"_speech_duration": 30.0, "speech_ratio": 0.9, "speech_duration": 30.0, "en": 0.6, "es": 0.4}
        assert language_detection._top_language_of_vote(vote) == "en"

    def test_a_vote_with_no_real_language_yields_none(self):
        """Metadata only: there is no language here to count toward a disagreement."""
        assert language_detection._top_language_of_vote({"_speech_duration": 30.0}) is None


class TestVoteIsCredible:
    """The one-third share rule, at its boundary."""

    @pytest.mark.parametrize(("count", "total"), [(2, 9), (3, 9), (1, 2), (1, 3), (4, 4)])
    def test_credible_counts(self, count, total):
        """Two votes, or a single vote holding at least a third of the evidence."""
        assert language_detection._vote_is_credible(count, total) is True

    @pytest.mark.parametrize(("count", "total"), [(1, 4), (1, 9), (1, 100)])
    def test_lone_dissenters_below_the_share_are_not(self, count, total):
        """One offset out of four or more is likelier a misfire than a second language."""
        assert language_detection._vote_is_credible(count, total) is False

    def test_two_votes_are_always_credible_regardless_of_share(self):
        """A second independent offset naming the same language is not a coincidence."""
        assert language_detection._vote_is_credible(2, 100) is True


def test_the_disagreement_verdict_reaches_the_detection_result():
    """`multilingual_suspected` is what the caller reads; assert it through the real path.

    It only records the suspicion -- it does not itself run a scan. Pinning it here keeps the
    cheap montage-derived signal and the expensive decision it informs from drifting apart.
    """
    results = [_vote("en"), _vote("en"), _vote("es"), _vote("es")]
    manager = mock.MagicMock()
    manager.run_batch_language_detection_direct.return_value = results
    perf = {"dur_queue": 0.0, "dur_montage": 0.0, "dur_iso": 0.0}

    res = language_detection._step_run_inference((mock.MagicMock(), manager), "/tmp/montage.wav", 4, perf)

    assert res["multilingual_suspected"] is True
    assert res["detected_language"] in {"en", "es"}


def test_an_agreeing_scan_does_not_raise_the_suspicion():
    """The other side of the same path: agreement must leave the expensive scan unrequested."""
    results = [_vote("en") for _ in range(4)]
    manager = mock.MagicMock()
    manager.run_batch_language_detection_direct.return_value = results
    perf = {"dur_queue": 0.0, "dur_montage": 0.0, "dur_iso": 0.0}

    res = language_detection._step_run_inference((mock.MagicMock(), manager), "/tmp/montage.wav", 4, perf)

    assert res["multilingual_suspected"] is False
    assert res["detected_language"] == "en"
