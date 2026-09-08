"""How the long-form test tells a decoder loop from speech that repeats.

The judgement is over sentences *in a row*, and a segment is not a sentence: faster-whisper
can put a whole loop inside one segment, with or without punctuation between the copies.
These pin the unit the test counts, without a GPU or a clip.
"""

from tests.real_audio import test_longform_stress as longform


class TestSentencesInASegment:
    """One segment can hold several sentences, or one sentence several times."""

    def test_a_plain_sentence_is_one_unit(self):
        """The common case: one segment, one sentence, normalised."""
        assert longform._sentences("Tell me the truth about it.") == ["tell me the truth about it"]

    def test_punctuated_sentences_are_split_before_the_marks_are_stripped(self):
        """normalize() removes the full stops, so the split has to happen first."""
        units = longform._sentences("Tell me the truth about it. Tell me the truth about it. Tell me the truth about it.")
        assert units == ["tell me the truth about it"] * 3

    def test_a_phrase_looped_without_punctuation_is_counted_once_per_copy(self):
        """The loop a segment boundary never interrupts: one segment, one phrase, many times."""
        units = longform._sentences("thank you thank you thank you thank you thank you thank you")
        assert units == ["thank you"] * 6

    def test_a_short_sentence_on_its_own_is_not_judged(self):
        """A lone "yes" is speech, and repeats in speech; only a looped copy of it counts."""
        assert longform._sentences("Yes.") == []
        assert longform._sentences("Yes. Yes.") == ["yes", "yes"]

    def test_a_looped_short_unit_is_kept_even_when_speech_follows_it(self):
        """A loop the decoder recovered from: "yes" more times in a row than any speech
        contains, then a real sentence. The short units used to be dropped because the
        segment was not *only* copies, and the loop went unjudged."""
        text = " ".join(["Yes."] * (longform.MAX_SENTENCE_REPEATS + 1)) + " Tell me the truth about it."
        units = longform._sentences(text)
        assert units == ["yes"] * (longform.MAX_SENTENCE_REPEATS + 1) + ["tell me the truth about it"]
        assert longform._longest_run(units)[1] > longform.MAX_SENTENCE_REPEATS

    def test_a_few_short_repeats_beside_speech_are_still_speech(self):
        """Two yeses before a sentence is how people talk; the short units drop as before."""
        assert longform._sentences("Yes. Yes. Tell me the truth about it.") == ["tell me the truth about it"]

    def test_trailing_whitespace_after_the_last_stop_adds_no_empty_unit(self):
        """A transcript segment often ends "Yes. " -- the split's empty tail must not count."""
        assert longform._sentences("Yes. Yes. ") == ["yes", "yes"]
        assert longform._sentences("Tell me the truth about it. ") == ["tell me the truth about it"]

    def test_a_sentence_that_merely_contains_a_repeated_word_is_not_a_loop(self):
        """Only a phrase the whole sentence is copies of counts as copies."""
        assert longform._sentences("It was very, very, very late.") == ["it was very very very late"]


class TestTheLoopIsCaughtInsideOneSegment:
    """The detector-loop assertion, end to end on the units."""

    def test_six_copies_in_one_segment_exceed_a_budget_of_five(self):
        """The loop the old segment-level count could not see."""
        text = " ".join(["Tell me the truth about it."] * 6)
        sentence, repeats = longform._longest_run(longform._sentences(text))
        assert (sentence, repeats) == ("tell me the truth about it", 6)
        assert repeats > longform.MAX_SENTENCE_REPEATS

    def test_the_same_copies_spread_over_segments_count_the_same(self):
        """Segment boundaries neither hide a loop nor invent one."""
        segments = [{"text": "Tell me the truth about it."}] * 6
        units = [unit for seg in segments for unit in longform._sentences(seg["text"])]
        assert longform._longest_run(units)[1] == 6
