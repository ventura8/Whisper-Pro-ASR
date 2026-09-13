"""The accuracy fixture's tolerance: a sentence within a word of itself, in order.

The integration module is skipped without a live service; its scorer is pinned here so the
bar it applies -- one substituted word, never a missing or invented sentence -- is checked
on every gate run.
"""

import pytest

from tests.integration import test_transcription_accuracy as accuracy

FOX = "the quick brown fox jumps over the lazy dog"


def test_the_exact_sentence_is_zero_off():
    """The sentence as spoken costs nothing."""
    assert accuracy._words_off(FOX, FOX.split())[0] == 0


def test_a_substituted_article_is_one_off():
    """The measured case: "A quick brown fox" from the 3-second region window."""
    assert accuracy._words_off(FOX, "a quick brown fox jumps over the lazy dog whisper pro asr".split())[0] == 1


def test_the_sentence_is_found_anywhere_in_the_transcript():
    """The promo card and the second sentence around it do not count against it."""
    transcript = ("made with whisper " + FOX + " and more").split()
    assert accuracy._words_off(FOX, transcript) == (0, 3 + len(FOX.split())), "found, and its end reported"


def test_a_dropped_word_is_one_off_and_a_missing_sentence_is_the_whole_sentence():
    """What the bar still catches: a sentence that is not there costs the sentence."""
    assert accuracy._words_off(FOX, "the quick brown fox jumps over lazy dog".split())[0] == 1
    assert accuracy._words_off(FOX, "whisper pro asr is running a hardware acceleration test".split())[0] >= 7
    assert accuracy._words_off(FOX, [])[0] == len(FOX.split())


def test_set_overlap_would_have_passed_the_article_swap():
    """Why this is an ordered edit distance and not the matrix's set overlap: the swapped
    transcript still contains every expected word, "the" included."""
    swapped = "a quick brown fox jumps over the lazy dog".split()
    assert set(FOX.split()) <= set(swapped)
    assert accuracy._words_off(FOX, swapped)[0] == 1


def _payload(text: str) -> dict:
    return {"segments": [{"start": 0.0, "end": 8.0, "text": text}]}


def test_the_sentences_must_come_in_order():
    """Both sentences present but swapped is not the fixture: the second is searched only
    past where the first was found."""
    fox, whisper = accuracy.EXPECTED_PHRASES
    accuracy._assert_says(accuracy.EXPECTED_PHRASES, _payload(f"{fox}. {whisper}."))
    with pytest.raises(AssertionError, match="after word"):
        accuracy._assert_says(accuracy.EXPECTED_PHRASES, _payload(f"{whisper}. {fox}."))


def test_a_swapped_article_still_passes_in_order():
    """The measured case, through the same path the live suite uses."""
    _, whisper = accuracy.EXPECTED_PHRASES
    accuracy._assert_says(accuracy.EXPECTED_PHRASES, _payload(f"A quick brown fox jumps over the lazy dog. {whisper}."))
