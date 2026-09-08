"""The accuracy fixture's tolerance: a sentence within a word of itself, in order.

The integration module is skipped without a live service; its scorer is pinned here so the
bar it applies -- one substituted word, never a missing or invented sentence -- is checked
on every gate run.
"""

from tests.integration import test_transcription_accuracy as accuracy

FOX = "the quick brown fox jumps over the lazy dog"


def test_the_exact_sentence_is_zero_off():
    """The sentence as spoken costs nothing."""
    assert accuracy._words_off(FOX, FOX.split()) == 0


def test_a_substituted_article_is_one_off():
    """The measured case: "A quick brown fox" from the 3-second region window."""
    assert accuracy._words_off(FOX, "a quick brown fox jumps over the lazy dog whisper pro asr".split()) == 1


def test_the_sentence_is_found_anywhere_in_the_transcript():
    """The promo card and the second sentence around it do not count against it."""
    transcript = ("made with whisper " + FOX + " and more").split()
    assert accuracy._words_off(FOX, transcript) == 0


def test_a_dropped_word_is_one_off_and_a_missing_sentence_is_the_whole_sentence():
    """What the bar still catches: a sentence that is not there costs the sentence."""
    assert accuracy._words_off(FOX, "the quick brown fox jumps over lazy dog".split()) == 1
    assert accuracy._words_off(FOX, "whisper pro asr is running a hardware acceleration test".split()) >= 7
    assert accuracy._words_off(FOX, []) == len(FOX.split())


def test_set_overlap_would_have_passed_the_article_swap():
    """Why this is an ordered edit distance and not the matrix's set overlap: the swapped
    transcript still contains every expected word, "the" included."""
    swapped = "a quick brown fox jumps over the lazy dog".split()
    assert set(FOX.split()) <= set(swapped)
    assert accuracy._words_off(FOX, swapped) == 1
