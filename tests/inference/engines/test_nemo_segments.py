"""onnx-asr results into segments and words (modules/inference/engines/nemo_segments.py)."""

from types import SimpleNamespace

import pytest

from modules.inference.engines import nemo_segments
from modules.inference.engines.nemo_segments import NemoWord
from tests.inference.engines.nemo_fixtures import FOX, timed_result


def test_words_are_rebuilt_from_space_prefixed_tokens_on_the_files_timeline():
    """Tokens become words at their leading space, offset to the file."""
    words = nemo_segments.words_from_result(FOX, offset=10.0, clip_end=12.0)
    assert [w.word for w in words] == [" The", " quick", " brown", " fox."]
    assert [w.start for w in words] == [10.0, 10.24, 10.64, 10.96]
    # A word ends at its last token plus the assumed tail, never past the next word.
    assert words[0].end == 10.24
    assert words[1].end == 10.64
    # The last word is bounded by the clip, here by the tail after its "." token.
    assert words[-1].end == pytest.approx(10.0 + 1.2 + nemo_segments.WORD_TAIL_SEC)


def test_word_probability_is_the_mean_token_probability():
    """A word's probability is its tokens' mean probability."""
    words = nemo_segments.words_from_result(FOX, offset=0.0, clip_end=2.0)
    assert words[0].probability == pytest.approx(0.99, abs=0.001)
    assert 0.0 < words[1].probability < 1.0


def test_missing_logprobs_mean_full_confidence():
    """No logprobs reads as full confidence."""
    result = timed_result("Hi", [" Hi"], [0.0], None)
    assert nemo_segments.words_from_result(result, 0.0, 1.0)[0].probability == 1.0


@pytest.mark.parametrize(
    "result",
    [timed_result("", [], []), timed_result("x", [" x"], []), SimpleNamespace(text="x")],
)
def test_untimed_or_inconsistent_results_have_no_words(result):
    """No tokens, or tokens without times, yield no words."""
    assert nemo_segments.words_from_result(result, 0.0, 1.0) == []


def test_last_word_end_is_capped_by_the_clip():
    """The last word never ends past its clip."""
    result = timed_result("a", [" a"], [0.96])
    assert nemo_segments.words_from_result(result, 0.0, 1.0)[0].end == 1.0


def _w(start, end, text):
    """A word with full confidence."""
    return NemoWord(start, end, text, 1.0)


def test_a_long_pause_starts_a_new_line():
    """A pause longer than GAP_SEC is a line break."""
    lines = nemo_segments.lines_from_words([_w(0, 0.5, " one"), _w(0.6, 1.0, " two"), _w(2.0, 2.4, " three")])
    assert [[w.word for w in line] for line in lines] == [[" one", " two"], [" three"]]


def test_a_line_never_exceeds_the_duration_limit():
    """No subtitle line is longer than MAX_SEGMENT_SEC."""
    words = [_w(i, i + 0.9, f" w{i}") for i in range(10)]
    lines = nemo_segments.lines_from_words(words)
    assert all(line[-1].end - line[0].start <= nemo_segments.MAX_SEGMENT_SEC for line in lines)


def test_a_line_never_exceeds_the_character_limit():
    """No subtitle line is longer than MAX_SEGMENT_CHARS."""
    words = [_w(i * 0.1, i * 0.1 + 0.1, " abcdefghij") for i in range(12)]
    lines = nemo_segments.lines_from_words(words)
    assert all(len("".join(w.word for w in line)) <= nemo_segments.MAX_SEGMENT_CHARS for line in lines)


def test_a_sentence_end_closes_a_line_once_it_has_some_length():
    """A full stop ends a line, but not a one-word flash."""
    words = [_w(0, 0.6, " Hello"), _w(0.6, 1.2, " there."), _w(1.2, 1.6, " Next")]
    assert len(nemo_segments.lines_from_words(words)) == 2
    short = [_w(0, 0.3, " Yes."), _w(0.3, 0.6, " No.")]
    assert len(nemo_segments.lines_from_words(short)) == 1


def test_segments_carry_words_only_when_asked():
    """Words are attached only for word_timestamps."""
    words = nemo_segments.words_from_result(FOX, 0.0, 2.0)
    with_words = nemo_segments.segments_from_words(words, keep_words=True)
    without = nemo_segments.segments_from_words(words, keep_words=False)
    assert with_words[0].words and without[0].words is None
    assert with_words[0].text == " The quick brown fox."


def test_untimed_text_is_one_segment_when_the_chunk_is_short():
    """A short Canary chunk is one line spanning the chunk."""
    segments = nemo_segments.segments_from_text("Ein Satz. Noch einer.", 3.0, 8.0)
    assert [(s.start, s.end, s.text) for s in segments] == [(3.0, 8.0, " Ein Satz. Noch einer.")]


def test_untimed_text_splits_a_long_chunk_by_sentence_share():
    """A long Canary chunk is divided between its sentences by length."""
    segments = nemo_segments.segments_from_text("Short one. A considerably longer second sentence.", 0.0, 20.0)
    assert len(segments) == 2
    assert segments[0].start == 0.0 and segments[-1].end == 20.0
    assert segments[0].end == segments[1].start
    assert segments[1].end - segments[1].start > segments[0].end - segments[0].start


def test_untimed_unpunctuated_speech_is_wrapped_at_word_boundaries():
    """A long sentence with no full stop is not one line spanning the whole chunk."""
    text = " ".join(["cuvant"] * 60)
    segments = nemo_segments.segments_from_text(text, 0.0, 18.0)
    assert len(segments) > 1
    assert all(len(segment.text.strip()) <= nemo_segments.MAX_SEGMENT_CHARS for segment in segments)
    assert " ".join(segment.text.strip() for segment in segments) == text
    assert segments[-1].end == 18.0


def test_a_word_longer_than_a_line_is_kept_whole():
    """Wrapping never cuts inside a word."""
    word = "x" * (nemo_segments.MAX_SEGMENT_CHARS + 5)
    assert nemo_segments._wrapped(f"a {word} b") == ["a", word, "b"]


def test_empty_text_yields_nothing():
    """Nothing said, nothing emitted."""
    assert not nemo_segments.segments_from_text("  ", 0.0, 5.0)


def test_to_segments_uses_token_times_when_the_model_has_them():
    """Timed results are placed by their token times."""
    [segment] = nemo_segments.to_segments(FOX, (5.0, 7.0), timed=True, keep_words=False)
    assert segment.start == 5.0


def test_to_segments_falls_back_to_the_chunk_span_for_plain_text():
    """Canary: onnx-asr returns a plain string with no token times."""
    [segment] = nemo_segments.to_segments("Guten Tag.", (5.0, 7.0), timed=False, keep_words=True)
    assert (segment.start, segment.end, segment.text, segment.words) == (5.0, 7.0, " Guten Tag.", None)


def test_to_segments_with_times_but_no_tokens_still_returns_the_text():
    """A timed model that returned no tokens still keeps its text."""
    [segment] = nemo_segments.to_segments(timed_result("Hallo.", [], []), (1.0, 2.0), timed=True, keep_words=False)
    assert segment.text == " Hallo."
