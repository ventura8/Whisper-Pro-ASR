"""Turning onnx-asr results into the segments and words the pipeline consumes.

Parakeet's transducer reports a start time for every token, and nothing else: no word
boundaries, no durations. A token whose text begins with a space starts a word (the
SentencePiece word marker, which onnx-asr has already rendered as a space), so words are
rebuilt from that, and segments -- subtitle lines -- from the words.

Canary's encoder-decoder export reports no times at all. Its segments take the span of the
chunk they were decoded from, divided between sentences in proportion to their length. That
is an approximation and is documented as one: a chunk is at most NEMO_MAX_CHUNK_SEC, so a
line can be off by a few seconds inside it, never by more.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from modules.inference.engines.base import SegmentWrapper

#: The transducer's time step: 10 ms features subsampled 8x. Token times are multiples of it.
TOKEN_STEP_SEC = 0.08

#: How long a word's last token is assumed to last. The model reports where a token starts,
#: not where it ends; three steps is a typical sub-word's length in the timings measured on
#: the accuracy fixture ("The" 0.00 -> "qu" 0.24). It decides only how early a line ends
#: before a pause, and never moves an end past the next word's start.
WORD_TAIL_SEC = 3 * TOKEN_STEP_SEC

#: Subtitle-line limits. Seven seconds and two 42-character lines are the common broadcast
#: ceilings; a pause longer than GAP_SEC is a new line whatever the text says.
MAX_SEGMENT_SEC = 7.0
MAX_SEGMENT_CHARS = 84
GAP_SEC = 0.8
#: A sentence ending closes a line only once the line has this much in it, so "Yes. No." is
#: not two flashes on screen.
MIN_SENTENCE_LINE_SEC = 1.0

_SENTENCE_END = (".", "?", "!", "…", "。", "？", "！")
_SENTENCE_SPLIT = re.compile(r"(?<=[.?!…])\s+")


@dataclass
class NemoWord:
    """One word, shaped like faster-whisper's: text keeps its leading space."""

    start: float
    end: float
    word: str
    probability: float


def words_from_result(result: Any, offset: float, clip_end: float) -> list[NemoWord]:
    """The words of one timestamped result, on the file's timeline.

    ``offset`` is where the chunk starts in the file; ``clip_end`` caps the last word, whose
    end would otherwise be a guess with nothing after it to bound it.
    """
    tokens, times = _field(result, "tokens"), _field(result, "timestamps")
    if not tokens or len(times) != len(tokens):
        return []
    timeline = [offset + seconds for seconds in times]
    groups = _word_groups(tokens)
    logprobs = _field(result, "logprobs")
    return [_word(tokens, timeline, logprobs, group, bound) for group, bound in zip(groups, _bounds(groups, timeline, clip_end))]


def _bounds(groups: list[list[int]], timeline: list[float], clip_end: float) -> list[float]:
    """The latest each word may end: where the next one starts, and the clip's end for the last."""
    return [*(min(timeline[group[0]], clip_end) for group in groups[1:]), clip_end]


def _word(tokens: list[str], timeline: list[float], logprobs: list[float], group: list[int], bound: float) -> NemoWord:
    """One word from its token indices: its first token's start, its last token's assumed end."""
    start = timeline[group[0]]
    end = min(timeline[group[-1]] + WORD_TAIL_SEC, bound)
    text = "".join(tokens[index] for index in group)
    return NemoWord(round(start, 3), round(max(end, start), 3), text, _probability(logprobs, group))


def _field(result: Any, name: str) -> list:
    """One of a result's per-token lists, or an empty list where the model gave none."""
    return list(getattr(result, name, None) or [])


def _word_groups(tokens: list[str]) -> list[list[int]]:
    """Token indices grouped into words: a leading space opens a new one."""
    groups: list[list[int]] = []
    for index, token in enumerate(tokens):
        if not groups or token.startswith(" "):
            groups.append([index])
        else:
            groups[-1].append(index)
    return groups


def _probability(logprobs: list[float], group: list[int]) -> float:
    """The word's mean token probability, or 1.0 when the model reported none."""
    if len(logprobs) <= group[-1]:
        return 1.0
    return round(math.exp(sum(logprobs[i] for i in group) / len(group)), 4)


def lines_from_words(words: list[NemoWord]) -> list[list[NemoWord]]:
    """Words grouped into subtitle lines; see the limits above."""
    lines: list[list[NemoWord]] = []
    for word in words:
        if lines and not _breaks_before(lines[-1], word):
            lines[-1].append(word)
        else:
            lines.append([word])
    return lines


def _breaks_before(line: list[NemoWord], word: NemoWord) -> bool:
    """Whether ``word`` starts a new line rather than joining ``line``."""
    last = line[-1]
    if word.start - last.end > GAP_SEC:
        return True
    if word.end - line[0].start > MAX_SEGMENT_SEC:
        return True
    if len(_text(line)) + len(word.word) > MAX_SEGMENT_CHARS:
        return True
    return last.word.rstrip().endswith(_SENTENCE_END) and last.end - line[0].start >= MIN_SENTENCE_LINE_SEC


def _text(words: list[NemoWord]) -> str:
    return "".join(word.word for word in words)


def segments_from_words(words: list[NemoWord], *, keep_words: bool) -> list[SegmentWrapper]:
    """One segment per subtitle line, carrying its words when the caller asked for them."""
    return [
        SegmentWrapper(start=line[0].start, end=line[-1].end, text=_text(line), words=line if keep_words else None)
        for line in lines_from_words(words)
    ]


def segments_from_text(text: str, start: float, end: float) -> list[SegmentWrapper]:
    """Untimed text over ``[start, end]``: one line per sentence, timed by character share.

    A chunk short enough to be one line stays one segment. Otherwise its sentences divide the
    span in proportion to their length -- speech rate is roughly constant within a chunk,
    which is what makes the share a usable estimate.
    """
    sentences = _sentences(text, end - start)
    total = sum(len(sentence) for sentence in sentences)
    segments = []
    cursor = start
    for sentence in sentences:
        stop = cursor + (end - start) * len(sentence) / total
        segments.append(SegmentWrapper(start=round(cursor, 3), end=round(stop, 3), text=f" {sentence}"))
        cursor = stop
    return segments


def _sentences(text: str, span_sec: float) -> list[str]:
    """The lines an untimed chunk becomes: none when empty, one when short, else one per sentence."""
    clean = str(text or "").strip()
    if not clean:
        return []
    if span_sec <= MAX_SEGMENT_SEC:
        return [clean]
    return _sentence_lines(clean)


def _sentence_lines(text: str) -> list[str]:
    """One line per sentence of ``text``, each wrapped to MAX_SEGMENT_CHARS."""
    lines: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(text):
        lines.extend(_wrapped(sentence))
    return lines


def _wrapped(sentence: str) -> list[str]:
    """``sentence`` cut at word boundaries into lines of at most MAX_SEGMENT_CHARS.

    A long unpunctuated sentence would otherwise be one line spanning the whole chunk (up to
    NEMO_MAX_CHUNK_SEC); the Parakeet path caps the same lines by characters, so this does too.
    A single word longer than the limit stays whole rather than being cut mid-word.
    """
    lines: list[str] = []
    for word in sentence.split():
        if lines and len(lines[-1]) + 1 + len(word) <= MAX_SEGMENT_CHARS:
            lines[-1] = f"{lines[-1]} {word}"
        else:
            lines.append(word)
    return lines


def to_segments(result: Any, span: tuple, *, timed: bool, keep_words: bool) -> list[SegmentWrapper]:
    """The segments of one chunk's result. ``timed`` is whether the model reported token times."""
    start, end = span
    if timed:
        words = words_from_result(result, start, end)
        if words:
            return segments_from_words(words, keep_words=keep_words)
    return segments_from_text(getattr(result, "text", result if isinstance(result, str) else ""), start, end)
