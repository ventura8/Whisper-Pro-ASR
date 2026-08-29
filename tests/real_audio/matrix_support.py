"""Shared helpers for the real-audio matrix tests.

Every module under ``tests/real_audio`` imports its text normalization, transcript
extraction and parametrization from here rather than redefining it. That is deliberate:
pylint's ``duplicate-code`` check runs with an empty ``disable=`` list and inline
suppressions are banned repo-wide, so the only way seven similarly-shaped test modules
coexist is for the shared shape to live in exactly one place.

``tests/integration/test_transcription_accuracy.py`` also imports from here, so the
promo-subtitle-card stripping rule has a single definition across the whole suite.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

import pytest

from scripts.audio_matrix.render import probe_duration

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "e2e" / "fixtures"
MATRIX_DIR = FIXTURES_DIR / "audio_matrix"
MANIFEST_PATH = MATRIX_DIR / "manifest.json"

# The service may prepend a promotional subtitle card to generated output. It is not
# spoken audio, so every content assertion strips it before comparing.
PROMO_TEXT = "Made with Whisper Pro ASR"

# When a response carries no segments, its flat `text` field is an SRT document, not a
# plain transcript: cue numbers, timestamp lines, and bracketed markers such as
# "[No dialogue detected]". Feeding that into a content assertion makes silence look like
# a hundred characters of hallucinated speech, so the scaffolding is stripped first.
SRT_TIMESTAMP = re.compile(r"\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->")
BRACKETED_MARKER = re.compile(r"\[[^\]]*\]")

# Every module in this package drives a live service, so they share one gate. Defining it
# once also keeps pylint's duplicate-code check quiet without an inline suppression, which
# the repository forbids.
REAL_ASR_MARKS = [
    pytest.mark.real_asr,
    pytest.mark.real_audio,
    pytest.mark.skipif(
        os.environ.get("RUN_REAL_ASR") != "1",
        reason="Real-engine ASR test; start the stack and set RUN_REAL_ASR=1 to run.",
    ),
]


def normalize(text: str) -> str:
    """Lowercase and collapse everything that is not a letter, digit or space.

    Unicode letters are preserved (``\\w`` with the default ``re.UNICODE`` flag), so this
    works for Cyrillic, Greek, CJK and Indic scripts rather than only ASCII.
    """
    return re.sub(r"[^\w ]+", " ", text.lower(), flags=re.UNICODE).strip()


def words(text: str) -> list[str]:
    """Return the normalized whitespace-separated tokens of ``text``."""
    return normalize(text).split()


def _is_scaffolding(line: str) -> bool:
    """Return whether a line is SRT structure rather than spoken content."""
    return not line or line.isdigit() or bool(SRT_TIMESTAMP.search(line))


def strip_subtitle_markup(text: str) -> str:
    """Return only the spoken content of a possibly SRT-formatted transcript."""
    spoken = [line.strip() for line in text.splitlines() if not _is_scaffolding(line.strip())]
    return BRACKETED_MARKER.sub(" ", " ".join(spoken))


def spoken_words(payload: dict) -> list[str]:
    """Return the transcript tokens from a response, ignoring the promo subtitle card.

    Segment text is preferred over the flat ``text`` field because segments are what the
    timing assertions use, and a mismatch between the two is itself worth surfacing.
    """
    segments = payload.get("segments") or []
    if segments:
        spoken = " ".join(str(seg.get("text", "")) for seg in segments)
    else:
        spoken = str(payload.get("text", ""))
    tokens = words(strip_subtitle_markup(spoken))
    promo = words(PROMO_TEXT)
    promo_len = len(promo)
    if tokens[:promo_len] == promo:
        tokens = tokens[promo_len:]
    return tokens


def _char_bigrams(tokens: Iterable[str]) -> set[str]:
    """Return the set of adjacent character pairs across ``tokens``.

    Used for scripts that do not delimit words with spaces (CJK, Thai), where token
    overlap is meaningless but character-bigram overlap tracks accuracy well.
    """
    joined = "".join(tokens)
    return {first + second for first, second in zip(joined, joined[1:])}


def _word_overlap_tokens(expected: list[str], actual: list[str]) -> float:
    """Return the fraction of ``expected`` tokens that appear in ``actual``."""
    if not expected:
        return 1.0
    # Both sets built once. `token in set(actual)` inside the comprehension rebuilt the
    # actual-token set per expected token -- O(expected x actual) on every scored clip.
    actual_tokens = set(actual)
    expected_tokens = set(expected)
    found = sum(1 for token in expected_tokens if token in actual_tokens)
    return found / len(expected_tokens)


def _word_overlap_chars(expected: list[str], actual: list[str]) -> float:
    """Return the character-bigram overlap ratio, for space-free scripts."""
    expected_grams = _char_bigrams(expected)
    if not expected_grams:
        return 1.0
    return len(expected_grams & _char_bigrams(actual)) / len(expected_grams)


_OVERLAP_STRATEGIES = {"words": _word_overlap_tokens, "chars": _word_overlap_chars}


def word_overlap(expected: list[str], actual: list[str], tokenizer: str = "words") -> float:
    """Return an overlap ratio in ``[0, 1]`` between expected and actual transcript tokens.

    ``tokenizer`` comes from the manifest entry so the choice of comparison is data, not a
    branch in a test.
    """
    strategy = _OVERLAP_STRATEGIES.get(tokenizer, _word_overlap_tokens)
    return strategy([normalize(word) for word in expected], actual)


def clip_duration(path: Path) -> float:
    """Return a clip's duration in seconds.

    Reuses the generator's own ffprobe wrapper so the tests and the generator agree on how
    a duration is measured.
    """
    return probe_duration(path)


def detected_code(payload: dict) -> str:
    """Return the lowercased language code from a ``/detect-language`` payload.

    The endpoint has historically returned the code under three different keys; accept all
    of them so the tests assert on behaviour rather than on one key's spelling.
    """
    for key in ("language_code", "language", "detected_language"):
        value = str(payload.get(key) or "").strip().lower()
        if value:
            return value
    return ""


def _policy_empty(spoken: str) -> str:
    """Return an error when text was produced for input that should yield none."""
    return f"expected no transcript, got {spoken[:200]!r}" if len(spoken) >= 40 else ""


def _policy_non_empty(spoken: str) -> str:
    """Return an error when nothing was transcribed from real speech."""
    return "" if spoken else "expected a transcript, got nothing"


_TEXT_POLICIES = {"empty_or_absent": _policy_empty, "non_empty": _policy_non_empty, "any": lambda _spoken: ""}


def assert_text_policy(case: dict, payload: dict) -> None:
    """Assert a response's transcript matches the text policy the manifest declares."""
    policy = case["expect"].get("text_policy", "any")
    spoken = " ".join(spoken_words(payload)).strip()
    problem = _TEXT_POLICIES[policy](spoken)
    assert not problem, f"{case['id']} ({policy}): {problem}"


def assert_declared_response(case: dict, response) -> None:
    """Assert a response matches the status and text contract its manifest entry declares."""
    expect = case["expect"]
    assert response.status_code in expect["status_in"], f"{case['id']}: {response.status_code}: {response.text[:400]}"
    if response.status_code == 200:
        assert_text_policy(case, response.json())


def load_manifest() -> dict[str, Any]:
    """Load the audio-matrix manifest, or an empty manifest when it is absent.

    Returning an empty manifest rather than raising keeps collection working on a checkout
    that predates the fixture set; the affected tests simply parametrize to nothing.
    """
    if not MANIFEST_PATH.exists():
        return {"clips": [], "combined": [], "adversarial": [], "longform": {}}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _entry_marks(entry: dict, scope: str | None) -> list[pytest.MarkDecorator]:
    """Return the pytest marks an entry's data asks for, for this concern.

    ``smoke`` selects the representative subset that the pipeline runs by default; the
    full matrix takes about two hours and is opt-in stress testing.

    A known defect usually breaks one property of a clip, not every assertion about it --
    a mixed-language file whose second half is dropped still detects a language and still
    returns HTTP 200. ``xfail_scope`` names the concern the defect affects, so the other
    tests stay strict instead of reporting a wall of meaningless XPASS.
    """
    marks = [pytest.mark.smoke] if entry.get("smoke") else []
    reason = entry.get("xfail_reason")
    if not reason or entry.get("xfail_scope") not in (None, scope):
        return marks
    return marks + [pytest.mark.xfail(strict=False, reason=str(reason))]


def _entry_is_selected(entry: dict, tier: str | None) -> bool:
    """Return whether an entry should be parametrized for the requested tier."""
    if entry.get("voice", "") is None:
        return False
    return tier is None or entry.get("tier") == tier


def clip_params(section: str, tier: str | None = None, scope: str | None = None) -> list[Any]:
    """Return ``pytest.param`` entries for a manifest section.

    Test ids come from the manifest ``id`` field, so node ids read
    ``test_tier_a_clip_transcribes[es_tier_a]`` and stay stable when the manifest is
    reordered. Entries with no available voice are dropped here rather than skipped at
    runtime, keeping the report free of noise about languages nothing can synthesize.
    """
    entries = load_manifest().get(section) or []
    selected = [entry for entry in entries if _entry_is_selected(entry, tier)]
    return [pytest.param(entry, id=str(entry["id"]), marks=_entry_marks(entry, scope)) for entry in selected]
