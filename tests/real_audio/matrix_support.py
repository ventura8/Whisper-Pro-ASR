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

import functools
import json
import os
import re
import unicodedata
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from scripts.audio_matrix.render import probe_duration
from tests.real_audio import service_client

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

    Unicode *letters* are preserved (``\\w`` with the default ``re.UNICODE`` flag), so this
    is right for Cyrillic, Greek and CJK.

    **It is not safe for Indic scripts.** Combining marks -- vowel signs, the virama, the
    nukta -- are Unicode category Mn, which ``\\w`` does not match, so they are replaced by
    spaces and words are split into fragments:

        விரைவான -> "வ ர வ ன"      (Tamil, one word becomes four)
        ਤੇਜ਼      -> "ਤ ਜ"          (Punjabi)
        કથ્થઈ    -> "કથ થઈ"        (Gujarati)

    Token overlap against fragments like that measures nothing. Every Indic entry in the
    manifest therefore sets ``"tokenizer": "chars"``, which scores character bigrams over
    the same normalized text -- the fragmentation is identical on both sides, so it
    cancels. An Indic entry left on the default ``words`` tokenizer is a scoring bug, not a
    stylistic choice.

    Russian ``ё`` is folded to ``е``: written Russian drops the diaeresis almost everywhere
    and so does Whisper, so a reference spelt "чём" scored a correct "чем" as a miss
    (`ru_line2` at 0.50 against 0.60 on the RTX 5090). The two are one letter in practice.
    """
    return re.sub(r"[^\w ]+", " ", text.lower().replace("ё", "е"), flags=re.UNICODE).strip()


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


def _fold_marks(word: str) -> str:
    """Drop combining marks, leaving the base letters."""
    return "".join(c for c in unicodedata.normalize("NFD", word) if unicodedata.category(c) != "Mn")


def _word_overlap_chars_folded(expected: list[str], actual: list[str]) -> float:
    """Character-bigram overlap after folding away combining marks.

    For languages the model transcribes phonetically but not orthographically. Whisper writes
    Yoruba without its tone and vowel marks and places word breaks differently -- "kolo-kolo"
    for "kọ̀lọ̀kọ̀lọ̀", "ala wobura nulera" for "aláwọ̀ búráùnù yára" -- so word overlap scores
    a recognisably correct transcript at 0.00 and reports a total failure that did not happen.
    Measured on that clip: words 0.00, chars 0.09, words folded 0.20, chars folded 0.76.

    This is a laxer bar than the other strategies and is meant to be: it asks whether the right
    sounds were heard, not whether the orthography matches. Use it only where the model's
    spelling of a language differs systematically from the reference, never to rescue a clip
    that is simply being transcribed wrongly -- unrelated text still scores low.
    """
    return _word_overlap_chars([_fold_marks(w) for w in expected], [_fold_marks(w) for w in actual])


def overlap_ceiling(expected: list[str], tokenizer: str = "words") -> float:
    """The score a flawless transcript of ``expected`` can reach under ``tokenizer``.

    ``normalize`` replaces Unicode combining marks with spaces, so a reference word that
    carries them -- every Yoruba tone mark, every Indic vowel sign -- is split into fragments
    before it is compared. A transcript is tokenized by ``words`` first and never sees those
    fragments joined, so identical text still scores below 1.0: 0.71 for Yoruba under
    ``words``, ~0.90 under ``chars``. That cap is a property of the reference, not of the
    audio, which makes a single ``min_word_overlap`` mean different things in different
    scripts. Scoring against ``bar * ceiling`` instead asks the same question everywhere:
    what fraction of the attainable score did the transcript reach.

    Built exactly as a real transcript is scored -- the reference joined, tokenized by
    ``words``, and compared to itself -- so it moves with the scoring code, never with a
    hand-maintained table.
    """
    return word_overlap(expected, words(" ".join(expected)), tokenizer)


def content_threshold(clip: dict) -> float:
    """The bar a clip's transcript must clear: its declared ``min_word_overlap``, as a
    fraction of the ceiling its own reference allows."""
    tokenizer = clip.get("tokenizer", "words")
    return float(clip["min_word_overlap"]) * overlap_ceiling(clip["expect_words"], tokenizer)


_OVERLAP_STRATEGIES = {
    "words": _word_overlap_tokens,
    "chars": _word_overlap_chars,
    "chars_folded": _word_overlap_chars_folded,
}


def word_overlap(expected: list[str], actual: list[str], tokenizer: str = "words") -> float:
    """Return an overlap ratio in ``[0, 1]`` between expected and actual transcript tokens.

    ``tokenizer`` comes from the manifest entry so the choice of comparison is data, not a
    branch in a test.
    """
    if tokenizer not in _OVERLAP_STRATEGIES:
        # Same manifest-drift failure assert_text_policy raises for an unknown text_policy.
        # Falling back to the word tokenizer scored an Indic or CJK entry with a comparison
        # that cannot match, so a typo'd tokenizer read as a failing language rather than as
        # the manifest error it is.
        raise AssertionError(f"manifest declares tokenizer {tokenizer!r}, which is not one of {sorted(_OVERLAP_STRATEGIES)}")
    return _OVERLAP_STRATEGIES[tokenizer]([normalize(word) for word in expected], actual)


def text_in_window(segments: list[dict], start: float, end: float) -> str:
    """The text of every segment whose midpoint falls inside ``[start, end)``.

    The one rule for attributing text to a stretch of audio, shared by the long-form windows
    and the code-switched legs: a segment is placed by where it sits, not by what it touches.
    """
    inside = [seg for seg in segments if start <= (float(seg["start"]) + float(seg["end"])) / 2 < end]
    return " ".join(str(seg.get("text", "")) for seg in inside)


def foreign_share(segments: list[dict], windows: list[dict]) -> float:
    """The share of spoken seconds whose transcript carries a language other than the one spoken.

    Duration-weighted, so finer segmentation cannot inflate it -- the mistake that once read a
    real-media improvement as a 49% regression -- and judged per ground-truth window against
    the language the sidecar says was spoken there. A segment that reports no language is
    counted as wrong: the reporting path is part of what is measured. No speech at all
    scores 0.0, since nothing was mislabelled.
    """
    spoken = wrong = 0.0
    for window in windows:
        for seg in segments:
            seconds, same = _judged(seg, window)
            spoken += seconds
            wrong += 0.0 if same else seconds
    return wrong / spoken if spoken else 0.0


def _judged(seg: dict, window: dict) -> tuple[float, bool]:
    """The seconds a segment shares with a ground-truth window, and whether the language matches."""
    shared = min(float(seg["end"]), float(window["end"])) - max(float(seg["start"]), float(window["start"]))
    return max(0.0, shared), str(seg.get("language") or "").lower() == str(window["language"]).lower()


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


#: Transcript characters tolerated from input that should produce none. Not zero: a decoder
#: handed silence or a malformed file routinely emits a short artefact -- a stray "you", a
#: bracketed marker the SRT stripper missed -- and failing on those would make the
#: empty_or_absent policy assert on decoder noise rather than on the behaviour it is about,
#: which is that the service does not invent *speech*. Forty characters is roughly one short
#: sentence; anything longer is a hallucinated utterance.
EMPTY_TRANSCRIPT_TOLERANCE_CHARS = 40


def _policy_empty(spoken: str) -> str:
    """Return an error when text was produced for input that should yield none."""
    if len(spoken) <= EMPTY_TRANSCRIPT_TOLERANCE_CHARS:
        return ""
    return f"expected no transcript, got {spoken[:200]!r}"


def _policy_non_empty(spoken: str) -> str:
    """Return an error when nothing was transcribed from real speech."""
    return "" if spoken else "expected a transcript, got nothing"


_TEXT_POLICIES = {"empty_or_absent": _policy_empty, "non_empty": _policy_non_empty, "any": lambda _spoken: ""}


def assert_text_policy(case: dict, payload: dict) -> None:
    """Assert a response's transcript matches the text policy the manifest declares."""
    policy = case["expect"].get("text_policy", "any")
    if policy not in _TEXT_POLICIES:
        # Same manifest-drift failure params_for_ids raises: an unknown policy is a typo or a
        # renamed constant, and indexing straight into the table turned that into a bare
        # KeyError naming only the string -- not the entry, and not that the manifest is the
        # thing to fix.
        raise AssertionError(f"{case['id']}: manifest declares text_policy {policy!r}, which is not one of {sorted(_TEXT_POLICIES)}")
    spoken = " ".join(spoken_words(payload)).strip()
    problem = _TEXT_POLICIES[policy](spoken)
    assert not problem, f"{case['id']} ({policy}): {problem}"


def assert_declared_response(case: dict, response) -> None:
    """Assert a response matches the status and text contract its manifest entry declares."""
    expect = case["expect"]
    assert response.status_code in expect["status_in"], f"{case['id']}: {response.status_code}: {response.text[:400]}"
    if response.status_code == 200:
        assert_text_policy(case, response.json())


def english_words_for(clip_id: str) -> list[str] | None:
    """The English meaning of a matrix clip, as scoring tokens, or None when it has none.

    The matrix is built from one sentence pool rendered in every language -- ``ru_line4``
    says what ``en_line4`` says, every ``*_core`` is the same pangram and sign-off -- so a
    clip's English counterpart is the clip with the same suffix under ``en_``. That is what
    a translation of the clip is scored against; a clip with no English sibling cannot be.
    """
    counterpart = clip_id if clip_id.startswith("en_") else "en_" + clip_id.split("_", 1)[-1]
    entry = _clips_by_id().get(counterpart)
    return words(entry["text"]) if entry else None


@functools.lru_cache(maxsize=1)
def _clips_by_id() -> dict[str, dict]:
    """``clip id -> entry``, built once: a long-form translation asks for it per window."""
    return {clip["id"]: clip for clip in load_manifest().get("clips") or []}


@functools.lru_cache(maxsize=1)
def _clip_ids_by_line() -> dict[tuple[str, str], str]:
    """``(language, text) -> clip id``, to find the clip a long-form window was cut from."""
    return {(clip["language"], clip["text"]): clip["id"] for clip in load_manifest().get("clips") or []}


def window_english_words(window: dict) -> list[str] | None:
    """What a long-form window's line means in English, or None when the clip is unknown."""
    clip_id = _clip_ids_by_line().get((window["language"], window["text"]))
    return english_words_for(clip_id) if clip_id else None


def translation_words(entry: dict) -> list[str]:
    """The English a code-switched clip should translate to, from both its legs.

    An English leg contributes its own text; every other leg contributes the ``translation``
    the manifest records for it, so a leg that was transcribed rather than translated -- the
    Spanish left in Spanish -- scores against what it should have said in English.
    """
    expected: list[str] = []
    for leg in entry.get("legs") or []:
        expected.extend(words(leg["text"] if leg["language"] == "en" else leg.get("translation", "")))
    return expected


def foreign_leg_words(entry: dict) -> list[str]:
    """The words of a code-switched clip's non-English legs, as spoken -- what must be gone."""
    return [word for leg in entry.get("legs") or [] if leg["language"] != "en" for word in words(leg["text"])]


@functools.lru_cache(maxsize=1)
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

    ``xfail_engines`` narrows the same way along the other axis. Every code-switched clip
    is a defect on WHISPERX and none is on FASTER-WHISPER, so without it the default engine
    reports six XPASSes -- the wall the paragraph above exists to prevent -- and the entries
    have to say "engine-dependent" in prose that nothing enforces. An entry that omits the
    key applies to every engine, which is what all the pre-existing entries mean.
    """
    marks = [pytest.mark.smoke] if entry.get("smoke") else []
    reason = entry.get("xfail_reason")
    if not reason or not _scope_applies(entry.get("xfail_scope"), scope):
        return marks
    if not _defect_applies_to_running_engine(entry):
        return marks
    return marks + [pytest.mark.xfail(strict=False, reason=str(reason))]


def _scope_applies(declared, scope: str | None) -> bool:
    """Whether a defect declared for ``declared`` concerns the test asking about ``scope``.

    One name or a list of them: an engine that drops a leg drops it whether the clip is
    transcribed or translated, so that entry names both concerns rather than leaving the
    translation test strict against a defect already on record.
    """
    if declared is None:
        return True
    return scope in declared if isinstance(declared, list) else declared == scope


def _defect_applies_to_running_engine(entry: dict) -> bool:
    """Return whether this entry's defect belongs to the engine that is answering.

    Unknown engine means apply the mark: collection happens before any request, and on a
    host with no service every real-audio test skips anyway, so guessing "not affected"
    there would turn a skipped suite into a strict one the moment a service appeared.
    """
    engines = entry.get("xfail_engines")
    if not engines:
        return True
    running = service_client.running_engine()
    return running is None or running in engines


def _entry_is_selected(entry: dict, tier: str | None) -> bool:
    """Return whether an entry should be parametrized for the requested tier.

    Only an *explicit* ``"voice": null`` is rejected -- that is how the manifest records a
    language nothing can synthesize (the ``*_gap`` entries), and there is no clip to test.
    An entry that simply omits the key is kept: the adversarial and combined sections never
    carry one, and dropping those would silently empty two whole sections. The previous
    ``entry.get("voice", "") is None`` expressed this through a default that could never
    match the comparison, which read as a bug even though it behaved correctly.
    """
    if "voice" in entry and entry["voice"] is None:
        return False
    return tier is None or entry.get("tier") == tier


def params_for_ids(section: str, wanted: set[str], scope: str | None = None) -> list[Any]:
    """Return the params for exactly ``wanted``, failing if any id is missing.

    The failure is the point. Both callers previously kept their own copy of this, and the
    filter is a silent one: an adversarial entry that is renamed or dropped simply stops
    matching, the parametrized set shrinks, and the case quietly stops being tested while
    the suite still reports all green. Raising here turns that into a collection error
    naming the section and the ids that vanished.
    """
    params = [param for param in clip_params(section, scope=scope) if param.id in wanted]
    missing = sorted(wanted - {param.id for param in params})
    if missing:
        raise AssertionError(f"ids named by the test are absent from the {section!r} manifest section: {missing}")
    return params


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
