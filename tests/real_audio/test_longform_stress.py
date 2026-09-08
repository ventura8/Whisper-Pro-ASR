"""Twenty minutes of realistic audio, on real GPU hardware.

A twenty-minute file is not a short file repeated. It exercises chunk boundaries, VAD
across long pauses, behaviour over sustained load, and -- the reason this test exists --
the failure mode where a model fills silence, music or noise with confident invented text.
An eight-second fixture cannot show any of that.

The clip contains long pauses, short pauses, a synthesized music bed, broadband noise, an
ambient hum, varying loudness, and speech in ten languages, all laid out from a fixed seed
so the timeline is irregular but identical everywhere. The generator emits a ground-truth
sidecar, so these assertions are about *where* text should and should not appear.

This never runs in CI: there is no GPU runner and no provisioned model cache there. Run it
deliberately:

    RUN_REAL_ASR=1 RUN_GPU_LONG_ASR=1 python3 -m pytest tests/real_audio/test_longform_stress.py -ra -s
"""

from __future__ import annotations

import json
import os
import re
import time

import pytest

from tests.real_audio import matrix_support, service_client

pytestmark = [
    *matrix_support.REAL_ASR_MARKS,
    pytest.mark.gpu,
    pytest.mark.slow,
    # Gated on the opt-in flag alone. What this clip exercises -- chunk boundaries, VAD
    # across long pauses, decoder loops, invented speech in silence -- is a property of the
    # engine on long audio, not of CUDA, and it is worth measuring on Intel and AMD too.
    # Requiring nvidia-smi made the whole file unrunnable on any non-NVIDIA accelerator.
    # The flag is already explicit opt-in, so nothing runs this by accident.
    pytest.mark.skipif(
        os.environ.get("RUN_GPU_LONG_ASR") != "1",
        reason="Long-form stress test; set RUN_GPU_LONG_ASR=1 to run it.",
    ),
]

MAX_RTF = float(os.environ.get("LONG_ASR_MAX_RTF", "1.0"))

# Defects this engine is known to have on long multilingual audio, keyed by the property
# they break. Keeping them in the manifest rather than in decorators means a fix is
# recorded by deleting data, and the test flips to XPASS on its own when the service
# improves -- rather than someone quietly deleting the assertion.
# The bars a variant inherits from the top-level spec when it does not set its own. Not the
# defects, and not `translate`: a scene-shaped clip does not inherit the stress grid's
# recorded defects (a claim nobody made), and a second 20-minute decode is opted into.
_INHERITED = (
    "target_seconds",
    "languages",
    "min_window_overlap",
    "min_language_coverage",
    "max_quiet_window_chars",
    "max_foreign_share",
    "min_translation_overlap",
)


def _resolved_specs(manifest: dict) -> list[dict]:
    """The long-form spec and its variants, each variant filled in from the top level."""
    spec = manifest.get("longform") or {}
    inherited = {key: spec[key] for key in _INHERITED if key in spec}
    return [spec, *({**inherited, **variant} for variant in spec.get("variants") or [])]


_VARIANTS = _resolved_specs(matrix_support.load_manifest())


def variants(key: str, flag: str | None = None):
    """Parametrize a test over every long-form variant, carrying each one's own xfail.

    ``flag`` restricts the variants to those whose spec sets it -- a translation of a
    20-minute clip is a second full decode, so only the variants that ask for it pay.

    The variants record different defects and must, because they are different signals: the
    stress grid's numbers were measured on the stress grid, and inheriting them for a
    scene-shaped clip would be a claim nobody made. So the mark is attached per parameter
    rather than to the function, which also keeps the XPASS behaviour the manifest relies on
    -- a fix is recorded by deleting the entry, and the test flips on its own.
    """
    params = []
    for spec in _VARIANTS:
        if not spec.get("id") or (flag and not spec.get(flag)):
            continue
        reason = (spec.get("known_defects") or {}).get(key, "")
        marks = [pytest.mark.xfail(strict=False, reason=reason)] if reason else []
        params.append(pytest.param(spec["id"], marks=marks, id=spec["id"]))
    return pytest.mark.parametrize("variant", params)


# A sentence repeated more often than this *in a row* is a decoder loop, not speech.
MAX_SENTENCE_REPEATS = 5
# Where one segment's text breaks into sentences, before normalisation strips the marks.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?\u2026])\s+")
# Shorter than this and a sentence is a word or two, which real speech does repeat.
MIN_SENTENCE_CHARS = 12


@pytest.fixture(name="result", scope="module")
def result_fixture(manifest, clip_path, service_ready):
    """Return a lookup that transcribes each variant once and caches it.

    Twenty minutes of audio is far too expensive to re-submit per assertion, so every test
    asking for the same variant shares one run -- and a module asking for two variants pays
    for two, not twelve.
    """
    cached: dict[tuple[str, str], dict] = {}
    specs = {spec["id"]: spec for spec in _resolved_specs(manifest) if spec.get("id")}

    def _for(variant: str, task: str = "transcribe") -> dict:
        if (variant, task) not in cached:
            spec = specs.get(variant)
            if not spec:
                pytest.skip(f"no manifest entry for long-form variant {variant}")
            path = clip_path(spec["id"])
            timeline_path = path.with_suffix(".timeline.json")
            if not timeline_path.exists():
                pytest.skip(f"missing ground truth {timeline_path}; regenerate with scripts/generate_audio_matrix.py longform")
            timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
            print(f"\nsubmitting {path.name} ({timeline['duration']:.0f}s) to {service_ready} for {task}")
            started = time.monotonic()
            request = service_client.translate if task == "translate" else service_client.transcribe
            cached[(variant, task)] = {
                "payload": request(path),
                "elapsed": time.monotonic() - started,
                "timeline": timeline,
                "spec": spec,
            }
        return cached[(variant, task)]

    return _for


def _segments(payload: dict) -> list[dict]:
    """Return the non-empty segments of a response."""
    return [seg for seg in (payload.get("segments") or []) if str(seg.get("text", "")).strip()]


def _text_in_window(segments: list[dict], start: float, end: float) -> str:
    """Return the text of every segment whose midpoint falls inside a window."""
    inside = [seg for seg in segments if start <= (float(seg["start"]) + float(seg["end"])) / 2 < end]
    return " ".join(str(seg.get("text", "")) for seg in inside)


@variants("coverage")
def test_transcription_reaches_the_end_of_the_clip(result, variant):
    """Coverage must span the dialogue, not stop after the first few minutes.

    Judged against the end of the last line the timeline holds, not the end of the audio:
    a film ends on minutes of credits, and a transcript that stops where the dialogue stops
    is right, not short. The bookends clip closes on 180-300 s of credits and its correct
    transcript ends 280 s before the audio does.
    """
    data = result(variant)
    segments = _segments(data["payload"])
    assert segments, "no segments returned for a 20-minute clip"
    spoken_until = float(data["timeline"]["speech"][-1]["end"])
    last_end = max(float(seg["end"]) for seg in segments)
    assert last_end >= spoken_until * 0.9, f"transcription stopped at {last_end:.0f}s; the last line ends at {spoken_until:.0f}s"


@variants("windows")
def test_each_utterance_is_transcribed_where_it_occurs(result, variant):
    """Every ground-truth utterance window must contain roughly the words spoken there."""
    data = result(variant)
    segments = _segments(data["payload"])
    threshold = float(data["spec"]["min_window_overlap"])
    misses = []
    for window in data["timeline"]["speech"]:
        found = matrix_support.words(_text_in_window(segments, window["start"], window["end"]))
        overlap = matrix_support.word_overlap(window["expect_words"], found)
        if overlap < threshold:
            misses.append(f"{window['language']}@{window['start']:.0f}s={overlap:.2f}")
    allowed = max(1, len(data["timeline"]["speech"]) // 10)
    assert len(misses) <= allowed, f"{len(misses)} windows below {threshold}: {misses[:12]}"


@variants("translation", flag="translate")
def test_each_utterance_is_translated_where_it_occurs(result, variant):
    """Translation must reach every window in English, whatever language the window spoke.

    The same per-language decode carries ``task=translate``, so a Turkish scene in a
    Romanian film is translated *from Turkish*; decoded under the film's language token it
    would come back as noise. The matrix is one sentence pool in every language, so each
    window's English meaning is its ``en_`` sibling, and the bar is the same shape as the
    transcription one: the window's words, roughly, where the window is.
    """
    data = result(variant, "translate")
    segments = _segments(data["payload"])
    threshold = float(data["spec"]["min_translation_overlap"])
    misses = []
    unscored = 0
    for window in data["timeline"]["speech"]:
        expected = matrix_support.window_english_words(window)
        if expected is None:
            unscored += 1
            continue
        found = matrix_support.words(_text_in_window(segments, window["start"], window["end"]))
        overlap = matrix_support.word_overlap(expected, found)
        if overlap < threshold:
            misses.append(f"{window['language']}@{window['start']:.0f}s={overlap:.2f}")
    assert unscored == 0, f"{unscored} windows have no English counterpart in the manifest"
    allowed = max(1, len(data["timeline"]["speech"]) // 10)
    assert len(misses) <= allowed, f"{len(misses)} windows translated below {threshold}: {misses[:12]}"


@variants("language")
def test_multiple_languages_are_recognized_across_the_clip(result, variant):
    """The clip is deliberately multilingual; a single-language result means detection stuck."""
    data = result(variant)
    expected = {window["language"] for window in data["timeline"]["speech"]}
    reported = {str(seg.get("language") or "").lower() for seg in _segments(data["payload"])} - {""}
    # An absent key used to skip here, which quietly excused every engine from the check and
    # left the test counted but never run. It is a manifest-recorded gap now, so it reports as
    # a known defect and turns into an XPASS by itself the day a segment carries a language.
    assert reported, "no segment reported a language"
    coverage = len(reported & expected) / len(expected)
    minimum = float(data["spec"]["min_language_coverage"])
    assert coverage >= minimum, f"only {coverage:.0%} of {sorted(expected)} were reported; saw {sorted(reported)}"


@variants("fracture")
def test_reported_language_matches_the_line_spoken(result, variant):
    """Speech must be reported -- and so decoded -- in the language actually spoken.

    Decoding by speech region on its own labelled a median 20% of the speech in 19 real
    monolingual films as another language; run-level hysteresis brought that to 3%, level
    with whole-file decoding. The bar is per variant: the multilingual film clip holds a
    single foreign line in some scenes, which the hysteresis absorbs into the run around it
    by design, so it declares a higher `max_foreign_share` than the single-language one.
    """
    data = result(variant)
    share = matrix_support.foreign_share(_segments(data["payload"]), data["timeline"]["speech"])
    limit = float(data["spec"]["max_foreign_share"])
    assert share <= limit, f"{share:.1%} of the speech was reported in the wrong language; the bar is {limit:.0%}"


@variants("quiet")
def test_quiet_regions_do_not_hallucinate(result, variant):
    """Long pauses, music and noise must not be filled with invented sentences."""
    data = result(variant)
    segments = _segments(data["payload"])
    limit = int(data["spec"]["max_quiet_window_chars"])
    noisy = []
    for window in data["timeline"]["quiet_windows"]:
        text = _text_in_window(segments, window["start"], window["end"]).strip()
        if len(text) > limit:
            noisy.append(f"{window['start']:.0f}-{window['end']:.0f}s: {text[:80]!r}")
    assert not noisy, f"{len(noisy)} silent windows produced text: {noisy[:6]}"


def _longest_run(sentences: list[str]) -> tuple[str, int]:
    """The sentence that repeats the most times *in a row*, and how many."""
    best = ("", 0)
    current = ("", 0)
    for sentence in sentences:
        current = (sentence, current[1] + 1) if sentence == current[0] else (sentence, 1)
        best = max(best, current, key=lambda pair: pair[1])
    return best


def _allowed_consecutive(timeline: dict) -> int:
    """How many times in a row a sentence may legitimately repeat, per the ground truth.

    The clip is built from a small set of fixed sentences laid down many times, so the
    *total* count of a sentence says nothing about loops: on the stress grid every sentence
    occurs 12 times, and on the film clip its most frequent line 43 times. Worse, the same
    sentence exists in every language of the pool, so a lone foreign line that the run
    hysteresis absorbs -- decoded in the surrounding language -- comes out as one more copy
    of that language's version ("Dis-moi la vérité" as "Sag mir die Wahrheit"), which is
    the documented single-line trade, not a loop, and it made an exact total bar fail by one.

    A decoder loop is the same sentence emitted again and again *in a row*, so that is what
    is judged, against the longest such run the timeline itself contains.
    """
    _, ground_truth = _longest_run([" ".join(window["expect_words"]).lower() for window in timeline["speech"]])
    return max(MAX_SENTENCE_REPEATS, ground_truth)


def _repeating_phrase(words: list[str]) -> tuple[str, int]:
    """The shortest phrase ``words`` is a whole number of copies of, and how many."""
    for period in range(1, len(words) // 2 + 1):
        if len(words) % period == 0 and words == words[:period] * (len(words) // period):
            return " ".join(words[:period]), len(words) // period
    return " ".join(words), 1


def _sentences(text: str) -> list[str]:
    """The sentence units of one segment, a phrase looped inside it counted once per copy.

    A decoder loop does not stop at segment boundaries: faster-whisper can emit one segment
    that reads "Thank you. Thank you. Thank you." or one that repeats a phrase without any
    punctuation at all. Judged whole, that segment was one sentence and never a repeat.
    Short units are dropped unless the segment is nothing but copies of one: "yes" in a
    sentence is speech, "yes" as the only content of a segment eight times over is not.
    """
    units: list[str] = []
    for piece in _SENTENCE_BREAK.split(str(text)):
        tokens = matrix_support.normalize(piece).split()
        if not tokens:
            # Trailing whitespace after the last full stop splits off an empty piece, which
            # would add an empty unit and break "the segment is nothing but copies of one".
            continue
        phrase, copies = _repeating_phrase(tokens)
        units.extend([phrase] * copies)
    if len(units) > 1 and len(set(units)) == 1:
        return units
    return [unit for unit in units if len(unit) > MIN_SENTENCE_CHARS]


@variants("repetition")
def test_no_runaway_repetition(result, variant):
    """A sentence repeating more often in a row than the audio contains it is a decoder loop."""
    data = result(variant)
    sentences = [sentence for seg in _segments(data["payload"]) for sentence in _sentences(seg.get("text", ""))]
    if not sentences:
        pytest.skip("no segments long enough to judge repetition")
    sentence, repeats = _longest_run(sentences)
    allowed = _allowed_consecutive(data["timeline"])
    # Repeats landing in the wrong place -- other languages' windows, or silence -- are
    # real defects, but they are what `windows` and `quiet` measure. This test is only
    # about runaway volume, so it must not double-count them.
    assert repeats <= allowed, f"{repeats} consecutive repeats of {sentence[:80]!r} exceeds the {allowed} the timeline contains"


@variants("throughput")
def test_throughput_stays_within_budget(result, variant):
    """Report and bound the real-time factor, so a performance regression is visible."""
    data = result(variant)
    duration = float(data["timeline"]["duration"])
    rtf = data["elapsed"] / duration
    print(f"\nlong-form RTF: {rtf:.3f} ({data['elapsed']:.0f}s wall clock for {duration:.0f}s of audio)")
    assert rtf <= MAX_RTF, f"real-time factor {rtf:.2f} exceeds the {MAX_RTF} budget (raise LONG_ASR_MAX_RTF to accept)"
