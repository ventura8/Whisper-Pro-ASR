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

import collections
import json
import os
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
_SPEC = matrix_support.load_manifest().get("longform") or {}
_DEFECTS = _SPEC.get("known_defects") or {}


def known_defect(key: str):
    """Return an xfail mark when the manifest records a known defect for ``key``."""
    reason = _DEFECTS.get(key, "")
    return pytest.mark.xfail(bool(reason), strict=False, reason=reason or f"no known defect for {key}")


# A sentence repeated more often than this is a decoder loop, not speech.
MAX_SENTENCE_REPEATS = 5


@pytest.fixture(name="longform", scope="module")
def longform_fixture(manifest, clip_path):
    """Return the long-form clip path and its ground-truth timeline."""
    spec = manifest.get("longform") or {}
    if not spec:
        pytest.skip("no longform entry in the manifest")
    path = clip_path(spec["id"])
    timeline_path = path.with_suffix(".timeline.json")
    if not timeline_path.exists():
        pytest.skip(f"missing ground truth {timeline_path}; regenerate with scripts/generate_audio_matrix.py longform")
    return path, json.loads(timeline_path.read_text(encoding="utf-8")), spec


@pytest.fixture(name="result", scope="module")
def result_fixture(longform, service_ready):
    """Transcribe the clip once and share the payload plus wall-clock across the tests.

    Twenty minutes of audio is far too expensive to re-submit per assertion, so the whole
    module reasons about a single run.
    """
    path, timeline, spec = longform
    print(f"\nsubmitting {path.name} ({timeline['duration']:.0f}s) to {service_ready}")
    started = time.monotonic()
    payload = service_client.transcribe(path)
    return {"payload": payload, "elapsed": time.monotonic() - started, "timeline": timeline, "spec": spec}


def _segments(payload: dict) -> list[dict]:
    """Return the non-empty segments of a response."""
    return [seg for seg in (payload.get("segments") or []) if str(seg.get("text", "")).strip()]


def _text_in_window(segments: list[dict], start: float, end: float) -> str:
    """Return the text of every segment whose midpoint falls inside a window."""
    inside = [seg for seg in segments if start <= (float(seg["start"]) + float(seg["end"])) / 2 < end]
    return " ".join(str(seg.get("text", "")) for seg in inside)


@known_defect("coverage")
def test_transcription_reaches_the_end_of_the_clip(result):
    """Coverage must span the timeline, not stop after the first few minutes."""
    segments = _segments(result["payload"])
    assert segments, "no segments returned for a 20-minute clip"
    duration = float(result["timeline"]["duration"])
    last_end = max(float(seg["end"]) for seg in segments)
    assert last_end >= duration * 0.9, f"transcription stopped at {last_end:.0f}s of {duration:.0f}s"


@known_defect("windows")
def test_each_utterance_is_transcribed_where_it_occurs(result):
    """Every ground-truth utterance window must contain roughly the words spoken there."""
    segments = _segments(result["payload"])
    threshold = float(result["spec"]["min_window_overlap"])
    misses = []
    for window in result["timeline"]["speech"]:
        found = matrix_support.words(_text_in_window(segments, window["start"], window["end"]))
        overlap = matrix_support.word_overlap(window["expect_words"], found)
        if overlap < threshold:
            misses.append(f"{window['language']}@{window['start']:.0f}s={overlap:.2f}")
    allowed = max(1, len(result["timeline"]["speech"]) // 10)
    assert len(misses) <= allowed, f"{len(misses)} windows below {threshold}: {misses[:12]}"


def test_multiple_languages_are_recognized_across_the_clip(result):
    """The clip is deliberately multilingual; a single-language result means detection stuck."""
    expected = {window["language"] for window in result["timeline"]["speech"]}
    reported = {str(seg.get("language") or "").lower() for seg in _segments(result["payload"])} - {""}
    if not reported:
        pytest.skip("this engine does not report per-segment language")
    coverage = len(reported & expected) / len(expected)
    minimum = float(result["spec"]["min_language_coverage"])
    assert coverage >= minimum, f"only {coverage:.0%} of {sorted(expected)} were reported; saw {sorted(reported)}"


@known_defect("quiet")
def test_quiet_regions_do_not_hallucinate(result):
    """Long pauses, music and noise must not be filled with invented sentences."""
    segments = _segments(result["payload"])
    limit = int(result["spec"]["max_quiet_window_chars"])
    noisy = []
    for window in result["timeline"]["quiet_windows"]:
        text = _text_in_window(segments, window["start"], window["end"]).strip()
        if len(text) > limit:
            noisy.append(f"{window['start']:.0f}-{window['end']:.0f}s: {text[:80]!r}")
    assert not noisy, f"{len(noisy)} silent windows produced text: {noisy[:6]}"


def _allowed_repeats(timeline: dict) -> int:
    """Return how often a sentence may legitimately repeat, per the ground truth.

    The clip is built from a small set of fixed sentences -- one per language, each laid
    down many times -- so a *correct* transcript repeats each of them exactly as often as
    the timeline does. Measured on the current fixture: 10 distinct utterances across 118
    windows, every one of them occurring 12 times. Judging that against a flat
    ``MAX_SENTENCE_REPEATS`` of 5 made the test unpassable by construction; it failed on
    faithful transcription, not on a decoder loop. A loop is repetition the audio does not
    account for, so the bar is what the timeline itself contains.
    """
    per_utterance = collections.Counter(" ".join(window["expect_words"]).lower() for window in timeline["speech"])
    ground_truth_max = max(per_utterance.values(), default=0)
    return max(MAX_SENTENCE_REPEATS, ground_truth_max)


@known_defect("repetition")
def test_no_runaway_repetition(result):
    """A sentence repeating more often than the audio contains it is a decoder loop."""
    sentences = [matrix_support.normalize(str(seg.get("text", ""))) for seg in _segments(result["payload"])]
    counts = collections.Counter(sentence for sentence in sentences if len(sentence) > 12)
    worst = counts.most_common(1)
    if not worst:
        pytest.skip("no segments long enough to judge repetition")
    sentence, repeats = worst[0]
    allowed = _allowed_repeats(result["timeline"])
    # Repeats landing in the wrong place -- other languages' windows, or silence -- are
    # real defects, but they are what `windows` and `quiet` measure. This test is only
    # about runaway volume, so it must not double-count them.
    assert repeats <= allowed, f"{repeats} repeats of {sentence[:80]!r} exceeds the {allowed} the timeline contains"


def test_throughput_stays_within_budget(result):
    """Report and bound the real-time factor, so a performance regression is visible."""
    duration = float(result["timeline"]["duration"])
    rtf = result["elapsed"] / duration
    print(f"\nlong-form RTF: {rtf:.3f} ({result['elapsed']:.0f}s wall clock for {duration:.0f}s of audio)")
    assert rtf <= MAX_RTF, f"real-time factor {rtf:.2f} exceeds the {MAX_RTF} budget (raise LONG_ASR_MAX_RTF to accept)"
