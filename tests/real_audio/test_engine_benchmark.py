"""Engine evaluation: what one ASR engine does across the language matrix, measured.

Not a pass/fail suite. The rest of tests/real_audio asks whether a transcript reaches its
bar; this asks *how good* and *how fast*, so engines can be compared on the same clips --
the data docs/ASR_ENGINE_EVALUATION.md is built from. It asserts only that each request
succeeded, and records:

* transcription with the language given (every clip available), scored as WER, or CER for
  the scripts the matrix scores by characters, and split into the languages the NeMo models
  cover and the ones they route to Whisper -- for FASTER-WHISPER the split is the same
  clips, which is what makes the comparison like for like;
* transcription with the language auto-detected, and translation, on tier A;
* /detect-language correctness and latency on tier A plus the smoke clips;
* RTF from the service's own inference time, and wall time per request.

At the end of the module a JSON and a Markdown report go to ``ASR_BENCH_OUT`` (default
reports/engine_eval). scripts/remote_validate.sh --suite bench runs this and copies the
report back. Gated behind RUN_ASR_BENCHMARK=1 as well as RUN_REAL_ASR=1: at ~300 requests it
is too long to ride along with an ordinary real-audio run.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tests.real_audio import bench_metrics, matrix_support, service_client

pytestmark = [
    *matrix_support.REAL_ASR_MARKS,
    pytest.mark.benchmark,
    pytest.mark.skipif(os.environ.get("RUN_ASR_BENCHMARK") != "1", reason="Engine benchmark; set RUN_ASR_BENCHMARK=1 to run."),
]

OUT_DIR = Path(os.environ.get("ASR_BENCH_OUT", "reports/engine_eval"))
HOST_LABEL = os.environ.get("ASR_BENCH_HOST", "local")

#: Every record this module measures; written out once, at module teardown.
_RECORDS: list[dict] = []

_CLIPS = matrix_support.load_manifest()["clips"]
_TIER_A = [clip for clip in _CLIPS if clip.get("tier") == "A"]
_DETECT = [clip for clip in _CLIPS if clip.get("tier") == "A" or clip.get("smoke")]


def _ids(clips: list[dict]) -> list[str]:
    """Test ids: the clip ids."""
    return [clip["id"] for clip in clips]


@pytest.fixture(scope="module", autouse=True)
def _report(service_ready):
    """Write the report after the last measurement, whatever happened to the ones before it."""
    yield
    status = service_client.status()
    engines = status.get("engines") or {}
    meta = {
        "engine": status.get("asr_engine"),
        "host": HOST_LABEL,
        "captured": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version": status.get("version") or status.get("version_display"),
        "nemo": engines.get("nemo"),
        "units": [unit.get("id") for unit in status.get("hardware_units", []) if isinstance(unit, dict)],
        "status": status,
    }
    stem = f"{meta['engine']}-{HOST_LABEL}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    path = bench_metrics.write_report(OUT_DIR, stem, _RECORDS, meta)
    print(f"\n[benchmark] {len(_RECORDS)} records -> {path}\n{path.read_text(encoding='utf-8')}")


def _timed(call):
    """Run ``call`` and return its result with the wall time it took."""
    started = time.monotonic()
    payload = call()
    return payload, round(time.monotonic() - started, 3)


def _rtf(payload: dict, wall: float) -> float | None:
    """Inference seconds per second of audio, from the service's own accounting when it gives one."""
    duration = float(payload.get("video_duration_sec") or 0.0)
    if duration <= 0:
        return None
    inference = float((payload.get("performance") or {}).get("inference_sec") or wall)
    return round(inference / duration, 4)


def _record(clip: dict, mode: str, payload: dict, wall: float, reference: str, *, tokenizer: str | None = None) -> None:
    """One transcription measurement. ``tokenizer`` defaults to the clip's own; a translation is English."""
    spoken = matrix_support.spoken_words(payload)
    reference_words = matrix_support.words(reference)
    _RECORDS.append(
        {
            "id": clip["id"],
            "language": clip["language"],
            "tier": clip.get("tier"),
            "mode": mode,
            "bucket": bench_metrics.bucket(clip["language"]),
            "error_rate": bench_metrics.error_rate(reference, spoken, tokenizer or clip.get("tokenizer", "words")),
            # More than twice the reference's length is not a misheard clip; it is text that
            # was never spoken, which the capped error rate alone would hide.
            "hallucinated": len(spoken) > 2 * max(1, len(reference_words)),
            "detected": payload.get("language"),
            "rtf": _rtf(payload, wall),
            "wall_sec": wall,
        }
    )


@pytest.mark.parametrize("clip", _CLIPS, ids=_ids(_CLIPS))
def test_transcribe_with_language(clip, clip_path):
    """Every clip, told its language: the path Bazarr takes when it knows the audio's language."""
    payload, wall = _timed(lambda: service_client.transcribe(clip_path(clip["id"]), data={"language": clip["language"]}))
    _record(clip, "transcribe+lang", payload, wall, clip["text"])


@pytest.mark.parametrize("clip", _TIER_A, ids=_ids(_TIER_A))
def test_transcribe_auto_detected(clip, clip_path):
    """Tier A without a language: detection first, then the decode it routes to."""
    payload, wall = _timed(lambda: service_client.transcribe(clip_path(clip["id"])))
    _record(clip, "transcribe+auto", payload, wall, clip["text"])


@pytest.mark.parametrize("clip", _TIER_A, ids=_ids(_TIER_A))
def test_translate(clip, clip_path):
    """Tier A to English, scored against the English sibling's text."""
    english = matrix_support.english_words_for(clip["id"])
    payload, wall = _timed(lambda: service_client.translate(clip_path(clip["id"])))
    _record(clip, "translate", payload, wall, " ".join(english or []), tokenizer="words")


@pytest.mark.parametrize("clip", _DETECT, ids=_ids(_DETECT))
def test_detect_language(clip, clip_path):
    """/detect-language, which every NeMo engine answers through its Whisper companion."""
    payload, wall = _timed(lambda: service_client.detect(clip_path(clip["id"])))
    detected = matrix_support.detected_code(payload)
    _RECORDS.append(
        {
            "id": clip["id"],
            "language": clip["language"],
            "tier": clip.get("tier"),
            "mode": "detect",
            "bucket": bench_metrics.bucket(clip["language"]),
            "error_rate": None,
            "detected": detected,
            "detect_correct": detected in clip.get("expect_detect", [clip["language"]]),
            "rtf": None,
            "wall_sec": wall,
        }
    )
