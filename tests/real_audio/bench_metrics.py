"""Scoring and reporting for the engine benchmark (test_engine_benchmark).

The real-audio matrix asks pass/fail questions -- did a transcript reach its bar. Comparing
engines needs a number that can go up or down: word error rate, or character error rate for
the scripts the matrix already scores by characters. Both are edit distance over the same
normalization the matrix uses (matrix_support.words), so a benchmark figure and a matrix
verdict describe the same transcript in the same terms.

Pure functions only, so the scoring itself is covered by the ordinary mocked suite
(tests/unit/test_bench_metrics.py) and only the measuring needs a live service.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from tests.real_audio import matrix_support

#: The 25 languages the NeMo models cover, as the manifest spells them. Mirrored from
#: modules.core.nemo_catalog.EU25 rather than imported: the benchmark runs in the test image
#: against a live service, and importing the application package would start its hardware
#: detection there. test_bench_metrics pins the two sets equal.
NEMO_LANGUAGES = frozenset(
    {
        "bg",
        "cs",
        "da",
        "de",
        "el",
        "en",
        "es",
        "et",
        "fi",
        "fr",
        "hr",
        "hu",
        "it",
        "lt",
        "lv",
        "mt",
        "nl",
        "pl",
        "pt",
        "ro",
        "ru",
        "sk",
        "sl",
        "sv",
        "uk",
    }
)

#: Tokenizers the matrix scores by characters; everything else is scored by words.
_CHARACTER_TOKENIZERS = frozenset({"chars", "chars_folded"})


def edit_distance(reference: list, hypothesis: list) -> int:
    """Levenshtein distance between two token sequences (substitution, insertion, deletion)."""
    previous = list(range(len(hypothesis) + 1))
    for row, ref_token in enumerate(reference, start=1):
        current = [row]
        for column, hyp_token in enumerate(hypothesis, start=1):
            cost = 0 if ref_token == hyp_token else 1
            current.append(min(previous[column] + 1, current[column - 1] + 1, previous[column - 1] + cost))
        previous = current
    return previous[-1]


def units(text_words: list[str], tokenizer: str) -> list[str]:
    """The units an error rate counts: words, or characters for character-scored scripts."""
    if tokenizer not in _CHARACTER_TOKENIZERS:
        return list(text_words)
    joined = "".join(text_words)
    if tokenizer == "chars_folded":
        joined = matrix_support._fold_marks(joined)
    return list(joined)


def error_rate(reference_text: str, hypothesis_words: list[str], tokenizer: str = "words") -> float:
    """WER (or CER) of a transcript against the clip's reference text, capped at 1.0.

    Capped because an engine that invents a paragraph into a one-line clip would otherwise
    report an error rate of several hundred percent and drown every mean it enters; the
    count of such clips is reported separately (``hallucinated``) instead.
    """
    reference = units(matrix_support.words(reference_text), tokenizer)
    hypothesis = units(hypothesis_words, tokenizer)
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return min(1.0, edit_distance(reference, hypothesis) / len(reference))


def bucket(language: str) -> str:
    """Whether a language is one the NeMo models transcribe themselves or route to Whisper."""
    return "nemo-covered" if language in NEMO_LANGUAGES else "whisper-routed"


def _median(values: list[float]) -> float | None:
    """The median, rounded, or None for no values."""
    return round(statistics.median(values), 4) if values else None


def _mean(values: list[float]) -> float | None:
    """The mean, rounded, or None for no values."""
    return round(statistics.fmean(values), 4) if values else None


def summarize(records: list[dict]) -> list[dict]:
    """One row per (mode, bucket): clip count, mean error, median RTF and wall time."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        groups.setdefault((record["mode"], record["bucket"]), []).append(record)
    return [_summary_row(mode, group_bucket, rows) for (mode, group_bucket), rows in sorted(groups.items())]


def _summary_row(mode: str, group_bucket: str, rows: list[dict]) -> dict:
    """Aggregate one (mode, bucket) group."""
    errors = [row["error_rate"] for row in rows if row.get("error_rate") is not None]
    return {
        "mode": mode,
        "bucket": group_bucket,
        "clips": len(rows),
        "mean_error_rate": _mean(errors),
        "hallucinated": sum(1 for row in rows if row.get("hallucinated")),
        "detect_correct": _share(rows, "detect_correct"),
        "median_rtf": _median([row["rtf"] for row in rows if row.get("rtf") is not None]),
        "median_wall_sec": _median([row["wall_sec"] for row in rows if row.get("wall_sec") is not None]),
    }


def _share(rows: list[dict], key: str) -> float | None:
    """The fraction of rows where ``key`` is true, among rows that report it at all."""
    reported = [row[key] for row in rows if row.get(key) is not None]
    return round(sum(1 for value in reported if value) / len(reported), 4) if reported else None


def to_markdown(summary: list[dict], meta: dict) -> str:
    """A report a person can read: the run's identity, then the summary table."""
    lines = [
        f"# Engine benchmark: {meta.get('engine')} on {meta.get('host')}",
        "",
        f"- Captured: {meta.get('captured')}",
        f"- Service version: {meta.get('version')}",
        f"- NeMo: {json.dumps(meta.get('nemo')) if meta.get('nemo') else 'n/a'}",
        f"- Units: {meta.get('units')}",
        "",
        "| mode | bucket | clips | mean WER/CER | hallucinated | detect correct | median RTF | median wall s |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {row['mode']} | {row['bucket']} | {row['clips']} | {_cell(row['mean_error_rate'])} | {row['hallucinated']} | "
        f"{_cell(row['detect_correct'])} | {_cell(row['median_rtf'])} | {_cell(row['median_wall_sec'])} |"
        for row in summary
    )
    return "\n".join(lines) + "\n"


def _cell(value: Any) -> str:
    """A table cell, with an en dash for a missing value."""
    return "–" if value is None else str(value)


def write_report(out_dir: Path, stem: str, records: list[dict], meta: dict) -> Path:
    """Write ``<stem>.json`` (every record) and ``<stem>.md`` (the summary); return the latter."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(records)
    (out_dir / f"{stem}.json").write_text(json.dumps({"meta": meta, "summary": summary, "records": records}, indent=2), encoding="utf-8")
    markdown = out_dir / f"{stem}.md"
    markdown.write_text(to_markdown(summary, meta), encoding="utf-8")
    return markdown
