"""Loading and validation for the audio-matrix manifest.

The manifest is the contract between the generator and the tests: it holds what is said,
in which language, by which voice, and what the transcript is expected to look like.
Validating it here means a malformed entry fails fast with a precise message instead of
surfacing later as a mysterious assertion failure.
"""

from __future__ import annotations

import functools
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "tests" / "e2e" / "fixtures" / "audio_matrix" / "manifest.json"
CORE_DIR = MANIFEST_PATH.parent / "core"

SECTIONS = ("clips", "combined", "adversarial")


def load(path: Path | None = None) -> dict[str, Any]:
    """Load and return the manifest."""
    return json.loads((path or MANIFEST_PATH).read_text(encoding="utf-8"))


def clips(data: dict[str, Any]) -> list[dict]:
    """Return the single-language clip entries."""
    return list(data.get("clips") or [])


def renderable(entries: list[dict]) -> list[dict]:
    """Return only entries that have a voice to render with."""
    return [entry for entry in entries if entry.get("voice")]


@functools.lru_cache(maxsize=1)
def known_languages() -> frozenset[str]:
    """Return the service's supported language codes.

    Loaded straight from ``modules/core/languages.py`` by path rather than imported, so the
    generator stays runnable in a bare virtualenv holding only piper-tts: importing
    ``modules.core`` would drag in the whole application package (psutil, fastapi, ...).

    Cached because validation calls this twice per manifest entry, and every call executed
    the module afresh -- a few hundred needless executions on a full run. Frozen so the
    cached value cannot be mutated by a caller.
    """
    path = REPO_ROOT / "modules" / "core" / "languages.py"
    spec = importlib.util.spec_from_file_location("_audio_matrix_languages", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return frozenset(module.LANGUAGES)


def _err_missing_fields(entry: dict) -> str:
    """Return an error when a required field is absent."""
    missing = [field for field in ("id", "language", "tier") if not entry.get(field)]
    return f"missing fields {missing}" if missing else ""


def _err_bad_tier(entry: dict) -> str:
    """Return an error when the tier is not A or B."""
    return "" if entry.get("tier") in {"A", "B"} else f"bad tier {entry.get('tier')!r}"


def _err_unknown_language(entry: dict) -> str:
    """Return an error when the language is not one the service knows."""
    language = entry.get("language")
    return "" if language in known_languages() else f"unknown language {language!r}"


def _err_unknown_detect(entry: dict) -> str:
    """Return an error when an expected detection code is not a known language."""
    known = known_languages()
    unknown = [code for code in (entry.get("expect_detect") or []) if code not in known]
    return f"unknown expect_detect codes {unknown}" if unknown else ""


def _err_voiceless_without_reason(entry: dict) -> str:
    """Return an error when a voiceless entry does not explain itself."""
    if entry.get("voice"):
        return ""
    return "" if entry.get("unsupported_reason") else "voice is null but unsupported_reason is missing"


_CHECKS: tuple[Callable[[dict], str], ...] = (
    _err_missing_fields,
    _err_bad_tier,
    _err_unknown_language,
    _err_unknown_detect,
    _err_voiceless_without_reason,
)


def _err_missing_id(entry: dict) -> str:
    """Return an error when an entry has no id to name its cache file."""
    return "" if entry.get("id") else "missing field id"


#: Checks that apply to every generated entry, whatever section it is in. The clip checks
#: above are additional, and only meaningful for spoken clips: a combined entry carries
#: ``legs`` rather than one language, and an adversarial entry describes a transformation.
_COMMON_CHECKS: tuple[Callable[[dict], str], ...] = (_err_missing_id,)


def _entry_messages(section: str, entry: dict, checks: tuple[Callable[[dict], str], ...]) -> list[str]:
    """Every validation message one entry produces, prefixed with where it came from."""
    label = entry.get("id") or "<no id>"
    return [f"{section}/{label}: {text}" for text in (check(entry) for check in checks) if text]


def _section_errors(section: str, entries: list[dict]) -> list[str]:
    """Return every validation error for one section's entries."""
    checks = _CHECKS if section == "clips" else _COMMON_CHECKS
    return [message for entry in entries for message in _entry_messages(section, entry, checks)]


def _duplicate_id_errors(data: dict[str, Any]) -> list[str]:
    """Return an error when two entries share an id, anywhere in the manifest.

    Uniqueness is global, not per-section: every generated entry is cached as
    ``<id>.wav`` in one directory, so a clip and an adversarial entry sharing an id
    overwrite each other's audio and each other's stamp. Checking only within `clips`
    left that collision undetectable.
    """
    seen: dict[str, str] = {}
    duplicates = []
    for section, entry_id in _identified_entries(data):
        if entry_id in seen:
            duplicates.append(f"duplicate id {entry_id!r} in {seen[entry_id]} and {section}")
        else:
            seen[entry_id] = section
    return duplicates


def _identified_entries(data: dict[str, Any]) -> list[tuple[str, str]]:
    """Every (section, id) pair in the manifest, skipping entries with no id.

    A missing id is already reported by the per-section checks; repeating it here would
    only duplicate the message under a heading about duplicates.

    ``longform`` is included even though it is a single object rather than a list: it is
    cached as ``<id>.wav`` in the same directory as everything else, so a longform id that
    collides with a clip id overwrites that clip's audio and stamp exactly as any other
    duplicate would -- and was the one entry the check could not see.
    """
    pairs: list[tuple[str, str]] = []
    for section, entries in _all_sections(data):
        pairs.extend((section, entry["id"]) for entry in entries if entry.get("id"))
    return pairs


def _all_sections(data: dict[str, Any]) -> list[tuple[str, list[dict]]]:
    """Every named group of generated entries, as (section, entries).

    ``longform`` is a single object rather than a list, and wrapping it here is what lets
    the id checks treat it like everything else instead of skipping it.
    """
    sections: list[tuple[str, list[dict]]] = [(section, list(data.get(section) or [])) for section in SECTIONS]
    return sections + [("longform", [data.get("longform") or {}])]


def validate(data: dict[str, Any]) -> list[str]:
    """Return every validation error in the manifest, empty when it is well formed."""
    errors = [message for section in SECTIONS for message in _section_errors(section, list(data.get(section) or []))]
    return errors + _duplicate_id_errors(data)
