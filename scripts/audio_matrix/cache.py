"""Content-addressed caching for generated audio clips.

A clip is regenerated only when something that affects its audio changes: its manifest
entry, the generator version, the Piper version, or the ffmpeg version. That is recorded
as a digest in a ``.stamp.json`` sidecar, which makes ``generate_audio_matrix.py`` cheap
and idempotent -- rerunning it produces no work and no diff.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from scripts.audio_matrix import GENERATOR_VERSION

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = REPO_ROOT / "test_data" / "audio_matrix"


def cache_root(explicit: str | None = None) -> Path:
    """Return the cache directory: the explicit path, the env override, or the default.

    The default lives under ``test_data/``, which is already gitignored, so generated
    audio can never be committed by accident.
    """
    chosen = explicit or os.environ.get("ASR_AUDIO_MATRIX_DIR") or str(DEFAULT_CACHE)
    return Path(chosen).expanduser().resolve()


def spec_digest(spec: dict, tool_versions: dict[str, str]) -> str:
    """Return a stable digest over a clip spec and the toolchain that renders it."""
    payload = {
        "spec": spec,
        "generator": GENERATOR_VERSION,
        "tools": dict(sorted(tool_versions.items())),
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def stamp_path(target: Path) -> Path:
    """Return the sidecar path recording how ``target`` was produced."""
    return target.with_suffix(target.suffix + ".stamp.json")


def read_stamp(target: Path) -> str:
    """Return the digest recorded for ``target``, or an empty string.

    A missing *or unreadable* stamp reads as "no digest", which makes the clip stale and
    regenerates it. Propagating a JSONDecodeError instead would abort the whole run over a
    sidecar that a truncated write left half-finished -- and leave no way to recover except
    deleting the file by hand.
    """
    path = stamp_path(target)
    if not path.exists():
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return ""
    return str(payload.get("digest", "")) if isinstance(payload, dict) else ""


def write_stamp(target: Path, digest: str, extra: dict | None = None) -> None:
    """Record ``digest`` (plus optional provenance) alongside ``target``.

    Written to a temporary file and renamed, so an interrupted run leaves either the old
    stamp or the new one -- never a half-written sidecar that reads as corrupt.
    """
    payload = {"digest": digest, **(extra or {})}
    path = stamp_path(target)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def is_fresh(target: Path, digest: str) -> bool:
    """Return whether ``target`` exists and was produced from ``digest``."""
    return target.exists() and read_stamp(target) == digest
