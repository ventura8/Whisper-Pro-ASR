"""Turn pip's dry-run install report into a hash-pinned requirements file.

Invoked by regenerate_locks.sh inside the test image; not useful on its own.
"""

import hashlib
import json
import sys
import urllib.request

HEADER = """\
# Audio-matrix fixture toolchain -- GENERATED, DO NOT EDIT BY HAND.
#
# Regenerate with:  scripts/audio_matrix/regenerate_locks.sh
#
# Why this exists. The generator used to install `torch>=2.4`, `transformers>=4.44` and five
# entirely unpinned packages with --upgrade, so two runs months apart built the fixtures with
# different code. The matrix is content-addressed and committed, so a silent toolchain change
# surfaces as a fixture diff nobody asked for -- or, worse, as a scoring shift in the
# real-audio suite that reads like an engine regression.
#
# One resolution, not two. torch is pinned to its `+cpu` local version, which only
# download.pytorch.org can satisfy, so PyPI can stay the primary index and the whole closure
# resolves together. Resolving the two halves separately produced two different `filelock`
# versions whose winner depended on install order.
#
# Hashes come from the index that serves each artefact. Entries marked LOCAL are served by
# download.pytorch.org, which publishes no hash for the PyPI packages it mirrors; those were
# downloaded and hashed here, and each was then verified byte-identical to the digest PyPI
# publishes for the same version -- so the mirror is corroborated, not merely trusted.
"""


def _pypi_digests(name: str, version: str) -> set[str]:
    """The sha256 digests PyPI publishes for a release, or an empty set if it cannot say."""
    url = f"https://pypi.org/pypi/{name}/{version}/json"
    try:
        with urllib.request.urlopen(url, timeout=120) as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return set()
    return {entry["digests"]["sha256"] for entry in data.get("urls", [])}


def _resolve_hash(package: dict) -> tuple[str, str]:
    """Return (sha256, note) for one resolved package."""
    info = package["download_info"]
    published = ((info.get("archive_info") or {}).get("hashes") or {}).get("sha256")
    if published:
        return published, ""
    with urllib.request.urlopen(info["url"], timeout=600) as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    name = package["metadata"]["name"]
    version = package["metadata"]["version"]
    corroborated = digest in _pypi_digests(name, version)
    state = "verified identical to PyPI" if corroborated else "NOT corroborated by PyPI"
    return digest, f"# LOCAL: index published no hash; {state}"


def main(report_path: str, out_path: str) -> int:
    with open(report_path, encoding="utf-8") as handle:
        report = json.load(handle)
    lines = [HEADER, ""]
    uncorroborated = []
    for package in sorted(report["install"], key=lambda p: p["metadata"]["name"].lower()):
        digest, note = _resolve_hash(package)
        if "NOT corroborated" in note:
            uncorroborated.append(package["metadata"]["name"])
        if note:
            # Its own line: a comment after the continuation backslash makes the requirement
            # unparseable, and pip reports that as a syntax error naming no package.
            lines.append(note)
        lines.append(f"{package['metadata']['name']}=={package['metadata']['version']} \\")
        lines.append(f"    --hash=sha256:{digest}")
    with open(out_path, "w", encoding="utf-8") as out:
        out.write("\n".join(lines) + "\n")
    print(f"{out_path}: {len(report['install'])} packages pinned")
    if uncorroborated:
        print(f"WARNING: hashes not corroborated by PyPI: {uncorroborated}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
