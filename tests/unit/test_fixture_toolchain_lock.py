"""The fixture toolchain lock must not drift from what pyproject declares.

The audio matrix is content-addressed and committed, so the toolchain that renders it is
part of the contract. Two ways that contract has already broken:

* the lock resolved plain ``piper-tts`` while pyproject declares ``piper-tts[ja,zh]``. The
  extras carry the Japanese and Chinese phonemizers, and without them those voices
  synthesize an EMPTY waveform rather than failing -- surfacing much later as
  ``wave.Error: # channels not specified`` on a single clip in a full regeneration.
* the lock is generated, so nothing else in the suite reads it. A drift check has to be
  explicit or there is none.
"""

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCK = REPO_ROOT / "scripts" / "audio_matrix" / "requirements.lock"
RESOLVER = REPO_ROOT / "scripts" / "audio_matrix" / "regenerate_locks.sh"


def _declared_piper_extras() -> set[str]:
    """The extras pyproject's tools group declares for piper-tts."""
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    tools = data["tool"]["poetry"]["group"]["tools"]["dependencies"]
    return set(tools["piper-tts"].get("extras", []))


class TestTheLockTracksPyproject:
    """What renders the fixtures must match what the project says renders them."""

    def test_the_resolver_requests_every_extra_pyproject_declares(self):
        """Dropping an extra does not fail the resolve; it fails one clip, much later."""
        requested = re.search(r'"piper-tts\[([^\]]*)\]"', RESOLVER.read_text(encoding="utf-8"))
        assert requested, "regenerate_locks.sh must request piper-tts with explicit extras"
        assert {e.strip() for e in requested.group(1).split(",")} == _declared_piper_extras()

    def test_the_lock_pins_piper_itself(self):
        """A lock without the synthesizer cannot render anything."""
        assert re.search(r"(?m)^piper-tts==", LOCK.read_text(encoding="utf-8"))

    def test_the_lock_carries_the_phonemizers_those_extras_pull(self):
        """The failure this guards: extras requested but their packages absent from the lock.

        Named by role rather than by exact distribution, because the phonemizer packages
        change name between piper releases; what must hold is that *something* providing
        Japanese and Chinese phonemization is pinned.
        """
        text = LOCK.read_text(encoding="utf-8").lower()
        assert re.search(r"(?m)^(pyopenjtalk|openjtalk)", text), "no Japanese phonemizer pinned"
        assert re.search(r"(?m)^(g2pw|pypinyin|jieba)", text), "no Chinese phonemizer pinned"

    def test_every_pinned_line_carries_a_hash(self):
        """--require-hashes refuses the whole file if a single pin lacks one."""
        lines = LOCK.read_text(encoding="utf-8").splitlines()
        pins = [i for i, line in enumerate(lines) if re.match(r"^[A-Za-z0-9_.\-]+==", line)]
        assert pins, "the lock pins nothing"
        for i in pins:
            assert lines[i + 1].strip().startswith("--hash=sha256:"), f"{lines[i]} has no hash"
