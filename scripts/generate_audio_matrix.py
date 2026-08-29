#!/usr/bin/env python3
"""Generate the multilingual real-audio test matrix.

Run ``python3 scripts/generate_audio_matrix.py --help`` for the available commands. See
``tests/e2e/fixtures/audio_matrix/README.md`` for the workflow.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import deliberately follows the sys.path insertion above so the script runs from anywhere.
from scripts.audio_matrix.cli import run

if __name__ == "__main__":
    sys.exit(run())
