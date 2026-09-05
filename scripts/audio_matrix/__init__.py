"""Reproducible generator for the multilingual real-audio test matrix.

Every clip the tests use is described by ``tests/e2e/fixtures/audio_matrix/manifest.json``
and rendered from it by this package, so a fixture is never an unexplained binary: its
text, voice, and post-processing are all reviewable data.

Bumping ``GENERATOR_VERSION`` invalidates every cached clip. Do it whenever a change here
alters rendered audio, so stale cache entries cannot silently outlive the code that made
them.
"""

GENERATOR_VERSION = 1
