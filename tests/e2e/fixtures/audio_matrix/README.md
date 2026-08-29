# Audio matrix fixtures

Real speech fixtures for the multilingual ASR tests in `tests/real_audio/`.

## What is here

- `manifest.json` — the contract between the generator and the tests: what is said, in
  which language, by which voice, and what the transcript is expected to contain.
- `core/*.flac` — the committed core tier. These are checked in (~1.3 MB total) so the
  tier-A tests run on any machine with no TTS tooling installed.

Everything else is generated on demand into `test_data/audio_matrix/` (gitignored) and is
never committed. Voice models are downloaded into `test_data/audio_matrix/voices/`.

## Regenerating

The container path is the supported one. The TTS toolchain is heavy, and on a host whose
system Python is externally managed `poetry install --with tools` fails outright trying to
uninstall pip:

```bash
scripts/generate_fixtures_docker.sh
```

```bash
scripts/generate_fixtures_docker.sh verify
```

<details>
<summary>Alternative: the host toolchain</summary>

Requires a Python that is not externally managed. Same output, but it installs piper-tts,
transformers and torch onto the host:

```bash
poetry install --with tools
python3 scripts/generate_audio_matrix.py all
```

</details>

Generation is content-addressed and idempotent: a clip is rebuilt only when its manifest
entry, the generator version, or the Piper/ffmpeg version changes. Re-running the command
must leave `git status` clean.

`python3 scripts/audio_catalog.py` rewrites `docs/AUDIO_CATALOG.md`, the agent-facing
description of every clip here; `--check` fails when that file is stale.

`python3 scripts/generate_audio_matrix.py verify` reports language coverage against the
service's supported languages, including the gaps Piper cannot fill.

## Why the output is reproducible

Piper's CLI exposes no seed and its VITS graph samples noise on every run, so default
settings render a different waveform each time. The manifest pins `noise_scale` and
`noise_w_scale` to zero, which removes the stochastic term and makes rendering
bit-identical — that is what lets the committed fixtures round-trip with an empty diff.
Voice models are verified against the MD5 digests upstream publishes in `voices.json`, so
a re-download cannot silently swap a voice underneath a calibrated accuracy threshold.

## Which fixtures the pipeline actually uses

Entries carrying `"smoke": true` form the representative subset the pipeline runs by
default (`-m "real_audio and smoke"`, under 20 minutes). The rest exist for opt-in stress
runs. Keep the smoke set small and diverse rather than complete: it is there to catch a
broken accelerator path quickly, not to prove every language works.

## Adding a language

Add an entry to `manifest.json` (see an existing clip for the shape) and run the
generator. Voices come from [`rhasspy/piper-voices`](https://huggingface.co/rhasspy/piper-voices);
take the `voice_md5` from that repository's `voices.json`. Languages Piper cannot cover are
recorded with `"voice": null` and an `unsupported_reason` rather than being left out, so
the coverage gap stays visible.
