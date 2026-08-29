---
name: hardware-verification
description: Verify that Whisper Pro ASR actually works on real accelerator hardware — driving a live container over HTTP with real speech instead of mocks. Use this whenever the user wants to validate a machine, a build target, or an accelerator path (NVIDIA/CUDA, Intel GPU/NPU, AMD, CPU fallback); whenever they ask "does the GPU actually work", "is it using the accelerator", "validate this build", "test on real hardware", "run the real ASR tests", or "check multilingual/language support"; and before claiming any release, image, or device is hardware-validated. Also use it when triaging a failure from tests/real_audio or tests/integration/test_transcription_accuracy.py, or when regenerating the audio-matrix fixtures. Every other test in this repo mocks the ASR engine, so a broken accelerator path passes them all — reach for this skill rather than trusting a green suite.
---

# Hardware verification

The entire test suite mocks the ASR engine. A wrong CUDA major, a missing ONNX Runtime,
a model that loads but decodes garbage, an OpenVINO device that silently falls back — all
of it passes CI. Hardware verification is the only thing that catches these, and it works
by driving a **running container over HTTP** with real audio.

Two claims are easy to confuse, and conflating them is the most common mistake here:

- **Decoding works.** The transcript matches what the audio actually says.
- **The accelerator was used.** CPU fallback also produces a perfect transcript.

A passing transcript proves only the first. Never report a machine as hardware-validated
on transcript evidence alone.

## Order of operations

### 1. Audit the hardware before claiming anything

```bash
scripts/audit_hardware.sh
```

This reports what the host actually has and recommends a `BUILD_TARGET` and compose
override. A target whose accelerator is absent may only be described as "boots and falls
back to CPU cleanly" — never as hardware-validated. Say that explicitly rather than
letting a green run imply more than it showed.

### 2. Bring up the stack for the target

```bash
docker compose -f docker-compose.yml -f docker-compose.nvidia.yml up -d
```

Overrides available: `nvidia`, `intel`, `nvidia-intel`, `amd`, `cpu`, `full`, `wsl`.
Wait for health before testing — the container reports healthy immediately while the model
downloads in the background, and tasks submitted meanwhile queue at stage
`Downloading Model (xx%)`:

```bash
curl -s http://127.0.0.1:9000/status | head -c 300
```

### 3. Prove decoding with the English fixture

The fastest signal, ~1 minute:

```bash
RUN_REAL_ASR=1 python3 -m pytest tests/integration/test_transcription_accuracy.py -ra
```

### 4. Prove multilingual decoding with the smoke set

The English fixture says nothing about the other 50 languages the service advertises. This
is the routine multilingual check — a representative subset (4 languages spanning Latin,
Cyrillic and CJK, one code-switched clip, five degraded and malformed cases, and the
request-contract checks), budgeted to finish in **under 20 minutes**:

```bash
RUN_REAL_ASR=1 python3 -m pytest tests/real_audio -m "real_audio and smoke" -ra
```

This is what a pipeline or a pre-merge check should run. Reach for it first, always.

The full matrix (`-m real_audio`, 156 tests) takes roughly **two hours** — per-request cost
is dominated by UVR Vocal Separation preprocessing (~30–40 s each), not by decoding. It is
stress testing: run it when changing the engine, the preprocessing path, or language
handling, not to validate a machine. Nobody should pay two hours to learn something the
smoke set would have caught in twelve minutes.

Which entries are in the smoke set is manifest data (`"smoke": true`), so widening or
narrowing it is a data edit. Keep it small and diverse rather than complete.

### 5. Confirm the accelerator was actually used

Run this **while a transcription is in flight** — that is the whole point, and an idle
GPU proves nothing:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

Use `intel_gpu_top` for Intel GPU/NPU targets. On AMD, use `rocm-smi` — the container
ships it, so it works from inside:

```bash
docker exec whisper-pro-asr rocm-smi --showuse --showmemuse
```

Pair the evidence with the transcript result when reporting; neither alone is a hardware
validation.

### AMD specifically: check the kernel architecture is shipped

AMD has a failure mode the other vendors do not. The images carry pre-compiled ROCm kernels
for a fixed list of consumer Radeon GPU architectures (see `scripts/docker/prune_rocm.sh`).
A card whose architecture is absent still initialises and still transcribes — silently on the
CPU. So a passing transcript on AMD proves even less than usual.

Identify the card's architecture, then confirm the image carries kernels for it:

```bash
docker exec whisper-pro-asr rocm-smi --showhw
```

```bash
docker exec whisper-pro-asr bash -c 'ls /opt/rocm/lib/rocblas/library/ | grep -o "gfx[0-9a-z]*" | sort -u'
```

If the architecture is missing, the published image does not support that card. Do not claim
hardware validation from a CPU-fallback transcript; data-center and legacy kernels are excluded.

Two architectures never appear in that listing and are still supported: `gfx1031`
and `gfx1150` use `gfx1030` and `gfx1100` respectively via `HSA_OVERRIDE_GFX_VERSION`.

### 6. Long-form stress, NVIDIA only

Twenty minutes of realistic audio — long and short pauses, synthesized music, broadband
noise, ambient hum, varying loudness, ten languages. It exercises chunk boundaries, VAD
across long pauses, and the failure mode where a model fills silence with confident
invented text. None of that is visible in an eight-second clip.

```bash
RUN_REAL_ASR=1 RUN_GPU_LONG_ASR=1 python3 -m pytest tests/real_audio/test_longform_stress.py -ra -s
```

Requires `nvidia-smi` on the host. Marked `gpu` and `slow`, and deliberately absent from
CI. **This is stress testing, not routine validation** — run it deliberately, not on every
change. Takes ~9 minutes of wall clock for 20 minutes of audio.

It found real defects on first run (2026-09-01, RTX 3080): invented speech in 11 silent
windows and 18 repeats of one sentence — a decoder loop. Those are recorded as
`known_defects` in the manifest's `longform` spec, so the suite is honest rather than
merely green and flips to XPASS when they are fixed.

## Running the tests inside the Docker test image

Quality gates are Docker-only (AGENTS.md). Three flags matter and are easy to get wrong:

```bash
# Build from the checked-out commit and tag it with that commit, so the run cannot land on
# a stale :latest built from a different tree -- the "validated the wrong thing" failure
# this document opens with. --pull=never keeps a missing local image an error rather than
# a silent pull of someone else's build.
TEST_IMAGE="whisper-pro-asr-test:$(git rev-parse --short HEAD)"
docker build -f Dockerfile.test --target test -t "$TEST_IMAGE" .

docker run --rm --pull=never --gpus all --network host -v "$PWD:/app" -w /app -u 1000:1000 \
  -e WHISPER_PRO_ASR_TEST_IMAGE=1 -e RUN_REAL_ASR=1 -e HOME=/tmp \
  -e WHISPER_BASE_URL=http://127.0.0.1:9000 \
  "$TEST_IMAGE" /bin/bash -c 'python3 -m pytest tests/real_audio -m real_audio -ra --no-cov'
```

- `--network host` — the tests reach the service on the host's port 9000.
- `-u 1000:1000` with `-e HOME=/tmp` — **without this the container writes as root into
  your working tree.** Formatters and caches will leave root-owned files you then cannot
  edit. If that has already happened, repair it with the same tool rather than sudo:
  `docker run --rm --pull=never -v "$PWD:/app" -w /app "$TEST_IMAGE" chown -R 1000:1000 <paths>`
  -- the same commit-tagged `$TEST_IMAGE` built above, not `:latest`. The moving tag is
  reassigned by every build, so repairing ownership through it can pull or run an image
  built from a different tree, which is the failure this whole section is written against.
- `--gpus all` — only needed for the long-form test, whose gate checks for `nvidia-smi`
  inside the container.

Two canonical in-image stages exist, both opt-in and never part of `all`:

| Stage | Selection | Budget |
| --- | --- | --- |
| `PIPELINE_STAGE=real-audio` | `real_audio and smoke` | under 20 min |
| `PIPELINE_STAGE=real-audio-stress` | whole matrix, then the long-form clip | ~2 h + 9 min |

```bash
docker run --rm --pull=never -e WHISPER_PRO_ASR_TEST_IMAGE=1 -e PIPELINE_STAGE=real-audio \
  "$TEST_IMAGE" /bin/bash -c "tests/run_suite.sh"
```

Neither runs in GitHub Actions. Hosted runners have no GPU, no provisioned `model_cache`
and no running service, so no real-engine test can execute there — wiring the smoke stage
into CI requires a self-hosted runner with the stack already up.

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `RUN_REAL_ASR` | unset | Set to `1`; everything here is skipped without it |
| `RUN_GPU_LONG_ASR` | unset | Set to `1` for the 20-minute clip (also needs `nvidia-smi`) |
| `WHISPER_BASE_URL` | `http://127.0.0.1:9000` | Service under test |
| `REAL_ASR_TIMEOUT` | `900` | Per-request budget; a cold cache downloads the model first |
| `REAL_ASR_ADVERSARIAL_TIMEOUT` | `120` | Tight budget for malformed input — a timeout is a failure, because a corrupt file that pins a worker looks like a slow success |
| `ASR_AUDIO_MATRIX_DIR` | `test_data/audio_matrix` | Fixture cache and voice models (gitignored) |
| `LONG_ASR_MAX_RTF` | `1.0` | Long-form wall-clock budget as a fraction of real time |

## Fixtures

The ~10-language core tier is committed (`tests/e2e/fixtures/audio_matrix/core/*.flac`,
~1.4 MB) and needs no tooling. Everything else — the long tail, code-switched clips,
adversarial audio, the 20-minute clip — is generated on demand:

```bash
scripts/generate_fixtures_docker.sh all
```

```bash
scripts/generate_fixtures_docker.sh verify   # language coverage, including known gaps
```

That runs the generator inside the test image, which is the supported path: the TTS
toolchain (piper-tts, transformers+torch, uroman) is heavy, and on a host whose system
Python is externally managed `poetry install --with tools` fails outright trying to
uninstall pip. `python3 scripts/generate_audio_matrix.py all` is the same generator invoked
directly, and works only where that toolchain and ffmpeg are already installed.

Generation is content-addressed and idempotent; re-running must leave `git status` clean.
Rendering is bit-reproducible only because the manifest pins Piper's `noise_scale` and
`noise_w_scale` to zero — Piper exposes no seed and samples noise per run otherwise. If
you ever raise those pins, every committed fixture churns on each rebuild. Details in
`tests/e2e/fixtures/audio_matrix/README.md`.

## Triaging a failure

Decide which of three things you are looking at before changing anything. Getting this
wrong is how a real defect gets quietly erased.

**A service defect.** Record it, do not hide it. Set `xfail_reason` on the manifest entry
naming the defect and the date verified, and `xfail_scope` so the xfail relaxes only the
assertion the defect actually breaks — a mixed-language file whose second half is dropped
still detects a language and still returns HTTP 200, and marking those xfail too produces
a wall of meaningless XPASS. The test then passes by itself once the service is fixed.
Never lower a threshold to make a defect disappear.

**A miscalibrated expectation.** Fix the data, not the test. Tiers, expected words,
tolerances and acceptable detection codes all live in
`tests/e2e/fixtures/audio_matrix/manifest.json`. `expect_detect` is a list because engines
legitimately conflate `no`/`nn`, `bs`/`hr`/`sr`, `id`/`ms`, `ur`/`hi`. Non-space-delimited
scripts need `"tokenizer": "chars"`; comparing Chinese or Hindi as whitespace tokens can
never match.

**A bug in the harness.** Check this before blaming the engine. Two real examples:

- When there are no segments, the service returns an **SRT document** in `text` — cue
  numbers, timestamps, `[No dialogue detected]`. Read naively that is 102 characters of
  "hallucinated speech" on a silent file. `matrix_support.spoken_words` strips it.
- `tests/class_progress.py` registers a `tryfirst` status hook. It must preserve the
  `xfailed`/`xpassed` categories: an xfail report carries outcome `"skipped"`, and filing
  it as a plain skip puts an exception-shaped `longrepr` into the skipped bucket, which
  crashes pytest's `-ra` summary *after* a two-hour run finishes.

## Known pre-existing failures

`tests/integration/test_robustness.py::test_system_config_validation` and
`tests/performance/test_ssd_optimization.py::TestSSDOptimization::test_get_temp_dir_low_space_fallback`
fail on pristine HEAD, independent of any local change — they patch `shutil.disk_usage`
globally so both temp directories look low and `get_temp_dir` raises instead of falling
back. Confirm against a clean worktree before attributing a failure to your work:

```bash
git worktree add --detach /tmp/pristine HEAD
```

## Reporting

State the target, what was proven, and what was not. Separate the two claims from the top
of this document, and give the accelerator evidence alongside the transcript result. If
the accelerator was absent, say the image boots and falls back to CPU cleanly — and stop
there. Report skips and xfails honestly rather than folding them into a pass count; a
suite that is green because everything skipped has verified nothing.
