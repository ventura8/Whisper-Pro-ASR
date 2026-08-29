# Local Pipeline Execution Skill

This skill provides a structured workflow for running the Whisper Pro ASR CI/CD pipeline locally. It automates linting, testing, and coverage verification.

## Prerequisites

- Python 3.12+ (or Docker Desktop for containerized runs)
- PowerShell 7+ (Windows) or Bash (Linux/macOS)

## Workflow Steps

### 1. Execute Pipeline (Containerized Docker - Mandatory)

All lints, tests, security scans, audits, or type-checking MUST happen inside the Docker test image. Do not run checks on the host environment directly.

Run the main build and test script:

- Both local wrappers verify `poetry.lock` in the workspace before Docker builds, using a disposable Python container. That container upgrades to the pinned latest `pip` (`PIP_VERSION` in `scripts/ci/dependencies.env`, kept in sync with `ARG PIP_VERSION` in `Dockerfile` / `Dockerfile.test`) before installing Poetry, then regenerates the lock only when it is missing or stale.

- **Linux/macOS**:

  ```bash
  ./scripts/ci/build-and-test.sh
  ```

- **Windows**:

  ```powershell
  powershell -ExecutionPolicy Bypass -File .\scripts\ci\build-and-test.ps1
  ```

### 2b. Stage Selection & Caching (CI Parallelization)

## Image size policy

The published variants carry vendor runtimes that dwarf the application, so size work lives
in `scripts/docker/` and is enforced in-layer:

- `strip_build_artifacts.sh` runs inside the dependency layer and after the isolated WhisperX
  install (compiler toolchain, static archives, Python bytecode caches, torch's `test/` and
  `include/` trees, bundled wheel test suites). It scans all filesystem prefixes so vendor and
  WhisperX wheels cannot retain build artifacts outside `/opt/venv`.
- `prune_os_docs.sh` runs as the **last** step of every vendor install. It cannot live in the
  dependency layer: each vendor `apt` transaction re-creates `/usr/share/doc`, so cleaning
  early ships the bytes anyway plus a useless whiteout layer.
- `prune_rocm.sh` keeps only consumer RDNA2/RDNA3/RDNA4 kernels in the published `amd` and `full`
  images. Instinct CDNA (`gfx908`, `gfx90a`, `gfx940`/`941`/`942`) is intentionally pruned and
  falls back to CPU; every listed published capability must stay synchronized with
  `docs/SETUP.md`.
- `prune_cuda.sh` removes only NVIDIA's NPP family. Resist the temptation to dedupe the
  system CUDA libraries against the pip `nvidia/*` wheels: they are different builds serving
  different consumers (ONNX Runtime vs torch), and CUDA 12.9 cannot be removed at all because
  `libctranslate2` resolves `libcublas.so.12` through `dlopen`, which no build-time check sees.
- `verify_no_build_artifacts.sh` asserts the cleanups held; extend it when adding a new one.

Every cleanup MUST run in the same `RUN` as whatever created the files. A later layer only
whites them out while the bytes still ship.

`tests/run_suite.sh` is stage-selectable via the `PIPELINE_STAGE` env var (`all` by default -- what the commands above use; or one of `lint`, `python-tests`, `js-unit-tests`, `e2e-fixture`, `e2e-real` for a single slice; `real-audio` and `real-audio-stress` also exist but are opt-in only and never part of `all`). Stage order for `all` is always **lint first** (including Radon rank-A), then tests: `js-unit-tests` → `python-tests` → `e2e-fixture` → `e2e-real`. `.github/workflows/ci.yml` runs `lint-and-security` after `build-image`, and every test job `needs:` lint before starting (test jobs may still run in parallel with each other). Locally, the wrapper scripts use `PIPELINE_STAGE=all` (unset), so one invocation runs that same lint-then-tests sequence (lint tools concurrently via background shell jobs; pytest uses a parallel `-n auto` bulk pass plus a separate serial pass for timing-sensitive concurrency tests, then merges coverage/JUnit data before the 90% gate and badge generation). Both pytest invocations pass `-m "not slow"`, so multi-minute tests (the long-form GPU audio clip) can never be pulled into the CI stage even if their skip guard is removed. The `real-audio` stage drives a **live** service (`WHISPER_BASE_URL`) with the multilingual audio-matrix **smoke set** (`-m "real_audio and smoke"`, budgeted under 20 minutes); `real-audio-stress` runs the whole matrix (~2h) and then the 20-minute long-form GPU clip. Both are deliberately absent from `.github/workflows/ci.yml`: GitHub-hosted runners have no GPU, no provisioned model cache and no running service, so no real-engine test can execute there -- wiring the smoke stage into CI needs a self-hosted runner with the stack already up. Any nonzero real-audio pytest exit, including exit 5 for no collection, fails the stage.

Caching: the Docker build itself uses `docker buildx build --cache-from/--cache-to=type=local` under `.docker-build-cache/` (mirroring CI's `type=gha`) so repeat local builds are fast even when a layer must re-execute. Local wrappers export cache to `.docker-build-cache.new` and atomically replace `.docker-build-cache` afterward — writing `cache-to` into the same directory used for `cache-from` can fail with `mkdir: cannot create directory ''` while buildx finalizes the local OCI cache. A named Docker volume (`whisper-pro-asr-tool-cache`) persists ESLint/Stylelint/ruff/pytest run-time caches across separate local `docker run` invocations (build-time cache mounts alone never reach the running container). Requires a `docker-container`-driver buildx builder (created automatically by the wrapper scripts) since the default driver does not support local cache export.

**Hard requirement**: `.dockerignore` MUST exclude `.docker-build-cache`, `.docker-build-cache.new`, `.docker-build-cache.old`, and `.buildx-cache`. Those dirs live under the project root as BuildKit cache destinations; if they are not ignored, `docker buildx build ... .` recursively ships tens of GB of cache back into the build context (the exact failure mode that turns a normal build into a multi-minute 50GB+ context transfer).

### 3. Resolve Test Failures & Coverage

If tests fail:

- Test-stage order contract: `tests/run_suite.sh` finishes the full lint stage (including Radon complexity summary + rank-A enforcement) before any test stage. Within `python-tests`, pytest and coverage generation run only after lint has already passed.
- Radon source discovery in the test container must use filesystem enumeration (`find`) rather than `git ls-files`, because `.git` metadata is unavailable in Docker test images.
- Review `reports/pytest.xml` or `reports/coverage_output.txt` for specific test failures after Docker-backed parity runs. Quiet pytest progress (via `tests/class_progress.py` + xdist `--dist=loadscope`) writes each `module.py::TestClass` line through pytest's terminal writer **when that class finishes** (all of its results received on the controller), with one result character per test (`.`/`F`/`s`). It does not wait for an xdist worker to drain and does not dump unfinished groups at session end. Do not use `-v` in the pipeline — it dumps every node id. Final `coverage_output.txt` includes a pytest-cov-style `coverage:` header so GitHub's coverage-comment action can parse it on pull requests.
- Fix broken assertions or environment-specific mocks.
- If coverage is below 90%, add missing test cases in the `tests/` directory to ensure all paths are verified before refactoring.
- Local parity pipeline contract: wrappers are fail-fast across the Docker test image build and execution path. If the image build or any in-container quality gate fails, the wrapper exits immediately with a non-zero status.
- Coverage badge generation is performed by the `genbadge` library from `coverage.xml` and must fail hard if the output badge is missing or empty.
- Wrapper scripts (`scripts/ci/build-and-test.sh` and `scripts/ci/build-and-test.ps1`) regenerate the badge from the latest `coverage.xml` only after the Docker test image run succeeds, before printing completion.

### 4. Resolve Linting Issues

Once the system is stable and covered by tests:

- **Requirement**: Run `npm run fix:md` when Markdown formatting drift exists, then pass `npm run lint:md`.
- **Requirement**: Achieve a Pylint score of **10.0/10.0** in *both* runs. The gate lints
  production and tests separately: `pylint modules whisper_pro_asr.py` with `.pylintrc`
  (which disables nothing), and `pylint --rcfile=.pylintrc-tests tests tests/check_coverage.py`
  with a config that relaxes five test-idiomatic checks. Note that pylint exits non-zero on
  *any* message, so a 10.00/10 score with convention messages still fails the stage.
- **Requirement**: Pass Flake8 checks on `modules`, `whisper_pro_asr.py`, `tests`, and `tests/check_coverage.py`.
- **Constraint**: Inline suppressions are banned outright and `scripts/ci/check-inline-ignores.py`
  enforces it -- no `# pylint: disable`, `# noqa`, `# type: ignore`, `# pragma: no cover` or
  `shellcheck disable`, anywhere. The established ways to satisfy a check honestly:
  - a deferred import -> `importlib.import_module("pkg.mod")`, which is a call, not an
    import statement, so neither ruff PLC0415 nor pylint import-outside-toplevel applies;
  - a deliberately broad handler -> `except tuple([Exception])`, as in `pcm_helpers.py`;
  - a module-level name rebound at runtime -> assign through `sys.modules[__name__]`
    rather than `global`, as `history_manager` does;
  - an unquoted shell expansion -> a bash array, not `# shellcheck disable=SC2086`;
  - a test-only exemption -> a line in `.pylintrc-tests`, which is reviewed once instead of
    scattered across the suite.
- Fix import errors, remove unused code, and enforce PEP8 standards across all modules and tests.
- Fix Markdown lint issues via the repo-configured markdownlint auto-fix flow before resorting to manual cleanup.
- Note: High coverage ensures that lint-driven refactoring does not introduce regressions.

### 5. Verify New Features

After implementing changes, verify the following areas have test coverage:

- **Speaker Diarization**: `tests/inference/pipeline/test_diarization.py` — WhisperX orchestration, caching, fallbacks.
- **ASR Improvements**: `tests/inference/test_improvements.py` — parameter forwarding, idle timeout, subtitle wrapping.
- **Priority Concurrency**: `tests/inference/scheduler/priority/test_priority_concurrency.py`, `tests/inference/scheduler/priority/test_priority_concurrency_core_tests.py`, `tests/inference/scheduler/priority/test_priority_concurrency_extended_tests.py` — hardware pool configurations, yielding, and targeted preemption regressions.

### 6. Audit the Local Machine FIRST (Before Any Hardware Validation)

**Never assume which accelerators exist.** Audit the host before choosing a build target
or running hardware validation, then run **only the validations that host can actually
support**. Claiming a target passed on hardware that is not present is a false result.

```bash
scripts/audit_hardware.sh          # Linux host   (add --env to write BUILD_TARGET/HOST_INTEL_RENDER_GID to .env, --json for CI)
scripts/audit_hardware.ps1         # Windows / Docker Desktop (WSL2) host
```

The tool probes NVIDIA (`nvidia-smi` plus a real `docker run --gpus all ... nvidia-smi`
probe), Intel/AMD render nodes and their GID (`/dev/dri/renderD*`),
the Intel NPU (`/dev/accel`, absent on iGPU/Arc-only hosts), the AMD ROCm node
(`/dev/kfd`), `lspci` vendor IDs (8086 Intel, 10de NVIDIA, 1002 AMD), `intel_gpu_top`
availability and free build space, then prints a recommended `BUILD_TARGET` and the
matching `docker-compose.<target>.yml`.

Map the audit to what you may run:

| Audit result | Build target / override | Validation you may claim |
| --- | --- | --- |
| NVIDIA **and** AMD both present | `full` on native Linux; otherwise `nvidia` or `amd`, one at a time | Both, but each needs its own evidence. See below |
| `nvidia-smi` works **and** Docker GPU probe succeeds | `nvidia` | CUDA transcription + `nvidia-smi` VRAM evidence |
| `/dev/dri/renderD*` present, vendor `8086` | `intel` | Intel transcription + `intel_gpu_top` evidence |
| NVIDIA **and** Intel both present | `nvidia-intel` | Both, plus hybrid `/dev/dri/by-path` enumeration |
| `/dev/kfd` present, vendor `1002` | `amd` | ROCm transcription + `rocm-smi` evidence |
| `/dev/accel` **absent** | -- | **Do not** claim NPU validation; it is untestable here |
| No accelerator, or Docker GPU probe fails | `cpu` | CPU transcription only |

**The audit tool's emitted `BUILD_TARGET` wins over this table's order.** The rows are a
reading aid, not a precedence list, and reading them top-down gets two hosts wrong: a
NVIDIA+Intel machine matches the `nvidia` row before reaching `nvidia-intel`, and a
NVIDIA+AMD one matches `nvidia` before its AMD card is ever considered. `audit_hardware.sh`
resolves the combinations first (it checks NVIDIA+Intel and NVIDIA+AMD before either
single-vendor case); take the target it prints and use this table to see what that target
then entitles you to claim.

**`full` is the supported hybrid target, on native Linux only.** It carries both vendors'
ONNX Runtimes, and `bootstrap._check_dual_gpu_path` deliberately loads `/app/libs/amd` when
both vendors are present and `ASR_DEVICE`/`ASR_PREPROCESS_DEVICE` are both `AUTO`: ASR runs
on CUDA (CTranslate2 binds CUDA directly and has no ROCm backend), while UVR vocal isolation
runs on the AMD GPU through onnxruntime-rocm. That split is the reason both stages must be
validated **separately**, each with its own evidence: `rocm-smi --showuse` sampled during
vocal separation for the AMD/UVR half, and `nvidia-smi --query-compute-apps` sampled during
inference for the NVIDIA/ASR half. One green suite covers both only if both samples were
taken; otherwise say which half was proven.

Note that an explicit `ASR_DEVICE` or `ASR_PREPROCESS_DEVICE` disables that path -- it is
gated on the AUTO/AUTO pair -- so setting either forces the normal single-vendor resolution.

**The WSL2 limitation is unchanged.** ROCm needs `/dev/kfd`, which WSL2 does not expose;
`/dev/dxg` detects the AMD adapter and UVR then falls back to CPU. So on Windows the hybrid
arrangement proves nothing about AMD, whatever the machine contains.

`amd` and `nvidia` remain the right targets for validating one vendor in isolation -- a
smaller build, and no ambiguity about which device did the work.

If a target's hardware is absent, the honest check is that the image **boots and falls
back to CPU cleanly** -- state that explicitly rather than reporting the target as
validated.

### 7. Local Hardware Validation (Mandatory on a Real Machine)

The suite mocks the ASR engine everywhere else, so a broken accelerator path -- a wrong
CUDA major, a missing ONNX Runtime, a model that loads but decodes garbage -- passes every
other test. **When validating on a local machine with real hardware, always run the
real-engine accuracy test** in addition to the pipeline:

```bash
# 1. Start the stack for the hardware under test, using the override the audit chose.
#    BUILD_TARGET comes from `scripts/audit_hardware.sh --env`; source it so the filename
#    below expands to a real file rather than "docker-compose..yml".
set -a; . ./.env; set +a
docker compose -f docker-compose.yml -f "docker-compose.${BUILD_TARGET}.yml" up -d

# 2. Drive the running service with a known-text speech fixture
docker build -f Dockerfile.test --target test -t whisper-pro-asr-test .
# -u and HOME, exactly as the runtime command above: without them the container writes as
# root into your working tree, and the caches and reports it leaves behind are root-owned
# files you then cannot edit or delete without sudo.
docker run --rm --network host -e RUN_REAL_ASR=1 -e WHISPER_BASE_URL \
  -u "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$(pwd):/app" -w /app whisper-pro-asr-test \
  python3 -m pytest tests/integration/test_transcription_accuracy.py
```

It posts `tests/e2e/fixtures/speech_known_text.wav` to the live container and asserts the
transcript contains both known sentences:

- *"The quick brown fox jumps over the lazy dog."*
- *"Whisper Pro ASR is running a hardware acceleration test on this machine."*

Skipped unless `RUN_REAL_ASR=1`, so it never slows CI. It talks to a running service over
HTTP rather than an in-process app, because the real engine needs the per-vendor ONNX
Runtime under `/app/libs` and a provisioned `model_cache` -- neither exists in the test
image. Override the target with `WHISPER_BASE_URL`.

**Acceptance is a successful pytest exit AND hardware evidence -- both, never either
alone.** The real-audio pytest run must exit zero; a failing or skipped suite has proven
nothing. But a passing transcript proves only that decoding worked, because a CPU fallback
returns correct text too. Pair the successful exit with the hardware evidence the runtime
skills require (`nvidia-smi` compute-apps VRAM for CUDA, `intel_gpu_top` for Intel,
`rocm-smi` for AMD), sampled while a transcription is in flight; CPU fallback is not
acceptable as acceleration evidence.

## Test Suite Structure

```text
tests/
├── inference/
│   ├── pipeline/
│   │   ├── test_diarization.py      # Speaker diarization pipeline tests
│   │   └── ...
│   ├── scheduler/
│   │   ├── priority/
│   │   │   ├── test_priority_concurrency.py # Shared helpers + core concurrency coverage
│   │   │   ├── test_priority_concurrency_core_tests.py
│   │   │   ├── test_priority_concurrency_extended_tests.py
│   │   │   └── ...
│   │   └── ...
│   └── ...
├── test_api_*.py                    # API route integration tests
├── test_config.py                   # Configuration resolution tests
└── ...
```

## Execution

To run this skill, execute:

**Linux/macOS (Containerized Docker):**

```bash
./scripts/ci/build-and-test.sh
```

**Windows (Containerized Docker):**

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\ci\build-and-test.ps1
```

Direct host-side `pytest` is unsupported for parity verification and may be used only for local diagnostics outside release-quality gate decisions.
