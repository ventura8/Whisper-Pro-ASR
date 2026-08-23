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

`tests/run_suite.sh` is stage-selectable via the `PIPELINE_STAGE` env var (`all` by default -- what the commands above use; or one of `lint`, `python-tests`, `js-unit-tests`, `e2e-fixture`, `e2e-real` for a single slice). Stage order for `all` is always **lint first** (including Radon rank-A), then tests: `js-unit-tests` → `python-tests` → `e2e-fixture` → `e2e-real`. `.github/workflows/ci.yml` runs `lint-and-security` after `build-image`, and every test job `needs:` lint before starting (test jobs may still run in parallel with each other). Locally, the wrapper scripts use `PIPELINE_STAGE=all` (unset), so one invocation runs that same lint-then-tests sequence (lint tools concurrently via background shell jobs; pytest uses a parallel `-n auto` bulk pass plus a separate serial pass for timing-sensitive concurrency tests, then merges coverage/JUnit data before the 90% gate and badge generation).

Caching: the Docker build itself uses `docker buildx build --cache-from/--cache-to=type=local` under `.docker-build-cache/` (mirroring CI's `type=gha`) so repeat local builds are fast even when a layer must re-execute. Local wrappers export cache to `.docker-build-cache.new` and atomically replace `.docker-build-cache` afterward — writing `cache-to` into the same directory used for `cache-from` can fail with `mkdir: cannot create directory ''` while buildx finalizes the local OCI cache. A named Docker volume (`whisper-pro-asr-tool-cache`) persists ESLint/Stylelint/ruff/pytest run-time caches across separate local `docker run` invocations (build-time cache mounts alone never reach the running container). Requires a `docker-container`-driver buildx builder (created automatically by the wrapper scripts) since the default driver does not support local cache export.

**Hard requirement**: `.dockerignore` MUST exclude `.docker-build-cache`, `.docker-build-cache.new`, `.docker-build-cache.old`, and `.buildx-cache`. Those dirs live under the project root as BuildKit cache destinations; if they are not ignored, `docker buildx build ... .` recursively ships tens of GB of cache back into the build context (the exact failure mode that turns a normal build into a multi-minute 50GB+ context transfer).

### 3. Resolve Test Failures & Coverage

If tests fail:

- Test-stage order contract: `tests/run_suite.sh` finishes the full lint stage (including Radon complexity summary + rank-A enforcement) before any test stage. Within `python-tests`, pytest and coverage generation run only after lint has already passed.
- Radon source discovery in the test container must use filesystem enumeration (`find`) rather than `git ls-files`, because `.git` metadata is unavailable in Docker test images.
- Review `reports/pytest.xml` or `reports/coverage_output.txt` for specific test failures after Docker-backed parity runs. Quiet pytest progress (via `tests/class_progress.py` + xdist `--dist=loadscope`) prints each `module.py::TestClass` line **when that class finishes** (all of its results received on the controller), with one result character per test (`.`/`F`/`s`). It does not wait for an xdist worker to drain and does not dump unfinished groups at session end. Do not use `-v` in the pipeline — it dumps every node id.
- Fix broken assertions or environment-specific mocks.
- If coverage is below 90%, add missing test cases in the `tests/` directory to ensure all paths are verified before refactoring.
- Local parity pipeline contract: wrappers are fail-fast across the Docker test image build and execution path. If the image build or any in-container quality gate fails, the wrapper exits immediately with a non-zero status.
- Coverage badge generation is performed by the `genbadge` library from `coverage.xml` and must fail hard if the output badge is missing or empty.
- Wrapper scripts (`scripts/ci/build-and-test.sh` and `scripts/ci/build-and-test.ps1`) regenerate the badge from the latest `coverage.xml` only after the Docker test image run succeeds, before printing completion.

### 4. Resolve Linting Issues

Once the system is stable and covered by tests:

- **Requirement**: Run `npm run fix:md` when Markdown formatting drift exists, then pass `npm run lint:md`.
- **Requirement**: Achieve a Pylint score of **10.0/10.0**.
- **Requirement**: Pass Flake8 checks on `modules`, `whisper_pro_asr.py`, `tests`, and `tests/check_coverage.py`.
- **Constraint**: Do NOT ignore or disable lints unless absolutely necessary for architectural reasons.
- Fix import errors, remove unused code, and enforce PEP8 standards across all modules and tests.
- Fix Markdown lint issues via the repo-configured markdownlint auto-fix flow before resorting to manual cleanup.
- Note: High coverage ensures that lint-driven refactoring does not introduce regressions.

### 5. Verify New Features

After implementing changes, verify the following areas have test coverage:

- **Speaker Diarization**: `tests/inference/pipeline/test_diarization.py` — WhisperX orchestration, caching, fallbacks.
- **ASR Improvements**: `tests/inference/test_improvements.py` — parameter forwarding, idle timeout, subtitle wrapping.
- **Priority Concurrency**: `tests/inference/scheduler/priority/test_priority_concurrency.py`, `tests/inference/scheduler/priority/test_priority_concurrency_core_tests.py`, `tests/inference/scheduler/priority/test_priority_concurrency_extended_tests.py` — hardware pool configurations, yielding, and targeted preemption regressions.

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
