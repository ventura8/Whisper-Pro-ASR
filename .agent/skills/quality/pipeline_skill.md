# Local Pipeline Execution Skill

This skill provides a structured workflow for running the Whisper Pro ASR CI/CD pipeline locally. It automates linting, testing, and coverage verification.

## Prerequisites

- Python 3.12+ (or Docker Desktop for containerized runs)
- PowerShell 7+ (Windows) or Bash (Linux/macOS)

## Workflow Steps

### 1. Execute Pipeline (Containerized Docker - Mandatory)

All lints, tests, security scans, audits, or type-checking MUST happen inside the Docker test image. Do not run checks on the host environment directly.

Run the main build and test script:

- Both local wrappers verify `poetry.lock` in the workspace before Docker builds, using a disposable Python container to regenerate it only when it is missing or stale.

- **Linux/macOS**:

  ```bash
  ./scripts/ci/build-and-test.sh
  ```

- **Windows**:

  ```powershell
  powershell -ExecutionPolicy Bypass -File .\scripts\ci\build-and-test.ps1
  ```

### 3. Resolve Test Failures & Coverage

If tests fail:

- Test-stage order contract: `tests/run_suite.sh` runs Radon complexity summary + rank-A enforcement before starting pytest and coverage generation.
- Radon source discovery in the test container must use filesystem enumeration (`find`) rather than `git ls-files`, because `.git` metadata is unavailable in Docker test images.
- Review `reports/pytest.xml` or `reports/coverage_output.txt` for specific test failures after Docker-backed parity runs.
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
