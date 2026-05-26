# Local Pipeline Execution Skill

This skill provides a structured workflow for running the Whisper Pro ASR CI/CD pipeline locally. It automates linting, testing, and coverage verification.

## Prerequisites
- Python 3.12+ (or Docker Desktop for containerized runs)
- PowerShell 7+ (Windows) or Bash (Linux/macOS)

## Workflow Steps

### 1. Execute Pipeline (Linux/macOS)
Run the test suite directly using pytest:
```bash
python3 -m pytest tests/
```
This will run all unit and integration tests, enforce >90% coverage, and output a coverage report.

### 2. Execute Pipeline (Windows/Docker)
Run the main build and test script:
```powershell
powershell -ExecutionPolicy Bypass -File .\build-and-test.ps1
```

### 3. Resolve Test Failures & Coverage
If tests fail:
- Review `pytest.xml` or `coverage_output.txt` for specific test failures.
- Fix broken assertions or environment-specific mocks.
- If coverage is below 90%, add missing test cases in the `tests/` directory to ensure all paths are verified before refactoring.

### 4. Resolve Linting Issues
Once the system is stable and covered by tests:
- **Requirement**: Achieve a Pylint score of **10.0/10.0**.
- **Constraint**: Do NOT ignore or disable lints unless absolutely necessary for architectural reasons.
- Fix import errors, remove unused code, and enforce PEP8 standards across all modules and tests.
- Note: High coverage ensures that lint-driven refactoring does not introduce regressions.

### 5. Verify New Features
After implementing changes, verify the following areas have test coverage:
- **Speaker Diarization**: `tests/inference/test_diarization.py` — WhisperX orchestration, caching, fallbacks.
- **ASR Improvements**: `tests/inference/test_improvements.py` — parameter forwarding, idle timeout, subtitle wrapping.
- **Priority Concurrency**: `tests/inference/test_priority_concurrency.py` — hardware pool configurations, yielding.

## Test Suite Structure
```text
tests/
├── inference/
│   ├── test_diarization.py          # Speaker diarization pipeline tests
│   ├── test_improvements.py         # ASR params, idle timeout, subtitle wrapping
│   ├── test_priority_concurrency.py # Multi-unit scheduling & preemption
│   └── ...
├── test_api_*.py                    # API route integration tests
├── test_config.py                   # Configuration resolution tests
└── ...
```

## Execution
To run this skill, execute:

**Linux/macOS:**
```bash
python3 -m pytest tests/
```

**Windows (Docker):**
```powershell
powershell -ExecutionPolicy Bypass -File .\build-and-test.ps1
```
