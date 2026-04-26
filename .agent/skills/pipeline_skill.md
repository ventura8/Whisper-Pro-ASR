# Local Pipeline Execution Skill

This skill provides a structured workflow for running the Whisper Pro ASR CI/CD pipeline locally on a Windows host using Docker. It automates linting, testing, and coverage verification.

## Prerequisites
- Docker Desktop (Running)
- PowerShell 7+

## Workflow Steps

### 1. Execute Pipeline
Run the main build and test script:
```powershell
powershell -ExecutionPolicy Bypass -File .\build-and-test.ps1
```

### 2. Analyze Linting Failures
If the `whisper-pro-asr-lint` stage fails:
- Check the output for `Pylint` or `Yamllint` errors.
- Fix import errors (ensure `test_requirements.txt` is up to date).
- Fix formatting issues using `autopep8`.

### 3. Analyze Test Failures
If the `whisper-pro-asr-test` stage fails:
- Review `pytest.xml` or `coverage_output.txt` for specific test failures.
- Fix broken assertions or environment-specific mocks.

### 4. Verify Coverage
If the `check_coverage.py` script fails:
- Review `coverage_output.txt` for files below the 90% threshold.
- Add missing test cases in the `tests/` directory.

## Execution
To run this skill, execute the following command in the terminal:
`powershell -ExecutionPolicy Bypass -File .\build-and-test.ps1`
