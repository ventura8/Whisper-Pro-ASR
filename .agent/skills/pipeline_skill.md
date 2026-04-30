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

### 2. Resolve Test Failures & Coverage
If the `whisper-pro-asr-test` stage fails:
- Review `pytest.xml` or `coverage_output.txt` for specific test failures.
- Fix broken assertions or environment-specific mocks.
- If coverage is below 90%, add missing test cases in the `tests/` directory to ensure all paths are verified before refactoring.

### 3. Resolve Linting Issues
Once the system is stable and covered by tests:
- **Requirement**: Achieve a Pylint score of **10.0/10.0**.
- **Constraint**: Do NOT ignore or disable lints unless absolutely necessary for architectural reasons.
- Fix import errors, remove unused code, and enforce PEP8 standards across all modules and tests.
- Note: High coverage ensures that lint-driven refactoring does not introduce regressions.

## Execution
To run this skill, execute the following command in the terminal:
`powershell -ExecutionPolicy Bypass -File .\build-and-test.ps1`
