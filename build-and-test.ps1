# CI-equivalent: install linting tools, run YAML/Python lint, build test image, run test suite.
# Run this in PowerShell from the project root.
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = $PSScriptRoot
Set-Location $root

Write-Host "`n--- Installing Linting Tools ---"
python -m pip install pylint yamllint

Write-Host "`n--- YAML Lint ---"
python -m yamllint -c .yamllint .

Write-Host "`n--- Pylint Lint ---"
python -m pylint modules tests whisper_pro_asr.py --recursive=y --disable=import-error

Write-Host "`n--- Building Test Image ---"
docker build -f Dockerfile.test -t whisper-pro-asr-test .

Write-Host "`n--- Execute Test Suite ---"
New-Item -ItemType Directory -Force -Path assets | Out-Null
docker run --rm `
  -e SKIP_LINT=1 `
  -v "${root}/assets:/app/assets" `
  -v "${root}:/reports" `
  whisper-pro-asr-test /bin/bash -c "tests/run_suite.sh && cp coverage.xml /reports/coverage.xml && cp coverage_output.txt /reports/coverage_output.txt && cp pytest.xml /reports/pytest.xml"

Write-Host "`n--- Done ---"
