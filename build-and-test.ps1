# CI-equivalent: build test image, run pylint, run test suite.
# Run this in PowerShell from the project root (e.g. outside Cursor terminal).
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = $PSScriptRoot
Set-Location $root

Write-Host "--- Building test image ---"
docker build -f Dockerfile.test -t whisper-npu-test .

Write-Host "`n--- Pylint ---"
docker run --rm whisper-npu-test python3 -m pylint modules tests whisper_server.py --recursive=y

Write-Host "`n--- Test suite ---"
New-Item -ItemType Directory -Force -Path assets | Out-Null
docker run --rm `
  -e SKIP_LINT=1 `
  -v "${root}/assets:/app/assets" `
  -v "${root}:/reports" `
  whisper-npu-test /bin/bash -c "tests/run_suite.sh && cp coverage.xml /reports/coverage.xml && cp coverage_output.txt /reports/coverage_output.txt"

Write-Host "`n--- Done ---"
