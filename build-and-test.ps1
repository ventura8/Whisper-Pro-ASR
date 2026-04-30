# CI-equivalent: build and run Docker-based pipeline with full caching.
# Run this in PowerShell from the project root.
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = $PSScriptRoot
Set-Location $root

Write-Host "`n--- Running Static Analysis (Linting) ---"
# Build the lint stage specifically. This uses Docker layer caching for both dependencies and results.
docker build -f Dockerfile.test --target lint -t whisper-pro-asr-lint .

Write-Host "`n--- Building Test Image ---"
# Build the final test image.
docker build -f Dockerfile.test --target test -t whisper-pro-asr-test .

Write-Host "`n--- Execute Test Suite ---"
New-Item -ItemType Directory -Force -Path assets | Out-Null
docker run --rm `
  -e SKIP_LINT=1 `
  -v "${root}/assets:/app/assets" `
  -v "${root}:/reports" `
  whisper-pro-asr-test /bin/bash -c "tests/run_suite.sh && cp coverage.xml /reports/coverage.xml && cp coverage_output.txt /reports/coverage_output.txt && cp pytest.xml /reports/pytest.xml"

Write-Host "`n--- Done ---"
