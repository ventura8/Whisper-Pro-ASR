# Local Test Runner with Coverage Badge Generation
# Usage: .\generate_badge.ps1

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Running Tests with Coverage" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Run pytest with coverage
python -m pytest --cov=modules --cov-report=term-missing --cov-report=xml --cov-fail-under=90

if ($LASTEXITCODE -ne 0) {
    Write-Host "Tests failed or coverage below 90%!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Generating Coverage Badge" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Generate the SVG badge
python -m coverage_badge -o assets/coverage.svg -f

if (Test-Path "assets/coverage.svg") {
    Write-Host "Coverage badge generated: assets/coverage.svg" -ForegroundColor Green
}
else {
    Write-Host "Failed to generate coverage badge!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Done! Coverage badge saved to assets/coverage.svg" -ForegroundColor Green
