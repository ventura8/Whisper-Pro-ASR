# CI-equivalent: build and run Docker-based pipeline with full caching.
# Run this in PowerShell from the project root.
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = $PSScriptRoot
Set-Location $root

# Auto-detect if sudo is needed for Docker commands on Unix platforms
# We check version to safely reference $IsLinux/$IsMacOS under Set-StrictMode
$isUnix = $false
if ($PSVersionTable.PSVersion.Major -ge 6) {
    $isUnix = $IsLinux -or $IsMacOS
}

$dockerExe = "docker"
if ($isUnix) {
    $processInfo = New-Object System.Diagnostics.ProcessStartInfo
    $processInfo.FileName = "docker"
    $processInfo.Arguments = "ps"
    $processInfo.RedirectStandardError = $true
    $processInfo.UseShellExecute = $false
    try {
        $process = [System.Diagnostics.Process]::Start($processInfo)
        $process.WaitForExit()
        $stdErr = $process.StandardError.ReadToEnd()
        if ($stdErr -like "*permission denied*") {
            if (Get-Command sudo -ErrorAction SilentlyContinue) {
                $dockerExe = "sudo"
                Write-Host "Note: Prepended 'sudo' to docker commands because of permission check on /var/run/docker.sock"
            }
        }
    } catch {}
}

function Invoke-Docker {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )
    if ($dockerExe -eq "sudo") {
        & sudo docker @Arguments
    } else {
        & docker @Arguments
    }
}

Write-Host "`n--- Running Static Analysis (Linting) ---"
# Build the lint stage specifically. This uses Docker layer caching for both dependencies and results.
Invoke-Docker -- build -f Dockerfile.test --target lint -t whisper-pro-asr-lint .

Write-Host "`n--- Building Test Image ---"
# Build the final test image.
Invoke-Docker -- build -f Dockerfile.test --target test -t whisper-pro-asr-test .

Write-Host "`n--- Execute Test Suite ---"
New-Item -ItemType Directory -Force -Path assets | Out-Null
Invoke-Docker -- run --rm `
  -e SKIP_LINT=1 `
  -v "${root}/assets:/app/assets" `
  -v "${root}:/reports" `
  whisper-pro-asr-test /bin/bash -c "tests/run_suite.sh && cp coverage.xml /reports/coverage.xml && cp coverage_output.txt /reports/coverage_output.txt && cp pytest.xml /reports/pytest.xml"

Write-Host "`n--- Done ---"
