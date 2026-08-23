# CI-equivalent: build and run Docker-based pipeline with full caching.
# Run this in PowerShell from the project root.
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if ($PSVersionTable.PSVersion.Major -ge 7) {
    $PSNativeCommandUseErrorActionPreference = $false
}

# Load shared dependencies
$depsFile = Join-Path $PSScriptRoot "dependencies.env"
if (Test-Path $depsFile) {
    Get-Content $depsFile | Foreach-Object {
        if ($_ -match '^\s*([^#=\s]+)\s*=\s*(.*?)\s*$') {
            $name = $Matches[1]
            $value = $Matches[2]
            Set-Variable -Name $name -Value $value -Scope Script
        }
    }
} else {
    throw "Dependencies configuration file not found at $depsFile"
}

$root = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
Set-Location $root

function Install-WithApt {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Packages
    )
    if (-not (Get-Command apt-get -ErrorAction SilentlyContinue)) {
        return $false
    }

    if (Get-Command sudo -ErrorAction SilentlyContinue) {
        & sudo -n apt-get update
        $updateExit = $LASTEXITCODE
        if ($updateExit -ne 0) {
            return $false
        }

        & sudo -n apt-get install -y @Packages
        $installExit = $LASTEXITCODE
        return ($installExit -eq 0)
    }

    & apt-get update
    $updateExit = $LASTEXITCODE
    if ($updateExit -ne 0) {
        return $false
    }

    & apt-get install -y @Packages
    $installExit = $LASTEXITCODE
    return ($installExit -eq 0)
}

function Invoke-CommandBootstrap {
    param(
        [Parameter(Mandatory = $true)]
        [string]$CommandName,
        [Parameter(Mandatory = $true)]
        [string[]]$AptPackages
    )

    if (Get-Command $CommandName -ErrorAction SilentlyContinue) {
        return
    }

    Write-Output "Dependency '$CommandName' is missing. Attempting auto-install..."
    $installed = Install-WithApt -Packages $AptPackages
    if (-not $installed -or -not (Get-Command $CommandName -ErrorAction SilentlyContinue)) {
        throw "Required dependency '$CommandName' is missing and could not be auto-installed. Install it manually and rerun."
    }
}

Invoke-CommandBootstrap -CommandName docker -AptPackages @("docker.io")

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
        # Read stderr before WaitForExit to prevent pipe-buffer deadlock
        $null = $process.StandardError.ReadToEnd()
        $process.WaitForExit()
        if ($process.ExitCode -ne 0) {
            if (Get-Command sudo -ErrorAction SilentlyContinue) {
                $sudoProcessInfo = New-Object System.Diagnostics.ProcessStartInfo
                $sudoProcessInfo.FileName = "sudo"
                $sudoProcessInfo.Arguments = "-n docker ps"
                $sudoProcessInfo.RedirectStandardError = $true
                $sudoProcessInfo.UseShellExecute = $false
                $sudoProcess = [System.Diagnostics.Process]::Start($sudoProcessInfo)
                # Read stderr before WaitForExit to prevent pipe-buffer deadlock
                $null = $sudoProcess.StandardError.ReadToEnd()
                $sudoProcess.WaitForExit()
                if ($sudoProcess.ExitCode -eq 0) {
                    $dockerExe = "sudo"
                    Write-Output "Note: Prepended 'sudo' to docker commands because of permission check on /var/run/docker.sock"
                } else {
                    throw "Docker permission check failed (docker ps exited with non-zero, and sudo docker ps also failed)."
                }
            } else {
                throw "Docker permission check failed (docker ps exited with non-zero, and sudo is not available)."
            }
        }
    } catch {
        throw "Failed to verify Docker access: $_"
    }
}

function Invoke-Docker {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        if ($dockerExe -eq "sudo") {
            & sudo docker @Arguments
        } else {
            & docker @Arguments
        }
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }

    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "Docker command failed with exit code ${exitCode}: docker $($Arguments -join ' ')"
    }

    return $exitCode
}

# The default buildx driver ("docker") does not support --cache-to=type=local;
# only the docker-container driver does. Create/reuse a dedicated builder so
# local iterative builds get the same fast-rebuild cache-export property CI
# already gets via cache-to=type=gha.
$buildxBuilderName = "whisper-pro-asr-builder"
function Initialize-BuildxBuilder {
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    if ($dockerExe -eq "sudo") {
        & sudo docker buildx inspect $buildxBuilderName *> $null
    } else {
        & docker buildx inspect $buildxBuilderName *> $null
    }
    $inspectExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference

    if ($inspectExitCode -eq 0) {
        Invoke-Docker -Arguments @("buildx", "use", $buildxBuilderName)
    } else {
        Invoke-Docker -Arguments @("buildx", "create", "--name", $buildxBuilderName, "--driver", "docker-container", "--use")
    }
}
Initialize-BuildxBuilder

# Named volume persisting tool run-time caches (ruff/pylint/eslint/stylelint/
# pytest) across separate `docker run` invocations on this machine -- build-time
# cache mounts alone never reach the running container.
$toolCacheVolume = "whisper-pro-asr-tool-cache"

function Invoke-PoetryLockSync {
    Write-Output "`n--- Verifying Poetry Lock File ---"

    if (-not (Test-Path "poetry.lock")) {
        New-Item -ItemType File -Path "poetry.lock" | Out-Null
    }

    $arguments = @("run", "--rm")
    if ($isUnix) {
        $uid = (& id -u).Trim()
        $gid = (& id -g).Trim()
        if ($LASTEXITCODE -eq 0 -and $uid -and $gid) {
            $arguments += @("--user", "${uid}:${gid}")
        }
    }

    $arguments += @(
        "-v", "${root}/pyproject.toml:/workspace/pyproject.toml",
        "-v", "${root}/poetry.lock:/workspace/poetry.lock",
        "-w", "/workspace",
        $PYTHON_IMAGE,
        "/bin/bash",
        "-lc",
        "export HOME=/tmp && export PATH=`"/tmp/.local/bin:`$PATH`" && export PIP_ROOT_USER_ACTION=ignore && export PIP_NO_WARN_SCRIPT_LOCATION=1 && python -m pip install --quiet --user --upgrade `"pip==$PIP_VERSION`" && python -m pip install --quiet --user `"poetry==$POETRY_VERSION`" && python -m poetry config virtualenvs.create false && if [ ! -f poetry.lock ] || ! python -m poetry check --lock >/dev/null 2>&1; then python -m poetry lock --no-interaction; fi"
    )

    Invoke-Docker -- @arguments
}

Invoke-PoetryLockSync
Write-Output "`n--- Building Test Image ---"
# Build the final test image, exporting/importing a local BuildKit cache so
# repeat builds on this machine are fast even when a layer must re-execute.
# Export to a fresh dest, then atomically replace — writing cache-to the same
# directory used for cache-from can fail with mkdir '' while finalizing ingest.
$dockerBuildCacheDir = Join-Path $root ".docker-build-cache"
$dockerBuildCacheDirNew = Join-Path $root ".docker-build-cache.new"
$dockerBuildCacheDirOld = Join-Path $root ".docker-build-cache.old"
if (Test-Path $dockerBuildCacheDirNew) {
    Remove-Item -Recurse -Force $dockerBuildCacheDirNew
}
New-Item -ItemType Directory -Force -Path $dockerBuildCacheDir | Out-Null
New-Item -ItemType Directory -Force -Path $dockerBuildCacheDirNew | Out-Null
$testBuildArgs = @(
    "buildx", "build", "--progress=plain",
    "-f", "Dockerfile.test", "--target", "test", "-t", "whisper-pro-asr-test",
    "--cache-from", "type=local,src=$dockerBuildCacheDir",
    "--cache-to", "type=local,dest=$dockerBuildCacheDirNew,mode=max",
    "--load", "."
)
Invoke-Docker -Arguments $testBuildArgs
if (Test-Path $dockerBuildCacheDirOld) {
    Remove-Item -Recurse -Force $dockerBuildCacheDirOld
}
if (Test-Path $dockerBuildCacheDir) {
    Rename-Item -Path $dockerBuildCacheDir -NewName ".docker-build-cache.old"
}
Rename-Item -Path $dockerBuildCacheDirNew -NewName ".docker-build-cache"
if (Test-Path $dockerBuildCacheDirOld) {
    Remove-Item -Recurse -Force $dockerBuildCacheDirOld
}

Write-Output "`n--- Execute Test Suite ---"
New-Item -ItemType Directory -Force -Path assets | Out-Null
$reportsDir = Join-Path $root "reports"
New-Item -ItemType Directory -Force -Path $reportsDir | Out-Null
$reportFiles = @(
    "coverage.xml",
    "coverage_output.txt",
    "complexity_output.txt",
    "pytest.xml"
)
foreach ($reportFile in $reportFiles) {
    $reportPath = Join-Path $reportsDir $reportFile
    if (Test-Path $reportPath) {
        Remove-Item -Force $reportPath
    }
}
if ($isUnix) {
    & chmod 0777 assets $reportsDir
    $null = $LASTEXITCODE
}
$runArgs = @(
    "run",
    "--rm",
    "-e", "CI=true",
    "-v", "${root}/assets:/app/assets",
    "-v", "${reportsDir}:/reports",
    "-v", "${toolCacheVolume}:/var/cache/whisper-pro-asr-tools",
    "whisper-pro-asr-test",
    "/bin/bash",
    "-lc",
    'bash tests/run_suite.sh; TEST_EXIT_CODE=$?; [ -f coverage.xml ] && cp coverage.xml /reports/coverage.xml || true; [ -f coverage_output.txt ] && cp coverage_output.txt /reports/coverage_output.txt || true; [ -f complexity_output.txt ] && cp complexity_output.txt /reports/complexity_output.txt || true; [ -f pytest.xml ] && cp pytest.xml /reports/pytest.xml || true; exit "$TEST_EXIT_CODE"'
)
Invoke-Docker -Arguments $runArgs

Write-Output "`n--- Regenerating Coverage Badge (Mandatory Final Stage) ---"
$badgeArgs = @(
    "run",
    "--rm",
    "-v", "${root}/assets:/app/assets",
    "-v", "${reportsDir}:/reports",
    "-v", "${toolCacheVolume}:/var/cache/whisper-pro-asr-tools",
    "whisper-pro-asr-test",
    "/bin/bash",
    "-lc",
    'test -s /reports/coverage.xml || { echo missing_coverage_xml; exit 1; }; genbadge coverage -i /reports/coverage.xml -o /app/assets/coverage.svg'
)
Invoke-Docker -Arguments $badgeArgs

$badgePath = Join-Path $root "assets/coverage.svg"
if (-not (Test-Path $badgePath)) {
    throw "Mandatory coverage badge is missing at assets/coverage.svg"
}
$badgeInfo = Get-Item $badgePath
if ($badgeInfo.Length -le 0) {
    throw "Mandatory coverage badge is empty at assets/coverage.svg"
}

Write-Output "`n--- Cyclomatic Complexity Summary (Radon cc) ---"
$complexityFile = Join-Path $reportsDir "complexity_output.txt"
if (Test-Path $complexityFile) {
    Get-Content $complexityFile | Write-Output
}

Write-Output "`n--- Code Coverage Summary ---"
$covFile = Join-Path $reportsDir "coverage_output.txt"
if (Test-Path $covFile) {
    $covContent = Get-Content $covFile
    $printing = $false
    foreach ($line in $covContent) {
        if ($line -match "---------- coverage") { $printing = $true }
        if ($printing) { Write-Output $line }
        if ($line -match "TOTAL" -and $printing) { $printing = $false }
    }
}

Write-Output "`n--- Done ---"
