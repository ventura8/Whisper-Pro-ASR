# Audit a Windows host for what can actually be validated, and print one JSON line.
#
# Counterpart to scripts/audit_hardware.sh (which audits Linux hosts). Invoked over SSH by
# scripts/remote_validate.sh via -EncodedCommand, because Windows sshd hands commands to
# cmd.exe, where quoting a non-trivial PowerShell expression is unreliable -- base64 makes
# the quoting question disappear entirely.
#
# What matters on Windows is narrower than it looks. The container stack is Linux, so
# every accelerator claim depends on what reaches a Linux container through WSL2:
#   NVIDIA - yes, via the WSL CUDA driver.
#   AMD    - the adapter is visible and /dev/dxg exists, but ROCm's WSL support covers
#            only specific discrete cards; an integrated Radeon falls back to CPU while
#            still looking present. Reported, never assumed.
#   Intel  - GPU and NPU are not exposed to Linux containers on Windows at all.

$ErrorActionPreference = 'SilentlyContinue'

$gpus = @(Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name })
$nvidia = @($gpus -match 'NVIDIA').Count -gt 0
$amd    = @($gpus -match 'AMD|Radeon').Count -gt 0
$intel  = @($gpus -match 'Intel').Count -gt 0

# An integrated Radeon and a discrete one are the same string to WMI in many builds, so
# distinguish by dedicated video memory rather than by name.
$amdDiscrete = $false
foreach ($c in Get-CimInstance Win32_VideoController) {
    if ($c.Name -match 'AMD|Radeon' -and $c.AdapterRAM -gt 2GB) { $amdDiscrete = $true }
}

$env:WSL_UTF8 = '1'
$distros = @()
# A failed enumeration and a host with no distro both leave $distros empty, and every probe
# below is then skipped -- reporting "no accelerator" for a machine whose GPU may be fine.
# The failure is recorded so the caller can tell the two apart.
$wslError = ''
try {
    # Native stderr and the exit code, not just a thrown exception. wsl.exe reports most
    # failures -- WSL not installed, the service stopped, a corrupt distro registration --
    # by writing to stderr and exiting non-zero, without raising anything PowerShell would
    # catch. Only the catch block was consulted, so every one of those produced an empty
    # distro list, an empty $wslError, and the misleading "returned no distro" below.
    $wslStdErr = ''
    $rawDistros = (wsl -l -q 2>&1 | ForEach-Object {
        if ($_ -is [System.Management.Automation.ErrorRecord]) { $wslStdErr += "$_ "; } else { $_ }
    })
    $wslExit = $LASTEXITCODE
    $distros = @($rawDistros |
        ForEach-Object { $_ -replace "`0", '' } |
        Where-Object { $_ -and $_.Trim() } |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ -notmatch '^docker-desktop' })
    if ($wslExit -ne 0) {
        $wslError = "wsl -l -q exited $wslExit$(if ($wslStdErr) { ": $($wslStdErr.Trim())" })"
    }
} catch {
    $wslError = $_.Exception.Message
}
# Only after a *successful* enumeration does an empty list actually mean "no distro".
if (-not $distros -and -not $wslError) { $wslError = 'wsl -l -q succeeded but listed no distro' }

$distro = if ($distros.Count) { $distros[0] } else { '' }
$docker = ''
$dxg = $false
$nvidiaSmi = ''
$diskFree = ''
if ($distro) {
    # Separate round trips rather than one clever compound command: $( ), parentheses and
    # && do not survive cmd.exe -> wsl -> bash intact, and a mangled probe reports "absent"
    # for hardware that is present -- the most expensive kind of wrong answer here.
    $docker    = (wsl -d $distro -e docker info --format '{{.ServerVersion}}' 2>$null | Select-Object -First 1)
    $dxgOut    = (wsl -d $distro -e ls /dev/dxg 2>$null | Select-Object -First 1)
    $dxg       = [bool]$dxgOut
    # A login shell, not a direct exec. WSL exposes the GPU driver through
    # /usr/lib/wsl/lib, which reaches PATH from the distro's profile -- so `wsl -e
    # nvidia-smi` fails with "execvpe(nvidia-smi) failed: No such file or directory" on a
    # machine whose GPU works perfectly. Verified on Sergiu-PC with an RTX 5090: direct
    # exec reported nothing and the audit recommended the `cpu` target, while
    # `bash -lc` returned "NVIDIA GeForce RTX 5090". That is the expensive wrong answer
    # this file warns about two lines above -- absent hardware that is present.
    #
    # `bash -lc "..."` with one quoted command still survives cmd.exe -> wsl intact; what
    # does not survive is $( ), parentheses and &&, so keep the probes separate.
    $nvidiaSmi = (wsl -d $distro -e bash -lc "nvidia-smi --query-gpu=name --format=csv,noheader" 2>$null | Select-Object -First 1)
    $diskFree  = (wsl -d $distro -e df -BG --output=avail / 2>$null | Select-Object -Last 1)
    if ($diskFree) { $diskFree = $diskFree.Trim() }
    if ($nvidiaSmi) { $nvidiaSmi = $nvidiaSmi.Trim() }
    if ($docker) { $docker = $docker.Trim() }
}
# Docker prints a multi-line "could not be found" advert on failure; that is not a version.
if ($docker -notmatch '^\d+\.\d+') { $docker = '' }

# Only NVIDIA survives the trip into a Linux container here, so it is the only target that
# can support an accelerator claim. Everything else means "boots and falls back to CPU".
#
# nvidia-smi inside the distro is necessary but not sufficient: it proves the WSL CUDA
# driver is present in the distro, not that Docker can hand a GPU to a container. With
# Docker Desktop's GPU support off, or no distro integration, the probe below fails while
# nvidia-smi still answers -- and a `nvidia` target recommended on that basis builds a
# 17.5GB image that runs every request on the CPU.
#
# The probe must carry the DOCKER_CONFIG workaround. Docker Desktop configures
# credsStore=desktop.exe, a Windows credential helper that needs an interactive logon
# session; over SSH there is none, so pulling the CUDA image dies with
#
#     error getting credentials - ... A specified logon session does not exist.
#
# Measured on this machine: without the workaround the probe returned nothing and the audit
# recommended `cpu` for a host whose RTX 5090 answers `docker run --gpus all` perfectly.
# That is the same expensive wrong answer -- absent hardware that is present -- this file
# warns about at the top, arrived at from the other direction.
#
# Everything pulled here is public, so point DOCKER_CONFIG at a helper-free config rather
# than editing the user's own. Composed in PowerShell rather than quoted through cmd.exe:
# this script is copied over and run as a file, so PowerShell owns the quoting.
$dockerGpu = ''
$dockerGpuError = ''
if ($distro -and $docker) {
    $prelude = 'export DOCKER_CONFIG="$HOME/.docker-wpa"; mkdir -p "$DOCKER_CONFIG"; printf %s "{}" > "$DOCKER_CONFIG/config.json"; '
    $probe = (wsl -d $distro -e bash -lc "$prelude docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi -L 2>&1")
    $probeText = ($probe | Out-String).Trim()
    # nvidia-smi -L prints one "GPU <n>: <name> (UUID: ...)" line per device. Matching that
    # shape, rather than trusting the exit code, keeps a successful pull with no devices
    # from reading as a pass.
    $gpuLine = ($probe | Where-Object { $_ -match '^GPU \d+:' } | Select-Object -First 1)
    if ($gpuLine) {
        $dockerGpu = $gpuLine.Trim()
    } else {
        # Recorded rather than discarded: "Docker cannot reach the GPU" and "the probe could
        # not run" lead to different fixes, and collapsing both into an empty string is what
        # made a credentials failure look like missing hardware.
        $dockerGpuError = if ($probeText) { ($probeText -split "`r?`n" | Select-Object -Last 1).Trim() } else { 'probe produced no output' }
    }
}

# Both conditions, deliberately. nvidia-smi in the distro proves the WSL CUDA driver is
# there; only the container probe proves Docker can hand a GPU to a container. With Docker
# Desktop's GPU support off, nvidia-smi still answers -- and a `nvidia` target recommended
# on that basis builds a 17.5GB image that runs every request on the CPU.
$target = if ($nvidiaSmi -and $dockerGpu) { 'nvidia' } else { 'cpu' }

# Hand-assembling JSON with string concatenation produced malformed output (a stray comma
# after every colon); ConvertTo-Json removes the entire class of quoting and escaping bug.
# -Compress keeps it to the single line the callers parse.
$result = [ordered]@{
    os                 = 'windows'
    nvidia             = $nvidia
    amd                = $amd
    amd_discrete       = $amdDiscrete
    intel_gpu          = $intel
    gpus               = @($gpus | Where-Object { $_ })
    wsl_distro         = "$distro"
    wsl_error          = "$wslError"
    wsl_docker         = "$docker"
    wsl_dxg            = $dxg
    wsl_nvidia_smi     = "$nvidiaSmi"
    wsl_docker_gpu     = "$dockerGpu"
    wsl_docker_gpu_error = "$dockerGpuError"
    disk_free          = "$diskFree"
    recommended_target = "$target"
}
Write-Output ($result | ConvertTo-Json -Compress)
