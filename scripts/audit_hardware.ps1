<#
.SYNOPSIS
    Audit the Windows / Docker Desktop (WSL2) host's accelerators BEFORE choosing a
    build target or claiming any hardware validation.

.DESCRIPTION
    Prints what the machine actually has, then recommends a BUILD_TARGET and the
    matching docker-compose.<target>.yml override.

    A target whose accelerator is absent may only be reported as "boots and falls
    back to CPU cleanly" -- never as hardware-validated. On WSL2 the Intel NPU
    (/dev/accel) is not exposed to Linux containers, so NPU paths are untestable.

.PARAMETER Env
    Append / update BUILD_TARGET (and HOST_INTEL_RENDER_GID) in .env.

.PARAMETER Json
    Emit a machine-readable summary instead of the report.

.EXAMPLE
    ./scripts/audit_hardware.ps1
    ./scripts/audit_hardware.ps1 -Env
#>
[Diagnostics.CodeAnalysis.SuppressMessageAttribute(
    'PSAvoidUsingWriteHost', '',
    Justification = 'This script is an operator-facing console tool: the coloured, immediate output IS its interface. Write-Output would put the report on the success stream, where a caller capturing the result would collect the prose along with it.'
)]
[CmdletBinding()]
param(
    [switch]$Env,
    [switch]$Json
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Have($name) { $null -ne (Get-Command $name -ErrorAction SilentlyContinue) }
function Report($msg) { if (-not $Json) { Write-Host $msg } }
function Section($t) { if (-not $Json) { Write-Host "`n=== $t ===" } }

$hasNvidia = $false
$hasNvidiaToolkit = $false
$hasIntelGpu = $false
$hasAmd = $false
$renderGid = ""
# Whether the render node in the Docker VM is confirmed Intel silicon, as opposed to
# merely present. Only this gates an Intel target.
$hasIntelRenderNode = $false
$diskFree = ""

Section "GPUs (Windows device inventory)"
try {
    $vids = Get-CimInstance Win32_VideoController -ErrorAction Stop
    foreach ($v in $vids) {
        Report ("  {0}  (driver {1})" -f $v.Name, $v.DriverVersion)
        switch -Regex ($v.Name) {
            'NVIDIA'        { $hasNvidia = $true }
            'Intel'         { $hasIntelGpu = $true }
            'AMD|Radeon'    { $hasAmd = $true }
        }
    }
} catch {
    Report "  Win32_VideoController query failed: $($_.Exception.Message)"
}

Section "NVIDIA (CUDA)"
if (Have nvidia-smi) {
    $hasNvidia = $true
    Report (& nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>$null | Out-String).Trim()
} else {
    Report "nvidia-smi not on PATH"
}

Section "Docker GPU wiring (WSL2 / Docker Desktop)"
if (Have docker) {
    try {
        & docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi -L 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) {
            $hasNvidiaToolkit = $true
            Report "Docker GPU probe succeeded"
        } else {
            Report "Docker GPU probe failed -- enable GPU support in Docker Desktop / WSL2"
        }
        # Probe the render node visible inside the Docker Desktop WSL VM.
        $gid = & docker run --rm alpine sh -c "stat -c '%g' /dev/dri/renderD128 2>/dev/null" 2>$null
        if ($LASTEXITCODE -eq 0 -and $gid) {
            $renderGid = $gid.Trim()
            # The node's DRM vendor id, not merely its existence. AMD and NVIDIA publish
            # render nodes too, so treating any node as Intel evidence selected the
            # nvidia-intel image on hosts with no Intel silicon reachable at all.
            $vendor = & docker run --rm alpine sh -c "cat /sys/class/drm/renderD128/device/vendor 2>/dev/null" 2>$null
            $renderVendor = if ($LASTEXITCODE -eq 0 -and $vendor) { $vendor.Trim() } else { "" }
            if ($renderVendor -eq '0x8086') {
                $hasIntelRenderNode = $true
                Report "  /dev/dri/renderD128 in the Docker VM -> GID $renderGid  vendor=$renderVendor (Intel; HOST_INTEL_RENDER_GID)"
            } elseif ($renderVendor) {
                Report "  /dev/dri/renderD128 in the Docker VM -> GID $renderGid  vendor=$renderVendor (not Intel)"
            } else {
                Report "  /dev/dri/renderD128 in the Docker VM -> GID $renderGid  (vendor unreadable; not treated as Intel)"
            }
        } else {
            Report "  no /dev/dri/renderD128 inside the Docker VM (expected on WSL2 without GPU paravirt)"
        }
    } catch {
        Report "docker probe failed: $($_.Exception.Message)"
    }
} else {
    Report "docker not on PATH"
}

Section "Disk space"
try {
    $sys = Get-PSDrive -Name ($ENV:SystemDrive.TrimEnd(':')) -ErrorAction Stop
    $diskFree = "{0:N1} GB" -f ($sys.Free / 1GB)
    Report "free on $($ENV:SystemDrive) : $diskFree"
} catch { Report "could not determine free disk space" }

# --- Recommendation ----------------------------------------------------------
$target = "cpu"
# Intel selection is gated on a *confirmed Intel* render node in the Docker VM, not on the
# Windows device inventory and not on a render node of any vendor. An Intel iGPU always
# appears in Win32_VideoController, but the container stack is Linux: without an Intel
# /dev/dri node inside the Docker VM the intel image has no device to reach and runs on the
# CPU while its name promises otherwise. Conversely an AMD or NVIDIA render node is not
# Intel evidence, and treating it as such selected nvidia-intel on NVIDIA+AMD hosts.
#
# There is deliberately no `amd` branch. This script audits a Windows/Docker Desktop host,
# where ROCm has no path into a Linux container at all: /dev/kfd is not exposed and
# /dev/dxg only makes the adapter visible. Recommending `amd` there built a ~14GB image
# whose every request ran on the CPU while the target name claimed otherwise -- so an AMD
# card falls through to `cpu`, which is what actually executes. scripts/audit_hardware.sh
# still selects `amd`, because on Linux it is real.
if ($hasNvidia -and $hasNvidiaToolkit -and $hasIntelRenderNode) {
    $target = "nvidia-intel"
} elseif ($hasNvidia -and $hasNvidiaToolkit) {
    $target = "nvidia"
} elseif ($hasIntelRenderNode) {
    $target = "intel"
}

if ($Json) {
    [pscustomobject]@{
        nvidia             = $hasNvidia
        nvidia_toolkit     = $hasNvidiaToolkit
        intel_gpu          = $hasIntelGpu
        intel_render_node  = $hasIntelRenderNode
        amd                = $hasAmd
        render_gid         = $renderGid
        disk_free          = $diskFree
        recommended_target = $target
    } | ConvertTo-Json -Compress
    return
}

Section "Recommendation"
Report "BUILD_TARGET=$target"
Report "docker compose -f docker-compose.yml -f docker-compose.$target.yml up -d --build"
if (($target -eq "intel" -or $target -eq "nvidia-intel") -and $renderGid) {
    Report "HOST_INTEL_RENDER_GID=$renderGid"
}
Report "Intel NPU (/dev/accel) is not exposed to WSL2 containers: do NOT claim NPU validation on this host."
if ($hasAmd) {
    Report "An AMD GPU is present, but ROCm has no route into a Linux container on Windows;"
    Report "the cpu target is what actually runs. Use a native Linux host to validate AMD."
}
if ($hasNvidia -and -not $hasNvidiaToolkit) {
    Report "Docker cannot currently access the NVIDIA GPU; do not claim CUDA validation."
}
Report ""
Report "Validate for real (against a running stack):"
Report "  RUN_REAL_ASR=1 python3 -m pytest tests/integration/test_transcription_accuracy.py"
Report "A correct transcript proves decoding, not acceleration -- pair with nvidia-smi"
Report "--query-compute-apps (CUDA) or intel_gpu_top (Intel) evidence."

if ($Env) {
    if (-not (Test-Path .env)) { New-Item -ItemType File -Path .env | Out-Null }
    $lines = @(Get-Content .env -ErrorAction SilentlyContinue)
    # @() around the filter: Where-Object unrolls to a scalar when one item survives and to
    # $null when none do, and `$scalar += "..."` then makes a two-element array from a
    # string -- or, from $null, concatenates into ONE string. Set-Content wrote that single
    # joined line, collapsing a .env into an unparseable line of run-together settings.
    $lines = @($lines | Where-Object { $_ -notmatch '^(BUILD_TARGET|HOST_INTEL_RENDER_GID)=' })
    $lines += "BUILD_TARGET=$target"
    if (($target -eq "intel" -or $target -eq "nvidia-intel") -and $renderGid) {
        $lines += "HOST_INTEL_RENDER_GID=$renderGid"
    }
    Set-Content -Path .env -Value $lines
    Write-Host "`nUpdated .env:"
    Get-Content .env | Where-Object { $_ -match '^(BUILD_TARGET|HOST_INTEL_RENDER_GID)=' }
}
