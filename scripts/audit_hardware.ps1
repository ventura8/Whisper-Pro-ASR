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
[CmdletBinding()]
param(
    # Named $WriteEnv rather than $Env: `$Env` sits one character from PowerShell's `$env:`
    # provider drive, and under Set-StrictMode the resemblance is a trap for anyone editing
    # this file. The alias keeps every documented `-Env` call site working unchanged.
    [Alias('Env')]
    [switch]$WriteEnv,
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

# Both probes below start a container, and a stalled Docker Desktop, a WSL backend mid-restart
# or an unreachable registry all block indefinitely -- with no output, because this script's
# whole contract is a report at the end. Callers (setup scripts, remote_validate.sh) then wait
# on an audit that never returns.
#
# Start-Process with a real wait, not Start-Job: killing a job leaves the docker.exe child
# alive, still holding the daemon that the next probe is about to ask for. Here the deadline
# belongs to the process that can actually be killed.
function Invoke-BoundedDocker {
    param([string[]]$DockerArgs, [int]$TimeoutSec = 300)

    $outFile = [System.IO.Path]::GetTempFileName()
    $errFile = [System.IO.Path]::GetTempFileName()
    try {
        $proc = Start-Process -FilePath docker -ArgumentList $DockerArgs -NoNewWindow -PassThru `
            -RedirectStandardOutput $outFile -RedirectStandardError $errFile
        if (-not $proc.WaitForExit($TimeoutSec * 1000)) {
            # Kill the tree where the runtime supports it; PowerShell 5.1's Process.Kill()
            # takes no argument and leaves grandchildren, which is better than leaving all of it.
            try { $proc.Kill($true) } catch { try { $proc.Kill() } catch { } }
            return [pscustomobject]@{ TimedOut = $true; ExitCode = -1; Output = @() }
        }
        $lines = @(Get-Content -Path $outFile -ErrorAction SilentlyContinue | Where-Object { $_ -and $_.Trim() })
        return [pscustomobject]@{ TimedOut = $false; ExitCode = $proc.ExitCode; Output = $lines }
    } finally {
        Remove-Item -Path $outFile, $errFile -Force -ErrorAction SilentlyContinue
    }
}

Section "Docker GPU wiring (WSL2 / Docker Desktop)"
if (Have docker) {
    try {
        $gpuProbe = Invoke-BoundedDocker -DockerArgs @('run', '--rm', '--gpus', 'all', 'nvidia/cuda:12.6.3-base-ubuntu24.04', 'nvidia-smi', '-L')
        if ($gpuProbe.TimedOut) {
            # Reported apart from a plain failure: "Docker cannot reach the GPU" and "Docker
            # never answered" have different causes and different fixes, and collapsing the
            # second into the first recommends a CPU target for a working GPU host.
            Report "Docker GPU probe timed out -- Docker Desktop or the WSL backend may be stalled"
        } elseif ($gpuProbe.ExitCode -eq 0) {
            $hasNvidiaToolkit = $true
            Report "Docker GPU probe succeeded"
        } else {
            Report "Docker GPU probe failed -- enable GPU support in Docker Desktop / WSL2"
        }
        # Every render node, not just renderD128. Numbering starts higher whenever another
        # DRM device enumerates first, so probing the fixed name reported "no render node"
        # on a host whose Intel iGPU was sitting at renderD129 -- and, on a machine with a
        # discrete card at renderD128, read the wrong device's vendor entirely.
        #
        # The node whose vendor is 0x8086 is the one selected, not merely the first found:
        # that is what makes HOST_INTEL_RENDER_GID the Intel node's GID on a mixed host.
        $probe = 'for n in /dev/dri/renderD*; do [ -e "$n" ] || continue; b=$(basename "$n"); v=$(cat /sys/class/drm/$b/device/vendor 2>/dev/null); echo "$b $(stat -c %g "$n") $v"; done'
        # -v /dev/dri: the probe reads render nodes from inside the container, and without
        # the mount there is nothing there to read -- so every host reported "no render
        # nodes" and no Intel target was ever recommended on Windows.
        $nodeProbe = Invoke-BoundedDocker -DockerArgs @('run', '--rm', '-v', '/dev/dri:/dev/dri:ro', 'alpine', 'sh', '-c', $probe)
        if ($nodeProbe.TimedOut) { Report "  render-node probe timed out -- Docker Desktop or the WSL backend may be stalled" }
        $nodes = @($nodeProbe.Output)
        if ($nodes.Count) {
            foreach ($line in $nodes) { Report "  $($line.Trim()) in the Docker VM" }
            $intelLine = $nodes | Where-Object { $_ -match '\s0x8086\s*$' } | Select-Object -First 1
            if ($intelLine) {
                $hasIntelRenderNode = $true
                $renderGid = ($intelLine -split '\s+')[1]
                Report "  selected Intel render node: $($intelLine.Trim())  (HOST_INTEL_RENDER_GID=$renderGid)"
            } else {
                # A GID is still reported for the first node so an operator can see one, but
                # it does not make the host Intel and does not select an Intel target.
                $renderGid = ($nodes[0] -split '\s+')[1]
                Report "  no Intel (0x8086) render node in the Docker VM; not treated as Intel"
            }
        } else {
            Report "  no /dev/dri render nodes inside the Docker VM (expected on WSL2 without GPU paravirt)"
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

if ($WriteEnv) {
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
