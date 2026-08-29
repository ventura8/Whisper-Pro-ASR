# Prepare a Windows host to be driven by scripts/remote_validate.sh.
#
# Enables the OpenSSH server, authorises one public key in the correct location for the
# current account type, opens the firewall, and reports whether WSL2 + Docker are usable.
# Everything here is idempotent: re-running changes nothing that is already correct.
#
# Run in an ADMINISTRATOR PowerShell. scripts/setup_windows_remote.sh generates a
# one-line command that fetches and runs this with the key already filled in.
#
#   .\setup_windows_remote.ps1 -PublicKey 'ssh-ed25519 AAAA... comment'

[Diagnostics.CodeAnalysis.SuppressMessageAttribute(
    'PSAvoidUsingWriteHost', '',
    Justification = 'This script is an operator-facing console tool: the coloured, immediate output IS its interface. Write-Output would put the report on the success stream, where a caller capturing the result would collect the prose along with it.'
)]
param(
    [Parameter(Mandatory = $true)][string]$PublicKey,
    [switch]$NoElevate
)

$ErrorActionPreference = 'Stop'
function Step($m) { Write-Host "==> $m" -ForegroundColor Cyan }
function Ok($m)   { Write-Host "    $m" -ForegroundColor Green }
function Warn($m) { Write-Host "    $m" -ForegroundColor Yellow }

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$isAdmin = ([Security.Principal.WindowsPrincipal]$identity).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

# Running unelevated is not merely degraded, it is actively misleading: sshd cannot be
# installed, and the key gets written to the per-user file, which Windows OpenSSH ignores
# outright for administrator accounts. The result looks like a successful setup and then
# refuses every connection. So re-launch through UAC instead of carrying on.
if (-not $isAdmin -and -not $NoElevate) {
    Step "Elevating"
    Warn "Not running as Administrator. Accept the UAC prompt -- a new window will open."
    $self = $PSCommandPath
    if (-not $self) { Warn "Cannot locate this script to re-launch; re-run it from an Administrator PowerShell."; exit 1 }
    try {
        $argLine = '-NoProfile -ExecutionPolicy Bypass -NoExit -File "' + $self +
                   '" -PublicKey "' + $PublicKey + '" -NoElevate'
        Start-Process -FilePath 'powershell' -Verb RunAs -ArgumentList $argLine
        Write-Host ""
        Write-Host "Setup continues in the elevated window. Read the READY line there." -ForegroundColor Magenta
        exit 0
    } catch {
        Warn "Elevation was declined or failed: $($_.Exception.Message)"
        Warn "Re-run this from an Administrator PowerShell."
        exit 1
    }
}

Step "OpenSSH server"
if (-not $isAdmin) {
    Warn "Not elevated. The server cannot be installed or started from here."
    Warn "The key below goes to the per-user file, which Windows OpenSSH IGNORES for"
    Warn "administrator accounts -- so SSH may still refuse the key. Re-run elevated."
} else {
    $cap = Get-WindowsCapability -Online -Name 'OpenSSH.Server*' | Select-Object -First 1
    # $null is its own case. Folded into the else it reported "already installed" for a
    # Windows edition that does not carry the capability at all, and the run then failed
    # further down on a missing sshd service with nothing pointing back to here.
    if (-not $cap) {
        Warn "The OpenSSH.Server capability was not found on this Windows edition."
        Warn "Install the OpenSSH server manually, then re-run this script."
    } elseif ($cap.State -ne 'Installed') {
        Add-WindowsCapability -Online -Name $cap.Name | Out-Null
        Ok "installed $($cap.Name)"
    } else {
        Ok "already installed"
    }
    # Add-WindowsCapability can succeed and still leave the service unregistered until a
    # reboot. Set-Service then throws a raw "Cannot find any service with service name
    # 'sshd'", which reads like the capability install failed rather than like the one
    # thing the operator has to do next.
    $sshd = Get-Service -Name sshd -ErrorAction SilentlyContinue
    if (-not $sshd) {
        Warn "The OpenSSH capability is installed but the 'sshd' service is not registered yet."
        Warn "Windows needs a reboot before it appears. Reboot, then re-run this script."
        exit 1
    }
    Set-Service -Name sshd -StartupType Automatic
    if ($sshd.Status -ne 'Running') { Start-Service sshd }
    Ok "sshd running, starts automatically"

    if (-not (Get-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -ErrorAction SilentlyContinue)) {
        New-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -DisplayName 'OpenSSH Server (sshd)' `
            -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22 | Out-Null
        Ok "firewall rule added for port 22"
    } else {
        Ok "firewall rule already present"
    }
}

Step "Authorising key"
# The trap this exists to avoid: for administrator accounts, Windows OpenSSH ignores
# ~\.ssh\authorized_keys entirely and reads the machine-wide file below. A key placed in
# the user file for an admin account simply never works, and looks like a bad key.
if ($isAdmin) {
    $keyFile = Join-Path $env:ProgramData 'ssh\administrators_authorized_keys'
} else {
    $keyFile = Join-Path $env:USERPROFILE '.ssh\authorized_keys'
}
New-Item -ItemType Directory -Force -Path (Split-Path $keyFile) | Out-Null
if (-not (Test-Path $keyFile)) { New-Item -ItemType File -Path $keyFile | Out-Null }

# Trailing whitespace on a stored line is invisible and would make an already-authorised
# key look absent, appending a duplicate on every run.
$existing = @(Get-Content $keyFile -ErrorAction SilentlyContinue | ForEach-Object { $_.Trim() })
if ($existing -and ($existing -contains $PublicKey.Trim())) {
    Ok "key already present in $keyFile"
} else {
    Add-Content -Path $keyFile -Value $PublicKey.Trim()
    Ok "key added to $keyFile"
}

# sshd silently ignores a key file with loose permissions, which is indistinguishable
# from a wrong key. Tighten them explicitly.
# Well-known SIDs rather than names: 'Administrators' and 'SYSTEM' are localised, so on a
# non-English Windows icacls fails to resolve them, leaves the permissions loose, and sshd
# then ignores the key file -- indistinguishable from a wrong key.
#   *S-1-5-32-544 = Administrators, *S-1-5-18 = SYSTEM
if ($isAdmin) {
    icacls $keyFile /inheritance:r /grant '*S-1-5-32-544:F' /grant '*S-1-5-18:F' | Out-Null
} else {
    # SYSTEM as well as the user: sshd runs as SYSTEM, and with inheritance removed it
    # would otherwise lose read access to the very file it must consult -- which looks
    # exactly like a rejected key. *S-1-5-18 = SYSTEM, by SID because the name is localised.
    icacls $keyFile /inheritance:r /grant:r "$($env:USERNAME):F" /grant:r '*S-1-5-18:F' | Out-Null
}
if ($LASTEXITCODE -ne 0) {
    Warn "icacls failed (exit $LASTEXITCODE); sshd ignores a key file with loose permissions."
    Warn "Fix the ACL on $keyFile by hand, then re-run."
    exit 1
}
Ok "permissions tightened"

Step "WSL2 and Docker"
# The container stack is Linux, so what matters is Docker reachable *inside* WSL, not
# Docker on Windows.
# wsl.exe emits UTF-16 by default, which arrives as NUL-padded junk; WSL_UTF8 fixes that
# at the source. `docker-desktop` (and the older `docker-desktop-data`) are Docker Desktop's
# own utility VMs -- they appear in the distro list but carry no shell, so `wsl -e bash`
# fails against them. They are not distros for our purposes.
$env:WSL_UTF8 = '1'
$distros = @()
try {
    # @() is load-bearing: a pipeline yielding one match returns a bare string, and
    # indexing that returns its first character rather than the distro name.
    $distros = @((wsl -l -q) -split "`r?`n" |
        ForEach-Object { $_ -replace "`0", '' } |
        Where-Object { $_ -and $_.Trim() } |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ -notmatch '^docker-desktop' })
} catch {
    Warn "Could not enumerate WSL distros: $($_.Exception.Message)"
}

if (-not $distros -and $isAdmin) {
    Warn "No usable Linux distro (only Docker Desktop's utility VMs). Installing Ubuntu..."
    # --no-launch keeps this non-interactive; the distro runs as root, which is all the
    # validator needs. A reboot is sometimes required before it is usable.
    wsl --install -d Ubuntu --no-launch
    if ($LASTEXITCODE -eq 0) {
        Ok "Ubuntu installed. A reboot may be required before it starts."
        $distros = @('Ubuntu')
    } else {
        Warn "Automatic install failed. Run manually:  wsl --install -d Ubuntu"
    }
}

if (-not $distros) {
    Warn "No usable WSL distro. Install one:  wsl --install -d Ubuntu"
    $dockerVersion = $null
} else {
    Ok "WSL distros: $($distros -join ', ')"
    $distro = $distros[0]
    $dockerVersion = (wsl -d $distro -e bash -lc "docker info --format '{{.ServerVersion}}' 2>/dev/null") 2>$null
    if ($dockerVersion) {
        Ok "docker inside WSL ($distro): $dockerVersion"
    } else {
        Warn "Docker is not reachable inside WSL."
        Warn "Docker Desktop -> Settings -> Resources -> WSL Integration -> enable for the distro."
    }
}

Step "Hardware"
# Reported so the caller knows what is worth validating. On Windows, only NVIDIA reaches
# Linux containers: Intel GPU/NPU are not exposed at all, and AMD is detected via
# /dev/dxg but falls back to CPU.
$gpus = (Get-CimInstance Win32_VideoController).Name
foreach ($g in $gpus) { Ok "GPU: $g" }
if ($gpus -match 'NVIDIA') {
    Ok "NVIDIA present - CUDA is validatable through WSL2"
} else {
    Warn "No NVIDIA GPU. This host can prove the images boot and fall back to CPU cleanly,"
    Warn "but Intel GPU/NPU and AMD acceleration are not reachable from Linux containers on Windows."
}

Write-Host ""
Write-Host "READY user=$($env:USERNAME) admin=$isAdmin keyfile=$keyFile docker=$dockerVersion" -ForegroundColor Magenta
Write-Host "Give the operator this username: $($env:USERNAME)" -ForegroundColor Magenta
