# Bootstrap a Linux host for remote hardware validation, from a Windows operator machine.
#
# PowerShell counterpart of scripts/setup_linux_remote.sh, for when the machine driving
# the validation runs Windows. Requires the OpenSSH client, which ships with Windows 10/11
# (Settings -> Optional features -> OpenSSH Client).
#
# The block it prints is the same one the shell version prints, and for the same reason:
# installing sshd, Docker and the NVIDIA container toolkit, and joining the docker, render
# and video groups, all need the remote user's own password, so they are folded into a
# single paste rather than a sequence of round trips.
#
# Linux is the only target platform where every accelerator is reachable from a container:
# NVIDIA, AMD ROCm, Intel GPU and Intel NPU. See docs/REMOTE_VALIDATION.md.
#
#   .\setup_linux_remote.ps1 -RemoteHost 10.0.0.5
#   .\setup_linux_remote.ps1 -RemoteHost 10.0.0.5 -User alice
#   .\setup_linux_remote.ps1 -RemoteHost 10.0.0.5 -User alice -VerifyOnly

[Diagnostics.CodeAnalysis.SuppressMessageAttribute(
    'PSAvoidUsingWriteHost', '',
    Justification = 'This script is an operator-facing console tool: the coloured, immediate output IS its interface. Write-Output would put the report on the success stream, where a caller capturing the result would collect the prose along with it.'
)]
param(
    [Parameter(Mandatory = $true)][string]$RemoteHost,
    [string]$User = '',
    [string]$Key = "$env:USERPROFILE\.ssh\whisper_remote_validation",
    [switch]$VerifyOnly,
    [int]$TimeoutSeconds = 600
)

$ErrorActionPreference = 'Stop'
function Hdr($m)  { Write-Host "`n=== $m ===" -ForegroundColor Cyan }
function Note($m) { Write-Host "  $m" }
function Fail($m) { Write-Host "`nERROR: $m" -ForegroundColor Red; exit 1 }

$repoRoot = Split-Path (Split-Path $PSCommandPath -Parent) -Parent

if (-not (Test-Path $Key)) {
    Note "No identity at $Key -- generating a dedicated one."
    New-Item -ItemType Directory -Force -Path (Split-Path $Key) | Out-Null
    ssh-keygen -t ed25519 -N '""' -C 'whisper-pro-asr remote hardware validation' -f $Key | Out-Null
}
$pubKey = (Get-Content "$Key.pub" -Raw).Trim()

if (-not $VerifyOnly) {
    Hdr "One block to paste on the Linux machine"
    @"

Paste this into a terminal on $RemoteHost. sudo will ask for your password -- that is
yours to type, and the only interactive part. Everything that needs root is in this one
block, because once I am on the machine over SSH I cannot answer a sudo prompt.

It installs the SSH server (a desktop install usually has none, which is why the
connection is refused rather than rejected), installs Docker if missing, adds the NVIDIA
container toolkit when an NVIDIA card is present, authorises the key, and puts you in the
groups that reach the GPUs: docker for all of them, plus render and video, which are what
an AMD card needs for /dev/kfd and /dev/dri.

set -e
sudo apt-get update
sudo apt-get install -y openssh-server curl
sudo systemctl enable --now ssh
command -v docker >/dev/null || curl -fsSL https://get.docker.com | sudo sh
if lspci | grep -qi nvidia && ! command -v nvidia-ctk >/dev/null; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
fi
mkdir -p ~/.ssh && chmod 700 ~/.ssh
printf '%s\n' '$pubKey' >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
sudo usermod -aG docker,render,video "`$(whoami)"
echo "READY user=`$(whoami) host=`$(hostname) arch=`$(uname -m)"

Group membership applies to new logins, so no logout is needed -- each command opens a
fresh SSH session.

"@ | Write-Host
}

if (-not $User) { $User = Read-Host 'Remote username (from the READY line)' }
if (-not $User) { Fail 'a username is required' }

# BatchMode turns a would-be password prompt into an immediate error rather than a hang.
$sshOpts = @('-o','BatchMode=yes','-o','ConnectTimeout=5','-o','StrictHostKeyChecking=accept-new','-o','IdentitiesOnly=yes')
function Run([string]$cmd) {
    $priorErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & ssh -i $Key @sshOpts "$User@$RemoteHost" $cmd 2>$null
    } finally {
        $ErrorActionPreference = $priorErrorActionPreference
    }
}

Hdr "Waiting for $User@$RemoteHost"
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
while ($true) {
    $priorErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & ssh -i $Key @sshOpts "$User@$RemoteHost" 'exit' 2>$null
    } finally {
        $ErrorActionPreference = $priorErrorActionPreference
    }
    if ($LASTEXITCODE -eq 0) { break }
    if ((Get-Date) -gt $deadline) { Fail "no SSH after ${TimeoutSeconds}s. Check sshd is running and port 22 is reachable." }
    Start-Sleep -Seconds 5
}
Note "ssh: OK ($(Run 'echo "$(whoami)@$(hostname) $(uname -m) kernel=$(uname -r)"'))"

Hdr "Verifying"
Run 'docker info >/dev/null 2>&1'
if ($LASTEXITCODE -ne 0) { Fail 'docker still needs sudo. Confirm the usermod ran, then re-run with -VerifyOnly.' }
Note "docker: usable without sudo ($(Run 'docker --version'))"
Note "free space: $(Run 'df -BG --output=avail / | tail -1 | tr -d " "')"

Hdr "Hardware audit"
# Piped so the remote needs no checkout. Never assume which vendor is present: a host
# offered for one accelerator may carry another worth validating instead.
$auditScript = Join-Path $repoRoot 'scripts\audit_hardware.sh'
# Normalised to LF before it is piped: the file is read on Windows, and bash treats the CR
# a checkout may leave on each line as part of the command, failing with a baffling
# "$'\r': command not found" on a script that is perfectly valid.
$auditBody = (Get-Content $auditScript -Raw) -replace "`r`n", "`n"
$audit = $auditBody | & ssh -i $Key @sshOpts "$User@$RemoteHost" 'bash -s -- --json'
Note $audit

$target = [regex]::Match($audit, '"recommended_target":"([^"]*)"').Groups[1].Value
$hasNpu = $audit -match '"intel_npu":true'

@"

=== Ready ===
  Validate with:

    scripts/remote_validate.sh $User@$RemoteHost --target $target --full
"@ | Write-Host

if ($hasNpu) {
    @"

  This host has an Intel NPU. AUTO ranks CUDA > AMD > GPU > NPU, so it will never be
  chosen on its own -- ask for it explicitly:

    scripts/remote_validate.sh $User@$RemoteHost --target $target --device NPU --full
"@ | Write-Host
}
