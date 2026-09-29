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

param(
    [Parameter(Mandatory = $true)][string]$RemoteHost,
    [string]$User = '',
    [string]$Key = "$env:USERPROFILE\.ssh\whisper_remote_validation",
    [switch]$VerifyOnly,
    [int]$TimeoutSeconds = 600
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'remote_setup_common.ps1')

$repoRoot = Split-Path (Split-Path $PSCommandPath -Parent) -Parent

$pubKey = Initialize-ValidationKey -Key $Key

if (-not $VerifyOnly) {
    Hdr "One block to paste on the Linux machine"
    # The group loop in the block below matches setup_linux_remote.sh exactly. A single
    # `usermod -aG docker,render,video` fails outright on a host that has no `render` or
    # `video` group -- and because the pasted block runs under `set -e`, it took READY down
    # with it, leaving the operator with no username to hand back and docker never joined.
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
sudo apt-get install -y openssh-server curl pciutils
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
for g in docker render video; do getent group "`$g" >/dev/null && sudo usermod -aG "`$g" "`$(whoami)" || true; done
echo "READY user=`$(whoami) host=`$(hostname) arch=`$(uname -m)"

Group membership applies to new logins, so no logout is needed -- each command opens a
fresh SSH session.

"@ | Write-Host
}

if (-not $User) { $User = Read-Host 'Remote username (from the READY line)' }
if (-not $User) { Fail 'a username is required' }

$remote = "$User@$RemoteHost"
function Run([string]$cmd) { Invoke-RemoteCommand -Key $Key -Target $remote -Command $cmd }

Wait-RemoteSsh -Key $Key -Target $remote -TimeoutSeconds $TimeoutSeconds -Hint "Check sshd is running and port 22 is reachable."
Show-Note "ssh: OK ($(Run 'echo "$(whoami)@$(hostname) $(uname -m) kernel=$(uname -r)"'))"

Hdr "Verifying"
Run 'docker info >/dev/null 2>&1'
if ($LASTEXITCODE -ne 0) { Fail 'docker still needs sudo. Confirm the usermod ran, then re-run with -VerifyOnly.' }
Show-Note "docker: usable without sudo ($(Run 'docker --version'))"
Show-Note "free space: $(Run 'df -BG --output=avail / | tail -1 | tr -d " "')"

Hdr "Hardware audit"
# Piped so the remote needs no checkout. Never assume which vendor is present: a host
# offered for one accelerator may carry another worth validating instead.
$auditScript = Join-Path $repoRoot 'scripts\audit_hardware.sh'
# Normalised to LF before it is piped: the file is read on Windows, and bash treats the CR
# a checkout may leave on each line as part of the command, failing with a baffling
# "$'\r': command not found" on a script that is perfectly valid.
$auditBody = (Get-Content $auditScript -Raw) -replace "`r`n", "`n"
$audit = $auditBody | & ssh -i $Key @script:RemoteSshOptions $remote 'bash -s -- --json'
Show-Note $audit

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
