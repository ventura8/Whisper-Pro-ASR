# Bootstrap a macOS host for remote validation, from a Windows operator machine.
#
# PowerShell counterpart of scripts/setup_macos_remote.sh. Requires the OpenSSH client,
# which ships with Windows 10/11 (Settings -> Optional features -> OpenSSH Client).
#
# Read this before spending time on it: Docker Desktop on macOS runs containers inside a
# Linux VM with no GPU passthrough. The Apple Silicon GPU and Neural Engine are not
# reachable from a Linux container, and this stack has no Metal or CoreML path. A macOS
# host can validate the `cpu` target and nothing else. That is still a real result -- it
# proves the software, the API surface and CPU decoding on arm64 -- but it can never
# support an accelerator claim.
#
#   .\setup_macos_remote.ps1 -RemoteHost 10.0.0.9
#   .\setup_macos_remote.ps1 -RemoteHost 10.0.0.9 -User alice -VerifyOnly

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

if (-not (Test-Path $Key)) {
    Note "No identity at $Key -- generating a dedicated one."
    New-Item -ItemType Directory -Force -Path (Split-Path $Key) | Out-Null
    ssh-keygen -t ed25519 -N '""' -C 'whisper-pro-asr remote hardware validation' -f $Key | Out-Null
}
$pubKey = (Get-Content "$Key.pub" -Raw).Trim()

if (-not $VerifyOnly) {
    Hdr "Steps on the Mac"
    @"

1. Enable Remote Login (macOS's SSH server), either in
   System Settings -> General -> Sharing -> Remote Login, or in Terminal:

     sudo systemsetup -setremotelogin on

2. Authorise the key. Paste this single block into Terminal on the Mac:

mkdir -p ~/.ssh && chmod 700 ~/.ssh && \
printf '%s\n' '$pubKey' >> ~/.ssh/authorized_keys && \
chmod 600 ~/.ssh/authorized_keys && \
echo "READY user=`$(whoami) arch=`$(uname -m)"

   The last line prints the username -- that is the only thing to report back.

3. Docker Desktop must be installed and running; its VM is what runs the containers.
   https://www.docker.com/products/docker-desktop/

"@ | Write-Host
}

if (-not $User) { $User = Read-Host 'macOS username (from the READY line)' }
if (-not $User) { Fail 'a username is required' }

$sshOpts = @('-o','BatchMode=yes','-o','ConnectTimeout=5','-o','StrictHostKeyChecking=accept-new','-o','IdentitiesOnly=yes')
# Guarded exactly as scripts/setup_linux_remote.ps1 does. With
# $PSNativeCommandUseErrorActionPreference on and $ErrorActionPreference='Stop' (set at the
# top of this file), a native command exiting non-zero *throws* -- so a probe that is
# supposed to be retryable, or simply allowed to fail, aborted the whole bootstrapper
# instead of letting the $LASTEXITCODE checks below do their job.
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
    # Same guard: this loop exists to poll a host that is not up yet, so the failing exits
    # it is waiting through must not be fatal.
    $priorErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & ssh -i $Key @sshOpts "$User@$RemoteHost" 'exit' 2>$null
    } finally {
        $ErrorActionPreference = $priorErrorActionPreference
    }
    if ($LASTEXITCODE -eq 0) { break }
    if ((Get-Date) -gt $deadline) { Fail "no SSH after ${TimeoutSeconds}s. Confirm Remote Login is enabled and port 22 is reachable." }
    Start-Sleep -Seconds 5
}
Note "ssh: OK ($(Run 'echo "$(whoami)@$(hostname -s) $(uname -m) macOS $(sw_vers -productVersion)"'))"

Hdr "Verifying"
$dockerVersion = Run 'docker info --format "{{.ServerVersion}}" 2>/dev/null'
if (-not $dockerVersion) { Fail 'Docker is not reachable. Start Docker Desktop on the Mac, then re-run with -VerifyOnly.' }
Note "docker: $dockerVersion"
# BSD df differs from GNU's; -g reports gigabytes on macOS.
Note "free space: $(Run 'df -g / | tail -1 | awk ''{print $4"G"}''')"
Note "cpu: $(Run 'sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown')"

@"

=== Ready (cpu target only) ===
  Validate with:

    scripts/remote_validate.sh $User@$RemoteHost --target cpu --full

  There is no GPU passthrough into Docker Desktop's Linux VM, so the Apple GPU and Neural
  Engine cannot be exercised and no accelerator claim can be made from this host. What it
  does prove: the image boots, the API works, and CPU decoding is correct on arm64.
  See docs/REMOTE_VALIDATION.md.
"@ | Write-Host
