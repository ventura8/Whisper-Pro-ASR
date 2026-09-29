# Shared by scripts/setup_linux_remote.ps1 and scripts/setup_macos_remote.ps1 (dot-sourced).
#
# Both bootstrappers generate the same dedicated key, drive the remote over the same
# BatchMode SSH options, and poll for the host the same way; only the paste block and the
# checks after connecting differ per platform. Keeping one copy here means a fix to any of
# the traps below reaches both, instead of landing in one file and not the other.

function Hdr($m)  { Write-Host "`n=== $m ===" -ForegroundColor Cyan }
function Show-Note($m) { Write-Host "  $m" }
function Fail($m) { Write-Host "`nERROR: $m" -ForegroundColor Red; exit 1 }

# BatchMode turns a would-be password prompt into an immediate error rather than a hang.
$script:RemoteSshOptions = @('-o','BatchMode=yes','-o','ConnectTimeout=5','-o','StrictHostKeyChecking=accept-new','-o','IdentitiesOnly=yes')

# Create the dedicated identity if it is missing, and return its public key.
function Initialize-ValidationKey([string]$Key) {
    if (-not (Test-Path $Key)) {
        Show-Note "No identity at $Key -- generating a dedicated one."
        New-Item -ItemType Directory -Force -Path (Split-Path $Key) | Out-Null
        # PowerShell 7.3 changed how arguments are passed to native commands: the old
        # workaround -N '""' now reaches ssh-keygen as a literal two-character passphrase, so
        # the key it writes is encrypted with `""` -- and every later BatchMode=yes connection
        # fails to read it, which looks exactly like an unauthorised key. Older hosts still
        # need the quoted form, because a bare "" was dropped there entirely.
        if ($PSVersionTable.PSVersion -ge [version]'7.3') {
            ssh-keygen -t ed25519 -N "" -C 'whisper-pro-asr remote hardware validation' -f $Key | Out-Null
        } else {
            ssh-keygen -t ed25519 -N '""' -C 'whisper-pro-asr remote hardware validation' -f $Key | Out-Null
        }
    }
    return (Get-Content "$Key.pub" -Raw).Trim()
}

# Run one command on the remote and return its output.
#
# With $PSNativeCommandUseErrorActionPreference on and $ErrorActionPreference='Stop' (set by
# each caller), a native command exiting non-zero *throws* -- so a probe that is supposed to
# be retryable, or simply allowed to fail, aborted the whole bootstrapper instead of letting
# the caller's $LASTEXITCODE checks do their job. Relaxed for the call, then restored.
function Invoke-RemoteCommand([string]$Key, [string]$Target, [string]$Command) {
    $priorErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & ssh -i $Key @script:RemoteSshOptions $Target $Command 2>$null
    } finally {
        $ErrorActionPreference = $priorErrorActionPreference
    }
}

# Poll until the host accepts a key-authenticated connection, or fail with $Hint.
function Wait-RemoteSsh([string]$Key, [string]$Target, [int]$TimeoutSeconds, [string]$Hint) {
    Hdr "Waiting for $Target"
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ($true) {
        # Same guard as Invoke-RemoteCommand: this loop exists to poll a host that is not up
        # yet, so the failing exits it is waiting through must not be fatal.
        Invoke-RemoteCommand -Key $Key -Target $Target -Command 'exit' | Out-Null
        if ($LASTEXITCODE -eq 0) { return }
        if ((Get-Date) -gt $deadline) { Fail "no SSH after ${TimeoutSeconds}s. $Hint" }
        Start-Sleep -Seconds 5
    }
}
