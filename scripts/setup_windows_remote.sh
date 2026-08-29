#!/bin/bash
# Bootstrap a Windows host for remote hardware validation, with one paste.
#
# A Windows box cannot be configured over SSH before SSH works, so exactly one step is
# manual: running a generated command in an Administrator PowerShell. This script prepares
# the key, packages scripts/setup_windows_remote.ps1 together with that key into a single
# self-contained command, then waits for the host to come up and verifies the whole chain.
#
# Usage:
#   scripts/setup_windows_remote.sh <host>              # generate, then wait and verify
#   scripts/setup_windows_remote.sh <host> --user NAME  # skip the username prompt
#   scripts/setup_windows_remote.sh <host> --verify-only
#
# Options:
#   --user NAME   Windows username (the setup command prints it).
#   --key PATH    SSH identity. Default: ~/.ssh/whisper_remote_validation (auto-created).
#   --verify-only Skip generation; just test an already-configured host.
#   --timeout SEC How long to wait for SSH after the paste. Default 600.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KEY="${HOME}/.ssh/whisper_remote_validation"
HOST=""
USERNAME=""
VERIFY_ONLY=false
TIMEOUT=600

while [ $# -gt 0 ]; do
  case "$1" in
    --user)        USERNAME="${2:?--user needs a value}"; shift 2 ;;
    --key)         KEY="${2:?--key needs a value}"; shift 2 ;;
    --timeout)     TIMEOUT="${2:?--timeout needs a value}"; shift 2 ;;
    --verify-only) VERIFY_ONLY=true; shift ;;
    -h|--help)     grep '^#' "$0" | sed 's/^# \{0,1\}//' | sed '1d'; exit 0 ;;
    -*)            echo "unknown argument: $1" >&2; exit 2 ;;
    *)             HOST="$1"; shift ;;
  esac
done

hdr()  { printf '\n=== %s ===\n' "$*"; }
note() { printf '  %s\n' "$*"; }
die()  { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

[ -n "$HOST" ] || read -r -p "Windows host (IP or name): " HOST
[ -n "$HOST" ] || die "a host is required"

if [ ! -f "$KEY" ]; then
  note "No identity at $KEY -- generating a dedicated one."
  # ssh-keygen does not create intermediate directories, so a --key under a path that
  # does not exist yet fails with a bare "No such file or directory" naming neither.
  mkdir -p "$(dirname "$KEY")"
  ssh-keygen -t ed25519 -N '' -C 'whisper-pro-asr remote hardware validation' -f "$KEY" >/dev/null
fi
PUBKEY="$(cat "${KEY}.pub")"

if [ "$VERIFY_ONLY" != true ]; then
  hdr "One command to run on the Windows machine"
  # The .ps1 and the key are packed into a single line so there is exactly one thing to
  # paste -- no file copying, no editing a key into a script by hand.
  #
  # gzip before base64 is not premature tidiness: cmd.exe refuses any command line over
  # 8191 characters, and the plain base64 of this script exceeds that. Compressing keeps
  # the line inside the limit with room to grow. PowerShell's own limit is far higher,
  # but the paste must survive being dropped into a cmd window too.
  # `tr -d` rather than `base64 -w0`: the wrapping flag is GNU-only and BSD base64 (macOS)
  # rejects it, so an operator driving this from a Mac got an empty command line.
  PS_B64="$(gzip -9 -c "${REPO_ROOT}/scripts/setup_windows_remote.ps1" | base64 | tr -d '\n')"
  ONELINER="powershell -NoProfile -ExecutionPolicy Bypass -Command \"\$b=[Convert]::FromBase64String('${PS_B64}'); \$m=New-Object IO.MemoryStream(,\$b); \$g=New-Object IO.Compression.GzipStream(\$m,[IO.Compression.CompressionMode]::Decompress); \$s=(New-Object IO.StreamReader(\$g)).ReadToEnd(); \$f=Join-Path \$env:TEMP 'whisper_setup.ps1'; Set-Content -Path \$f -Value \$s -Encoding UTF8; & \$f -PublicKey '${PUBKEY}'\""
  LEN=${#ONELINER}
  cat <<EOF

Open PowerShell on ${HOST} (Start -> type "powershell" -> Enter). It does NOT need to be
elevated -- the script raises its own UAC prompt. Paste this single line:

${ONELINER}

It enables the OpenSSH server, authorises the key in the right place for your account
type, opens the firewall, and checks WSL2 + Docker. It prints your username on the last
line -- that is the only thing to tell me.

EOF
  if [ "$LEN" -gt 8191 ]; then
    note "WARNING: the command is ${LEN} characters, over cmd.exe's 8191 limit."
    note "Paste it into PowerShell (not a cmd window), or it will be truncated."
  fi
fi

if [ -z "$USERNAME" ]; then
  read -r -p "Windows username (from the last line of that output): " USERNAME
fi
[ -n "$USERNAME" ] || die "a username is required"

hdr "Waiting for ${USERNAME}@${HOST}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes)
deadline=$(( $(date +%s) + TIMEOUT ))
until ssh -i "$KEY" "${SSH_OPTS[@]}" "${USERNAME}@${HOST}" 'exit' 2>/dev/null; do
  [ "$(date +%s)" -lt "$deadline" ] || die "no SSH after ${TIMEOUT}s. Check the setup output, and that port 22 is reachable: nc -vz ${HOST} 22"
  sleep 5
done
note "ssh: OK"

hdr "Verifying the Windows host"
# SSH on Windows lands in PowerShell or cmd, so Linux commands only work through WSL.
# Everything the validator needs lives inside WSL, which is what these check.
run() { ssh -i "$KEY" "${SSH_OPTS[@]}" "${USERNAME}@${HOST}" "$@"; }

note "shell: $(run 'echo %COMSPEC%' 2>/dev/null || echo 'powershell')"

if run 'wsl -e bash -lc "echo ok"' 2>/dev/null | grep -q ok; then
  note "wsl: reachable ($(run 'wsl -e bash -lc "grep PRETTY_NAME /etc/os-release | cut -d= -f2"' 2>/dev/null | tr -d '\"'))"
else
  die "WSL is not reachable over SSH. Install a distro (wsl --install -d Ubuntu) and re-run with --verify-only."
fi

if DOCKER_VER="$(run 'wsl -e bash -lc "docker info --format {{.ServerVersion}} 2>/dev/null"' 2>/dev/null)" && [ -n "$DOCKER_VER" ]; then
  note "docker in wsl: $DOCKER_VER"
else
  die "Docker is not reachable inside WSL. Docker Desktop -> Settings -> Resources -> WSL Integration -> enable for the distro, then re-run with --verify-only."
fi

note "nvidia in wsl: $(run 'wsl -e bash -lc "nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none"' 2>/dev/null)"
note "free space: $(run 'wsl -e bash -lc "df -BG --output=avail / | tail -1"' 2>/dev/null | tr -d ' ')"

cat <<EOF

=== Ready ===
  Validate with:

    scripts/remote_validate.sh ${USERNAME}@${HOST} --wsl --full

  On Windows only NVIDIA reaches Linux containers. Intel GPU/NPU are not exposed, and AMD
  is detected through /dev/dxg but falls back to CPU -- see docs/REMOTE_VALIDATION.md.
EOF
