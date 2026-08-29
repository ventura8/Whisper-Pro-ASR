#!/bin/bash
# Bootstrap a macOS host for remote validation, with one paste.
#
# Read this first, because it decides whether the exercise is worth doing at all:
# Docker Desktop on macOS runs containers inside a Linux VM with no GPU passthrough. The
# Apple Silicon GPU and the Neural Engine are not reachable from a Linux container, and
# there is no Metal or CoreML path in this stack. A macOS host can therefore validate the
# `cpu` target and nothing else. That is still a real result -- it proves the software,
# the API surface and CPU decoding on a different architecture (arm64) -- but it can never
# support an accelerator claim.
#
# Usage:
#   scripts/setup_macos_remote.sh <host>              # print setup, then wait and verify
#   scripts/setup_macos_remote.sh <host> --user NAME  # skip the username prompt
#   scripts/setup_macos_remote.sh <host> --verify-only
#
# Options:
#   --user NAME    macOS username.
#   --key PATH     SSH identity. Default: ~/.ssh/whisper_remote_validation (auto-created).
#   --verify-only  Skip the instructions; just test an already-configured host.
#   --timeout SEC  How long to wait for SSH. Default 600.
set -euo pipefail

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

[ -n "$HOST" ] || read -r -p "macOS host (IP or name): " HOST
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
  hdr "Steps on the Mac"
  cat <<EOF

1. Enable Remote Login (this is macOS's SSH server), either in
   System Settings -> General -> Sharing -> Remote Login, or in Terminal:

     sudo systemsetup -setremotelogin on

2. Authorise the key. Paste this single block into Terminal on the Mac:

mkdir -p ~/.ssh && chmod 700 ~/.ssh && \\
printf '%s\\n' '${PUBKEY}' >> ~/.ssh/authorized_keys && \\
chmod 600 ~/.ssh/authorized_keys && \\
echo "READY user=\$(whoami) arch=\$(uname -m)"

   The last line prints your username -- that is the only thing to tell me.

3. Docker Desktop must be installed and running (its VM is what actually runs the
   containers). If it is not: https://www.docker.com/products/docker-desktop/

EOF
fi

if [ -z "$USERNAME" ]; then
  read -r -p "macOS username (from the READY line): " USERNAME
fi
[ -n "$USERNAME" ] || die "a username is required"

hdr "Waiting for ${USERNAME}@${HOST}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes)
deadline=$(( $(date +%s) + TIMEOUT ))
until ssh -i "$KEY" "${SSH_OPTS[@]}" "${USERNAME}@${HOST}" 'exit' 2>/dev/null; do
  [ "$(date +%s)" -lt "$deadline" ] || die "no SSH after ${TIMEOUT}s. Confirm Remote Login is on and port 22 is reachable: nc -vz ${HOST} 22"
  sleep 5
done
run() { ssh -i "$KEY" "${SSH_OPTS[@]}" "${USERNAME}@${HOST}" "$@"; }
note "ssh: OK ($(run 'echo "$(whoami)@$(hostname -s) $(uname -m) macOS $(sw_vers -productVersion)"'))"

hdr "Verifying"
if DOCKER_VER="$(run 'docker info --format "{{.ServerVersion}}" 2>/dev/null')" && [ -n "$DOCKER_VER" ]; then
  note "docker: $DOCKER_VER"
else
  die "Docker is not reachable. Start Docker Desktop on the Mac, then re-run with --verify-only."
fi
# BSD df differs from GNU's; -g gives gigabytes on macOS.
note "free space: $(run 'df -g / | tail -1 | awk "{print \$4\"G\"}"' 2>/dev/null)"
note "cpu: $(run 'sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown')"

cat <<EOF

=== Ready (cpu target only) ===
  Validate with:

    scripts/remote_validate.sh ${USERNAME}@${HOST} --target cpu --full

  There is no GPU passthrough into Docker Desktop's Linux VM, so the Apple GPU and Neural
  Engine cannot be exercised and no accelerator claim can be made from this host. What it
  does prove: the image boots, the API works, and CPU decoding is correct on arm64.
  See docs/REMOTE_VALIDATION.md.
EOF
