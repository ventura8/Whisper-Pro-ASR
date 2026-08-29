#!/bin/bash
# Bootstrap a Linux host for remote hardware validation, with one paste.
#
# Linux is the only platform where every accelerator is reachable from a container:
# NVIDIA, AMD ROCm, Intel GPU and Intel NPU. Windows exposes CUDA only, macOS none --
# see docs/REMOTE_VALIDATION.md.
#
# Two things need the remote user's own password and so cannot be done for them:
# authorising the key, and joining the `docker` group. Both are folded into a single
# block to paste, rather than a sequence of round trips.
#
# Usage:
#   scripts/setup_linux_remote.sh <host>              # print setup, then wait and verify
#   scripts/setup_linux_remote.sh <host> --user NAME  # skip the username prompt
#   scripts/setup_linux_remote.sh <host> --verify-only
#
# Options:
#   --user NAME    Remote username.
#   --key PATH     SSH identity. Default: ~/.ssh/whisper_remote_validation (auto-created).
#   --verify-only  Skip the instructions; just test an already-configured host.
#   --timeout SEC  How long to wait for SSH. Default 600.
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

[ -n "$HOST" ] || read -r -p "Linux host (IP or name): " HOST
[ -n "$HOST" ] || die "a host is required"

if [ ! -f "$KEY" ]; then
  note "No identity at $KEY -- generating a dedicated one."
  # ssh-keygen does not create intermediate directories, so a --key under a path that does
  # not exist yet fails with a bare "No such file or directory" naming neither.
  mkdir -p "$(dirname "$KEY")"
  ssh-keygen -t ed25519 -N '' -C 'whisper-pro-asr remote hardware validation' -f "$KEY" >/dev/null
fi
PUBKEY="$(cat "${KEY}.pub")"

if [ "$VERIFY_ONLY" != true ]; then
  hdr "One block to paste on the Linux machine"
  cat <<EOF

Paste this into a terminal on ${HOST}. sudo will ask for your password -- that is yours to
type, and the only interactive part. Everything that needs root is in this one block,
because once I am on the machine over SSH I cannot answer a sudo prompt.

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
printf '%s\\n' '${PUBKEY}' >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
sudo usermod -aG docker,render,video "\$(whoami)"
echo "READY user=\$(whoami) host=\$(hostname) arch=\$(uname -m)"

Group membership applies to new logins, so no logout is needed -- each command I run opens
a fresh SSH session.

EOF
fi

if [ -z "$USERNAME" ]; then
  read -r -p "Remote username (from the READY line): " USERNAME
fi
[ -n "$USERNAME" ] || die "a username is required"

hdr "Waiting for ${USERNAME}@${HOST}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes)
deadline=$(( $(date +%s) + TIMEOUT ))
until ssh -i "$KEY" "${SSH_OPTS[@]}" "${USERNAME}@${HOST}" 'exit' 2>/dev/null; do
  [ "$(date +%s)" -lt "$deadline" ] || die "no SSH after ${TIMEOUT}s. Check sshd is running and port 22 is reachable: nc -vz ${HOST} 22"
  sleep 5
done
run() { ssh -i "$KEY" "${SSH_OPTS[@]}" "${USERNAME}@${HOST}" "$@"; }
note "ssh: OK ($(run 'echo "$(whoami)@$(hostname) $(uname -m) kernel=$(uname -r)"'))"

hdr "Verifying"
if run 'docker info >/dev/null 2>&1'; then
  note "docker: usable without sudo ($(run 'docker --version'))"
else
  die "docker still needs sudo. Confirm the usermod ran, then re-run with --verify-only."
fi
note "free space: $(run 'df -BG --output=avail / | tail -1 | tr -d " "')"

hdr "Hardware audit"
# Piped so the remote needs no checkout. Never assume which vendor is present.
AUDIT="$(run 'bash -s -- --json' < "${REPO_ROOT}/scripts/audit_hardware.sh")"
note "$AUDIT"
TARGET="$(printf '%s' "$AUDIT" | sed -n 's/.*"recommended_target":"\([^"]*\)".*/\1/p')"
NPU="$(printf '%s' "$AUDIT" | sed -n 's/.*"intel_npu":\([a-z]*\).*/\1/p')"

cat <<EOF

=== Ready ===
  Validate with:

    scripts/remote_validate.sh ${USERNAME}@${HOST} --target ${TARGET} --full
EOF
if [ "$NPU" = "true" ]; then
  cat <<EOF

  This host has an Intel NPU. AUTO ranks CUDA > AMD > GPU > NPU, so it will never be
  chosen on its own -- ask for it explicitly:

    scripts/remote_validate.sh ${USERNAME}@${HOST} --target ${TARGET} --device NPU --full
EOF
fi
