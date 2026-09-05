#!/bin/bash
# Validate a build target on a remote host's accelerators, over SSH.
#
# This machine cannot validate silicon it does not have. Point this at a host that does:
# it establishes non-interactive access, audits what that host actually has, and can then
# sync, build, run and smoke-test a target there.
#
# Everything is automatic except the two steps that need the remote user's own password
# (authorising the key, and adding the user to the docker group). Those cannot be done on
# the user's behalf, so the script stops and prints the exact command to run.
#
# Usage:
#   scripts/remote_validate.sh user@host                 # preflight + hardware audit
#   scripts/remote_validate.sh user@host --full          # + sync, build, run, smoke test
#   scripts/remote_validate.sh user@host --target intel --full
#   scripts/remote_validate.sh user@host --device NPU --full
#
# Options:
#   --target NAME   Build target. Default: whatever the remote audit recommends.
#   --device DEV    ASR_DEVICE for the run (AUTO|CPU|GPU|NPU|CUDA). Default: AUTO.
#                   AUTO ranks CUDA > AMD > GPU > NPU, so pass NPU explicitly to test it.
#   --full          Sync the tree, build the target, start it, and run the accuracy suite.
#   --engine NAME   ASR_ENGINE for the run (FASTER-WHISPER|INTEL-WHISPER|OPENAI-WHISPER|
#                   WHISPERX). Default: AUTO, which picks by hardware.
#   --preprocess DEV  ASR_PREPROCESS_DEVICE for the run (AUTO|CPU|GPU|NPU|CUDA). UVR runs
#                   here, which is independent of the ASR device: on Intel the NPU does
#                   preprocessing well and cannot do ASR at all.
#   --suite NAME    Which tests to run with --full:
#                     accuracy (default, 9 tests ~4min) | smoke (~20min)
#                     full (156 tests ~2h) | stress (full matrix + the 20-min clip)
#                     longform (the 20-minute clip alone, NVIDIA only)
#   --fixtures      Sync the generated audio matrix (~3.2G). Needed for smoke and above;
#                   only the ~10-language core tier is committed.
#   --wsl [DISTRO]  The remote is a Windows host; run everything inside WSL2.
#                   Default distro: Ubuntu. Only NVIDIA reaches Linux containers there.
#   --key PATH      SSH identity. Default: ~/.ssh/whisper_remote_validation (auto-created).
#   --keep          Leave the remote stack running afterwards.
set -euo pipefail

# Normally derived from this script's own location, but validation_matrix.sh runs a
# snapshot copy from /tmp -- where that derivation yields "/" and every repo path becomes
# "//scripts/...". The override lets a copy still know where the repository is.
REPO_ROOT="${WHISPER_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
KEY="${HOME}/.ssh/whisper_remote_validation"
TARGET=""
DEVICE="AUTO"
FULL=false
KEEP=false
REMOTE=""
WSL=false
WSL_DISTRO="Ubuntu"
# rsync must run inside WSL on a Windows target. SSH there lands in cmd.exe, which has no
# rsync at all ("'rsync' is not recognized as an internal or external command"), while the
# distro has it at /usr/bin/rsync. --rsync-path runs the remote half through wsl, so the
# files land in the WSL filesystem -- which is also where Docker and the build run.
# An array, not a string: `${WSL:+"$(...)"}` looked conditional but WSL is literally the
# word "false" on a Linux target -- non-empty -- so the alternate value was always
# substituted, and an empty one at that. rsync received a stray "" argument and refused
# every Linux sync with "Empty source arg specified".
# `if`, not `[ ... ] && ...`: the latter would be the function's last command and would
# return 1 on a Linux host, which `set -e` turns into an aborted run right after the
# "Sync source" header, with nothing printed to say why.
rsync_path_args() {
  RSYNC_PATH_ARGS=()
  if [ "$WSL" = true ]; then
    RSYNC_PATH_ARGS=("--rsync-path=wsl -d ${WSL_DISTRO} rsync")
  fi
}
ENGINE="AUTO"
PREPROCESS="AUTO"
SUITE="accuracy"
FIXTURES=false

while [ $# -gt 0 ]; do
  case "$1" in
    --target) TARGET="${2:?--target needs a value}"; shift 2 ;;
    --device) DEVICE="${2:?--device needs a value}"; shift 2 ;;
    --key)    KEY="${2:?--key needs a value}"; shift 2 ;;
    --wsl)    WSL=true
              # Optional distro argument, but never swallow the next flag or a host token.
              # `local` and `localhost` are hosts too -- the argument parser accepts them
              # alongside user@host -- so `--wsl local` consumed "local" as the distro name
              # and then left no host at all, dropping the run into the interactive
              # "Remote user@host:" prompt it must never reach non-interactively.
              case "${2:-}" in ""|-*|*@*|local|localhost) shift ;; *) WSL_DISTRO="$2"; shift 2 ;; esac ;;
    --engine) ENGINE="${2:?--engine needs a value}"; shift 2 ;;
    --preprocess) PREPROCESS="${2:?--preprocess needs a value}"; shift 2 ;;
    # Validated here, not at the case below that assembles SUITE_ARGS. That one runs after
    # the sync, the image build, the container start and the health wait -- so a typo in
    # --suite was reported as "unknown --suite" twenty minutes and one full image build
    # into the run, having already torn down and replaced the remote's stack.
    --suite)  SUITE="${2:?--suite needs a value}"
              case "$SUITE" in
                accuracy|smoke|full|stress|longform) ;;
                *) echo "unknown --suite '$SUITE' (accuracy|smoke|full|stress|longform)" >&2; exit 2 ;;
              esac
              shift 2 ;;
    --fixtures) FIXTURES=true; shift ;;
    --full)   FULL=true; shift ;;
    --keep)   KEEP=true; shift ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//' | sed '1d'; exit 0 ;;
    local|localhost|*@*) REMOTE="$1"; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

hdr()  { printf '\n=== %s ===\n' "$*"; }
note() { printf '  %s\n' "$*"; }
die()  { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

# The username is asked for rather than assumed: assuming the local one produces a
# "Permission denied" that looks like broken key auth but is just the wrong account.
if [ -z "$REMOTE" ]; then
  read -r -p "Remote user@host: " REMOTE
fi
case "$REMOTE" in
  local|localhost) ;;
  *@*) ;;
  *) die "expected user@host (or 'local'), got '$REMOTE'" ;;
esac
REMOTE_HOST="${REMOTE#*@}"

# BatchMode turns a would-be password prompt into an immediate error instead of a hang,
# which is the difference between a clear failure and a stuck session.
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes)

# On Windows, sshd hands the command to cmd.exe, where `;`, `$( )`, quotes and parentheses
# are either meaningless or actively mangled -- a probe rewritten by cmd reports absent
# hardware as confidently as present hardware. So nothing is quoted through cmd: the
# command is written to a file, copied over, and executed by path inside WSL. The Windows
# SSH home is C:\Users\<user>, which WSL always sees at /mnt/c/Users/<user>.
REMOTE_USER="${REMOTE%@*}"
LOCAL=false
case "$REMOTE" in local|localhost|local@*) LOCAL=true ;; esac

# The stages below all cd into the checkout. Remotely that is the synced copy; locally it
# is the working tree itself, and hardcoding the remote path made every local stage cd
# into a directory that does not exist.
if [ "$LOCAL" = true ]; then WORK_DIR="$REPO_ROOT"; else WORK_DIR="~/whisper-pro-asr"; fi

ssh_raw() {
  if [ "$LOCAL" = true ]; then bash -c "$*"; else ssh -i "$KEY" "${SSH_OPTS[@]}" "$REMOTE" "$@"; fi
}

ssh_run() {
  # The local machine is just another target: same stages, same evidence, no transport.
  if [ "$LOCAL" = true ]; then
    ( cd "$REPO_ROOT" && bash -c "$*" )
    return
  fi
  if [ "$WSL" != true ]; then
    ssh -i "$KEY" "${SSH_OPTS[@]}" "$REMOTE" "$@"
    return
  fi
  local tmp
  tmp="$(mktemp)"
  # Docker Desktop configures credsStore=desktop.exe, a Windows credential helper that
  # needs an interactive logon session. Over SSH there is none, so every image pull dies
  # with "error getting credentials -- A specified logon session does not exist", even
  # while merely booting buildkit. Everything this script pulls is public, so point
  # DOCKER_CONFIG at a helper-free config rather than editing the user's own.
  printf '#!/bin/bash\nset -o pipefail\nexport DOCKER_CONFIG="$HOME/.docker-wpa"\nmkdir -p "$DOCKER_CONFIG"\nprintf %s > "$DOCKER_CONFIG/config.json"\n%s\n' "'{}'" "$*" > "$tmp"
  scp -i "$KEY" "${SSH_OPTS[@]}" -q "$tmp" "${REMOTE}:wpa_cmd.sh" || { rm -f "$tmp"; return 1; }
  rm -f "$tmp"
  # sed strips the CR that the copy through Windows leaves on every line; bash treats a
  # trailing CR as part of the command name and fails with a baffling "not found".
  ssh -i "$KEY" "${SSH_OPTS[@]}" "$REMOTE" \
    "wsl -d ${WSL_DISTRO} -e bash -c \"sed -i 's/\\r$//' /mnt/c/Users/${REMOTE_USER}/wpa_cmd.sh; bash /mnt/c/Users/${REMOTE_USER}/wpa_cmd.sh\""
}

hdr "Preflight: $REMOTE"

if [ ! -f "$KEY" ]; then
  note "No identity at $KEY -- generating a dedicated one (revocable on its own)."
  # ssh-keygen does not create intermediate directories, so a --key under a path that
  # does not exist yet fails with a bare "No such file or directory" naming neither.
  mkdir -p "$(dirname "$KEY")"
  ssh-keygen -t ed25519 -N '' -C 'whisper-pro-asr remote hardware validation' -f "$KEY" >/dev/null
fi

if ! ssh_run true 2>/dev/null; then
  cat <<EOF

Key authentication is not set up for ${REMOTE}.

Authorise it yourself -- this is the one step that needs your password, and it must be
typed by you, in your own terminal:

    ssh-copy-id -i ${KEY}.pub ${REMOTE}

Then re-run this script. If it still fails, the username is probably wrong, or the remote
~/.ssh permissions are too open (chmod 700 ~/.ssh; chmod 600 ~/.ssh/authorized_keys).
EOF
  exit 2
fi
if [ "$WSL" = true ]; then
  note "ssh: OK (Windows host, running inside WSL distro '${WSL_DISTRO}')"
  note "wsl: $(ssh_run 'echo "$(whoami)@$(hostname) kernel=$(uname -r)"')"
else
  note "ssh: OK ($(ssh_run 'echo "$(whoami)@$(hostname) kernel=$(uname -r)"'))"
fi

if ! ssh_run 'docker info >/dev/null 2>&1'; then
  if [ "$WSL" = true ]; then
    cat <<EOF

Docker is not reachable inside the WSL distro '${WSL_DISTRO}' on ${REMOTE_HOST}.

Docker Desktop does not expose its engine to a distro until integration is switched on for
that distro specifically, and this is the one step no script can do for you:

    Docker Desktop -> Settings -> Resources -> WSL Integration -> enable ${WSL_DISTRO} -> Apply & Restart

Then re-run this script.
EOF
    exit 3
  fi
  cat <<EOF

Docker on ${REMOTE_HOST} needs sudo, which cannot be answered non-interactively.

Run this on that machine, then re-run the script (group membership applies at next login,
so a fresh SSH session picks it up -- no reboot needed):

    sudo usermod -aG docker ${REMOTE%@*}
EOF
  exit 3
fi
note "docker: usable without sudo ($(ssh_run 'docker --version'))"

# On a Windows target, `docker --version` succeeding proves nothing: with WSL integration
# off, PATH interop finds the *Windows* docker.exe under /mnt/c and it answers --version
# quite happily. It cannot run a build from inside the distro, though -- bash hands it
# `compose -f ...` and the Windows binary mangles the arguments:
#
#     unknown shorthand flag: 'f' in -f
#
# which arrives long after preflight has declared everything fine. Check where the binary
# actually lives, so the failure is named here instead.
if [ "$WSL" = true ]; then
  DOCKER_BIN="$(ssh_run 'command -v docker 2>/dev/null || true' | tr -d '\r')"
  case "$DOCKER_BIN" in
    /mnt/*)
      cat <<EOF

Docker inside WSL distro '${WSL_DISTRO}' resolves to the Windows binary:

    ${DOCKER_BIN}

That is PATH interop reaching docker.exe on the Windows side, not Docker Desktop's WSL
integration. It answers 'docker --version', but it cannot build from inside the distro.

    Docker Desktop -> Settings -> Resources -> WSL Integration -> enable ${WSL_DISTRO} -> Apply & Restart

Then re-run this script. Verify with:

    wsl -d ${WSL_DISTRO} -e bash -lc 'command -v docker'   # expect /usr/bin/docker
EOF
      exit 3
      ;;
  esac
  note "docker binary: ${DOCKER_BIN} (native to the distro)"

  # WSL's clock drifts behind after the Windows host sleeps or hibernates, and it does not
  # always resync on resume. A behind-clock breaks image builds in a way that names
  # neither WSL nor the clock:
  #
  #     OpenPGP signature verification failed ... Not live until 2026-09-05T06:12:55Z
  #
  # apt rejects a repository signature that is "not yet valid", so every apt-get update in
  # every Dockerfile stage fails. Measured on Sergiu-PC: WSL was three hours behind UTC.
  LOCAL_EPOCH="$(date -u +%s)"

  # The Windows clock first, because it is the authoritative one. WSL and every container
  # derive their time from the host, so a skewed host cannot be fixed from inside the
  # distro: `sudo date -u -s` there appears to work, prints "clock: corrected", and is then
  # pulled back by the host's time service within minutes. Measured on this host, which sat
  # 3h behind real UTC -- the distro correction held just long enough for the audit to pass
  # and reverted before the build reached its first apt transaction, so every run failed
  # ten minutes in on "Release file ... is not valid yet" while the preflight had reported
  # the clock as fine.
  WIN_UTC="$(ssh_raw 'powershell -NoProfile -Command "(Get-Date).ToUniversalTime().ToString(\"yyyy-MM-dd HH:mm:ss\")"' 2>/dev/null | tr -d '\r' | tail -1)"
  WIN_EPOCH="$(date -u -d "$WIN_UTC" +%s 2>/dev/null || true)"
  if [ -n "$WIN_EPOCH" ]; then
    WIN_SKEW=$(( LOCAL_EPOCH - WIN_EPOCH )); [ "$WIN_SKEW" -lt 0 ] && WIN_SKEW=$(( -WIN_SKEW ))
    if [ "$WIN_SKEW" -gt 120 ]; then
      cat <<EOF

The Windows clock on ${REMOTE_HOST} is ${WIN_SKEW}s out from this machine.

    Windows UTC: ${WIN_UTC}
    this machine: $(date -u '+%Y-%m-%d %H:%M:%S')

Every WSL distro and every container takes its time from Windows, so this cannot be
corrected from inside the distro -- setting it there is overwritten by the host's time
service a few minutes later, which is long enough for preflight to pass and for the build
to fail afterwards on "Release file ... is not valid yet".

Fix it on the Windows host (an Administrator PowerShell), then re-run:

    w32tm /resync /force
    # if that fails, Settings -> Time & Language -> Date & time -> Set time automatically,
    # and check the time zone is right: a clock set to UTC under a UTC+N zone reads N hours
    # behind, which is exactly this symptom.

Verify with:

    wsl -d ${WSL_DISTRO} -e date -u
EOF
      exit 3
    fi
    note "clock: Windows host within ${WIN_SKEW}s of this machine"
  fi

  WSL_EPOCH="$(ssh_run 'date -u +%s' | tr -dc '0-9')"
  if [ -n "$WSL_EPOCH" ]; then
    SKEW=$(( LOCAL_EPOCH - WSL_EPOCH )); [ "$SKEW" -lt 0 ] && SKEW=$(( -SKEW ))
    if [ "$SKEW" -gt 120 ]; then
      note "clock: WSL is ${SKEW}s out from this machine; correcting (builds fail on skew)"
      if ssh_run "sudo -n date -u -s '$(date -u '+%Y-%m-%d %H:%M:%S')' >/dev/null 2>&1"; then
        note "clock: corrected ($(ssh_run 'date -u "+%Y-%m-%d %H:%M:%S UTC"' | tr -d '\r'))"
      else
        cat <<EOF

WSL's clock in '${WSL_DISTRO}' is ${SKEW}s out from this machine, and it could not be
corrected without a password. apt rejects repository signatures dated in its future, so
image builds fail with "Not live until <timestamp>".

Fix it on the Windows host, then re-run:

    wsl --shutdown                       # simplest: restarting the distro resyncs it
    wsl -d ${WSL_DISTRO} -e bash -lc "sudo date -u -s '$(date -u '+%Y-%m-%d %H:%M:%S')'"
EOF
        exit 3
      fi
    else
      note "clock: within ${SKEW}s of this machine"
    fi
  fi

  # The distro's clock is not the one the build sees. Docker Desktop runs its engine in a
  # separate utility VM, and every build container takes its time from THAT VM -- which the
  # correction above does not touch, because it only reaches the distro.
  #
  # Measured on this host: the distro read correct UTC while build containers were 1h19m
  # behind, and apt rejected Ubuntu's repository signatures with
  #
  #     E: Release file for .../noble-updates/InRelease is not valid yet
  #        (invalid for another 1h 19min 17s)
  #
  # twenty minutes into a build, naming neither the clock nor Docker Desktop. Checked here
  # so the failure is immediate and says what to do.
  CONTAINER_EPOCH="$(ssh_run 'docker run --rm alpine date -u +%s 2>/dev/null' | tr -dc '0-9')"
  if [ -n "$CONTAINER_EPOCH" ]; then
    CSKEW=$(( $(date -u +%s) - CONTAINER_EPOCH )); [ "$CSKEW" -lt 0 ] && CSKEW=$(( -CSKEW ))
    if [ "$CSKEW" -gt 120 ]; then
      cat <<EOF

Docker build containers on ${REMOTE_HOST} are ${CSKEW}s out from this machine.

This is Docker Desktop's own utility VM, not the '${WSL_DISTRO}' distro -- correcting the
distro's clock (done above) does not touch it. apt rejects repository signatures dated in
its future, so every image build fails with "Release file ... is not valid yet".

Only a restart resyncs that VM, and it cannot be done safely from this session because it
stops every distro on the host. Run this on the Windows machine, then re-run:

    wsl --shutdown
    # then start Docker Desktop again and wait for it to report Running

Verify with:

    wsl -d ${WSL_DISTRO} -e bash -lc "docker run --rm alpine date -u"
EOF
      exit 3
    fi
    note "clock: build containers within ${CSKEW}s of this machine"
  fi
fi
note "disk: $(ssh_run 'df -BG --output=avail / | tail -1 | tr -d " "') free"

hdr "Hardware audit (vendor-agnostic)"
if [ "$WSL" = true ]; then
  # Copied and run by path for the same reason ssh_run does it: nothing survives cmd.exe
  # quoting reliably, and -EncodedCommand's UTF-16 base64 overruns cmd's 8191-char limit.
  scp -i "$KEY" "${SSH_OPTS[@]}" -q "${REPO_ROOT}/scripts/audit_hardware_windows.ps1" \
    "${REMOTE}:wpa_audit.ps1" || die "could not copy the Windows audit script"
  # `|| true`: grep exits 1 when the audit printed no JSON line, and under `set -e` a
  # failing pipeline inside a command substitution aborts the script right here -- before
  # the guard below can report *why*. The run died with no message at all whenever the
  # remote audit failed, which is exactly the case the message exists for.
  AUDIT="$(ssh_raw "powershell -NoProfile -ExecutionPolicy Bypass -File wpa_audit.ps1" 2>/dev/null | tr -d '\r' | grep '^{' | tail -1 || true)"
  [ -n "$AUDIT" ] || die "the Windows audit returned nothing"
  note "$AUDIT"
  if [ -z "$TARGET" ]; then
    TARGET="$(printf '%s' "$AUDIT" | sed -n 's/.*"recommended_target":"\([^"]*\)".*/\1/p')"
    note "target: $TARGET (from Windows audit)"
  fi
  if ! printf '%s' "$AUDIT" | grep -q '"wsl_nvidia_smi":"[^"]'; then
    note "NOTE: nvidia-smi is not visible inside WSL, so CUDA cannot be validated here yet."
    note "      Only NVIDIA reaches Linux containers on Windows; Intel GPU/NPU never do, and"
    note "      AMD appears via /dev/dxg but falls back to CPU. Treat a pass as CPU-only."
  fi
else
# Piped so the remote needs nothing checked out. Never assume which vendor is present:
# a host offered for one accelerator may carry another worth validating instead.
AUDIT="$(ssh_run 'bash -s -- --json' < "${REPO_ROOT}/scripts/audit_hardware.sh")"
note "$AUDIT"

json_field() { printf '%s' "$AUDIT" | sed -n "s/.*\"$1\":\"\{0,1\}\([^,\"}]*\)\"\{0,1\}.*/\1/p"; }
RECOMMENDED="$(json_field recommended_target)"
RENDER_GID="$(json_field render_gid)"
[ -n "$TARGET" ] || TARGET="$RECOMMENDED"
note "target: ${TARGET}$([ "$TARGET" = "$RECOMMENDED" ] && echo ' (from audit)' || echo " (overridden; audit said ${RECOMMENDED})")"
fi

[ -n "$TARGET" ] || die "hardware audit did not produce a build target; fix the audit result or pass --target explicitly."

# Only the Intel override consumes this, and only on Linux; WSL never reaches an Intel
# GPU, so the default keeps the variable defined without implying a device is there.
RENDER_GID="${RENDER_GID:-990}"

if [ "$FULL" != true ]; then
  cat <<EOF

Audit complete. Re-run with --full to sync, build and validate:

    scripts/remote_validate.sh ${REMOTE} --target ${TARGET} --device ${DEVICE} --full
EOF
  exit 0
fi

if [ "$LOCAL" = true ]; then
  note "local run: no sync needed (working tree is the source)"
else
hdr "Sync source -> ${REMOTE_HOST}"
# Weights and caches are deliberately excluded: they are large and are provisioned on the
# remote at first start.
ssh_run "mkdir -p ${WORK_DIR}"
# Build caches are the trap here: .buildx-cache is 75G of local Docker layer cache that is
# worthless on the remote, and rsync will happily push all of it across the network and
# fill the target's disk. Exclude every cache directory, not just the obvious data ones.
rsync_path_args
rsync -az --delete ${RSYNC_PATH_ARGS[@]+"${RSYNC_PATH_ARGS[@]}"} \
  --exclude '.git' --exclude 'model_cache' --exclude 'test_data' --exclude 'data' \
  --exclude '__pycache__' --exclude '.claude/worktrees' \
  --exclude '.buildx-cache' --exclude '.cache' --exclude '*.log' \
  --exclude '.venv' --exclude 'node_modules' --exclude '.pytest_cache' --exclude '.ruff_cache' \
  -e "ssh -i $KEY ${SSH_OPTS[*]}" \
  "${REPO_ROOT}/" "${REMOTE}:~/whisper-pro-asr/"
note "synced commit: $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
fi

if [ "$FIXTURES" = true ] && [ "$LOCAL" != true ]; then
  # Only the core tier is committed; the long tail, code-switched, adversarial and
  # long-form clips are generated locally. Copying them beats regenerating on each remote,
  # which would mean installing the TTS toolchain and downloading voices per machine.
  if [ -d "${REPO_ROOT}/test_data/audio_matrix" ]; then
    hdr "Sync audio fixtures -> ${REMOTE_HOST}"
    ssh_run "mkdir -p ${WORK_DIR}/test_data/audio_matrix"
    rsync_path_args
    rsync -az --info=progress2 ${RSYNC_PATH_ARGS[@]+"${RSYNC_PATH_ARGS[@]}"} \
      -e "ssh -i $KEY ${SSH_OPTS[*]}" \
      "${REPO_ROOT}/test_data/audio_matrix/" "${REMOTE}:~/whisper-pro-asr/test_data/audio_matrix/" 2>&1 | tail -1
    note "fixtures synced: $(ssh_run "find ${WORK_DIR}/test_data/audio_matrix -type f | wc -l") files"
  else
    note "no local fixtures at test_data/audio_matrix; generate them first with scripts/generate_audio_matrix.py all"
  fi
fi

# `devices:` is mandatory in Compose, so the NPU passthrough lives in its own override and
# is layered in only when the host really has /dev/accel. Without it OpenVINO enumerates
# "CPU, GPU" and an ASR_DEVICE=NPU request runs on the iGPU while every log line still
# says NPU -- a false pass that looks exactly like a real one.
COMPOSE_FILES="-f docker-compose.yml -f docker-compose.${TARGET}.yml"
# Pass the NPU through whenever the host has one, not only when --device NPU is given.
# Tying this to --device was wrong: that flag selects the *ASR* device, and the NPU cannot
# run ASR at all -- its working use is UVR preprocessing. Gating on it left the NPU
# invisible to the container in exactly the configuration that wants it.
case "$TARGET" in
  intel*|nvidia-intel|full)
    if ssh_run 'test -e /dev/accel'; then
      COMPOSE_FILES="${COMPOSE_FILES} -f docker-compose.intel-npu.yml"
      note "Intel NPU passthrough enabled (/dev/accel present)"
    elif [ "$DEVICE" = "NPU" ]; then
      die "--device NPU requested but the host has no /dev/accel. Load the intel_vpu module, or drop --device NPU."
    fi
    ;;
  *)
    [ "$DEVICE" != "NPU" ] || die "--device NPU needs an Intel target, not ${TARGET}."
    ;;
esac

# docker-compose.yml exports a local build cache, which the default "docker" driver cannot
# do -- the build dies with "Cache export is not supported for the docker driver" on any
# machine that has not been set up for buildx. Provision a container driver rather than
# requiring the remote to have been prepared by hand.
# `inspect --bootstrap`, not a bare `inspect`. A builder can exist and still be unusable:
# on Docker Desktop/WSL the buildkit container keeps a bind mount into
# /run/desktop/mnt/host/wsl/docker-desktop-bind-mounts/<hash>, and that path does not
# survive a WSL shutdown or a Docker Desktop restart. A bare inspect reported the builder
# as present and healthy while every build died at "booting buildkit" with
#
#     invalid mount config for type "bind": bind source path does not exist
#
# which reads like a problem with this repo's Dockerfile rather than a stale builder left
# over from the previous session. Bootstrapping proves it can actually start.
if ! ssh_run 'docker buildx inspect --bootstrap whisper-builder >/dev/null 2>&1'; then
  note "buildx builder is missing or cannot boot; recreating it"
  ssh_run 'docker buildx rm -f whisper-builder >/dev/null 2>&1; true'
  ssh_run 'docker buildx create --name whisper-builder --driver docker-container --bootstrap >/dev/null 2>&1' \
    || die "could not create a working buildx builder on ${REMOTE_HOST}; check 'docker buildx ls' there"
fi
ssh_run 'docker buildx use whisper-builder' >/dev/null 2>&1 || true

hdr "Build ${TARGET}"
ssh_run "cd ${WORK_DIR} && BUILD_TARGET=${TARGET} docker compose ${COMPOSE_FILES} build" \
  || die "remote build of ${TARGET} failed"

hdr "Start ${TARGET} (ASR_DEVICE=${DEVICE})"
# Clear any stack left by a previous run first. Without this, a run that died before its
# teardown leaves a container holding the name and every later run fails with
# "Conflict. The container name is already in use" -- which reads like a bug in the
# target rather than leftover state.
ssh_run "cd ${WORK_DIR} && docker compose ${COMPOSE_FILES} down --remove-orphans >/dev/null 2>&1; docker rm -f whisper-pro-asr >/dev/null 2>&1; true"
ssh_run "cd ${WORK_DIR} && HOST_INTEL_RENDER_GID=${RENDER_GID} BUILD_TARGET=${TARGET} ASR_DEVICE=${DEVICE} \
  ASR_ENGINE=${ENGINE} ASR_PREPROCESS_DEVICE=${PREPROCESS} docker compose ${COMPOSE_FILES} up -d" || die "remote start failed"

note "waiting for health..."
ssh_run 'for i in $(seq 1 120); do curl -sf http://127.0.0.1:9000/status >/dev/null && exit 0; sleep 3; done; exit 1' \
  || die "service did not become healthy; check: ssh $REMOTE 'docker logs whisper-pro-asr'"

hdr "What actually loaded"
# The banner is the authority, not the HTTP status: a 200 is entirely compatible with a
# silent CPU fallback.
ssh_run 'docker logs whisper-pro-asr 2>&1 | grep -E "Whisper Pro ASR [0-9]|ASR Engine  |ASR Runtime|Preprocess Device|Resource Pool|OpenVINO devices" | tail -8'

hdr "Test image"
# The accuracy suite runs inside the test image, which is separate from the shippable
# targets and is not produced by `docker compose build`. Build it on demand rather than
# assuming the remote has one -- a stale or missing image is the difference between a
# real result and a confusing failure.
if ssh_run 'docker image inspect whisper-pro-asr-test:latest >/dev/null 2>&1'; then
  note "whisper-pro-asr-test:latest already present"
else
  note "building whisper-pro-asr-test:latest (not present on remote)"
  ssh_run "cd ${WORK_DIR} && docker build -f Dockerfile.test --target test -t whisper-pro-asr-test:latest ." \
    || die "remote build of the test image failed"
fi

# Each tier is a deliberate budget, not an arbitrary subset: accuracy proves decoding in
# minutes, smoke spans scripts and degraded input in under 20, full covers all 156 language
# entries in ~2h, stress adds the 20-minute clip that catches decoder loops and invented
# speech in silence -- none of which an 8-second fixture can show.
# --gpus all fails outright on a host without the NVIDIA runtime, so it tracks the target
# rather than being passed unconditionally.
case "$TARGET" in
  *nvidia*|full) GPU_FLAG="--gpus all" ;;
  *)             GPU_FLAG="" ;;
esac

case "$SUITE" in
# The marker expression travels as an environment variable, not inline. It contains spaces,
# and the pytest command is already inside `bash -c '...'` inside an ssh argument -- the
# quotes did not survive, so `--suite smoke` silently ran the entire 156-test matrix and
# only the 13-minute runtime gave it away.
  accuracy) SUITE_ARGS="tests/integration/test_transcription_accuracy.py"; SUITE_ENV="" ;;
  smoke)    SUITE_ARGS='tests/real_audio -m "$PYTEST_MARK"'; SUITE_ENV="-e PYTEST_MARK='real_audio and smoke'" ;;
  full)     SUITE_ARGS="tests/real_audio -m real_audio"; SUITE_ENV="" ;;
  # The long-form clip is deselected here and run by its own stage below, which reports
  # it separately. Leaving it in the matrix selection ran the 20-minute clip twice.
  stress)   SUITE_ARGS='tests/real_audio -m real_audio --deselect tests/real_audio/test_longform_stress.py'
            SUITE_ENV="-e RUN_GPU_LONG_ASR=1" ;;
  # Just the 20-minute clip: chunk boundaries, VAD across long pauses, decoder loops and
  # invented speech in silence -- none of which an 8-second fixture can show. NVIDIA only,
  # and far cheaper than the full matrix when comparing engines against each other.
  longform) SUITE_ARGS="tests/real_audio/test_longform_stress.py"; SUITE_ENV="-e RUN_GPU_LONG_ASR=1" ;;
  # Unreachable: --suite is validated during argument parsing. Kept so a future tier added
  # to one list and not the other fails here rather than running with an empty selection.
  *)        die "unknown --suite '$SUITE' (accuracy|smoke|full|stress|longform)" ;;
esac

hdr "Suite: ${SUITE} (engine=${ENGINE}, preprocess=${PREPROCESS}, real speech over HTTP)"
note "this can take ~2h for full/stress; output is trimmed to the last 80 lines"
# The pipeline runs inside the container with pipefail set, and the trim happens there too.
# With `pytest | tail` split across the boundary the exit status was tail's -- always 0 --
# so a failing suite set SUITE_FAILED=false and the run reported a clean pass.
ssh_run "cd ${WORK_DIR} && docker run --rm --network host ${GPU_FLAG} -v \$PWD:/app -w /app -u \$(id -u):\$(id -g) \
  -e WHISPER_PRO_ASR_TEST_IMAGE=1 -e RUN_REAL_ASR=1 -e HOME=/tmp -e WHISPER_BASE_URL=http://127.0.0.1:9000 \
  -e REAL_ASR_TIMEOUT=900 ${SUITE_ENV} \
  whisper-pro-asr-test:latest /bin/bash -c 'set -o pipefail; python3 -m pytest ${SUITE_ARGS} -ra --no-cov --tb=short 2>&1 | tail -80'" \
  || SUITE_FAILED=true

if [ "$SUITE" = "stress" ]; then
  hdr "Long-form stress (20 minutes of audio)"
  ssh_run "cd ${WORK_DIR} && docker run --rm --network host ${GPU_FLAG} -v \$PWD:/app -w /app -u \$(id -u):\$(id -g) \
    -e WHISPER_PRO_ASR_TEST_IMAGE=1 -e RUN_REAL_ASR=1 -e RUN_GPU_LONG_ASR=1 -e HOME=/tmp \
    -e WHISPER_BASE_URL=http://127.0.0.1:9000 \
    whisper-pro-asr-test:latest /bin/bash -c 'set -o pipefail; python3 -m pytest tests/real_audio/test_longform_stress.py -ra --no-cov --tb=short -s 2>&1 | tail -40'" \
    || { note "long-form stress reported failures (see above)"; SUITE_FAILED=true; }
fi

# A failing run is the one time the container logs matter most, so do not tear the stack
# down underneath it. `tail -15` used to cut the tracebacks off too, leaving a list of
# test names with no reason attached -- which is how an NPU run looked identical whether
# it failed on timeouts or on wrong output.
if [ "${SUITE_FAILED:-false}" = true ]; then
  note "the suite reported failures -- keeping the stack up so the logs survive"
  note "  ssh ${REMOTE} 'docker logs whisper-pro-asr 2>&1 | tail -60'"
  KEEP=true
fi

if [ "$KEEP" != true ]; then
  hdr "Teardown"
  ssh_run "cd ${WORK_DIR} && docker compose ${COMPOSE_FILES} down" >/dev/null 2>&1 || true
  note "stack stopped (use --keep to leave it running)"
fi

hdr "Done"
note "host:   $REMOTE_HOST"
note "target: $TARGET (ASR_DEVICE=$DEVICE)"
note "A passing suite proves decoding worked. Confirm the accelerator was USED by sampling"
note "it during a transcription -- see docs/REMOTE_VALIDATION.md."

# The exit status is the whole result as far as a caller is concerned. Without this the
# script always exited 0, so validation_matrix.sh aggregated a matrix of failures into a
# clean pass no matter what the suites did.
if [ "${SUITE_FAILED:-false}" = true ]; then
  note "result: FAILED (see the suite output above)"
  exit 1
fi
note "result: passed"
