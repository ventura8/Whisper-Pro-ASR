#!/bin/bash
# Run the hardware validation matrix across every available machine, in parallel.
#
# Machines run concurrently because they are independent hosts. Configurations *within* a
# machine run in sequence: each one rebuilds the stack and binds port 9000, and two
# services on one GPU would contend for memory and make every timing meaningless.
#
# Depth per configuration follows the tiering in the hardware-verification process: one
# full 156-test matrix per machine on its primary engine, the ~20-minute smoke set for
# every other engine/device combination, and the 20-minute long-form clip wherever there
# is an NVIDIA GPU to run it on.
#
# Which engine can use which device is a hard constraint, not a preference:
#   FASTER-WHISPER  CTranslate2  -> CUDA or CPU only; no OpenVINO backend exists
#   INTEL-WHISPER   OpenVINO     -> Intel GPU (the NPU cannot execute Whisper's dynamic IR)
#   OPENAI-WHISPER  torch        -> CUDA, or Intel XPU in the intel-xpu image
#   WHISPERX        torch        -> CUDA
# A combination outside that table would silently run on the CPU and report a clean pass.
#
# Usage:
#   scripts/validation_matrix.sh                 # every machine that answers
#   scripts/validation_matrix.sh --only local    # one machine
#   scripts/validation_matrix.sh --dry-run       # print the plan and stop
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${VALIDATION_LOG_DIR:-${REPO_ROOT}/.validation-logs}"
ONLY=""
DRY=false

while [ $# -gt 0 ]; do
  case "$1" in
    --only)    ONLY="${2:?--only needs a machine name}"; shift 2 ;;
    --dry-run) DRY=true; shift ;;
    --logs)    LOG_DIR="${2:?--logs needs a path}"; shift 2 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//' | sed '1d'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

hdr()  { printf '\n=== %s ===\n' "$*"; }
note() { printf '  %s\n' "$*"; }

# machine | host | target | engine | preprocess | suite | transport (optional)
#
# The preprocess column matters as much as the engine: UVR runs there, independently of
# ASR. On the Intel NUC the NPU does preprocessing well and cannot do ASR at all, so both
# NPU and GPU preprocessing are exercised -- otherwise the NPU's only working capability
# would go untested.
# The machine plan lives outside the repository: it names hosts and usernames, which are
# nobody else's business and change per operator. Point VALIDATION_MATRIX_PLAN at a file,
# or drop one at .validation-matrix.conf (gitignored). See .validation-matrix.conf.example.
#
# Format, one configuration per line:
#   machine | user@host | target | engine | preprocess | suite | transport (optional)
#
# ``machine`` is a physical-host identifier, not an operating-system label. Rows for a
# dual-boot host must share it so they cannot run in parallel and contend for one GPU.
# ``transport`` is ``linux`` (the default) or ``wsl[:distro]`` for a Windows SSH target.
#
# A remote host must be written as ``user@host``; a bare hostname is rejected below. The
# username is not assumed, because assuming the local one produces a "Permission denied"
# that looks like broken key auth but is only the wrong account. ``local`` and
# ``localhost`` are the exceptions and need no user.
#
# The `local` rows below need no host and are always available.
PLAN_FILE="${VALIDATION_MATRIX_PLAN:-${REPO_ROOT}/.validation-matrix.conf}"
PLAN=$(cat <<'PLAN'
local|local|full|FASTER-WHISPER|AUTO|stress
local|local|full|INTEL-WHISPER|GPU|smoke
local|local|full|OPENAI-WHISPER|AUTO|smoke
local|local|full|WHISPERX|AUTO|smoke
PLAN
)
if [ -f "$PLAN_FILE" ]; then
  PLAN="${PLAN}
$(grep -vE '^[[:space:]]*(#|$)' "$PLAN_FILE")"
else
  note "no machine plan at ${PLAN_FILE}; running the local configurations only"
  note "copy .validation-matrix.conf.example and fill in your hosts to add remote machines"
fi
# Drop blank lines and trim whitespace around every pipe-delimited field. The documented
# format is spaced ("nuc | user@host | intel | ..."), so without this a copied-from-the-docs
# row yields " intel" as the target and " smoke" as the suite -- which reach
# remote_validate.sh as unknown values, or silently mismatch the machine name in the
# `[ "$m" = "$machine" ]` filter so the row is never run at all.
PLAN="$(printf '%s\n' "$PLAN" | awk -F'|' 'NF {
  out = ""
  for (i = 1; i <= NF; i++) {
    f = $i
    gsub(/^[ \t]+|[ \t]+$/, "", f)
    out = (i == 1) ? f : out "|" f
  }
  print out
}')"

run_machine() {
  local machine="$1" log fixtures_flag first=true rc failures=0
  log="${LOG_DIR}/${machine}.log"
  : > "$log"
  while IFS='|' read -r m host target engine preprocess suite transport; do
    [ "$m" = "$machine" ] || continue
    # Fixtures are ~3.2G and identical between configurations, so sync once per machine.
    fixtures_flag=""
    if [ "$first" = true ]; then fixtures_flag="--fixtures"; first=false; fi
    {
      printf '\n########## %s | %s | %s | prep=%s | suite=%s ##########\n' "$machine" "$target" "$engine" "$preprocess" "$suite"
      date -Is
    } >> "$log"
    # </dev/null: this loop reads the plan on stdin, and ssh/scp/rsync inside the child
    # read stdin too -- they would swallow the remaining configurations, so the loop would
    # silently run only the first one.
    case "${transport:-linux}" in
      linux) transport_args=() ;;
      wsl) transport_args=(--wsl) ;;
      wsl:*) transport_args=(--wsl "${transport#wsl:}") ;;
      *)
        printf 'invalid transport %q for %s\n' "$transport" "$machine" >> "$log"
        failures=$((failures + 1))
        continue
        ;;
    esac
    case "$host" in
      local|localhost|*@*) ;;
      *)
        printf 'invalid host %q for %s: remote rows need user@host (or local)\n' "$host" "$machine" >> "$log"
        failures=$((failures + 1))
        continue
        ;;
    esac
    bash "$VALIDATE_SNAPSHOT" "$host" "${transport_args[@]}" \
      --target "$target" --engine "$engine" --preprocess "$preprocess" \
      --suite "$suite" --full --keep $fixtures_flag </dev/null >> "$log" 2>&1
    rc=$?
    [ "$rc" -eq 0 ] || failures=$((failures + 1))
    printf '########## exit=%s %s ##########\n' "$rc" "$(date -Is)" >> "$log"
  done <<< "$PLAN"
  # Returned so the wait loop below can aggregate it. Without this the function's status
  # was the final printf's -- always 0 -- and a matrix in which every configuration failed
  # still exited successfully.
  [ "$failures" -eq 0 ]
}

mkdir -p "$LOG_DIR"

# bash re-reads a script from its byte offset as it runs, so editing remote_validate.sh
# mid-run makes live runs resume inside changed text -- it has produced "syntax error near
# unexpected token" in the middle of a validated file twice. Run from a snapshot so edits
# during a multi-hour matrix are harmless.
export WHISPER_REPO_ROOT="$REPO_ROOT"
VALIDATE_SNAPSHOT="$(mktemp -t remote_validate.XXXXXX.sh)"
trap 'rm -f "$VALIDATE_SNAPSHOT"' EXIT
# The trap is armed before the copy, so a failure here still cleans up. Failing loudly
# matters: mktemp has already created the file, so a failed cp leaves a valid, EMPTY script
# that every row then "runs" successfully -- a whole matrix reporting passes for validation
# that never executed.
cp "${REPO_ROOT}/scripts/remote_validate.sh" "$VALIDATE_SNAPSHOT" \
  || { echo "ERROR: could not snapshot scripts/remote_validate.sh to $VALIDATE_SNAPSHOT" >&2; exit 1; }
# Space-separated: the membership test below is a glob on " $MACHINES ", and newlines
# would make every match fail silently.
MACHINES=$(printf '%s\n' "$PLAN" | cut -d'|' -f1 | awk '!seen[$0]++' | tr '\n' ' ')
if [ -n "$ONLY" ]; then
  # Checked against the plan: an unknown name used to yield an empty matrix that ran
  # nothing and exited 0, which reads exactly like a clean pass.
  case " $MACHINES " in
    *" $ONLY "*) MACHINES="$ONLY" ;;
    *) echo "unknown machine '$ONLY'; the plan has: ${MACHINES% }" >&2; exit 2 ;;
  esac
fi

hdr "Plan"
printf '%s\n' "$PLAN" | while IFS='|' read -r m host target engine preprocess suite transport; do
  case " $MACHINES " in *" $m "*) printf '  %-8s %-14s %-15s prep=%-5s %-8s %s\n' "$m" "$target" "$engine" "$preprocess" "${transport:-linux}" "$suite" ;; esac
done

if [ "$DRY" = true ]; then
  hdr "Dry run"; note "nothing executed"; exit 0
fi

hdr "Running (machines in parallel, configurations sequential within each)"
pids=""
for m in $MACHINES; do
  run_machine "$m" &
  pids="$pids $!"
  note "$m started (pid $!) -> ${LOG_DIR}/${m}.log"
done

EXIT_STATUS=0
for p in $pids; do
  wait "$p" || EXIT_STATUS=1
done

hdr "Summary"
for m in $MACHINES; do
  printf '\n  --- %s ---\n' "$m"
  grep -E '^##########|[0-9]+ (passed|failed)' "${LOG_DIR}/${m}.log" 2>/dev/null |
    sed 's/^/    /' | tail -40
done

if [ "$EXIT_STATUS" -ne 0 ]; then
  printf '\n  At least one configuration failed; see the logs above.\n'
fi
exit "$EXIT_STATUS"
