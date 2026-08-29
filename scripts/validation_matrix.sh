#!/bin/bash
# Run the hardware validation matrix across every available machine, in parallel.
#
# Machines run concurrently because they are independent hosts. Configurations *within* a
# machine run in sequence: each one rebuilds the stack and binds port 9000, and two
# services on one GPU would contend for memory and make every timing meaningless.
#
# Depth per configuration is the SMOKE tier by default (~20 minutes), which is what
# docs/SETUP.md calls "the one a pipeline or a pre-merge check should run". The `full`
# (156 tests, ~2h) and `stress` (full matrix plus the 20-minute long-form clip) tiers are
# opt-in and belong to a release, not to every review wave: a wave that touches one module
# does not need every language re-decoded on every machine, and two hosts each spending two
# hours is how a validation step stops being run at all.
#
# Change a row's last column to full or stress when a release actually wants that depth.
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
	--only)
		ONLY="${2:?--only needs a machine name}"
		shift 2
		;;
	--dry-run)
		DRY=true
		shift
		;;
	--logs)
		LOG_DIR="${2:?--logs needs a path}"
		shift 2
		;;
	-h | --help)
		grep '^#' "$0" | sed 's/^# \{0,1\}//' | sed '1d'
		exit 0
		;;
	*)
		echo "unknown argument: $1" >&2
		exit 2
		;;
	esac
done

hdr() { printf '\n=== %s ===\n' "$*"; }
note() { printf '  %s\n' "$*"; }

# machine | host | target | engine | preprocess | suite | transport | device | separation
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
#   machine | user@host | target | engine | preprocess | suite | transport | device | separation
#
# The last three are optional and may be left empty; defaults are linux / AUTO / off.
#
# ``separation`` is ENABLE_VOCAL_SEPARATION, and it is the reason the ``preprocess`` column
# means anything at all. Until it existed this runner never passed --separation, so UVR never
# ran on any row -- every plan named a preprocessing device that was then not exercised, and
# the matrix reported passes for routing it had never touched. A row that names a preprocess
# device almost always wants ``on``.
#
# ``device`` is ASR_DEVICE. AUTO ranks CUDA > AMD > GPU > NPU, so a device that AUTO would
# never choose -- the Intel NPU especially -- has to be asked for by name.
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
#
# Not the `full` target, which these rows used to name. `full` passes /dev/kfd through for
# the AMD half of the image, and Compose refuses to start when a listed device does not
# exist -- so on any developer machine without an AMD GPU all four local rows died with
# `error gathering device information while adding custom device "/dev/kfd"`, before a
# single test ran. docker-compose.full.yml says so in its own header: a host missing a
# vendor should use the narrower override. `nvidia-intel` is that override here, and
# WhisperX gets `nvidia-whisperx` because nvidia-intel does not ship it.
#
# OPENAI-WHISPER is deliberately absent from these local rows. torch loads large-v3
# unquantized and it does not fit an 8 GB card: measured on an RTX 3080 laptop GPU as
# "OutOfMemoryError: CUDA out of memory ... GPU 0 has a total capacity of 7.66 GiB", which is
# a property of the machine, not a defect. FASTER-WHISPER fits because CTranslate2 quantizes.
# The engine is covered on hosts that can hold it: the RTX 5090 row and the NUC's intel-xpu
# row both pass.
#
# Row 2 is the one worth understanding: an explicit Intel preprocess device on a CUDA ASR
# host is the combination that used to load the Intel ONNX Runtime and then serve UVR on the
# CPU while every log line named the iGPU. It must now report CUDAExecutionProvider. Row 3
# is the same stack with isolation off, so a failure there is the decode path rather than
# the preprocessing one.
PLAN_FILE="${VALIDATION_MATRIX_PLAN:-${REPO_ROOT}/.validation-matrix.conf}"
PLAN=$(
	cat <<'PLAN'
local|local|nvidia-intel|FASTER-WHISPER|AUTO|smoke|||on
local|local|nvidia-intel|FASTER-WHISPER|GPU|accuracy||CUDA|on
local|local|nvidia-intel|FASTER-WHISPER|AUTO|accuracy|||off
local|local|nvidia-whisperx|WHISPERX|AUTO|smoke|||on
local|local|cpu|FASTER-WHISPER|CPU|accuracy||CPU|on
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

# Plan validation. A row naming something that does not exist does not fail loudly: it
# runs, silently on the CPU, and reports a clean pass -- the one outcome this matrix exists
# to make impossible.
#
# NOTE the columns. `preprocess` is ASR_PREPROCESS_DEVICE, which is where UVR runs and is
# deliberately independent of the ASR engine (see the header above: on the NUC the NPU does
# preprocessing well and cannot do ASR at all). So there is no engine-vs-preprocess rule to
# apply here -- the engine/device table in the header is about the ASR device, which these
# rows do not carry. What IS checkable is that each value exists, and that a preprocessing
# device is reachable from the image the row builds.
engine_is_known() {
	case "$1" in FASTER-WHISPER | INTEL-WHISPER | OPENAI-WHISPER | WHISPERX) return 0 ;; esac
	return 1
}

# UVR reaches a device only if the image carries that vendor's ONNX Runtime provider.
preprocess_is_reachable() {
	local device="$1" target="$2"
	case "$device" in
	AUTO | CPU) return 0 ;;
	CUDA) case "$target" in *nvidia* | full) return 0 ;; esac ;;
	GPU | NPU) case "$target" in intel | intel-xpu | intel-npu | nvidia-intel | full) return 0 ;; esac ;;
	AMD) case "$target" in amd | amd-rocm-torch | full) return 0 ;; esac ;;
	esac
	return 1
}

device_is_known() {
	case "${1:-AUTO}" in "" | AUTO | CPU | GPU | NPU | CUDA | AMD) return 0 ;; esac
	return 1
}

# Accepted spellings kept deliberately narrow. A row reading "yes" or "1" would otherwise be
# treated as off and silently validate the wrong thing -- which is the failure this whole
# column exists to end.
separation_is_known() {
	case "${1:-off}" in "" | on | off | true | false) return 0 ;; esac
	return 1
}

validate_plan() {
	local problems=0 m host target engine preprocess suite transport device separation
	while IFS='|' read -r m host target engine preprocess suite transport device separation; do
		[ -n "$m" ] || continue
		if ! device_is_known "$device"; then
			echo "${m}: unknown device '${device}' (AUTO|CPU|GPU|NPU|CUDA|AMD)" >&2
			problems=$((problems + 1))
		fi
		if ! separation_is_known "$separation"; then
			echo "${m}: unknown separation '${separation}' (on|off)" >&2
			problems=$((problems + 1))
		fi
		if [ ! -f "${REPO_ROOT}/docker-compose.${target}.yml" ]; then
			echo "${m}: invalid target '${target}' (no docker-compose.${target}.yml)" >&2
			problems=$((problems + 1))
		fi
		case "$suite" in accuracy | smoke | full | stress | longform) ;;
		*)
			echo "${m}: invalid suite '${suite}'" >&2
			problems=$((problems + 1))
			;;
		esac
		if ! engine_is_known "$engine"; then
			echo "${m}: unknown engine '${engine}'" >&2
			problems=$((problems + 1))
		fi
		if ! preprocess_is_reachable "$preprocess" "$target"; then
			echo "${m}: the ${target} image cannot reach ${preprocess} for preprocessing; UVR would run on the CPU and report a pass" >&2
			problems=$((problems + 1))
		fi
	done <<<"$PLAN"
	[ "$problems" -eq 0 ] || {
		echo "refusing to run: ${problems} invalid plan row(s)" >&2
		exit 2
	}
}

run_machine() {
	local machine="$1" log fixtures_flag first=true rc failures=0
	log="${LOG_DIR}/${machine}.log"
	: >"$log"
	while IFS='|' read -r m host target engine preprocess suite transport device separation; do
		[ "$m" = "$machine" ] || continue
		# Fixtures are ~3.2G and identical between configurations, so sync once per machine.
		# The flag is chosen here but `first` is only consumed below, after the transport and
		# host checks -- an invalid first row used to claim the one --fixtures run and `continue`
		# without ever syncing, so every later row on that machine ran against absent fixtures.
		fixtures_flag=""
		if [ "$first" = true ]; then fixtures_flag="--fixtures"; fi
		{
			printf '\n########## %s | %s | %s | prep=%s | suite=%s ##########\n' "$machine" "$target" "$engine" "$preprocess" "$suite"
			# +%%FT%%T%%z, not -Is: BSD/macOS date rejects -I entirely, and this script is
			# written to run there (see the bash 3.2 note below).
			date +%Y-%m-%dT%H:%M:%S%z
		} >>"$log"
		# </dev/null: this loop reads the plan on stdin, and ssh/scp/rsync inside the child
		# read stdin too -- they would swallow the remaining configurations, so the loop would
		# silently run only the first one.
		case "${transport:-linux}" in
		linux) transport_args=() ;;
		wsl) transport_args=(--wsl) ;;
		wsl:*) transport_args=(--wsl "${transport#wsl:}") ;;
		*)
			printf 'invalid transport %q for %s\n' "$transport" "$machine" >>"$log"
			failures=$((failures + 1))
			continue
			;;
		esac
		case "$host" in
		local | localhost | *@*) ;;
		*)
			printf 'invalid host %q for %s: remote rows need user@host (or local)\n' "$host" "$machine" >>"$log"
			failures=$((failures + 1))
			continue
			;;
		esac
		# Only now, with the row known to be runnable, does it consume the fixture sync.
		first=false
		# ${a[@]+"${a[@]}"}, not "${a[@]}": bash 3.2 (the system bash on macOS) treats an empty
		# array as unset under `set -u` and aborts the expansion. The default `linux` transport
		# sets no arguments at all, so every plain Linux row died there before running.
		# Arrays, not interpolated strings: --separation takes no value, so it is either
		# present or absent, and an empty "" argument would reach remote_validate.sh as an
		# unknown token and abort the row.
		local device_args=() separation_args=()
		[ -z "$device" ] || device_args=(--device "$device")
		case "${separation:-off}" in on | true) separation_args=(--separation) ;; esac
		bash "$VALIDATE_SNAPSHOT" "$host" ${transport_args[@]+"${transport_args[@]}"} \
			${device_args[@]+"${device_args[@]}"} ${separation_args[@]+"${separation_args[@]}"} \
			--target "$target" --engine "$engine" --preprocess "$preprocess" \
			--suite "$suite" --full --keep $fixtures_flag </dev/null >>"$log" 2>&1
		rc=$?
		[ "$rc" -eq 0 ] || failures=$((failures + 1))
		printf '########## exit=%s %s ##########\n' "$rc" "$(date +%Y-%m-%dT%H:%M:%S%z)" >>"$log"
	done <<<"$PLAN"
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
# No ".sh" suffix after the X's: BSD/macOS mktemp requires the template to END in X's and
# rejects anything after them, so the snapshot could not be created there at all. The file
# is run with an explicit `bash "$VALIDATE_SNAPSHOT"`, which needs no extension.
VALIDATE_SNAPSHOT="$(mktemp -t remote_validate.XXXXXX)"
trap 'rm -f "$VALIDATE_SNAPSHOT"' EXIT
# The trap is armed before the copy, so a failure here still cleans up. Failing loudly
# matters: mktemp has already created the file, so a failed cp leaves a valid, EMPTY script
# that every row then "runs" successfully -- a whole matrix reporting passes for validation
# that never executed.
cp "${REPO_ROOT}/scripts/remote_validate.sh" "$VALIDATE_SNAPSHOT" ||
	{
		echo "ERROR: could not snapshot scripts/remote_validate.sh to $VALIDATE_SNAPSHOT" >&2
		exit 1
	}
# Space-separated: the membership test below is a glob on " $MACHINES ", and newlines
# would make every match fail silently.
MACHINES=$(printf '%s\n' "$PLAN" | cut -d'|' -f1 | awk '!seen[$0]++' | tr '\n' ' ')
if [ -n "$ONLY" ]; then
	# Checked against the plan: an unknown name used to yield an empty matrix that ran
	# nothing and exited 0, which reads exactly like a clean pass.
	case " $MACHINES " in
	*" $ONLY "*) MACHINES="$ONLY" ;;
	*)
		echo "unknown machine '$ONLY'; the plan has: ${MACHINES% }" >&2
		exit 2
		;;
	esac
fi

# Before the plan is printed and long before anything runs: a bad row must cost seconds,
# not a multi-hour matrix that reports a pass for hardware it never touched.
validate_plan

hdr "Plan"
printf '%s\n' "$PLAN" | while IFS='|' read -r m host target engine preprocess suite transport device separation; do
	case " $MACHINES " in *" $m "*) printf '  %-8s %-14s %-15s dev=%-5s prep=%-5s uvr=%-3s %-8s %s\n' "$m" "$target" "$engine" "${device:-AUTO}" "$preprocess" "${separation:-off}" "${transport:-linux}" "$suite" ;; esac
done

if [ "$DRY" = true ]; then
	hdr "Dry run"
	note "nothing executed"
	exit 0
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
