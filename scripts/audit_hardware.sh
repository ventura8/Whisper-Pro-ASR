#!/bin/bash
# Audit the host's accelerators BEFORE choosing a build target or claiming any
# hardware validation. Prints what the machine actually has, then recommends a
# BUILD_TARGET + docker-compose.<target>.yml override.
#
# A target whose accelerator is absent may only be reported as "boots and falls
# back to CPU cleanly" -- never as hardware-validated.
#
# Usage:
#   scripts/audit_hardware.sh            # human-readable report + recommendation
#   scripts/audit_hardware.sh --env      # append BUILD_TARGET / HOST_INTEL_RENDER_GID to .env
#   scripts/audit_hardware.sh --json     # machine-readable summary
set -uo pipefail

MODE="report"
case "${1:-}" in
--env) MODE="env" ;;
--json) MODE="json" ;;
-h | --help)
	grep '^#' "$0" | sed 's/^# \{0,1\}//' | sed '1d'
	exit 0
	;;
"") ;;
*)
	echo "unknown argument: $1" >&2
	exit 2
	;;
esac

have() { command -v "$1" >/dev/null 2>&1; }
note() { [ "$MODE" = "report" ] && printf '%s\n' "$*" || true; }
hdr() { [ "$MODE" = "report" ] && printf '\n=== %s ===\n' "$*" || true; }

has_nvidia=false
has_nvidia_toolkit=false
has_intel_gpu=false
has_intel_npu=false
has_amd=false
render_gid=""
# Tracked separately from render_gid: a render node exists for AMD and NVIDIA too, so a
# non-empty render_gid says only "some DRM device is here". Intel target selection needs
# a node whose DRM vendor is actually 0x8086.
intel_render_gid=""
intel_gpu_top=false
disk_free=""

hdr "NVIDIA (CUDA)"
if have nvidia-smi && nvidia-smi >/dev/null 2>&1; then
	has_nvidia=true
	note "$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null)"
	# Bounded. This contacts the Docker daemon and may pull an image, so a hung daemon, a
	# stalled registry or a broken WSL backend left the whole audit waiting with nothing on
	# screen -- and the audit is the step every validation starts with. A timeout is reported
	# as its own outcome rather than as "no GPU", because the two call for different fixes.
	gpu_probe_status=0
	if have docker; then
		timeout 120 docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi -L >/dev/null 2>&1 || gpu_probe_status=$?
	else
		gpu_probe_status=127
	fi
	if [ "$gpu_probe_status" -eq 124 ]; then
		note "Docker GPU probe timed out after 120s; the daemon is not answering. Not treated as a working toolkit."
	elif [ "$gpu_probe_status" -eq 0 ]; then
		has_nvidia_toolkit=true
		note "Docker GPU probe succeeded"
	else
		note "WARNING: GPU present but Docker GPU probe failed"
	fi
else
	note "no NVIDIA GPU / driver"
fi

hdr "Render nodes (Intel iGPU / Arc, AMD)"
if [ -d /dev/dri ]; then
	note "$(ls -l /dev/dri/ 2>/dev/null)"
	for n in /dev/dri/renderD*; do
		[ -e "$n" ] || continue
		g=$(stat -c '%g' "$n" 2>/dev/null)
		[ -n "$render_gid" ] || render_gid="$g"
		vendor=$(cat "/sys/class/drm/$(basename "$n")/device/vendor" 2>/dev/null || true)
		case "$vendor" in
		0x8086)
			[ -n "$intel_render_gid" ] || intel_render_gid="$g"
			note "  $n -> GID $g  vendor=$vendor (Intel; HOST_INTEL_RENDER_GID)"
			;;
		"") note "  $n -> GID $g  (vendor unreadable)" ;;
		*) note "  $n -> GID $g  vendor=$vendor (not Intel)" ;;
		esac
	done
else
	note "no /dev/dri render nodes"
fi

hdr "PCI display devices"
if have lspci; then
	gpus=$(lspci -nn 2>/dev/null | grep -iE 'vga|3d|display')
	note "${gpus:-<none reported>}"
	echo "$gpus" | grep -qi '\[8086:' && has_intel_gpu=true
	echo "$gpus" | grep -qi '\[10de:' && has_nvidia=true
	echo "$gpus" | grep -qi '\[1002:' && has_amd=true
else
	note "lspci not available -- inferring from device nodes only"
	# Only an Intel-vendor render node counts. Inferring Intel from any render node made an
	# NVIDIA+AMD host look hybrid and selected the nvidia-intel image, whose Intel half then
	# has no device to reach.
	[ -n "$intel_render_gid" ] && has_intel_gpu=true
fi

hdr "Intel NPU (/dev/accel)"
if [ -e /dev/accel ] || ls /dev/accel* >/dev/null 2>&1; then
	has_intel_npu=true
	note "$(ls -l /dev/accel* 2>/dev/null)"
else
	note "no Intel NPU -- absent on iGPU/Arc-only hosts; NPU paths are untestable here"
fi

hdr "AMD ROCm (/dev/kfd)"
if [ -e /dev/kfd ]; then
	has_amd=true
	note "/dev/kfd present"
	have rocm-smi && note "$(rocm-smi --showproductname 2>/dev/null | grep -i 'card' || true)"
else
	note "no /dev/kfd (AMD ROCm)"
fi

hdr "Acceleration-evidence tooling / build space"
if have intel_gpu_top; then
	intel_gpu_top=true
	note "intel_gpu_top available"
else note "intel_gpu_top NOT installed"; fi
have nvidia-smi && note "nvidia-smi available (use --query-compute-apps for CUDA evidence)"
disk_free=$(df -h / | awk 'NR==2 {print $4}')
note "free on / : ${disk_free:-unknown}"

# --- Recommendation -------------------------------------------------------------
target="cpu"
# "Intel is present" means confirmed Intel silicon: an 0x8086 PCI display device, an
# Intel-vendor render node, or an Intel NPU. A bare non-empty render_gid is not evidence --
# AMD and NVIDIA publish render nodes too, and using it made every NVIDIA+AMD host resolve
# to nvidia-intel.
# A target is only recommended when the container can actually REACH the device. lspci
# reports what is soldered to the board, which is inventory, not access: a machine whose
# Intel iGPU has no render node, or whose Radeon has no /dev/kfd (every WSL2 host, and any
# Linux box without amdgpu loaded), was recommended `intel` or `amd` and then ran every
# request on the CPU under a target name promising otherwise -- the exact false-validation
# this script exists to prevent. has_intel_gpu / has_amd stay as reported inventory in the
# JSON; only these two gate the recommendation.
has_intel_device=false
if [ -n "$intel_render_gid" ] || $has_intel_npu; then has_intel_device=true; fi
has_amd_device=false
if [ -e /dev/kfd ]; then has_amd_device=true; fi

if $has_nvidia && $has_nvidia_toolkit && $has_intel_device; then
	target="nvidia-intel"
elif $has_nvidia && $has_nvidia_toolkit; then
	target="nvidia"
elif $has_amd_device; then
	target="amd"
elif $has_intel_device; then
	target="intel"
fi

# Said plainly, because "a GPU is present but unreachable" is the state most likely to be
# misread as a working accelerator.
if $has_intel_gpu && ! $has_intel_device; then
	note ""
	note "NOTE: Intel graphics are present but no Intel render node or NPU is reachable here,"
	note "      so the intel image would have no device to use. Recommending ${target} instead."
fi
if $has_amd && ! $has_amd_device; then
	note ""
	note "NOTE: An AMD GPU is present but /dev/kfd is absent, so ROCm cannot be used from a"
	note "      container (this is always the case on WSL2). Recommending ${target} instead."
fi

# Only the Intel node is meaningful to HOST_INTEL_RENDER_GID, so it is the Intel GID or
# nothing. Falling back to the generic render GID put an AMD or NVIDIA node's group under
# an Intel-specific name: the container joined a group that grants it no Intel device, and
# the audit printed a HOST_INTEL_RENDER_GID line that looked like working Intel wiring.
render_gid="${intel_render_gid:-}"

if [ "$MODE" = "json" ]; then
	printf '{"nvidia":%s,"nvidia_toolkit":%s,"intel_gpu":%s,"intel_npu":%s,"amd":%s,"render_gid":"%s","intel_gpu_top":%s,"disk_free":"%s","recommended_target":"%s"}\n' \
		"$has_nvidia" "$has_nvidia_toolkit" "$has_intel_gpu" "$has_intel_npu" "$has_amd" \
		"${render_gid}" "$intel_gpu_top" "${disk_free}" "$target"
	exit 0
fi

hdr "Recommendation"
note "BUILD_TARGET=$target"
note "docker compose -f docker-compose.yml -f docker-compose.${target}.yml up -d --build"
if { [ "$target" = "intel" ] || [ "$target" = "nvidia-intel" ]; } && [ -n "$render_gid" ]; then
	note "HOST_INTEL_RENDER_GID=$render_gid"
fi
$has_intel_npu || note "Intel NPU absent: do NOT claim NPU validation on this host."
{ $has_nvidia && ! $has_nvidia_toolkit; } && note "Docker cannot currently access the NVIDIA GPU; do not claim CUDA validation."
note ""
note "Validate for real (against a running stack):"
note "  RUN_REAL_ASR=1 python3 -m pytest tests/integration/test_transcription_accuracy.py"
note "A correct transcript proves decoding, not acceleration -- pair with nvidia-smi"
note "--query-compute-apps (CUDA) or intel_gpu_top (Intel) evidence."

if [ "$MODE" = "env" ]; then
	touch .env
	# Rewritten by filtering the file rather than with `sed -i`, whose in-place form is a GNU
	# extension: on macOS/BSD `sed -i "s/.../"` reads the next argument as a backup suffix and
	# fails, and because the failure was the left side of an `&&` the `||` branch then
	# APPENDED a second BUILD_TARGET line. Repeated runs accumulated duplicates, and compose
	# takes the last one -- so a corrected target could be silently overridden by a stale one.
	set_env_key() {
		local key="$1" value="$2" tmp
		tmp="$(mktemp)"
		grep -v "^${key}=" .env >"$tmp" || true
		printf '%s=%s\n' "$key" "$value" >>"$tmp"
		mv "$tmp" .env
	}
	set_env_key BUILD_TARGET "$target"
	if { [ "$target" = "intel" ] || [ "$target" = "nvidia-intel" ]; } && [ -n "$render_gid" ]; then
		set_env_key HOST_INTEL_RENDER_GID "$render_gid"
	fi
	echo
	echo "Updated .env:"
	grep -E '^(BUILD_TARGET|HOST_INTEL_RENDER_GID)=' .env
fi
