#!/bin/bash
# Generate the audio matrix inside a container.
#
# The TTS toolchain (piper-tts, transformers+torch, uroman) is heavy and, on a host whose
# system Python is externally managed, `poetry install --with tools` fails outright trying
# to uninstall pip. Generating in a container keeps the toolchain off the host, works the
# same on any machine, and writes fixtures straight into test_data/ where the suites and
# the remote sync expect them.
#
#   scripts/generate_fixtures_docker.sh          # generate everything missing
#   scripts/generate_fixtures_docker.sh verify   # report coverage and gaps only
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTION="${1:-all}"
# ACTION reaches the container as a positional argument to `bash -s`, not interpolated into
# the script text, so it cannot become anything but an argument. Still validated against the
# generator's own command list, so a typo fails here rather than inside the container.
case "$ACTION" in
all | core | clips | combined | adversarial | longform | verify) ;;
*)
	echo "unknown action '$ACTION' (all|core|clips|combined|adversarial|longform|verify)" >&2
	exit 2
	;;
esac
# The test image is the base because it already carries ffmpeg and the repo's runtime
# deps; python:3.12-slim needs ffmpeg installed as root, which then writes root-owned
# fixtures into the working tree.
IMAGE="whisper-pro-asr-test:latest"
# Built locally by scripts/ci/build-and-test.sh, so --pull=never below turns a missing image
# into an immediate error naming it, rather than a silent pull of an unrelated registry
# image that happens to answer to the same tag.
CACHE="${REPO_ROOT}/.fixture-tooling"

mkdir -p "$CACHE" "${REPO_ROOT}/test_data/audio_matrix"

# The pip cache is bind-mounted so a rerun does not re-download ~2.5G of torch.
# -i, because the generator script below arrives on this container's stdin. Without it
# docker gives the container no stdin at all, `bash -s` reads EOF immediately, and the run
# exits 0 having generated nothing -- a silent no-op that looks exactly like success.
docker run --rm --pull=never -i \
	-v "${REPO_ROOT}:/app" -w /app \
	-v "${CACHE}:/tooling" \
	-u "$(id -u):$(id -g)" \
	-e HOME=/tooling -e PIP_CACHE_DIR=/tooling/pip -e HF_HOME=/tooling/hf \
	-e PYTHONDONTWRITEBYTECODE=1 \
	"$IMAGE" bash -s -- "$ACTION" <<'INNER'
set -e
action="$1"
# One hash-pinned lock, resolved together. The toolchain used to be installed as
# `torch>=2.4` plus five unpinned names with --upgrade, so two runs months apart built the
# fixtures with different code -- and the matrix is content-addressed and committed, so that
# surfaces as a fixture diff nobody asked for, or as a scoring shift in the real-audio suite
# that reads like an engine regression. Regenerate deliberately with
# scripts/audio_matrix/regenerate_locks.sh.
#
# PyPI is primary and the CPU index is the extra: torch is pinned to its "+cpu" local
# version, which no other index carries, so the whole closure resolves as one set. --upgrade
# is gone; --require-hashes refuses anything whose bytes are not the locked ones.
lock=/app/scripts/audio_matrix/requirements.lock
stamp=/tooling/site/.toolchain-lock-sha256
want="$(sha256sum "$lock" | cut -d" " -f1)"
# Two conditions, because either alone lies. The stamp is written only after a successful
# install, so an interrupted run leaves it stale and the toolchain is rebuilt; and a tree
# that was completed once can still be unusable -- the recorded failure was
# /tooling/site/torch present but unimportable, surfacing much later inside MMS synthesis.
if [ "$(cat "$stamp" 2>/dev/null || true)" != "$want" ] || ! PYTHONPATH=/tooling/site python3 -c "import torch" >/dev/null 2>&1; then
  # Wiped first, so the tree ends up as exactly what the lock says. pip --target refuses to
  # replace a directory that already exists ("Specify --upgrade to force replacement") and
  # merely warns, so installing over the previous unpinned toolchain left most of it in
  # place and the lock enforced nothing -- while the stamp below recorded success. Only
  # site/ goes: /tooling also holds the pip and HuggingFace caches that make a rerun cheap.
  rm -rf /tooling/site
  # No '|| true': a failed toolchain install used to surface much later as a confusing
  # ModuleNotFoundError from the generator instead of here.
  pip install --quiet --disable-pip-version-check --target /tooling/site \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --require-hashes -r "$lock"
  printf '%s' "$want" >"$stamp"
fi
export PYTHONPATH=/tooling/site:/app
python3 scripts/generate_audio_matrix.py "$action"
INNER
