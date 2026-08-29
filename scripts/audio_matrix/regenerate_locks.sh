#!/bin/bash
# Regenerate the hash-pinned lock for the audio-matrix fixture toolchain.
#
# Run this only when the toolchain should move -- a torch or transformers upgrade, a new
# dependency. It rewrites scripts/audio_matrix/requirements.lock, and the fixtures must be
# regenerated and re-committed afterwards, because a different toolchain renders different
# audio and the matrix is content-addressed.
#
#   scripts/audio_matrix/regenerate_locks.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="whisper-pro-asr-test:latest"
OUT_DIR="${REPO_ROOT}/scripts/audio_matrix"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Resolved inside the test image, because that is the interpreter and platform the fixture
# generator actually runs on -- a lock produced against another Python resolves to wheels
# that will not install there.
#
# piper-tts carries its [ja,zh] extras here because pyproject declares them, and dropping
# them does not fail: the Japanese and Chinese phonemizers go missing and those two voices
# synthesize an EMPTY waveform, which surfaces much later as "wave.Error: # channels not
# specified" on ja_tail. Keep this list in step with the tools group in pyproject.toml.
docker run --rm --pull=never -u root \
	-v "${WORK}:/work" -v "${OUT_DIR}:/emit:ro" \
	--entrypoint bash "$IMAGE" -c '
set -euo pipefail
# Two resolutions, and the first one exists only to discover the newest CPU build. torch has
# to be pinned to its "+cpu" local version in the real resolution below: that version exists
# on no other index, which is what lets PyPI stay primary while torch still comes from
# download.pytorch.org. Resolving the two halves against separate indexes instead produced
# two different filelock versions, and install order silently picked the winner.
pip install --quiet --disable-pip-version-check --dry-run --ignore-installed \
	--report /work/torch-report.json --target /work/probe1 \
	--index-url https://download.pytorch.org/whl/cpu "torch>=2.4" >/dev/null
TORCH_VER="$(python3 -c "import json,sys; print(json.load(open(\"/work/torch-report.json\"))[\"install\"][0][\"metadata\"][\"version\"])")"
echo "resolved CPU torch: ${TORCH_VER}"

pip install --quiet --disable-pip-version-check --dry-run --ignore-installed \
	--report /work/combined-report.json --target /work/probe2 \
	--index-url https://pypi.org/simple \
	--extra-index-url https://download.pytorch.org/whl/cpu \
	"torch==${TORCH_VER}" "transformers>=4.44" uroman scipy numpy soundfile "piper-tts[ja,zh]" >/dev/null

python3 /emit/_emit_lock.py /work/combined-report.json /work/requirements.lock
'

cp "${WORK}/requirements.lock" "${OUT_DIR}/requirements.lock"
printf '\nwrote %s\n' "${OUT_DIR}/requirements.lock"
printf 'Regenerate the fixtures next: scripts/generate_fixtures_docker.sh all\n'
