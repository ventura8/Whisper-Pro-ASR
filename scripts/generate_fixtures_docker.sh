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
# ACTION is interpolated into a `bash -c` string inside the container, so it is checked
# against the generator's own command list before it can become anything else.
case "$ACTION" in
  all|core|clips|combined|adversarial|longform|verify) ;;
  *) echo "unknown action '$ACTION' (all|core|clips|combined|adversarial|longform|verify)" >&2; exit 2 ;;
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
docker run --rm --pull=never \
  -v "${REPO_ROOT}:/app" -w /app \
  -v "${CACHE}:/tooling" \
  -u "$(id -u):$(id -g)" \
  -e HOME=/tooling -e PIP_CACHE_DIR=/tooling/pip -e HF_HOME=/tooling/hf \
  -e PYTHONDONTWRITEBYTECODE=1 \
  "$IMAGE" bash -c "
    set -e
    # torch from the CPU index and transformers from PyPI, as separate commands: the CPU
    # index does not carry transformers, and piping pip to tail hides its exit status so a
    # '||' fallback never fires.
    if [ ! -d /tooling/site/torch ]; then
      pip install --quiet --disable-pip-version-check --target /tooling/site \
        --index-url https://download.pytorch.org/whl/cpu 'torch>=2.4'
    fi
    # No '|| true': a failed toolchain install used to surface much later as a
    # confusing ModuleNotFoundError from the generator instead of here.
    pip install --quiet --disable-pip-version-check --target /tooling/site \
      'transformers>=4.44' uroman scipy numpy soundfile piper-tts
    export PYTHONPATH=/tooling/site:/app
    python3 scripts/generate_audio_matrix.py ${ACTION}
  "
