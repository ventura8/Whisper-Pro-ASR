#!/bin/bash

# CI-equivalent: build and run Docker-based pipeline with full caching.
# Run this in bash from the project root.

# Exit immediately if a command exits with a non-zero status
set -e

# Resolve script directory in a POSIX-safe way (works with sh and bash).
SCRIPT_PATH="$0"
case "$SCRIPT_PATH" in
  /*) ;;
  *) SCRIPT_PATH="$(pwd)/$SCRIPT_PATH" ;;
esac
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
cd "${SCRIPT_DIR}"

install_with_apt() {
  packages="$1"
  if command -v sudo >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y $packages
    return 0
  fi

  if [ "$(id -u)" -eq 0 ]; then
    apt-get update
    apt-get install -y $packages
    return 0
  fi

  return 1
}

ensure_command() {
  cmd_name="$1"
  apt_packages="$2"

  if command -v "$cmd_name" >/dev/null 2>&1; then
    return 0
  fi

  echo "Dependency '$cmd_name' is missing. Attempting auto-install..."
  if command -v apt-get >/dev/null 2>&1 && install_with_apt "$apt_packages"; then
    if command -v "$cmd_name" >/dev/null 2>&1; then
      return 0
    fi
  fi

  echo "Error: '$cmd_name' is required and could not be auto-installed."
  echo "Install it manually, then re-run this script."
  exit 1
}

ensure_command npm "nodejs npm"
ensure_command docker "docker.io"

# Pick Docker command: direct docker when allowed, otherwise sudo docker.
DOCKER_CMD="docker"
if ! docker ps >/dev/null 2>&1; then
  if command -v sudo >/dev/null 2>&1 && sudo docker ps >/dev/null 2>&1; then
    DOCKER_CMD="sudo docker"
  else
    echo "Error: Docker is installed but not accessible for the current user."
    echo "Add your user to the docker group or enable sudo access to docker."
    exit 1
  fi
fi

printf "\n--- Running Frontend Quality Gates (JS/CSS Lint + JS Coverage) ---\n"
npm ci
printf "\n--- Running Frontend Security Audit (Fail on Any Vulnerability) ---\n"
npm audit --audit-level=low
npm run quality:frontend

printf "\n--- Running Static Analysis (Linting) ---\n"
$DOCKER_CMD build -f Dockerfile.test --target lint -t whisper-pro-asr-lint .

printf "\n--- Building Test Image ---\n"
$DOCKER_CMD build -f Dockerfile.test --target test -t whisper-pro-asr-test .

printf "\n--- Execute Test Suite ---\n"
mkdir -p assets
$DOCKER_CMD run --rm \
  -e SKIP_LINT=1 \
  -v "${SCRIPT_DIR}/assets:/app/assets" \
  -v "${SCRIPT_DIR}:/reports" \
  whisper-pro-asr-test /bin/bash -c "tests/run_suite.sh && cp coverage.xml /reports/coverage.xml && cp coverage_output.txt /reports/coverage_output.txt && cp pytest.xml /reports/pytest.xml"

printf "\n--- Done ---\n"

