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
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Load shared dependencies
DEPS_FILE="${SCRIPT_DIR}/dependencies.env"
if [ -f "$DEPS_FILE" ]; then
	# shellcheck source-path=SCRIPTDIR
	# shellcheck source=dependencies.env
	. "$DEPS_FILE"
else
	echo "Error: Dependencies configuration file not found at ${DEPS_FILE}"
	exit 1
fi

install_with_apt() {
	if command -v sudo >/dev/null 2>&1; then
		sudo -n apt-get update && sudo -n apt-get install -y "$@"
		return $?
	fi

	if [ "$(id -u)" -eq 0 ]; then
		apt-get update && apt-get install -y "$@"
		return $?
	fi

	return 1
}

ensure_command() {
	cmd_name="$1"
	shift
	apt_packages=("$@")

	if command -v "$cmd_name" >/dev/null 2>&1; then
		return 0
	fi

	echo "Dependency '$cmd_name' is missing. Attempting auto-install..."
	if command -v apt-get >/dev/null 2>&1 && install_with_apt "${apt_packages[@]}"; then
		if command -v "$cmd_name" >/dev/null 2>&1; then
			return 0
		fi
	fi

	echo "Error: '$cmd_name' is required and could not be auto-installed."
	echo "Install it manually, then re-run this script."
	exit 1
}

ensure_command docker docker.io

# Pick Docker command: direct docker when allowed, otherwise sudo docker.
DOCKER_CMD=(docker)
if ! docker ps >/dev/null 2>&1; then
	if command -v sudo >/dev/null 2>&1 && sudo docker ps >/dev/null 2>&1; then
		DOCKER_CMD=(sudo docker)
	else
		echo "Error: Docker is installed but not accessible for the current user."
		echo "Add your user to the docker group or enable sudo access to docker."
		exit 1
	fi
fi

ensure_poetry_lock() {
	printf "\n--- Verifying Poetry Lock File ---\n"
	user_args=()
	if command -v id >/dev/null 2>&1; then
		user_args=(--user "$(id -u):$(id -g)")
	fi

	touch "${PROJECT_ROOT}/poetry.lock"

	"${DOCKER_CMD[@]}" run --rm \
		"${user_args[@]}" \
		-v "${PROJECT_ROOT}/pyproject.toml:/workspace/pyproject.toml" \
		-v "${PROJECT_ROOT}/poetry.lock:/workspace/poetry.lock" \
		-w /workspace \
		"$PYTHON_IMAGE" \
		/bin/bash -lc "export HOME=/tmp && export PATH=\"/tmp/.local/bin:\$PATH\" && export PIP_ROOT_USER_ACTION=ignore && export PIP_NO_WARN_SCRIPT_LOCATION=1 && python -m pip install --quiet --user poetry==2.4.1 && python -m poetry config virtualenvs.create false && if [ ! -f poetry.lock ] || ! python -m poetry check --lock >/dev/null 2>&1; then python -m poetry lock --no-interaction; fi"
}

ensure_poetry_lock

printf "\n--- Building Test Image ---\n"
"${DOCKER_CMD[@]}" build -f Dockerfile.test --target test -t whisper-pro-asr-test .

printf "\n--- Execute Test Suite ---\n"
REPORTS_DIR="${PROJECT_ROOT}/reports"
mkdir -p assets "$REPORTS_DIR"
rm -f "$REPORTS_DIR/coverage.xml" \
	"$REPORTS_DIR/coverage_output.txt" \
	"$REPORTS_DIR/complexity_output.txt" \
	"$REPORTS_DIR/pytest.xml"
if command -v id >/dev/null 2>&1; then
	chown -R "$(id -u):$(id -g)" assets "$REPORTS_DIR"
fi
chmod 0755 assets "$REPORTS_DIR"
find "$REPORTS_DIR" -type f -exec chmod 0644 {} +
set +e
cat <<'DOCKER_TEST_SCRIPT' | "${DOCKER_CMD[@]}" run --rm \
	-e CI=true \
	-v "${PROJECT_ROOT}/assets:/app/assets" \
	-v "${REPORTS_DIR}:/reports" \
	whisper-pro-asr-test /bin/bash -s
tests/run_suite.sh
TEST_EXIT_CODE=$?
[ -f coverage.xml ] && cp coverage.xml /reports/coverage.xml || true
[ -f coverage_output.txt ] && cp coverage_output.txt /reports/coverage_output.txt || true
[ -f complexity_output.txt ] && cp complexity_output.txt /reports/complexity_output.txt || true
[ -f pytest.xml ] && cp pytest.xml /reports/pytest.xml || true
exit "$TEST_EXIT_CODE"
DOCKER_TEST_SCRIPT
TEST_EXIT_CODE=$?
set -e

printf "\n--- Regenerating Coverage Badge (Mandatory Final Stage) ---\n"
"${DOCKER_CMD[@]}" run --rm \
	-v "${PROJECT_ROOT}/assets:/app/assets" \
	-v "${REPORTS_DIR}:/reports" \
	whisper-pro-asr-test /bin/bash -lc "if [ ! -s /reports/coverage.xml ]; then echo 'Error: /reports/coverage.xml is missing or empty'; exit 1; fi; genbadge coverage -i /reports/coverage.xml -o /app/assets/coverage.svg"

if [ ! -s "${PROJECT_ROOT}/assets/coverage.svg" ]; then
	echo "Error: Mandatory coverage badge is missing or empty at assets/coverage.svg"
	exit 1
fi

printf "\n--- Cyclomatic Complexity Summary (Radon cc) ---\n"
if [ -f "${REPORTS_DIR}/complexity_output.txt" ]; then
	cat "${REPORTS_DIR}/complexity_output.txt"
fi

printf "\n--- Code Coverage Summary ---\n"
if [ -f "${REPORTS_DIR}/coverage_output.txt" ]; then
	sed -n '/---------- coverage/,/TOTAL/p' "${REPORTS_DIR}/coverage_output.txt"
fi

printf "\n--- Done ---\n"
exit $TEST_EXIT_CODE
