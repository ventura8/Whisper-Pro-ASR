#!/bin/bash
# Script to run the test suite and linting in Docker
set -e
set -o pipefail

# Change to the project root directory
cd "$(dirname "$0")/.."

# Containerized CI runs may mount a host-owned .git directory; mark this
# workspace safe so git file enumeration for quality gates is deterministic.
git config --global --add safe.directory "$(pwd)" || true

ensure_hadolint() {
	if command -v hadolint >/dev/null 2>&1; then
		return 0
	fi

	HADOLINT_VERSION="${HADOLINT_VERSION:-2.12.0}"
	HADOLINT_SHA256="${HADOLINT_SHA256:-56de6d5e5ec427e17b74fa48d51271c7fc0d61244bf5c90e828aab8362d55010}"
	target_dir="${HOME}/.local/bin"
	mkdir -p "$target_dir"
	target_file="${target_dir}/hadolint"
	tmp_file="$(mktemp)"
	url="https://github.com/hadolint/hadolint/releases/download/v${HADOLINT_VERSION}/hadolint-Linux-x86_64"

	if command -v wget >/dev/null 2>&1; then
		wget -q -O "$tmp_file" "$url"
	elif command -v curl >/dev/null 2>&1; then
		curl -fsSL -o "$tmp_file" "$url"
	else
		echo "Error: hadolint is missing and neither wget nor curl is available to auto-install it."
		rm -f "$tmp_file"
		exit 1
	fi

	if ! printf '%s  %s\n' "$HADOLINT_SHA256" "$tmp_file" | sha256sum -c - >/dev/null 2>&1; then
		echo "Error: hadolint checksum verification failed."
		rm -f "$tmp_file"
		exit 1
	fi

	chmod +x "$tmp_file"
	mv "$tmp_file" "$target_file"
	export PATH="$target_dir:$PATH"

	if ! command -v hadolint >/dev/null 2>&1; then
		echo "Error: Failed to auto-install hadolint."
		exit 1
	fi
}

ensure_shellcheck() {
	if command -v shellcheck >/dev/null 2>&1; then
		return 0
	fi

	SHELLCHECK_VERSION="${SHELLCHECK_VERSION:-0.10.0}"
	SHELLCHECK_SHA256="${SHELLCHECK_SHA256:-6c881ab0698e4e6ea235245f22832860544f17ba386442fe7e9d629f8cbedf87}"

	if command -v sudo >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then
		if sudo -n apt-get update && sudo -n apt-get install -y shellcheck; then
			if command -v shellcheck >/dev/null 2>&1; then
				return 0
			fi
		fi
	elif [ "$(id -u)" -eq 0 ] && command -v apt-get >/dev/null 2>&1; then
		if apt-get update && apt-get install -y shellcheck; then
			if command -v shellcheck >/dev/null 2>&1; then
				return 0
			fi
		fi
	fi

	target_dir="${HOME}/.local/bin"
	mkdir -p "$target_dir"
	tmp_dir="$(mktemp -d)"
	archive_path="${tmp_dir}/shellcheck.tar.xz"
	archive_url="https://github.com/koalaman/shellcheck/releases/download/v${SHELLCHECK_VERSION}/shellcheck-v${SHELLCHECK_VERSION}.linux.x86_64.tar.xz"

	if command -v wget >/dev/null 2>&1; then
		wget -q -O "$archive_path" "$archive_url"
	elif command -v curl >/dev/null 2>&1; then
		curl -fsSL -o "$archive_path" "$archive_url"
	else
		echo "Error: shellcheck is missing and neither wget nor curl is available to auto-install it."
		rm -rf "$tmp_dir"
		exit 1
	fi

	if ! printf '%s  %s\n' "$SHELLCHECK_SHA256" "$archive_path" | sha256sum -c - >/dev/null 2>&1; then
		echo "Error: shellcheck checksum verification failed."
		rm -rf "$tmp_dir"
		exit 1
	fi

	if ! tar -xJf "$archive_path" -C "$tmp_dir"; then
		echo "Error: Failed to extract shellcheck archive."
		rm -rf "$tmp_dir"
		exit 1
	fi

	shellcheck_binary="${tmp_dir}/shellcheck-v${SHELLCHECK_VERSION}/shellcheck"
	if [ ! -f "$shellcheck_binary" ]; then
		echo "Error: ShellCheck binary not found after extraction."
		rm -rf "$tmp_dir"
		exit 1
	fi

	mv "$shellcheck_binary" "${target_dir}/shellcheck"
	chmod +x "${target_dir}/shellcheck"
	export PATH="${target_dir}:$PATH"
	rm -rf "$tmp_dir"

	if ! command -v shellcheck >/dev/null 2>&1; then
		echo "Error: Failed to auto-install shellcheck."
		exit 1
	fi
}

run_powershell_script_analyzer() {
	pwsh -NoLogo -NoProfile -Command - <<'POWERSHELL'
$ErrorActionPreference = "Stop"
$issues = Invoke-ScriptAnalyzer -Path scripts -Recurse `
	-IncludeDefaultRules -Severity Warning,Error,Information
if ($issues) {
	$issues | Sort-Object ScriptName,Line,RuleName |
		Format-Table ScriptName,Line,Severity,RuleName,Message -AutoSize
	exit 1
}
POWERSHELL
}

# Activate virtual environment if running locally and it exists
if [ "$CI" != "true" ]; then
	if [ -d ".venv" ]; then
		VENV_BIN_PATH="$(pwd)/.venv/bin"
		export PATH="${VENV_BIN_PATH}:$PATH"
	elif [ -d "venv" ]; then
		VENV_BIN_PATH="$(pwd)/venv/bin"
		export PATH="${VENV_BIN_PATH}:$PATH"
	fi
fi

if [ "$SKIP_LINT" != "1" ]; then
	git init -q >/dev/null 2>&1
	echo "--- Running Lint/Static: PSScriptAnalyzer ---"
	run_powershell_script_analyzer

	echo "--- Running Config/Schema: actionlint ---"
	actionlint

	echo "--- Running Config/Schema: check-jsonschema ---"
	check-jsonschema --builtin-schema vendor.github-workflows .github/workflows/*.yml

	echo "--- Running Config/Schema: Yamllint (strict) ---"
	yamllint -s -f parsable -c .yamllint .

	echo "--- Running Syntax/Format: shfmt ---"
	shfmt_files=("scripts/ci/build-and-test.sh" "tests/run_suite.sh")
	if [ -f ".agent/skills/workflow/resolve-pr-comments-run.sh" ]; then
		shfmt_files+=(".agent/skills/workflow/resolve-pr-comments-run.sh")
	fi
	shfmt -d "${shfmt_files[@]}"

	echo "--- Running Syntax/Format: taplo ---"
	npm run lint:toml

	echo "--- Running Syntax/Format: Black Check ---"
	black --check modules scripts tests whisper_pro_asr.py tests/check_coverage.py

	echo "--- Running Syntax/Format: isort Check ---"
	isort --check-only modules scripts tests whisper_pro_asr.py tests/check_coverage.py

	echo "--- Running Syntax/Format: Ruff Format Check ---"
	ruff format --check .

	echo "--- Running Lint/Static: ShellCheck ---"
	ensure_shellcheck
	shellcheck_files=("scripts/ci/build-and-test.sh" "tests/run_suite.sh")
	if [ -f ".agent/skills/workflow/resolve-pr-comments-run.sh" ]; then
		shellcheck_files+=(".agent/skills/workflow/resolve-pr-comments-run.sh")
	fi
	shellcheck -x "${shellcheck_files[@]}"

	echo "--- Running Lint/Static: Hadolint ---"
	ensure_hadolint
	hadolint --failure-threshold warning --disable-ignore-pragma Dockerfile Dockerfile.test

	echo "--- Running Lint/Static: ESLint ---"
	npm run lint:js

	echo "--- Running Lint/Static: Stylelint ---"
	npm run lint:css

	echo "--- Running Lint/Static: HTMLHint & HTML-Validate ---"
	npm run lint:html

	echo "--- Running Lint/Static: Markdownlint ---"
	npm run lint:md

	echo "--- Running Lint/Static: Ruff Check ---"
	ruff check .

	echo "--- Running Lint/Static: Flake8 ---"
	flake8 modules whisper_pro_asr.py tests tests/check_coverage.py

	echo "--- Running Lint/Static: Pylint ---"
	pylint modules whisper_pro_asr.py tests tests/check_coverage.py

	echo "--- Running Security/Sanity: Bandit ---"
	bandit -r modules whisper_pro_asr.py -x modules/core/utils.py,modules/core/utils_helpers.py,modules/inference/language_detection.py,modules/inference/vad.py,modules/monitoring/metrics_discovery.py

	echo "--- Running Security/Sanity: pip-audit ---"
	pip-audit

	echo "--- Running Security/Sanity: gitleaks ---"
	gitleaks detect --source=. --no-git --verbose

	echo "--- Running Security/Sanity: npm audit (low threshold) ---"
	npm audit --audit-level=low

	echo "--- Running Security/Sanity: check-inline-ignores ---"
	python3 scripts/ci/check-inline-ignores.py
else
	echo "--- Skipping Linting (SKIP_LINT=1) ---"
fi

echo ""
echo "--- Running JS Unit Tests (Vitest) ---"
npm run test:js

echo ""
echo "--- Running JS E2E Tests (Playwright) ---"
npm run test:e2e

echo ""
echo "--- Running Cyclomatic Complexity Check (Radon) ---"
RADON_SOURCE_LIST="$(mktemp)"
trap 'rm -f "$RADON_SOURCE_LIST"' EXIT
# Docker test image does not include .git metadata; use filesystem discovery.
find . -type f -name '*.py' \
	-not -path './.venv/*' \
	-not -path './node_modules/*' \
	-not -path './coverage-js/*' \
	-print0 >"$RADON_SOURCE_LIST"
xargs -0 -r python3 -m radon cc -s <"$RADON_SOURCE_LIST" | tee complexity_output.txt

echo ""
echo "--- Enforcing Rank-A Complexity ---"
VIOLATIONS=$(xargs -0 -r python3 -m radon cc -n B <"$RADON_SOURCE_LIST")
if [ -n "$VIOLATIONS" ]; then
	echo "Error: The following blocks do not meet the rank-A complexity requirement (complexity <= 5):"
	echo "$VIOLATIONS"
	exit 1
fi

echo ""
echo "--- Running Pytest with Coverage ---"
# We output XML (for PR display) and terminal report
python3 -m pytest -ra --cov=. --cov-report=xml:coverage.xml --cov-report=term-missing --junitxml=pytest.xml | tee coverage_output.txt

echo ""
echo "--- Verifying Per-File Coverage (Threshold: 90%) ---"
python3 tests/check_coverage.py
echo ""
echo "--- Generating Coverage Badge ---"
mkdir -p assets
genbadge coverage -i coverage.xml -o assets/coverage.svg

if [ ! -s assets/coverage.svg ]; then
	echo "Error: Coverage badge was not generated or is empty at assets/coverage.svg"
	exit 1
fi

# Copy reports to /reports if mounted (before final checks so reports are available even on failure)
if [ -d /reports ]; then
	echo "--- Copying reports to /reports volume ---"
	cp coverage.xml /reports/coverage.xml
	cp coverage_output.txt /reports/coverage_output.txt
	cp complexity_output.txt /reports/complexity_output.txt
	cp pytest.xml /reports/pytest.xml
fi

echo ""
echo "--- Test Suite Completed Successfully ---"
