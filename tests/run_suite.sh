#!/bin/bash
# Script to run the test suite and linting in Docker
set -e
set -o pipefail

# Change to the project root directory
cd "$(dirname "$0")/.."

# Containerized CI runs may mount a host-owned .git directory; mark this
# workspace safe so git file enumeration for quality gates is deterministic.
git config --global --add safe.directory "$(pwd)" || true

# Every quality gate in this script (lint, audit, test) is a Docker-test-image-only
# gate (AGENTS.md's "Docker-Only Quality Execution"). Check this up front -- before
# any gate runs -- so a non-Docker invocation fails fast instead of only after
# minutes of unrelated work. SKIP_REAL_E2E stays scoped to skipping only the
# real-backend E2E phase further below; it must never bypass this Docker requirement.
if [ "${WHISPER_PRO_ASR_TEST_IMAGE:-}" != "1" ]; then
	echo "Error: quality gates require WHISPER_PRO_ASR_TEST_IMAGE=1 (Dockerfile.test target test)."
	echo "Use scripts/ci/build-and-test.sh or scripts/ci/build-and-test.ps1."
	exit 1
fi

# Stage selection: PIPELINE_STAGE lets a CI job (or a developer) run only one
# slice of this script instead of everything. Env var (not a CLI arg) so every
# call site (build-and-test.sh/.ps1, ci.yml) only needs one extra `-e` flag on
# `docker run` rather than threading a positional arg through wrapped `-c "..."`
# strings. Default "all" runs lint first (including Radon), then tests in order:
# js-unit-tests → python-tests → e2e-fixture → e2e-real. Each stage below is
# gated but never duplicated, so "all" and any single stage share the same path.
STAGE="${PIPELINE_STAGE:-all}"
case "$STAGE" in
all | lint | python-tests | js-unit-tests | e2e-fixture | e2e-real) ;;
*)
	echo "Error: unknown PIPELINE_STAGE '$STAGE'. Expected one of: all, lint, python-tests, js-unit-tests, e2e-fixture, e2e-real."
	exit 1
	;;
esac
stage_active() {
	[ "$STAGE" = "all" ] || [ "$STAGE" = "$1" ]
}

# Persistent tool run-time cache root. Mounted as a named Docker volume by
# scripts/ci/build-and-test.sh/.ps1 and .github/workflows/ci.yml so these
# survive across separate `docker run` invocations (build-time cache mounts
# alone never reach the running container). Locally this speeds up repeat
# runs; in split CI jobs each job is a fresh runner VM so this mainly keeps
# local/CI invocation shape identical rather than speeding up CI itself.
TOOL_CACHE_ROOT="/var/cache/whisper-pro-asr-tools"

ensure_hadolint() {
	if command -v hadolint >/dev/null 2>&1; then
		return 0
	fi

	HADOLINT_VERSION="${HADOLINT_VERSION:-2.15.1}"
	HADOLINT_SHA256="${HADOLINT_SHA256:-c7187db94eeeeca956519a6af171adc31453941a1e777961f6e680f697c8c507}"
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

	SHELLCHECK_VERSION="${SHELLCHECK_VERSION:-0.11.0}"
	SHELLCHECK_SHA256="${SHELLCHECK_SHA256:-8c3be12b05d5c177a04c29e3c78ce89ac86f1595681cab149b65b97c4e227198}"

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

run_radon_complexity_gate() {
	# Docker test image does not include .git metadata; use filesystem discovery.
	local source_list
	source_list="$(mktemp)"
	find . -type f -name '*.py' \
		-not -path './.venv/*' \
		-not -path './node_modules/*' \
		-not -path './coverage-js/*' \
		-print0 >"$source_list"
	xargs -0 -r python3 -m radon cc -s <"$source_list" | tee complexity_output.txt
	local violations
	violations="$(xargs -0 -r python3 -m radon cc -n B <"$source_list")"
	rm -f "$source_list"
	if [ -n "$violations" ]; then
		echo "Error: The following blocks do not meet the rank-A complexity requirement (complexity <= 5):"
		echo "$violations"
		return 1
	fi
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

if stage_active lint; then
	if [ "$SKIP_LINT" != "1" ]; then
		git init -q >/dev/null 2>&1

		# One-time bootstrap (network-dependent, must complete before the tools
		# that need these binaries are fanned out below).
		ensure_shellcheck
		ensure_hadolint

		export RUFF_CACHE_DIR="${TOOL_CACHE_ROOT}/ruff"
		mkdir -p "$RUFF_CACHE_DIR" \
			"${TOOL_CACHE_ROOT}/eslint" \
			"${TOOL_CACHE_ROOT}/stylelint" \
			2>/dev/null || true

		shfmt_files=("scripts/ci/build-and-test.sh" "tests/run_suite.sh")
		if [ -f ".agent/skills/workflow/resolve-pr-comments-run.sh" ]; then
			shfmt_files+=(".agent/skills/workflow/resolve-pr-comments-run.sh")
		fi
		shellcheck_files=("scripts/ci/build-and-test.sh" "tests/run_suite.sh")
		if [ -f ".agent/skills/workflow/resolve-pr-comments-run.sh" ]; then
			shellcheck_files+=(".agent/skills/workflow/resolve-pr-comments-run.sh")
		fi

		# The independent lint/security tools below share no inter-tool ordering
		# dependency and no output files -- run them concurrently so wall-clock
		# collapses toward the slowest single tool. Radon rank-A enforcement is
		# part of this lint phase so it finishes before any test stage starts.
		declare -a PIDS=()
		declare -a NAMES=()
		run_bg() {
			local name="$1"
			shift
			NAMES+=("$name")
			("$@" 2>&1 | sed -u "s/^/[${name}] /") &
			PIDS+=("$!")
		}

		run_bg "PSScriptAnalyzer" run_powershell_script_analyzer
		run_bg "actionlint" actionlint
		run_bg "check-jsonschema" check-jsonschema --builtin-schema vendor.github-workflows .github/workflows/*.yml
		run_bg "Yamllint" yamllint -s -f parsable -c .yamllint .
		run_bg "shfmt" shfmt -d "${shfmt_files[@]}"
		run_bg "taplo" npm run lint:toml
		run_bg "Black" black --check modules scripts tests whisper_pro_asr.py tests/check_coverage.py
		run_bg "isort" isort --check-only modules scripts tests whisper_pro_asr.py tests/check_coverage.py
		run_bg "Ruff Format" ruff format --check .
		run_bg "ShellCheck" shellcheck -x "${shellcheck_files[@]}"
		run_bg "Hadolint" hadolint --failure-threshold warning --disable-ignore-pragma Dockerfile Dockerfile.test
		run_bg "ESLint" npm run lint:js
		run_bg "ESLint (complexity)" npm run lint:js:complexity
		run_bg "Stylelint" npm run lint:css
		run_bg "HTMLHint & HTML-Validate" npm run lint:html
		run_bg "Markdownlint" npm run lint:md
		run_bg "Ruff Check" ruff check .
		run_bg "Flake8" flake8 modules whisper_pro_asr.py tests tests/check_coverage.py
		run_bg "Pylint" pylint modules whisper_pro_asr.py tests tests/check_coverage.py
		run_bg "Bandit" bandit -r modules whisper_pro_asr.py -x modules/core/utils.py,modules/core/utils_helpers.py,modules/inference/language_detection.py,modules/inference/vad.py,modules/monitoring/metrics_discovery.py,modules/inference/engines/whisperx_worker.py
		# whisperx_worker.py only imports pickle to catch PicklingError when serializing RPC
		# replies over multiprocessing.connection (no untrusted deserialization). Bandit's B403
		# blacklist check fires identically on a static `import pickle` and on
		# `importlib.import_module("pickle")`, so the exception type cannot be referenced here
		# without suppressing B403; dropping PicklingError from the except tuple was tried and
		# reverted -- it broke test_send_response_falls_back_to_error_reply_when_first_send_unpicklable,
		# a real, intentional behavior guarantee. Scan it separately with only B403 skipped so
		# every other Bandit check still runs against this file.
		run_bg "Bandit (whisperx_worker.py, B403 only)" bandit modules/inference/engines/whisperx_worker.py --skip B403
		run_bg "pip-audit" pip-audit
		run_bg "gitleaks" gitleaks detect --source=. --no-git --verbose
		run_bg "npm audit" npm audit --audit-level=low
		run_bg "check-inline-ignores" python3 scripts/ci/check-inline-ignores.py
		run_bg "Radon" run_radon_complexity_gate

		FAIL=0
		for i in "${!PIDS[@]}"; do
			if ! wait "${PIDS[$i]}"; then
				echo "FAILED: ${NAMES[$i]}"
				FAIL=1
			fi
		done
		if [ "$FAIL" -ne 0 ]; then
			exit 1
		fi
		if [ -d /reports ] && [ -f complexity_output.txt ]; then
			cp complexity_output.txt /reports/complexity_output.txt
		fi
		echo "--- Lint/Security Suite Completed Successfully ---"
	else
		echo "--- Skipping Linting (SKIP_LINT=1) ---"
	fi
fi

if stage_active js-unit-tests; then
	echo ""
	echo "--- Running JS Unit Tests (Vitest) ---"
	npm run test:js
fi

if stage_active python-tests; then
	# Timing-sensitive concurrency/preemption tests use real threads + sleep-based
	# synchronization windows; running them under pytest-xdist alongside every
	# other worker risks CPU contention degrading those timing assertions. Run
	# them in a dedicated serial invocation instead of the parallel bulk run --
	# NOT via @pytest.mark.xdist_group, which only guarantees co-location on one
	# worker and does nothing to prevent that worker from running concurrently
	# with every other one on the same machine (the actual risk here).
	SERIAL_TEST_PATHS=(
		"tests/inference/scheduler/priority/test_priority_stage_preemption.py"
		"tests/integration/concurrency/test_e2e_preemption_yielding_stages.py"
		"tests/inference/runtime/test_model_manager_resource_lifecycle.py"
	)
	IGNORE_ARGS=()
	for p in "${SERIAL_TEST_PATHS[@]}"; do
		IGNORE_ARGS+=(--ignore="$p")
	done

	rm -f .coverage .coverage.* coverage.xml pytest-bulk.xml pytest-serial.xml pytest.xml

	echo ""
	echo "--- Running Pytest (parallel bulk, -n auto) ---"
	set +e
	python3 -m pytest --verbosity=0 -ra "${IGNORE_ARGS[@]}" -n auto --dist=loadscope \
		--cov=. --cov-report= --junitxml=pytest-bulk.xml | tee coverage_output_bulk.txt
	BULK_EXIT=${PIPESTATUS[0]}
	set -e

	echo ""
	echo "--- Running Pytest (serial, timing-sensitive concurrency tests) ---"
	set +e
	python3 -m pytest --verbosity=0 -ra "${SERIAL_TEST_PATHS[@]}" \
		--cov=. --cov-append --cov-report= --junitxml=pytest-serial.xml | tee coverage_output_serial.txt
	SERIAL_EXIT=${PIPESTATUS[0]}
	set -e

	echo ""
	echo "--- Combining Coverage Data ---"
	# coverage combine exits 1 when there is nothing to combine -- the common case
	# here, since the bulk and serial invocations run sequentially (not concurrently)
	# and --cov-append already consolidates into one .coverage file directly. Treat
	# that as non-fatal; a genuine data problem still surfaces as a hard failure on
	# the next line (coverage xml has nothing to read from).
	python3 -m coverage combine || true
	python3 -m coverage xml -o coverage.xml
	set +e
	python3 -m coverage report --fail-under=90 | tee coverage_output.txt
	COMBINE_EXIT=${PIPESTATUS[0]}
	set -e

	python3 - <<'PYEOF'
import xml.etree.ElementTree as ET

merged = ET.Element("testsuites")
for junit_file in ("pytest-bulk.xml", "pytest-serial.xml"):
    tree = ET.parse(junit_file)
    root = tree.getroot()
    suites = root.findall("testsuite") if root.tag == "testsuites" else [root]
    for suite in suites:
        merged.append(suite)
ET.ElementTree(merged).write("pytest.xml", encoding="unicode")
PYEOF

	if [ "$BULK_EXIT" -ne 0 ] || [ "$SERIAL_EXIT" -ne 0 ] || [ "$COMBINE_EXIT" -ne 0 ]; then
		echo "Error: pytest stage failed (bulk=$BULK_EXIT serial=$SERIAL_EXIT coverage-gate=$COMBINE_EXIT)"
		exit 1
	fi

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
		cp pytest.xml /reports/pytest.xml
		if [ -f complexity_output.txt ]; then
			cp complexity_output.txt /reports/complexity_output.txt
		fi
	fi
fi

if stage_active e2e-fixture; then
	echo ""
	echo "--- Running JS E2E Tests (Playwright) ---"
	npm run test:e2e
fi

if stage_active e2e-real; then
	if [ "${SKIP_REAL_E2E:-0}" != "1" ]; then
		echo ""
		echo "--- Running Real-Backend E2E Tests (Playwright against the real FastAPI app) ---"
		# WHISPER_PRO_ASR_TEST_IMAGE is already validated at script startup, before any
		# quality gates ran.
		npm run test:e2e:real
	else
		echo "--- Skipping Real-Backend E2E Tests (SKIP_REAL_E2E=1) ---"
	fi
fi

echo ""
echo "--- Test Suite Completed Successfully ---"
