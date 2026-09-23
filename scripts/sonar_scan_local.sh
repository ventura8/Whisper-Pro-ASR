#!/bin/bash
# Run a SonarQube Cloud analysis of this working tree from the local machine.
#
# Analysis settings (organization, project key, sources, exclusions, coverage report
# paths) live in sonar-project.properties and are shared with the CI `sonarqube` job --
# this script only supplies the host, the token and the branch, so a local scan and a CI
# scan apply exactly the same rules to exactly the same files.
#
# The token is never stored in the repository. Supply it via the environment:
#
#   read -rsp 'SonarQube token: ' SONAR_TOKEN && export SONAR_TOKEN && echo
#   scripts/sonar_scan_local.sh
#
# Coverage is optional. Without it the scan still reports every bug, vulnerability,
# security hotspot and code smell; coverage and duplication figures are simply absent.
# To include coverage, run the test stages first so the reports exist:
#
#   scripts/sonar_scan_local.sh --with-coverage
#
# Usage:
#   scripts/sonar_scan_local.sh [--with-coverage] [-- <extra sonar-scanner args>]
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}" || exit 1

WITH_COVERAGE=0
case "${1:-}" in
--with-coverage)
	WITH_COVERAGE=1
	shift
	;;
-h | --help)
	grep '^#' "$0" | sed 's/^# \{0,1\}//' | sed '1d'
	exit 0
	;;
esac
[ "${1:-}" = "--" ] && shift

if [ -z "${SONAR_TOKEN:-}" ]; then
	echo "ERROR: SONAR_TOKEN is not set." >&2
	echo "  read -rsp 'SonarQube token: ' SONAR_TOKEN && export SONAR_TOKEN && echo" >&2
	exit 1
fi

if [ "${WITH_COVERAGE}" = "1" ]; then
	# Same invocation the CI python-tests / js-unit-tests stages use, so the reports
	# land where sonar-project.properties expects them.
	# No chmod 0777 here, unlike the CI stage this mirrors: that one runs the container
	# as the runner's uid and needs the bind mounts writable by it. These two containers
	# run as root, which is not subject to the mode at all, so widening it to everyone on
	# the developer's own checkout buys nothing.
	mkdir -p assets reports
	docker run --rm \
		-e CI=true \
		-e PIPELINE_STAGE=python-tests \
		-v "${PWD}/assets:/app/assets" \
		-v "${PWD}:/out" \
		-v whisper-pro-asr-tool-cache:/var/cache/whisper-pro-asr-tools \
		whisper-pro-asr-test /bin/bash -c "tests/run_suite.sh; \
			[ -f coverage.xml ] && cp coverage.xml /out/coverage.xml || true; \
			[ -f pytest.xml ] && cp pytest.xml /out/pytest.xml || true"
	docker run --rm \
		-e CI=true \
		-e PIPELINE_STAGE=js-unit-tests \
		-v "${PWD}:/out" \
		-v whisper-pro-asr-tool-cache:/var/cache/whisper-pro-asr-tools \
		whisper-pro-asr-test /bin/bash -c "tests/run_suite.sh; \
			mkdir -p /out/coverage-js; \
			[ -f coverage-js/lcov.info ] && cp coverage-js/lcov.info /out/coverage-js/lcov.info || true"
fi

# Blame drives new-code detection, so the scanner needs the real .git directory, not
# just the working tree. The container runs as the invoking user so that .scannerwork/
# and the scanner cache are not left root-owned in the repository.
docker run --rm \
	-u "$(id -u):$(id -g)" \
	-e SONAR_TOKEN \
	-e SONAR_HOST_URL="https://sonarcloud.io" \
	-e SONAR_USER_HOME=/usr/src/.sonar \
	-v "${PWD}:/usr/src" \
	sonarsource/sonar-scanner-cli:latest \
	-Dsonar.branch.name="$(git rev-parse --abbrev-ref HEAD)" \
	"$@"
