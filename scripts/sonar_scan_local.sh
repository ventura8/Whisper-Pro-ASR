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
	# These two containers stay root: the suite writes its reports inside the image's
	# root-owned /app and the tool cache is a root-owned volume, so running them as the
	# host user fails outright rather than fixing ownership. Instead the exported files
	# are handed back afterwards -- without this, `coverage-js/` lands root-owned and the
	# next JS run on the host cannot write into it.
	HOST_UID="$(id -u)"
	HOST_GID="$(id -g)"
	# Same invocation the CI python-tests / js-unit-tests stages use, so the reports
	# land where sonar-project.properties expects them.
	# No chmod 0777 here, unlike the CI stage this mirrors: that one runs the container
	# as the runner's uid and needs the bind mounts writable by it. These two containers
	# run as root, which is not subject to the mode at all, so widening it to everyone on
	# the developer's own checkout buys nothing.
	mkdir -p assets reports
	# Remove any report from a previous run FIRST. Otherwise a stage that fails, or an
	# export that does not happen, leaves yesterday's coverage.xml in the checkout and the
	# scanner uploads it as though it measured this tree -- a wrong number is worse than
	# the missing one it replaces.
	rm -f coverage.xml pytest.xml coverage-js/lcov.info

	# `|| exit` on each: `set -uo pipefail` does not stop the script on a failing
	# `docker run`, and the trailing `chown ... || true` inside the container would
	# otherwise mask a failed suite as a success. The suite's own exit code is what
	# propagates, so a red test run stops here instead of being scanned and reported as
	# though it had passed.
	docker run --rm \
		-e CI=true \
		-e PIPELINE_STAGE=python-tests \
		-v "${PWD}/assets:/app/assets" \
		-v "${PWD}:/out" \
		-v whisper-pro-asr-tool-cache:/var/cache/whisper-pro-asr-tools \
		whisper-pro-asr-test /bin/bash -c "tests/run_suite.sh; SUITE=\$?; \
			[ -f coverage.xml ] && cp coverage.xml /out/coverage.xml; \
			[ -f pytest.xml ] && cp pytest.xml /out/pytest.xml; \
			chown ${HOST_UID}:${HOST_GID} /out/coverage.xml /out/pytest.xml 2>/dev/null || true; \
			exit \$SUITE" || {
		echo "ERROR: the python-tests stage failed; not scanning a tree whose tests are red." >&2
		exit 1
	}
	docker run --rm \
		-e CI=true \
		-e PIPELINE_STAGE=js-unit-tests \
		-v "${PWD}:/out" \
		-v whisper-pro-asr-tool-cache:/var/cache/whisper-pro-asr-tools \
		whisper-pro-asr-test /bin/bash -c "tests/run_suite.sh; SUITE=\$?; \
			mkdir -p /out/coverage-js; \
			[ -f coverage-js/lcov.info ] && cp coverage-js/lcov.info /out/coverage-js/lcov.info; \
			chown -R ${HOST_UID}:${HOST_GID} /out/coverage-js 2>/dev/null || true; \
			exit \$SUITE" || {
		echo "ERROR: the js-unit-tests stage failed; not scanning a tree whose tests are red." >&2
		exit 1
	}

	# An export that silently did not happen is the other way a stale or absent report
	# reaches the scanner, so require what the properties file points at.
	for report in coverage.xml coverage-js/lcov.info; do
		[ -s "${report}" ] || {
			echo "ERROR: ${report} was not produced; sonar-project.properties expects it." >&2
			exit 1
		}
	done
fi

# Blame drives new-code detection, so the scanner needs the real .git directory, not
# just the working tree. The container runs as the invoking user so nothing it writes is
# left root-owned.
#
# A caller-owned host directory rather than a named volume: a fresh named volume is
# created root-owned, and this container runs as the invoking user, so the scanner would
# fail creating its cache before analysis ever started.
SONAR_CACHE_DIR="${XDG_CACHE_HOME:-${HOME}/.cache}/whisper-pro-asr-sonar"
mkdir -p "${SONAR_CACHE_DIR}"

# The scanner cache goes outside the working tree. It holds a
# provisioned JRE -- ~136 MB of OpenJDK, including markdown under legal/ and a
# java.security with a high-entropy line -- and every linter here walks the filesystem
# rather than the git index, so caching it inside the repository failed Markdownlint and
# gitleaks on third-party files, on any machine that had run a local scan. Gitignoring it
# does not help, for the same reason `.fixture-tooling` needs an explicit markdownlint
# exclusion and `./_*` needs one in ruff.
docker run --rm \
	-u "$(id -u):$(id -g)" \
	-e SONAR_TOKEN \
	-e SONAR_HOST_URL="https://sonarcloud.io" \
	-e SONAR_USER_HOME=/opt/sonar-cache \
	-v "${SONAR_CACHE_DIR}:/opt/sonar-cache" \
	-v "${PWD}:/usr/src" \
	sonarsource/sonar-scanner-cli:latest \
	-Dsonar.branch.name="$(git rev-parse --abbrev-ref HEAD)" \
	"$@"
