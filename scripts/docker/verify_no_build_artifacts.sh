#!/bin/bash
# Hard assertion that a built image carries no build-only artifacts.
# Run against a built image: docker run --rm --entrypoint bash <img> -c '...'
set -euo pipefail

fail=0
check() {
	local label="$1" count="$2"
	if [ "$count" -ne 0 ]; then
		echo "FAIL: $label ($count found)" >&2
		fail=1
	else
		echo "ok: $label"
	fi
}

check "__pycache__ dirs" "$(find / -xdev -type d -name __pycache__ 2>/dev/null | wc -l)"
check "compiled Python bytecode" "$(find / -xdev -type f \( -name '*.pyc' -o -name '*.pyo' \) 2>/dev/null | wc -l)"
check "static .a archives" "$(find / -xdev -type f -name '*.a' 2>/dev/null | wc -l)"
# Both temp trees, because both are cleaned. Checking only /tmp let a build leave files in
# /var/tmp -- which persists across the image exactly as /tmp does -- and still pass.
check "/tmp leftovers" "$(find /tmp -xdev -type f 2>/dev/null | wc -l)"
check "/var/tmp leftovers" "$(find /var/tmp -xdev -type f 2>/dev/null | wc -l)"
check "compiler toolchain" "$(find /usr/bin -maxdepth 1 -type f \( -name 'gcc' -o -name 'g++' -o -name 'gcc-[0-9]*' -o -name 'g++-[0-9]*' \) 2>/dev/null | wc -l)"
check "apt lists" "$(find /var/lib/apt/lists -type f 2>/dev/null | wc -l)"
# Anchor first: every check below globs into site-packages, and `find` on a path that does
# not exist reports zero hits, which is indistinguishable from "cleaned". A moved venv would
# have turned the whole torch assertion into a no-op that passes.
# `|| true` because `head -n1` closes the pipe on the first hit: under `set -o pipefail`
# find's resulting SIGPIPE would fail the assignment and abort the script through `set -e`,
# reporting a build-artifact failure that never happened. The emptiness check below stays
# the only thing that decides this verdict.
site_packages="$(find /opt/venv/lib -maxdepth 2 -type d -name site-packages 2>/dev/null | head -n1 || true)"
if [ -z "$site_packages" ]; then
	echo "FAIL: no site-packages under /opt/venv/lib -- the artifact checks would pass vacuously" >&2
	exit 1
fi
check "torch test/include trees" "$(find "$site_packages/torch/test" "$site_packages/torch/include" -maxdepth 0 2>/dev/null | wc -l)"
# /usr/share/doc is expected to retain per-package `copyright` files -- prune_os_docs.sh
# deliberately keeps them for licence compliance -- so only non-copyright files there count
# as leftovers. man, info and locale must be empty outright.
check "docs/man/locale" "$({
	find /usr/share/doc -type f ! -name copyright 2>/dev/null
	find /usr/share/man /usr/share/info /usr/share/locale -type f 2>/dev/null
} | wc -l)"

exit "$fail"
