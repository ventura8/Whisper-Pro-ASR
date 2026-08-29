#!/bin/bash
# Remove build-only artifacts so they never reach a shipped layer.
#
# This MUST run inside the same RUN that created the artifacts. A cleanup in a later
# layer only whites the files out; the bytes still ship in the earlier layer.
set -euo pipefail

# The compiler toolchain is only needed to build wheels.
#
# Purge only what is actually installed, and keep the purge itself unsuppressed.
#
# A bare `apt-get purge gcc` cannot be used here: this script deletes /var/lib/apt/lists at
# the end of every stage it runs in, so by the time a target's own final layer runs it the
# package index is empty and apt cannot resolve the name at all -- `E: Unable to locate
# package gcc`, exit 100, build dead. That is not the same as the "known but not installed"
# case ("is not installed, so not removed") that apt does tolerate, and conflating the two
# broke every standalone target while remaining invisible behind a warm build cache: the
# final-stage RUN only re-executes when the build context changes.
#
# Selecting the installed subset first keeps a genuine purge failure fatal, which is the
# point -- a purge that silently fails leaves the toolchain in a shipped layer, and
# verify_no_build_artifacts.sh only checks the final filesystem.
BUILD_ONLY_PACKAGES=(
	build-essential
	g++
	g++-13
	gcc
	gcc-13
	python3-dev
	software-properties-common
)
installed_build_packages=()
for pkg in "${BUILD_ONLY_PACKAGES[@]}"; do
	if dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q 'ok installed'; then
		installed_build_packages+=("$pkg")
	fi
done
if [ ${#installed_build_packages[@]} -gt 0 ]; then
	apt-get purge -y --auto-remove "${installed_build_packages[@]}"
fi

# Python sources remain installed, so cached bytecode is rebuilt lazily by writable runtime
# directories when useful. Static archives are link-time inputs only; no shipped component
# compiles or links native code. Cover every runtime prefix because vendor and WhisperX wheels
# install their artifacts outside `/opt/venv`.
find / -xdev -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
find / -xdev -type f \( -name "*.pyc" -o -name "*.pyo" -o -name "*.a" \) -delete \
	2>/dev/null || true

# Torch ships its C++ test binaries (~83MB) and the headers needed to compile custom
# extensions (~62MB). This service only calls torch from Python, so neither is reachable.
# torch/bin is deliberately kept: torch_shm_manager backs shared-memory tensor passing.
rm -rf /opt/venv/lib/python3*/site-packages/torch/test \
	/opt/venv/lib/python3*/site-packages/torch/include 2>/dev/null || true

# Test suites bundled inside installed wheels (~55MB) are never imported at runtime.
find /opt/venv/lib/python3*/site-packages -maxdepth 3 -type d \( -name tests -o -name test \) \
	-exec rm -rf {} + 2>/dev/null || true

# Build-time temp files. The globs need the dot-prefixed entries too: `/tmp/*` skips
# `.cache`, `.npm`, `.gnupg` and friends, which is where build tooling puts most of what it
# leaves behind, so the cleanup was shipping exactly the directories it meant to remove.
# find, not a `.[!.]*` glob: an empty match would leave the literal pattern as an argument.
find /tmp /var/tmp -mindepth 1 -maxdepth 1 -exec rm -rf {} + 2>/dev/null || true
rm -rf /root/.cache /root/.cargo 2>/dev/null || true
apt-get clean
rm -rf /var/lib/apt/lists/*

# OS documentation. Each vendor install script runs this itself as its last step, because
# it has to follow the final apt transaction of its stage and those run after this script.
# The call here is what covers the `cpu` target, which has no vendor install at all -- it
# is otherwise the one image that would ship /usr/share/doc. Pruning is idempotent, so the
# overlap with the vendor stages costs nothing.
/usr/local/build/prune_os_docs.sh
