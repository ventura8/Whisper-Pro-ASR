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
check "torch test/include trees" "$(find /opt/venv/lib/python3*/site-packages/torch/test /opt/venv/lib/python3*/site-packages/torch/include -maxdepth 0 2>/dev/null | wc -l)"
check "docs/man/locale" "$(find /usr/share/doc /usr/share/man /usr/share/info /usr/share/locale -type f 2>/dev/null | wc -l)"

exit "$fail"
