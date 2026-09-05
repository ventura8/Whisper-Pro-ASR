#!/bin/sh
# Clear executable-stack flags on shipped shared libraries, failing loudly rather than
# silently leaving one set. Applies to the interpreter's libraries and, when present,
# WhisperX's segregated stack. Extracted from the Dockerfile so every target that ships
# WhisperX runs the identical audit instead of carrying its own copy.
set -eu

failures=""
library_list=/tmp/shared-libraries

# Tested explicitly rather than inferred from a failed find: using find's status as the
# signal conflated "whisperx is not in this image" -- normal for most targets -- with a
# genuine scan failure, and the fallback would then hide the latter.
#
# There is no allowlist. Every shipped library must have a clear stack; an exemption would
# be a shipped executable stack, which is the thing this script exists to prevent.
scan_roots="/usr/local/lib/python3.*/"
if [ -d /app/libs/whisperx/ ]; then
  scan_roots="$scan_roots /app/libs/whisperx/"
fi

# shellcheck disable=SC2086  # deliberate word splitting and globbing of the root list
find $scan_roots -name "*.so*" -print > "$library_list"

while IFS= read -r library; do
  if ! patchelf --clear-execstack "$library"; then
    failures="$failures $library"
  fi
done < "$library_list"

for library in $failures; do echo "Failed to clear executable stack: $library" >&2; done
test -z "$failures"

# Allowlisted output, not a denylist of the one bad value. `!= "execstack: X"` passed on
# anything unexpected -- a patchelf that errored to stdout, a future marker, an empty line
# from a truncated read -- so an unverified library was indistinguishable from a cleared
# one, in the audit whose entire job is to prove they are cleared.
while IFS= read -r library; do
  actual="$(patchelf --print-execstack "$library" 2>&1 || true)"
  test "$actual" = "execstack: -" || { echo "Executable stack not confirmed cleared: $library (patchelf said: $actual)" >&2; exit 1; }
done < "$library_list"

rm -f "$library_list"
