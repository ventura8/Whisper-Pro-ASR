#!/bin/sh
# Clear executable-stack flags on shipped shared libraries, failing loudly rather than
# silently leaving one set. Applies to the interpreter's libraries and, when present,
# WhisperX's segregated stack. Extracted from the Dockerfile so every target that ships
# WhisperX runs the identical audit instead of carrying its own copy.
set -eu

failures=""
library_list=/tmp/shared-libraries
# Armed immediately, because every exit below the scan is a failure exit -- a library whose
# stack could not be cleared, or one whose cleared state could not be confirmed. Those paths
# left the list behind in /tmp, where verify_no_build_artifacts.sh then reports it as a
# leftover build artifact, burying the real error under an unrelated one.
trap 'rm -f "$library_list"' EXIT

# Tested explicitly rather than inferred from a failed find: using find's status as the
# signal conflated "whisperx is not in this image" -- normal for most targets -- with a
# genuine scan failure, and the fallback would then hide the latter.
#
# There is no allowlist. Every shipped library must have a clear stack; an exemption would
# be a shipped executable stack, which is the thing this script exists to prevent.
# This script is #!/bin/sh, so there are no arrays -- an earlier revision used one and broke
# `nvidia-whisperx` and `full` outright with "syntax error: unexpected \"(\"", on the two
# targets no local hardware validation builds. The roots are passed as separate `find`
# arguments instead, which needs neither word splitting nor a shellcheck SC2086 directive
# (banned here), and keeps a path containing a space intact.
#
# The python3.* glob is expanded by `set --` rather than by an unquoted variable, so a
# no-match leaves the literal pattern and `find` reports it by name instead of silently
# scanning nothing.
set -- /usr/local/lib/python3.*/
if [ -d /app/libs/whisperx/ ]; then
	set -- "$@" /app/libs/whisperx/
fi

find "$@" -name "*.so*" -print >"$library_list"

while IFS= read -r library; do
	if ! patchelf --clear-execstack "$library"; then
		failures="$failures $library"
	fi
done <"$library_list"

for library in $failures; do echo "Failed to clear executable stack: $library" >&2; done
test -z "$failures"

# Allowlisted output, not a denylist of the one bad value. `!= "execstack: X"` passed on
# anything unexpected -- a patchelf that errored to stdout, a future marker, an empty line
# from a truncated read -- so an unverified library was indistinguishable from a cleared
# one, in the audit whose entire job is to prove they are cleared.
while IFS= read -r library; do
	actual="$(patchelf --print-execstack "$library" 2>&1 || true)"
	test "$actual" = "execstack: -" || {
		echo "Executable stack not confirmed cleared: $library (patchelf said: $actual)" >&2
		exit 1
	}
done <"$library_list"
