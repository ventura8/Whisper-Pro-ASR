#!/bin/bash
# Reduce a remote_validate.sh transcript to the parts that are evidence, and strip the parts
# that are private infrastructure.
#
# Two separate reasons, both learned the hard way. A raw transcript is ~1200 lines of Docker
# build output around ~20 lines that actually say what ran, and it carries the SSH target,
# the remote hostname and the LAN address in its preflight and footer -- none of which may
# reach a commit. Redaction happens here rather than by hand because "remember to scrub it"
# is not a control. Order matters in the redaction: see the comment beside it.
set -o pipefail
IFS= read -r -d '' AWK_PROGRAM <<'AWK'
/^=== (Preflight|What actually loaded|Measured execution|Suite|Done)/ { keep = 1 }
/^=== (Sync|Build|Start|Test image|Teardown)/ { keep = 0 }
keep { print }
/passed|failed|error/ && !keep { print }
AWK
awk "$AWK_PROGRAM" "$1" |
	sed -E 's#[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+#<user>@<host>#g' |
	sed -E 's#([0-9]{1,3}\.){3}[0-9]{1,3}#<host>#g' |
	sed -E 's#(ssh: OK \()[^)]*#\1<user>@<host>#'
