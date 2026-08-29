#!/bin/bash
# Remove package documentation, man pages, locales and info files.
#
# This has to run after the LAST apt transaction in a stage: every vendor install
# re-creates /usr/share/doc, so cleaning only in the dependency layer leaves the bytes
# back in the image plus a pointless whiteout layer.
set -euo pipefail

# /usr/share/doc is pruned rather than removed outright: the per-package `copyright` files
# are the redistribution licences of everything shipped in the image, and deleting them puts
# the image out of compliance with the licences that require them to travel with the binary.
# They are a few hundred KB. Everything else under there -- changelogs, READMEs, examples --
# goes, as do man, info and locale entirely.
find /usr/share/doc -mindepth 1 ! -name copyright -type f -delete 2>/dev/null || true
find /usr/share/doc -mindepth 1 -type d -empty -delete 2>/dev/null || true
rm -rf /usr/share/man /usr/share/info /usr/share/locale 2>/dev/null || true
rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*.deb 2>/dev/null || true
