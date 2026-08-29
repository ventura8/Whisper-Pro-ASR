#!/bin/bash
# Remove package documentation, man pages, locales and info files.
#
# This has to run after the LAST apt transaction in a stage: every vendor install
# re-creates /usr/share/doc, so cleaning only in the dependency layer leaves the bytes
# back in the image plus a pointless whiteout layer.
set -euo pipefail

rm -rf /usr/share/doc /usr/share/man /usr/share/info /usr/share/locale 2>/dev/null || true
rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*.deb 2>/dev/null || true
