#!/bin/bash
# Drop ROCm content that this service can never use, immediately after installing it.
#
# ROCm ships compiled kernels for every GPU architecture AMD has ever supported. The
# published amd/full images support consumer Radeon only, so data-center and legacy
# kernels are intentionally omitted.
#
# This MUST run inside the same RUN layer as the install. Deleting in a later layer only
# whites the files out; the bytes still ship in the earlier one.
#
# Supported architectures are RDNA2, RDNA3, and RDNA4 (see KEEP_ARCHS).
#
# Usage: prune_rocm.sh [extra-dir ...]
set -euo pipefail

# Kept -- consumer Radeon architectures advertised in docs/SETUP.md.
# gfx1031 and gfx1150 use the compatible gfx1030/gfx1100 kernels respectively.
#
# Dropped -- unsupported hardware: Instinct CDNA (gfx908, gfx90a, gfx940/941/942),
# Polaris (gfx803), Vega (gfx900/906), and RDNA1 (gfx1010/1012). Neither gfx1031 nor
# gfx1150 ships its own kernels, so neither needs an entry.
#
KEEP_ARCHS="gfx1030 gfx1100 gfx1101 gfx1102 gfx1200 gfx1201"

# Extra directories may be passed as arguments -- PyTorch's ROCm wheel bundles its own
# copy of the same per-architecture kernels (rocBLAS, hipBLASLt, MIOpen, aotriton) under
# site-packages/torch/lib, ~6GB of which is for hardware this service does not support.
# Pruning it here rather than in a second script keeps one architecture list.
ROCM_ROOT="$(readlink -f /opt/rocm)"
ROCM_DIRS=("$ROCM_ROOT" /usr/lib/x86_64-linux-gnu/rocblas "$@")
FFT_CACHE="$ROCM_ROOT/lib/rocfft/rocfft_kernel_cache.db"

# `|| true`: an extra directory passed as an argument may legitimately not exist, and
# du then exits non-zero, which pipefail turns into an aborted build. awk still sums the
# paths that do exist, so the reported size stays correct.
before="$( { du -sm "${ROCM_DIRS[@]}" 2>/dev/null || true; } | awk '{s+=$1} END{print s+0}')"

is_kept() {
    local file="$1" arch
    for arch in $KEEP_ARCHS; do
        case "$file" in *"$arch"*) return 0 ;; esac
    done
    return 1
}

# Per-architecture kernel blobs are identified by the arch in their filename. Anything
# whose name mentions no architecture at all is shared or fallback code and stays.
prune_arch_files() {
    local file
    while IFS= read -r file; do
        is_kept "$file" || rm -f "$file"
    done < <(find "${ROCM_DIRS[@]}" -type f -name '*gfx[0-9]*' 2>/dev/null)
}

# The rocFFT cache is one SQLite file holding each architecture's kernels as rows, so it is
# pruned with a DELETE plus VACUUM rather than by removing files.
prune_fft_cache() {
    [ -f "$FFT_CACHE" ] || return 0
    python3 - "$FFT_CACHE" "$KEEP_ARCHS" <<'PYEOF'
import sqlite3
import sys

path, keep = sys.argv[1], sys.argv[2].split()
connection = sqlite3.connect(path)
# Architectures appear as bare names and with xnack suffixes (gfx90a-xnack+), so match on
# a prefix rather than equality.
predicate = " AND ".join(f"arch NOT LIKE ? || '%'" for _ in keep)
connection.execute(f"DELETE FROM cache_v1 WHERE {predicate}", keep)
connection.commit()
connection.execute("VACUUM")
connection.close()
PYEOF
}

# rocSOLVER was previously deleted here as an "orphan". It is not one: the ONNX Runtime
# ROCm provider links librocsolver.so.0 directly (confirmed with ldd). Removing it made
# libonnxruntime_providers_rocm.so unloadable, and ONNX Runtime reacts to a provider that
# fails to load by silently falling back to CPU -- get_available_providers() still lists
# ROCMExecutionProvider, so AMD acceleration looked present while every model ran on the
# processor. Do not prune shared libraries by inspection again; verify_no_unmet_deps below
# is what makes that safe.

# A pruned library that something still links is invisible until runtime, so fail the
# build here instead. Checks the ROCm provider plus the ROCm tree's own libraries.
# Dependencies that are legitimately absent from a ROCm library in this image.
#
# libpython is supplied by whichever interpreter imports a Python extension module, not by
# the module itself. ROCm ships bindings like libmigraphx_py_3.13.so that link it and are
# never loaded here -- MIGraphX's Python API is not used by this service. Failing the build
# on those is a false positive, and it did fail the amd image outright.
IGNORED_DEPS_RE='^libpython'

verify_no_unmet_deps() {
    local unmet failed=0 lib
    # Only the ROCm tree exists in this stage; the ONNX Runtime providers arrive in a
    # later stage and are checked there by verify_ort_provider_links.sh.
    for lib in "$ROCM_ROOT"/lib/*.so* /usr/lib/x86_64-linux-gnu/rocblas/*.so*; do
        [ -f "$lib" ] || continue
        if ! ldd_output="$(ldd "$lib" 2>&1)"; then
            echo "prune_rocm: ERROR ldd could not inspect $lib: $ldd_output" >&2
            failed=1
            continue
        fi
        # grep exits non-zero when it filters every line out, which is the healthy case.
        unmet="$(printf '%s\n' "$ldd_output" | awk '/not found/ {print $1}' \
            | grep -Ev "$IGNORED_DEPS_RE" | sort -u | tr '\n' ' ' || true)"
        if [ -n "$unmet" ]; then
            echo "prune_rocm: ERROR $lib has unmet dependencies: $unmet" >&2
            failed=1
        fi
    done
    [ "$failed" -eq 0 ] || {
        echo "prune_rocm: refusing to ship a ROCm stack with unmet dependencies." >&2
        exit 1
    }
}

prune_arch_files
prune_fft_cache
ldconfig
verify_no_unmet_deps

after="$( { du -sm "${ROCM_DIRS[@]}" 2>/dev/null || true; } | awk '{s+=$1} END{print s+0}')"
echo "prune_rocm: ${before}MB -> ${after}MB (kept: ${KEEP_ARCHS})"
