#!/bin/bash
# Fail the build if a bundled ONNX Runtime provider has unmet shared-library dependencies.
#
# ONNX Runtime loads execution providers with dlopen and treats a failure as "provider
# unavailable" rather than an error: get_available_providers() still advertises the
# provider, sessions still build, and every model silently runs on CPU. A pruned or
# missing dependency therefore produces no message anyone would notice -- only a machine
# that is mysteriously slow. That happened once already, when ROCm pruning deleted
# librocsolver.so.0, which libonnxruntime_providers_rocm.so links directly.
#
# This must run in a stage where the vendor runtime and /app/libs both exist, which is the
# final image stage -- the runtime is installed several stages before the providers arrive.
set -euo pipefail

# Dependencies that are legitimately absent from the image.
#
# libcuda.so.1 is the NVIDIA kernel driver's user-space half. It belongs to the host and
# is bind-mounted in by the NVIDIA container toolkit at run time; an image that shipped
# its own copy would be wrong, so it is never present at build time.
#
# TensorRT (libnvinfer, libnvonnxparser) is deliberately not installed -- the TensorRT
# provider is unused and ONNX Runtime falls back to the CUDA provider, which is the
# intended configuration rather than an accidental CPU fallback.
IGNORED_DEPS_RE='^(libcuda\.so|libnvinfer|libnvinfer_plugin|libnvonnxparser)'

status=0
found=0

while IFS= read -r provider; do
    found=1
    # ldd's own status is captured separately. Folding it into the pipeline made an
    # uninspectable provider -- a corrupt object, or one ldd refuses -- read as "no unmet
    # dependencies found", which is precisely the silent pass this script exists to stop.
    if ! ldd_output="$(ldd "$provider" 2>&1)"; then
        echo "verify_ort_provider_links: ERROR ${provider}: ldd could not inspect it: ${ldd_output}" >&2
        status=1
        continue
    fi
    # grep exits non-zero when it filters every line out, which is the healthy case here,
    # so that pipeline's status is deliberately discarded rather than tripping pipefail.
    unmet="$(printf '%s\n' "$ldd_output" | awk '/not found/ {print $1}' \
        | grep -Ev "$IGNORED_DEPS_RE" | sort -u | tr '\n' ' ' || true)"
    if [ -n "$unmet" ]; then
        echo "verify_ort_provider_links: ERROR ${provider}: unmet: ${unmet}" >&2
        status=1
    else
        echo "verify_ort_provider_links: ok ${provider}"
    fi
done < <(find /app/libs -name 'libonnxruntime_providers_*.so' 2>/dev/null | sort)

if [ "$found" -eq 0 ]; then
    echo "verify_ort_provider_links: no providers found under /app/libs" >&2
    exit 1
fi

[ "$status" -eq 0 ] || {
    echo "verify_ort_provider_links: a provider would dlopen-fail and fall back to CPU." >&2
    exit 1
}
