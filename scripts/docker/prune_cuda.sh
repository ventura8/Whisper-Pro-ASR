#!/bin/bash
# Remove CUDA libraries nothing in this image can load, in the same layer that installed
# them. A later RUN would only whiteout the files; the bytes would still ship.
#
# Deliberately conservative, because the apparent redundancy here is not redundancy:
#
#   * The system CUDA 13 libraries and the pip `nvidia/cu13` wheels look like duplicates
#     but are different builds serving different consumers -- ONNX Runtime loads the
#     system ones, torch loads the wheels. Symlinking one onto the other silently swaps a
#     library version underneath a consumer.
#   * CUDA 12.9 looks removable because nothing lists it in NEEDED, but libctranslate2
#     dlopens "libcublas.so.12" by name. Dropping it breaks faster-whisper on GPU at
#     runtime while the build stays green.
#
# What is genuinely unreachable is NPP, NVIDIA's image and signal processing family. It is
# pulled in by the cuda-libraries metapackage and referenced by nothing this service ships.
set -euo pipefail

CUDA_LIB_DIR=/usr/local/cuda-13.3/targets/x86_64-linux/lib

# Without this the script reports a cheerful "0MB -> 0MB" when the CUDA version is
# bumped and this path moves, having pruned nothing at all.
if [ ! -d "$CUDA_LIB_DIR" ]; then
    echo "prune_cuda: ERROR CUDA library directory not found: $CUDA_LIB_DIR" >&2
    exit 1
fi

before="$(du -sm "$CUDA_LIB_DIR" 2>/dev/null | awk '{print $1+0}')"
rm -f "$CUDA_LIB_DIR"/libnpp*.so.* 2>/dev/null || true
ldconfig
after="$(du -sm "$CUDA_LIB_DIR" 2>/dev/null | awk '{print $1+0}')"
echo "prune_cuda: ${before}MB -> ${after}MB (removed the NPP family)"
