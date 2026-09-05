#!/bin/bash
# Replace the CPU PyTorch build with Intel's XPU one, so torch-based engines
# (openai-whisper) can use an Intel GPU instead of falling back to the CPU.
#
# Cost is ~1GB: site-packages 1690MB -> 2706MB, measured. The XPU wheel's dependencies
# bring the SYCL/Level-Zero runtime torch needs, so unlike the ROCm install this one
# keeps its deps -- except triton, which is torch.compile-only and ~1.6GB on its own.
# XPU was verified working without it (matmul on an Intel UHD Graphics iGPU).
#
# MUST run in the same RUN layer as the install, or the removed bytes still ship.
set -euo pipefail

TORCH_VERSION="${TORCH_XPU_VERSION:-2.13.0+xpu}"
TORCH_INDEX="${TORCH_XPU_INDEX:-https://download.pytorch.org/whl/xpu}"
SITE_PACKAGES="$(python3 -c 'import site; print(site.getsitepackages()[0])')"

echo "install_xpu_torch: installing torch ${TORCH_VERSION}"
PYTHONDONTWRITEBYTECODE=1 python3 -m pip install --no-cache-dir \
  --index-url "${TORCH_INDEX}" "torch==${TORCH_VERSION}"

rm -rf "${SITE_PACKAGES}/triton" "${SITE_PACKAGES}"/pytorch_triton_xpu*

# Intel's runtime wheels drop oneAPI/SYCL shared objects into the venv's lib/ directory
# (not site-packages) and ship each soname as a full copy rather than a symlink --
# libccl.so, libccl.so.1 and libccl.so.1.0 are 200MB each. Collapsing identical files
# into symlinks recovers ~1.1GB with no functional change (verified: XPU still
# initialises and runs a matmul afterwards).
#
# Do NOT delete libccl entirely to save more: torch links it unconditionally and
# `import torch` then fails with "libccl.so.1: cannot open shared object file".
# Derived from the interpreter rather than hardcoded: site-packages is <venv>/lib/pythonX.Y,
# so the oneAPI libraries sit two levels up. A hardcoded /opt/venv/lib silently matched
# nothing if the venv ever moved, and the deduplication saved 0 MB without saying so.
VENV_LIB_DIR="$(python3 -c 'import pathlib, site; print(pathlib.Path(site.getsitepackages()[0]).parent)')"
if [ ! -d "$VENV_LIB_DIR" ]; then
    echo "install_xpu_torch: ERROR derived library directory does not exist: $VENV_LIB_DIR" >&2
    exit 1
fi

python3 - "$VENV_LIB_DIR" <<'DEDUPE'
import hashlib
import pathlib
import sys

lib = pathlib.Path(sys.argv[1])
by_hash: dict[str, list[pathlib.Path]] = {}
for path in sorted(lib.glob("*.so*")):
    if path.is_symlink() or not path.is_file():
        continue
    by_hash.setdefault(hashlib.md5(path.read_bytes()).hexdigest(), []).append(path)

saved = 0
for duplicates in by_hash.values():
    if len(duplicates) < 2:
        continue
    keep = max(duplicates, key=lambda p: len(p.name))
    for duplicate in duplicates:
        if duplicate == keep:
            continue
        saved += duplicate.stat().st_size
        duplicate.unlink()
        duplicate.symlink_to(keep.name)
print(f"install_xpu_torch: deduplicated {saved // 1024 // 1024} MB of oneAPI libraries into symlinks")
DEDUPE

python3 -c 'import torch; assert "xpu" in torch.__version__, "expected an XPU torch build"; print("install_xpu_torch: torch", torch.__version__)'
