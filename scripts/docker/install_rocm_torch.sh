#!/bin/bash
# Replace the CPU PyTorch build with the ROCm one, then prune it to the architectures
# this service supports.
#
# The wheel is ~5.9GB and unpacks to ~13.8GB because it bundles kernels for every AMD
# GPU architecture -- the same redundancy prune_rocm.sh already removes from the apt
# ROCm install, so the same script (and the same KEEP_ARCHS list) is reused here. That
# takes torch from 13789MB to 8298MB, measured.
#
# --no-deps is deliberate: it keeps pytorch-triton-rocm out, which is only needed for
# torch.compile and which this service never calls.
#
# MUST run in the same RUN layer as the install, or the pruned bytes still ship.
set -euo pipefail

TORCH_VERSION="${TORCH_ROCM_VERSION:-2.13.0+rocm7.2}"
TORCH_INDEX="${TORCH_ROCM_INDEX:-https://download.pytorch.org/whl/rocm7.2}"
SITE_PACKAGES="$(python3 -c 'import site; print(site.getsitepackages()[0])')"
TORCH_LIB="${SITE_PACKAGES}/torch/lib"

# hipSPARSE, which only this target needs. The ROCm torch wheel bundles MAGMA, and
# torch/lib/libmagma.so links libhipsparse.so.4 -- a soname nothing else in the image
# pulls in, because install_rocm.sh installs only what onnxruntime-rocm and MIGraphX need
# and the plain `amd` target has no torch. Without it prune_rocm.sh's own unmet-dependency
# check fails the build outright ("libmagma.so has unmet dependencies: libhipsparse.so.4"),
# which is how this target came to be unbuildable from a cold cache.
#
# Installed here rather than in install_rocm.sh so `amd` and `full` do not carry rocSPARSE
# for a torch build they never load. apt-get update first: strip_build_artifacts.sh deletes
# /var/lib/apt/lists in the stage this one inherits, so the package index is empty on entry.
echo "install_rocm_torch: installing hipSPARSE for the MAGMA backend"
apt-get update
apt-get install -y --no-install-recommends hipsparse=*
apt-get clean
rm -rf /var/lib/apt/lists/*
# Same as every other install_*.sh that runs an apt transaction: the package brings its own
# /usr/share/doc tree, and verify_no_build_artifacts.sh checks that stage image for it.
/usr/local/build/prune_os_docs.sh

echo "install_rocm_torch: installing torch ${TORCH_VERSION}"
PYTHONDONTWRITEBYTECODE=1 python3 -m pip install --no-cache-dir --no-deps \
	--index-url "${TORCH_INDEX}" "torch==${TORCH_VERSION}"

test -d "${TORCH_LIB}" || {
	echo "install_rocm_torch: ${TORCH_LIB} missing after install" >&2
	exit 1
}

/usr/local/build/prune_rocm.sh "${TORCH_LIB}"

# librccl.so looks like a multi-GPU-only collective library, but torch links it
# unconditionally: removing it makes `import torch` fail outright. Verify the import here
# so a future size-trimming change cannot ship a broken image.
python3 -c 'import torch; assert torch.version.hip, "expected a ROCm torch build"; print("install_rocm_torch: torch", torch.__version__)'
