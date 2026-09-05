#!/bin/bash
# NVIDIA CUDA runtime (Ubuntu 24.04). cuda-nvcc is omitted deliberately (~1.5GB).
set -euo pipefail

CUDA_KEYRING=cuda-keyring_1.1-1_all.deb
CUDA_KEYRING_SHA256=d2a6b11c096396d868758b86dab1823b25e14d70333f1dfa74da5ddaf6a06dba
wget --progress=dot:giga "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/${CUDA_KEYRING}"
echo "${CUDA_KEYRING_SHA256}  ${CUDA_KEYRING}" | sha256sum --check --status
dpkg -i "${CUDA_KEYRING}"
apt-get update
apt-get install -y --no-install-recommends \
  cuda-libraries-13-3=* \
  cuda-cudart-13-3=* \
  libcudnn9-cuda-13=* \
  libcublas-12-9=* \
  cuda-cudart-12-9=*
rm -f "${CUDA_KEYRING}"
ldconfig
/usr/local/build/prune_os_docs.sh
