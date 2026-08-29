#!/bin/bash
# NVIDIA CUDA runtime (Ubuntu 24.04). cuda-nvcc is omitted deliberately (~1.5GB).
set -euo pipefail

CUDA_KEYRING=cuda-keyring_1.1-1_all.deb
CUDA_KEYRING_SHA256=d2a6b11c096396d868758b86dab1823b25e14d70333f1dfa74da5ddaf6a06dba
wget --progress=dot:giga "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/${CUDA_KEYRING}"
echo "${CUDA_KEYRING_SHA256}  ${CUDA_KEYRING}" | sha256sum --check --status
dpkg -i "${CUDA_KEYRING}"
# libcudnn9-cuda-13 below is UNPINNED, and that is a live hazard worth understanding before
# touching torch. torch ships its own cuDNN as the pip wheel nvidia-cudnn-cu13, so a CUDA
# image carries TWO cuDNN installs and they have to agree. When they diverged -- apt's
# current cuDNN against pip's 9.24.0.43, pulled in by a torch 2.14 bump -- every torch-engine
# request returned HTTP 500 with "CUDNN_BACKEND_TENSOR_DESCRIPTOR cudnnFinalize failed ...
# CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED", while the full 2233-test suite stayed green and
# all nine images built. FASTER-WHISPER was unaffected, because CTranslate2 binds CUDA
# directly and never touches cuDNN; only WHISPERX and OPENAI-WHISPER failed.
#
# No static check can see this: the mismatch was a MINOR version difference and this side is
# whatever the repository currently serves. Running a torch engine on a real CUDA device is
# the only thing that catches it -- docs/RELEASE_VALIDATION_PLAN.md requires the CUDA rows to
# be re-run whenever torch moves.
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
