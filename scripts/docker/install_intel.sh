#!/bin/bash
# Intel compute runtime. The OpenVINO base image already ships the OpenVINO runtime,
# Level Zero and the NPU driver; this adds the OpenCL ICD. Do not install
# intel-level-zero-gpu here: it is supplied by the pinned base and is not present in
# Ubuntu's package index, which otherwise makes a clean Intel build fail.
set -euo pipefail

apt-get update
apt-get install -y --no-install-recommends \
  intel-opencl-icd=*
ldconfig
/usr/local/build/prune_os_docs.sh
