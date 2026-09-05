#!/bin/bash
# AMD ROCm 7.2.4 runtime (Ubuntu 24.04) plus the soname shims the wheels expect.
# 7.2.4 is the newest build published for Ubuntu at repo.radeon.com/rocm/apt; the
# 7.14.x and 10.x releases listed in AMD's docs are not in that apt repository.
set -euo pipefail

wget -q -O /tmp/rocm.gpg.key https://repo.radeon.com/rocm/rocm.gpg.key
echo "2de99e2354646a90d9903e2a669fc4e36b02c1bbff7075c481e12d7edab2c88b  /tmp/rocm.gpg.key" | sha256sum -c -
mkdir -p /etc/apt/keyrings
gpg --dearmor -o /etc/apt/keyrings/rocm.gpg /tmp/rocm.gpg.key
rm -f /tmp/rocm.gpg.key
cat > /etc/apt/sources.list.d/rocm.list <<'EOF'
deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/7.2.4 noble main
EOF
# The AMDGPU driver is a host responsibility. Adding AMD's graphics repository here pulls
# kernel-driver and compiler packages into the runtime image without helping containers use
# /dev/kfd, so it is deliberately omitted.
apt-get update
apt-get install -y --no-install-recommends \
  miopen-hip=* \
  hip-runtime-amd=* \
  hipfft=* \
  rocm-smi=* \
  rocm-smi-lib=* \
  migraphx=* \
  libhipblas0=* \
  librocblas0=*

# These shims exist because onnxruntime-rocm links against sonames ROCm does not ship
# under those exact names. They are pinned to the ROCm version above: a version bump can
# rename the source library, in which case the shim silently does not fire and AMD
# degrades to CPU at runtime -- invisible in a transcript, because a CPU fallback still
# transcribes correctly. Announce every miss so a build log shows it.
# Warns on the *end state*, not the source: newer ROCm ships some of these sonames
# itself (7.2.4 provides libamdhip64.so.7 and librocm_smi64.so.1 natively), so a missing
# source is only a problem when the destination is absent too.
shim_lib() {
  local src="$1" dst="$2"
  if [ -e "$dst" ]; then
    return 0
  fi
  if [ -f "$src" ]; then
    ln -sf "$src" "$dst"
  else
    echo "install_rocm: WARNING $dst is absent and $src does not exist to shim it (onnxruntime-rocm may fall back to CPU)" >&2
  fi
}

shim_lib /usr/lib/x86_64-linux-gnu/libhipblas.so.0 /usr/lib/x86_64-linux-gnu/libhipblas.so.3
shim_lib /usr/lib/x86_64-linux-gnu/librocblas.so.0 /usr/lib/x86_64-linux-gnu/librocblas.so.3
shim_lib /opt/rocm/lib/libamdhip64.so.6 /opt/rocm/lib/libamdhip64.so.7
shim_lib /opt/rocm/lib/librocm_smi64.so.7 /opt/rocm/lib/librocm_smi64.so.1
if [ -d /usr/lib/x86_64-linux-gnu/rocblas ]; then
  for rocblas_dir in /usr/lib/x86_64-linux-gnu/rocblas/*; do
    if [ -d "$rocblas_dir" ]; then
      ln -sfn "$rocblas_dir" /usr/lib/x86_64-linux-gnu/rocblas/current
      break
    fi
  done
fi

# librocdxg enables AMD WSL detection (/opt/rocm/lib/librocdxg.so). WSL2's ROCm runtime
# limitation means this supports detection and CPU fallback, not native ROCm inference.
wget -q -O /tmp/rocdxg-roct.deb \
  "https://github.com/ROCm/librocdxg/releases/download/v1.2.1/rocdxg-roct_1.2.1_amd64.deb"
echo "7889eef45a1132ed2dde88d8ea1356bf791ec9c05802a18940bc81b970e850e0  /tmp/rocdxg-roct.deb" | sha256sum -c -
dpkg -i /tmp/rocdxg-roct.deb
rm -f /tmp/rocdxg-roct.deb
ldconfig
/usr/local/build/prune_os_docs.sh
