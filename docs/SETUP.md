# Setup Guide

## Prerequisites

- Intel Core Ultra (Meteor Lake/Lunar Lake) with NPU, OR NVIDIA GPU (CUDA), OR AMD GPU (native Linux ROCm), OR generic CPU
- Windows 11 (WSL2) or Linux (Ubuntu 22.04+)
- Intel NPU drivers installed (for NPU acceleration)
- Docker

## Installation

### Method 1: Docker Hub (Recommended)

Create persistent host directories first, and make them writable by the container's runtime UID/GID up front, so task history, telemetry, and model downloads survive container restarts/updates. The image runs as the `nobody` user (UID/GID `65534:65534`) by default — both the CPU-only and Intel commands below run as `65534:65534` (the Intel one just sets `--user` explicitly to the same identity):

```bash
mkdir -p data model_cache
sudo chown -R 65534:65534 data model_cache
sudo chmod -R u+rwX data model_cache
```

(`sudo` is required since your own user normally doesn't own files as UID 65534. The explicit `chmod` guarantees write access for that UID even if the directories pre-existed with restrictive permissions -- ownership alone doesn't grant write access if the owner-write bit was already cleared.)

CPU-only (works on any host; the service auto-detects hardware and falls back to CPU when no accelerator device is mapped):

```bash
docker run -d --name whisper-pro-asr -p 9000:9000 \
  --tmpfs /tmp/whisper:size=2G --tmpfs /tmp/numba-cache:size=128M \
  -v "$(pwd)/data:/app/data" -v "$(pwd)/model_cache:/app/model_cache" \
  ventura8/whisper-pro-asr
```

If the container still reports permission errors writing to `data`/`model_cache` (e.g. you skipped the setup step above), re-run `sudo chown -R 65534:65534 data model_cache && sudo chmod -R u+rwX data model_cache`. `docker compose` (Method 2 below) automates this via its `init-permissions` service.

Intel iGPU/Arc hosts (GPU-only, no NPU) — only if `/dev/dri` and at least one `/dev/dri/renderD*` render node exist on the host (skip this variant, use the CPU-only command above, on NVIDIA/AMD/CPU-only systems; without a renderD* node the `--group-add` derivation below resolves to an empty string and the command fails):

```bash
docker run -d --name whisper-pro-asr -p 9000:9000 --device /dev/dri --user 65534:65534 \
  --group-add "$(stat -c '%g' /dev/dri/renderD* 2>/dev/null | head -n 1)" \
  --tmpfs /tmp/whisper:size=2G --tmpfs /tmp/numba-cache:size=128M \
  -v "$(pwd)/data:/app/data" -v "$(pwd)/model_cache:/app/model_cache" \
  ventura8/whisper-pro-asr
```

Intel NPU hosts — only if `/dev/dri`, `/dev/accel`, and at least one `/dev/dri/renderD*` render node exist on the host:

```bash
docker run -d --name whisper-pro-asr -p 9000:9000 --device /dev/accel --device /dev/dri --user 65534:65534 \
  --group-add "$(stat -c '%g' /dev/dri/renderD* 2>/dev/null | head -n 1)" \
  --tmpfs /tmp/whisper:size=2G --tmpfs /tmp/numba-cache:size=128M \
  -v "$(pwd)/data:/app/data" -v "$(pwd)/model_cache:/app/model_cache" \
  ventura8/whisper-pro-asr
```

The two Intel commands above explicitly set `--user 65534:65534`; the CPU-only command relies on the image's built-in default user (also `65534:65534`, the `nobody` identity). In all cases the `data`/`model_cache` ownership from the setup step above already covers the required write access.

### Method 2: Local Build

```bash
git clone https://github.com/ventura8/Whisper-Pro-ASR.git
cd Whisper-Pro-ASR

# 1. Audit the host, then pick the hardware target and record it in .env
scripts/audit_hardware.sh --env            # Linux; or scripts/audit_hardware.ps1 -Env on Windows/WSL2
# (or set it by hand:)
echo "BUILD_TARGET=nvidia" >> .env          # cpu | intel | intel-xpu | nvidia | nvidia-whisperx | full | nvidia-intel | amd | amd-rocm-torch

# 2. For Intel (or nvidia-intel) also pin the host render group. `--env` above already
#    wrote it; do this by hand only if you set BUILD_TARGET yourself. renderD128 is not
#    guaranteed to exist -- the first render node is renderD128 on most hosts but numbering
#    starts higher when other DRM devices are present -- so glob rather than hardcode:
echo "HOST_INTEL_RENDER_GID=$(stat -c '%g' /dev/dri/renderD* | head -n 1)" >> .env

# 3. Build and start with the override matching BUILD_TARGET. There is no default here:
#    docker-compose.nvidia.yml on an Intel, AMD or CPU host either fails on a device that
#    does not exist or starts an image with no passthrough at all.
# .env has to be sourced: compose reads it for the container's environment, but the shell
# expanding the filename below does not, so without this the override resolves to the
# literal "docker-compose..yml" and compose fails on a file that does not exist.
set -a; . ./.env; set +a
docker compose -f docker-compose.yml -f "docker-compose.${BUILD_TARGET}.yml" up -d --build
```

The audit verifies NVIDIA availability with both `nvidia-smi` and a real Docker GPU
probe. This is authoritative even when a Docker daemon configuration file does not list
an `nvidia` runtime explicitly.

Device passthrough differs per vendor, so it lives in a per-target override file rather
than the base compose. Merge the one matching `BUILD_TARGET`:

Sizes are uncompressed on-disk (`docker images`); a registry reports a smaller
compressed number.

| Target | Override file | Size |
| :--- | :--- | ---: |
| `cpu` | `docker-compose.cpu.yml` | ~4.9 GB |
| `intel` | `docker-compose.intel.yml` | ~5.2 GB |
| `nvidia` | `docker-compose.nvidia.yml` | ~17.5 GB |
| `nvidia-intel` | `docker-compose.nvidia-intel.yml` | ~17.9 GB |
| `amd` | `docker-compose.amd.yml` | ~14.1 GB |
| `nvidia-whisperx` | `docker-compose.nvidia-whisperx.yml` | ~18.4 GB (adds speaker diarization) |
| `full` | `docker-compose.full.yml` | ~29.8 GB locally (every vendor's ONNX Runtime + WhisperX; preferred for diarization) |
| `intel-xpu` | `docker-compose.intel-xpu.yml` | ~11.2 GB (adds openai-whisper on the Intel GPU) |
| `amd-rocm-torch` | `docker-compose.amd-rocm-torch.yml` | ~21.8 GB (adds openai-whisper on the AMD GPU) |

The `Dockerfile` also supports direct target builds. Intel-bearing targets use the
pinned OpenVINO runtime base; their Compose overrides set it explicitly because the
base Compose file defaults CPU builds to Ubuntu. Pass the Ubuntu base explicitly for
`cpu`, `nvidia`, and `amd` (the matching Compose overrides already do this):

```bash
docker buildx build --load --target intel -t whisper-pro-asr:intel .
docker buildx build --load --target cpu \
  --build-arg RUNTIME_BASE=ubuntu:24.04@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517 \
  -t whisper-pro-asr:cpu .
```

Production targets remove build-only bytecode caches and static archives in the same
Docker layer that creates them. Verify a completed image with
`scripts/docker/verify_no_build_artifacts.sh` through `docker run`.
Each production target performs a final post-copy purge, so shipped images contain no
Python bytecode caches or static link archives from shared/vendor runtime layers.

CUDA targets intentionally retain Torch's complete CUDA dependency set, including NCCL,
cuSPARSELt, and NVSHMEM, because UVR loads those libraries dynamically, including in
single-GPU deployments.

Using the base file alone starts the container with **no accelerator passthrough**. WSL2
hosts additionally merge `docker-compose.wsl.yml`.

Models are not baked into the images; the first start downloads them into `./model_cache`
(see the *Runtime model provisioning note* below) and later starts reuse them.

Build cache behavior note (Windows/WSL2): `docker-compose.yml` is configured with a persistent local BuildKit cache directory (`.buildx-cache`) so repeated `docker compose build` runs can reuse prior layers reliably.

To verify cache reuse explicitly:

```bash
docker compose build --progress=plain
```

You should see `CACHED` for unchanged steps after the first successful build.

Optional timezone overrides for `docker-compose.yml` can be set via a local `.env` file:

```bash
TZ=America/New_York
```

This controls the container timezone (`TZ`) without editing compose YAML. The default compose runtime keeps a non-root identity example in `docker-compose.yml`; Linux Intel hosts should also set `HOST_INTEL_RENDER_GID` in `.env` (derive via `stat -c '%g' /dev/dri/renderD128`, or `stat -c '%g' /dev/dri/renderD* | head -n 1` if multiple render nodes exist) so `group_add: ["${HOST_INTEL_RENDER_GID:?...}"]` matches their host's actual render group when their render/accel nodes require it — there is no silent numeric default.

Before starting, ensure local bind-mount directories exist:

```bash
mkdir -p data model_cache
```

**Automatic permission fix**: The `init-permissions` service in `docker-compose.yml` runs once on every `docker compose up`, fixing bind-mount ownership so the runtime user can write task history and model cache. This ensures history persists across image updates and stack redeployments without any manual intervention.

For local quality runs, `Dockerfile.test` uses BuildKit cache mounts for `apt`, `pip`, `poetry`, `npm`, and Playwright browser downloads to speed repeated builds. Keep Docker BuildKit enabled.

**Note**: The system automatically detects NVIDIA CUDA, Intel GPU, and Intel NPU resources. Manual device selection (`ASR_DEVICE`) is optional; Intel NPUs accelerate vocal isolation rather than Whisper ASR.

**NVIDIA CUDA compose note**: Uncomment the NVIDIA reservation block in `docker-compose.yml` when deploying on CUDA hosts, then launch normally:

```bash
docker compose up -d --build --force-recreate
```

Set `NVIDIA_VISIBLE_DEVICES` in `.env` if you need to target specific GPUs (for example `NVIDIA_VISIBLE_DEVICES=0`).

**Intel Docker access note**: Intel GPU/NPU inference requires device nodes inside the container. The shipped `docker-compose.yml` now documents separate Linux and Windows/WSL2 Intel snippets:

- Linux Intel hosts: uncomment `group_add: ["${HOST_INTEL_RENDER_GID:?...}"]` (set `HOST_INTEL_RENDER_GID` in `.env` via `stat -c '%g' /dev/dri/renderD* | head -n 1`; no silent default) plus `/dev/dri:/dev/dri` and `/dev/accel:/dev/accel`.
- Windows 11 / WSL2 Intel hosts: uncomment `/dev/dxg:/dev/dxg`, and also `/dev/dri:/dev/dri` and `/dev/accel:/dev/accel` when WSL exposes them.

**Intel telemetry container-access note (Ubuntu 24.04)**: Hardware telemetry tools need additional low-level visibility beyond device-node passthrough.

- `intel_gpu_top` requires Intel DRM access plus performance-counter capability (`PERFMON` preferred; `SYS_ADMIN` fallback for older runtimes).
- The Windows/WSL2 Intel snippet in `docker-compose.yml` also documents `pid: host`, `privileged: true`, and relaxed seccomp/capability settings to improve PMU/sysfs compatibility when telemetry or OpenVINO enumeration is blocked by container isolation.
- NPU busy-time telemetry requires read-only sysfs visibility so `npu_busy_time` or `npu_busy_time_us` counters are visible from inside the container.
- No extra host package installs are required. Intel telemetry CLI tooling is bundled automatically in the container image, with runtime fallback to sysfs-delta probing when optional tools are unavailable.

**Intel launch note**: After uncommenting the relevant Intel block in `docker-compose.yml`, launch normally:

```bash
docker compose up -d --build --force-recreate
```

For Linux Intel hosts, prefer `/dev/dri:/dev/dri` and `/dev/accel:/dev/accel` directory mappings instead of a single `/dev/accel/accel0` node so multi-GPU and multi-NPU systems stay visible to runtime discovery.

Equivalent direct `docker run` telemetry flags:

```bash
docker run -d \
  --device /dev/dri:/dev/dri \
  --device /dev/accel:/dev/accel \
  --pid=host \
  --privileged \
  --cap-add=PERFMON \
  --cap-add=SYS_ADMIN \
  -v /sys:/sys:ro \
  -v /sys/class/accel:/sys/class/accel:ro \
  -v /sys/bus/pci/drivers/intel_vpu:/sys/bus/pci/drivers/intel_vpu:ro \
  ventura8/whisper-pro-asr
```

**Production persistence note**: The runtime writes history/telemetry to `/app/data` by default. Keep `./data:/app/data` mounted so tasks survive restarts and image updates. The `init-permissions` init container guarantees the directory is writable by the configured UID/GID on every deployment.

**History compatibility note**: Default mapping remains `./data:/app/data`. If a prior setup wrote history into `./data`, runtime can import from legacy candidates automatically; you can also set `WHISPER_LEGACY_STATE_DIR` explicitly if needed.

**Intel detection note**: The runtime treats OpenVINO device enumeration as authoritative for runnable Intel GPU/NPU scheduler units. Linux nodes (`/dev/accel/accel0`, `/dev/dri`) are still reported in startup diagnostics, but node visibility alone no longer registers Intel units for scheduling; OpenVINO must report `NPU`/`GPU` inside the container.

**Intel diagnostics note**: Startup logs now include deeper Intel probes to pinpoint why OpenVINO may still report `devices=['CPU']`:

- `OpenVINO target probe`: direct `FULL_DEVICE_NAME` probes for `GPU`, `GPU.0`, `NPU`, `NPU.0` with per-target errors.
- `Intel process security`: `uid/gid/groups`, effective Linux capabilities (`CapEff`), and seccomp mode.
- `Intel runtime env`: key variables (`INTEL_OPENVINO_DIR`, `LD_LIBRARY_PATH`, `LIBVA_DRIVER_NAME`, `ONEAPI_DEVICE_SELECTOR`, `ZE_AFFINITY_MASK`, `OCL_ICD_VENDORS`).
- `Intel node details`: mode/uid/gid for `/dev/accel/*` and `/dev/dri/renderD*`.
- `Intel sysfs`: vendor/device/driver mapping from `/sys/class/drm/renderD*/device` and `/sys/class/accel/accel*/device`.

Deep OpenVINO target probing (`GPU/GPU.0/NPU/NPU.0`) is **disabled by default** to avoid crash loops on some driver/plugin combinations. Enable only when explicitly troubleshooting:

```bash
INTEL_DEEP_OV_PROBE=true docker compose up -d --build --force-recreate
```

If nodes are visible but `OpenVINO target probe` reports GPU/NPU unavailable, the blocker is usually host kernel driver exposure or container security policy rather than ONNX provider injection.

## AMD GPU Acceleration (native Linux ROCm)

### Supported Hardware Compatibility

AMD ROCm GPU acceleration inside Docker containers is **implemented but unverified**. The
images carry pre-compiled HIP kernels for the consumer Radeon architectures below, and
nothing in this table has been confirmed on real supported silicon:

| GPU Series | Models | Architecture | Status |
| --- | --- | --- | --- |
| **Radeon RX 7000 Series** | RX 7900 XTX, RX 7900 XT, RX 7900 GRE, RX 7800 XT, RX 7700 XT, RX 7600 | `gfx1100`, `gfx1101`, `gfx1102` | ❔ **Kernels shipped, unverified** |
| **Radeon RX 6000 Series** | RX 6950 XT, RX 6900 XT, RX 6800 XT, RX 6800, RX 6700 XT | `gfx1030`, `gfx1031` | ❔ **Kernels shipped, unverified** |
| **Ryzen iGPUs (APUs)** | Ryzen AI 9 HX 370 / Strix Point integrated graphics | `gfx1150` | ❔ **Unverified on Linux** *(needs `HSA_OVERRIDE_GFX_VERSION=11.0.0`)*<br>ℹ️ **CPU Fallback on Windows WSL2** |
| **Radeon RX 9000 Series** | RX 9070 XT, RX 9070, RX 9070 GRE, RX 9060 XT, RX 9060 | `gfx1201`, `gfx1200` | ❔ **Kernels shipped, unverified** |
| **Desktop Ryzen display iGPUs** | Ryzen 7000/9000 desktop integrated graphics (e.g. Ryzen 9 9950X3D) | `gfx1036` | ❌ **Not usable — CPU only** *(measured; no override works, see below)* |

❔ means the ROCm kernels for that architecture ship in the `amd` and `full` images and the
code path is implemented, but **it has never been exercised on that silicon**. The only
Radeon available for testing was a `gfx1036` integrated part, which is the one row here
backed by measurement -- and it is the negative result. Do not read ❔ as working.

This matches the AMD note in `README.md` and the compatibility matrix in
`docs/DOCKERHUB_DESCRIPTION.md`; if you validate a card, update all three together.

#### Shipped GPU kernel architectures

ROCm ships pre-compiled kernels for every architecture AMD has ever supported, and they are
large. The published `amd` and `full` images support consumer Radeon only: they keep RDNA2,
RDNA3, and RDNA4 kernels and omit data-center and legacy architectures. Native Linux still
needs the compatible AMDGPU driver and `/dev/kfd`; Windows WSL2 remains CPU fallback.

`gfx1031` (RX 6700 XT) and `gfx1150` (Ryzen APUs) ship no kernels of their own and run on
`gfx1030` / `gfx1100` respectively via `HSA_OVERRIDE_GFX_VERSION`, so both remain supported.

#### Why `gfx1036` cannot be rescued by an override

The small display iGPU built into desktop Ryzen CPUs reports as `gfx1036`. The usual trick
of pretending it is a supported neighbour -- `HSA_OVERRIDE_GFX_VERSION=10.3.0`, borrowing
RDNA2's `gfx1030` kernels -- gets part of the way and then stops, measured on a Ryzen 9
9950X3D:

| Layer | Without override | With `HSA_OVERRIDE_GFX_VERSION=10.3.0` |
| --- | --- | --- |
| ROCm agent enumeration | ✅ `gfx1036` visible via `/dev/kfd` | ✅ |
| ONNX Runtime ROCm provider loads | ✅ | ✅ |
| Elementwise kernels (`Add`, `Mul`) | ❌ `hipErrorNoBinaryForGpu` | ✅ **runs on GPU, numerically correct** |
| Matrix multiply (rocBLAS) | ❌ `hipErrorNoBinaryForGpu` | ❌ `hipErrorNotFound: named symbol not found` |

The override genuinely works at the HIP level: ONNX Runtime's own kernels compile and
execute correctly on the iGPU. It fails one layer up, in rocBLAS, whose Tensile kernel
library upstream is built for `gfx803 gfx900 gfx906 gfx908 gfx90a gfx1010 gfx1030 gfx1100
gfx1101 gfx1102` and has never included `gfx1036`. This is an upstream gap, not something
this project's kernel pruning removed. Setting `ROCBLAS_USE_HIPBLASLT=0` does not help.

Since every layer of Whisper and every UVR separation model is dominated by matrix
multiplication and convolution, a GPU that can add but cannot multiply matrices is of no
use here. Leave `gfx1036` systems on CPU, or use a discrete card.

---

### Setup on Windows 11 (WSL2 Docker Containers)

Linux Docker containers on Windows WSL2 **cannot accelerate UVR on the AMD GPU**. DirectML is a Windows-only ONNX provider, and `onnxruntime-rocm` requires `/dev/kfd`, which standard WSL2 does not expose. Mapping `/dev/dxg` only helps detect an AMD adapter; UVR falls back to CPU.

1. **Host Driver Requirement**: Install **AMD Software: Adrenalin Edition** (v26.2.2 or newer) on your Windows host if you want AMD hardware to be detected.
2. **`docker-compose.yml` Configuration** (detection only; UVR remains CPU):

   ```yaml
   services:
     whisper-pro-asr:
       image: whisper-pro-asr:latest
       ports:
         - "9000:9000"
       devices:
         - /dev/dxg:/dev/dxg # Windows 11 / WSL2 GPU bridge (detection only)
       environment:
         - HSA_ENABLE_DXG_DETECTION=1
       volumes:
         - /usr/lib/wsl:/usr/lib/wsl # WSL2 host driver library mount
   ```

3. **Equivalent `docker run` Command**:

   ```bash
   docker run -d \
     --name whisper-pro-asr \
     -p 9000:9000 \
     --device /dev/dxg \
     -v /usr/lib/wsl:/usr/lib/wsl \
     -e HSA_ENABLE_DXG_DETECTION=1 \
     whisper-pro-asr:latest
   ```

---

### Setup on Linux (Bare-Metal Docker Containers)

On native Linux hosts (Ubuntu 22.04+), Docker containers access the AMD GPU directly via native Linux ROCm kernel drivers (`/dev/kfd` and `/dev/dri`).

1. **Host Driver Requirement**: Install `amdgpu-dkms` or standard AMD ROCm kernel drivers on the host Linux OS. Ensure the runtime user belongs to the device-owning groups.
2. **Deriving Host GIDs**: On Linux, obtain the numeric GIDs for the render and video device nodes:

   ```bash
   RENDER_GID=$(stat -c '%g' /dev/dri/renderD* | head -n 1)
   VIDEO_GID=$(stat -c '%g' /dev/kfd)
   echo "HOST_AMD_RENDER_GID=$RENDER_GID"
   echo "HOST_AMD_VIDEO_GID=$VIDEO_GID"
   ```

3. **`docker-compose.yml` Configuration**:

   ```yaml
   services:
     whisper-pro-asr:
       image: whisper-pro-asr:latest
       ports:
         - "9000:9000"
       devices:
         - /dev/kfd:/dev/kfd # AMD ROCm Kernel Fusion Driver (Bare-metal Linux only)
         - /dev/dri:/dev/dri # Direct Rendering Manager render nodes
       group_add:
         - "${HOST_AMD_RENDER_GID:-990}" # Numeric GID for /dev/dri render node
         - "${HOST_AMD_VIDEO_GID:-44}"   # Numeric GID for /dev/kfd device node
   ```

4. **Ryzen iGPUs (`gfx1150` on Strix Point / Ryzen AI APUs)**:
   On Linux hosts, you can enable GPU acceleration for Ryzen iGPUs (`gfx1150`) by setting the ROCm architecture override flag in `environment`:

   ```yaml
       environment:
         - HSA_OVERRIDE_GFX_VERSION=11.0.0
   ```

5. **Equivalent `docker run` Command**:

   ```bash
   docker run -d \
     --name whisper-pro-asr \
     -p 9000:9000 \
     --security-opt seccomp=unconfined \
     --device /dev/kfd \
     --device /dev/dri \
     --group-add $(stat -c '%g' /dev/dri/renderD* | head -n 1) \
     --group-add $(stat -c '%g' /dev/kfd) \
     whisper-pro-asr:latest
   ```

   *Optional for Ryzen APUs requiring GFX version override:*

   ```bash
   docker run -d \
     --name whisper-pro-asr \
     -p 9000:9000 \
     --security-opt seccomp=unconfined \
     --device /dev/kfd \
     --device /dev/dri \
     --group-add $(stat -c '%g' /dev/dri/renderD* | head -n 1) \
     --group-add $(stat -c '%g' /dev/kfd) \
     -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
     whisper-pro-asr:latest
   ```

**Intel telemetry note**: Real Intel GPU/NPU utilization in dashboard hardware charts uses native Linux sysfs counters first, then Intel-native CLI probes (`intel_gpu_top` and `nputop`), then Windows performance counters, and only then compatibility inference values. Runtime resolves Intel sysfs metric nodes dynamically (instead of assuming `card0`/`accel0`) and accepts decimal/percent-style utilization payloads, reducing false `0%` GPU readings and `0/100` NPU oscillation. Synthetic fallback activity is stage-gated and only reports busy during actual UVR or inference/language-detection accelerator stages; initialization, standardization, uploads, and other non-accelerator stages stay at `0%`.

Runtime now performs an automatic telemetry self-check during startup and logs `[intel-telemetry] Runtime diagnostics` with:

- Device-node visibility (`/dev/dri`, `/dev/accel`, `/dev/dxg`)
- Sysfs visibility (`/sys/class/drm`, `/sys/class/accel`, `/sys/bus/pci/drivers/intel_vpu`)
- Intel telemetry tool availability (`intel_gpu_top`, `nputop`/`npu-top`, `timeout`)

These diagnostics are generated automatically after update with no host-side setup steps.

`intel_gpu_top` integration details:

- Probe execution is bounded: runtime wraps `intel_gpu_top` with GNU `timeout` (when available) so JSON sampling exits cleanly instead of hanging indefinitely.
- Non-zero timeout exits with stdout payload are treated as valid samples (for example `timeout` exit `124` with parseable JSON output).
- Parser now accepts JSON numeric utilization keys (`busy`, `util`, `utilization`, `load`, `active`) even when `%` is omitted, plus nested `{"value": ...}` payloads.
- Runtime also probes DRM engine busy counters (`/sys/class/drm/card*/engine/*/busy`) and derives utilization from deltas, which keeps Intel GPU telemetry working when PMU/perf-counter access for `intel_gpu_top` is restricted.

Linux NPU integration details:

- Runtime now probes `intel_vpu` cumulative busy-time counters (`npu_busy_time` and `npu_busy_time_us`) from sysfs paths (for example `/sys/bus/pci/drivers/intel_vpu/*/npu_busy_time_us`) and computes utilization from consecutive deltas.
- This delta-based path is consulted before synthetic activity fallback, which helps avoid persistent `0%` chart values when direct `utilization` nodes are absent.

Enable detailed telemetry trace logs when debugging chart mismatches:

```bash
INTEL_TELEMETRY_DEBUG=true docker compose up -d --build --force-recreate
```

This emits `[intel-telemetry]` log lines showing discovered sysfs paths, vendor filtering decisions, raw source values (`sysfs` / `intel_native_cli` / windows counters), and final selected source/value per GPU/NPU sample.

**AMD GPU note**: When an AMD GPU is detected (via `/dev/kfd`, `/dev/dri` DRM vendor `0x1002`, or WSL driver visibility), the runtime loads `onnxruntime-rocm` from `/app/libs/amd` and registers an `amd:0` scheduler unit. UVR vocal isolation runs on the AMD GPU only on native Linux ROCm hosts with `/dev/kfd`. On Windows 11 / WSL2, `/dev/dxg` allows adapter detection for this Linux image, but UVR falls back to CPU because DirectML requires `sys.platform == "win32"` and ROCm GPU execution still requires `/dev/kfd`. Whisper ASR inference falls back to CPU on AMD units because CTranslate2 does not have a ROCm backend.

For Linux AMD hosts, map `/dev/kfd` and `/dev/dri`. For Windows 11 / WSL2, map `/dev/dxg` only for detection and diagnostics. Set `MAX_AMD_UNITS=1` to enable the AMD unit (and cap AMD parallelism via `get_parallel_limit('AMD')`). When both NVIDIA and AMD GPUs are present, Whisper ASR runs on NVIDIA CUDA while UVR preprocessing uses AMD on native Linux ROCm hosts and falls back to CPU on WSL2.

**UVR OpenVINO device note**: For vocal separation, generic preprocess targets like `ASR_PREPROCESS_DEVICE=NPU` are resolved against OpenVINO runtime-reported device IDs. GPU slot selection is passed directly in OpenVINO `device_type` (for example `GPU.0`, `GPU.1`). NPU slot selection uses OpenVINO `load_config` with `DEVICE_ID` while keeping provider `device_type=NPU`, because the ORT OpenVINO provider in this runtime rejects dotted NPU `device_type` values such as `NPU.0`. When `ASR_PREPROCESS_DEVICE=AUTO`, the runtime selects the next available Intel accelerator in OpenVINO discovery order and falls back to CPU if no Intel accelerator is available.

**OpenVINO compatibility note**: The runtime image pins OpenVINO 2026.3.1. Intel startup logs include ONNX Runtime/OpenVINO versions, provider paths, available devices, and Linux node visibility to distinguish device-mapping failures from provider compatibility failures. Hardware validation must confirm provider/device metadata identifies `GPU`; CPU fallback is not accepted as acceleration evidence.

**OpenVINO runtime env note**: The container exports `INTEL_OPENVINO_DIR=/opt/intel/openvino` and extends `LD_LIBRARY_PATH` with OpenVINO runtime library paths so `onnxruntime-openvino` can load provider dependencies in non-interactive startup paths. In hybrid images, the system OpenCL ICD loader remains ahead of CUDA's bundled OpenCL library so Intel GPU discovery can use `intel.icd`; CUDA libraries remain available later in the path. `LD_LIBRARY_PATH` is composed per build target inside the image and must **not** be restated in `docker-compose.yml`.

**Intel compose note**: `docker-compose.yml` exposes `LIBVA_DRIVER_NAME`, `ONEAPI_DEVICE_SELECTOR`, `ZE_AFFINITY_MASK`, and `OCL_ICD_VENDORS` as optional pass-through variables so the banner can show the exact container runtime environment when OpenVINO reports `devices=['CPU']`. Intel device passthrough lives in `docker-compose.intel.yml` (or `docker-compose.nvidia-intel.yml`), merged onto the base file.

**ONNX Runtime policy note**: The container uses deterministic ONNX runtime paths under `/app/libs/*` (`/app/libs/cpu`, `/app/libs/nvidia`, `/app/libs/intel`, `/app/libs/amd`) instead of relying on ambiguous site-packages resolution. This avoids accidental CPU-only runtime selection on Intel preprocessing targets. Per-vendor images ship only their own runtime plus `/app/libs/cpu`; `modules/core/bootstrap.py` existence-checks every candidate, including explicit `ASR_DEVICE` selections, and degrades to the CPU runtime when the requested vendor library is absent.

### Local hardware validation

Every other test mocks the ASR engine, so a broken accelerator path still passes them.
**Always run the real-engine accuracy test when validating on a local machine:**

```bash
# 1. Bring up the stack for the target the audit recommended (BUILD_TARGET in .env):
#    cpu | intel | intel-xpu | nvidia | nvidia-intel | nvidia-whisperx | amd | amd-rocm-torch | full
# .env has to be sourced: compose reads it for the container's environment, but the shell
# expanding the filename below does not, so without this the override resolves to the
# literal "docker-compose..yml" and compose fails on a file that does not exist.
set -a; . ./.env; set +a
docker compose -f docker-compose.yml -f "docker-compose.${BUILD_TARGET}.yml" up -d

# 2. Run the real-engine checks through the Docker test image, never on the host.
RUN_REAL_ASR=1 PIPELINE_STAGE=real-audio scripts/ci/build-and-test.sh          # smoke, <20 min
RUN_REAL_ASR=1 PIPELINE_STAGE=real-audio-stress scripts/ci/build-and-test.sh   # full matrix, ~2h
```

The override must match `BUILD_TARGET`; `docker-compose.nvidia.yml` is an example, not a
default. `pytest` runs inside the test image because the real engine needs the per-vendor
ONNX Runtime under `/app/libs`, which a host interpreter does not have -- and because this
project runs every quality gate through `Dockerfile.test` rather than on the host.

It posts `tests/e2e/fixtures/speech_known_text.wav` to the running service and asserts the
transcript contains both known sentences -- *"The quick brown fox jumps over the lazy
dog."* and *"Whisper Pro ASR is running a hardware acceleration test on this machine."* --
plus that segment timings span the full ~8.3s clip rather than stopping after the first
sentence.

The test is skipped unless `RUN_REAL_ASR=1`, so it never slows CI. It drives a live
container over HTTP (override with `WHISPER_BASE_URL`) because the real engine needs the
per-vendor ONNX Runtime under `/app/libs` and a provisioned `model_cache`, neither of which
exists in the test image. On a cold cache the request waits in the queue while the model
downloads; raise `REAL_ASR_TIMEOUT` if your connection is slow.

A passing transcript proves decoding works, **not** that the accelerator was used -- CPU
fallback also transcribes correctly. Pair it with `nvidia-smi --query-compute-apps` (CUDA)
or `intel_gpu_top` (Intel) as required above.

#### Multilingual real-audio matrix

The single English fixture proves the engine decodes; it says nothing about the other
languages the service advertises. `tests/real_audio/` drives the same live service with
real neural speech in each language, plus degraded and malformed audio.

Run the **smoke set** by default -- a representative subset (4 languages across Latin,
Cyrillic and CJK, one code-switched clip, five degraded/malformed cases, and the request
contract checks) budgeted to finish in under 20 minutes:

```bash
RUN_REAL_ASR=1 PIPELINE_STAGE=real-audio scripts/ci/build-and-test.sh
```

The **full matrix** is opt-in stress testing. It takes about two hours, dominated by UVR
Vocal Separation preprocessing (~30-40s per request) rather than decoding, and adds the
20-minute long-form clip:

```bash
RUN_REAL_ASR=1 PIPELINE_STAGE=real-audio-stress scripts/ci/build-and-test.sh
```

The marker expressions those stages select (`real_audio and smoke`, and the whole matrix)
live in `tests/run_suite.sh`; the table further down maps each stage to its budget.

The committed core tier (`tests/e2e/fixtures/audio_matrix/core/*.flac`) needs no tooling.
To generate the rest -- the long tail, code-switched clips, adversarial audio and the
long-form stress clip -- install the fixture generator and run it once:

```bash
poetry install --with tools
python3 scripts/generate_audio_matrix.py all
```

Generation is idempotent and content-addressed; see
`tests/e2e/fixtures/audio_matrix/README.md`. `python3 scripts/generate_audio_matrix.py
verify` prints language coverage, including the languages Piper has no voice for.

Expectations are **data, not code**: each clip's manifest entry carries its tier, its
expected words, its acceptable detection codes and an optional `xfail_reason`. Tier A
asserts both transcript content and detected language; tier B (the long tail) asserts the
weaker "detected correctly, or at least transcribed to something". Tune a language by
editing `manifest.json`, never by editing a test.

| Variable | Default | Purpose |
| --- | --- | --- |
| `WHISPER_BASE_URL` | `http://127.0.0.1:9000` | Service under test |
| `REAL_ASR_TIMEOUT` | `900` | Per-request timeout (a cold cache downloads the model) |
| `REAL_ASR_ADVERSARIAL_TIMEOUT` | `120` | Tighter budget for malformed input; a timeout is a failure |
| `ASR_AUDIO_MATRIX_DIR` | `test_data/audio_matrix` | Generation cache and voice models |
| `RUN_GPU_LONG_ASR` | unset | Set to `1` to run the 20-minute long-form clip (NVIDIA only) |
| `LONG_ASR_MAX_RTF` | `1.0` | Wall-clock budget for the long-form clip, as a fraction of real time |

The long-form test additionally requires `nvidia-smi` on the host and is marked `slow`, so
it is excluded from the CI test stage by construction as well as by its skip guard.

Inside the Docker test image there are two stages, both opt-in and never part of `all`:

| Stage | Selection | Budget |
| --- | --- | --- |
| `PIPELINE_STAGE=real-audio` | `real_audio and smoke` | under 20 min |
| `PIPELINE_STAGE=real-audio-stress` | the whole matrix, then the long-form clip | ~2 h + 20 min |

Neither is referenced by `.github/workflows/ci.yml`. GitHub-hosted runners have no GPU, no
provisioned `model_cache` and no running service, so no real-engine test can execute there;
wiring the smoke stage into CI requires a self-hosted runner with the stack already up.

**Runtime model provisioning note**: Images no longer bake model weights. `modules/core/model_provisioning.py` downloads the models the active configuration needs at startup into `/app/model_cache`, reusing them when already present and valid. The container reports healthy immediately, and tasks submitted during the download wait in the scheduler queue with stage `Downloading Model (xx%)` rather than being rejected; `GET /status` reports `engines.whisper.status: "downloading"`. A failed download releases the queue gate so tasks surface a real engine error instead of waiting forever.

The UVR preload download stages temporary output inside the build cache (falling back to the configured persistent temp directory), so production builds do not depend on the retired `CACHE_DIR` configuration setting.

**Numba cache note**: Runtime now sets `NUMBA_CACHE_DIR=/tmp/numba-cache` by default so libraries like `librosa` can write JIT caches, preventing `no locator available` cache failures during preprocessing.

**UVR accelerator fallback note**: When vocal separation explicitly targets Intel preprocessing (`ASR_PREPROCESS_DEVICE=NPU` or `GPU`) and OpenVINO cannot initialize the requested device, runtime retries other available Intel OpenVINO devices first, logs any ONNX Runtime CPU provider fallback, and then falls back to CPU preprocessing. `AUTO` may choose CPU immediately when OpenVINO reports no Intel accelerators.

**Dashboard hardware graph note**: The Hardware Acceleration chart now uses a hybrid distinction model for overlapping series with custom legend badges: explicit labels (`TYPE UNIT_ID - Name`) plus per-type line pattern and marker shape. Defaults are CUDA = solid/circle, Intel GPU = dashed/square, NPU = short-dashed/triangle. Marker spacing is also unit-specific so symbol cadence differs between series for easier visual separation.

Legacy telemetry fallback remains multi-unit aware: when `hardware_util` is missing, per-unit values are resolved by unit ID/index from legacy NVIDIA arrays and Intel GPU/NPU keyed/indexed telemetry fields.

First build exports model to INT8 (~5-10 min, ~4GB RAM).

## 3. Configuration & Device Selection

The service utilizes **Autonomous Hardware Sensing**, but the engine and the device are two
separate decisions.

**Engine.** `ASR_ENGINE=AUTO` always resolves to **`FASTER-WHISPER`**, on every host. This
is deliberate: when the engine varied with the accelerator, the decoding behaviour -- and
therefore the transcript -- became a property of whichever machine a request landed on. One
engine everywhere makes a deployment reproducible across a fleet.

**Device.** The hardware still decides which *unit* runs the task, in this order:

1. **NVIDIA CUDA**
2. **AMD GPU** (UVR on native Linux AMD ROCm; WSL2 `/dev/dxg` is detection only)
3. **Intel GPU (Arc/iGPU)**
4. **Intel NPU**
5. **CPU (Fallback)**

CTranslate2 has CUDA and CPU backends and nothing else. When the selected unit is one it
cannot drive -- AMD, Intel GPU, Intel NPU -- the ASR device is reported as **CPU** rather
than naming a device it cannot address, and that unit remains available for vocal
isolation, which runs on ONNX Runtime and does reach it.

An explicit `ASR_DEVICE` still constrains which tier is chosen.

**To accelerate ASR on Intel or AMD, ask for the engine that can:**

| Hardware | Engine to request |
| :--- | :--- |
| Intel Arc / iGPU | `ASR_ENGINE=INTEL-WHISPER` |
| Intel NPU | CPU fallback for ASR; use NPU for vocal isolation |
| AMD Radeon (ROCm) | `ASR_ENGINE=OPENAI-WHISPER` with the `amd-rocm-torch` image |

**Hybrid engines.** `HYBRID_ENGINES` is `false` by default. On a host carrying both a
CUDA/AMD GPU *and* an Intel GPU/NPU, setting it to `true` lets each unit run its native
engine in its own worker process, so both accelerators work at once. It is opt-in because
it reintroduces exactly the per-unit engine variation the single default engine removes.
It is ignored on single-vendor hosts, where it cannot apply.

If `ASR_ENGINE` is set explicitly to an unsupported value, startup fails fast with a clear validation error.

**Preprocessing Toggles:**

- `ENABLE_VOCAL_SEPARATION=false` (default): UVR/MDX-NET vocal isolation is off. Measured on
  an RTX 5090 it gave no gain on clean speech (0.9469 either way), cost 1.7 points on harder
  audio, and ran 76% slower (RTF 0.063 -> 0.110). Its one win was hallucinated quiet windows
  (12 -> 10 of 25), and WHISPERX scores 0/25 there while being faster. Turn it on for
  music-heavy source material, where there is genuinely non-speech content to strip.

## Speaker Diarization Setup

To enable speaker diarization (identifying who said what), you need a **Hugging Face token** with access to PyAnnote speaker segmentation models:

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Accept the license terms for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
3. Generate an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Set the diarization token environment variable in your `docker-compose.yml` (`DIARIZATION_HF_TOKEN`):

```yaml
environment:
  - DIARIZATION_HF_TOKEN=hf_your_token_here
```

> [!IMPORTANT]
> **Without `DIARIZATION_HF_TOKEN`**, diarization requests will fall back to standard transcription (without speaker labels). The token is only required if you use `diarize=true` in API calls.

## Volume Mapping

Edit `docker-compose.yml`:

```yaml
volumes:
  - ./model_cache:/app/model_cache     # NPU compilation blobs + diarization models (Critical for fast reload)
  - ./data:/app/data                   # Task history, telemetry, and system logs (upgrade-safe; backward-compatible)
  - /mnt/nas/movies:/movies            # Your media (mapped to same path as in Bazarr)
  - /mnt/nas/tv:/tv
```

> [!TIP]
> The `model_cache` volume now also stores cached WhisperX alignment and PyAnnote diarization models. Mapping this volume avoids re-downloading these models on container restarts.

## SSD Protection

If running on an SSD, consider adding a `tmpfs` mount to minimize write wear. See `docs/TUNING.md` for details.

## Verify

```bash
docker compose logs -f
# Look for: "Model loaded successfully!"
```

## Watch JS E2E In A Real Browser

Use the headed Playwright helpers to see test steps live in Chromium.

Linux/macOS (sh):

```bash
scripts/quality/run-e2e-headed.sh
```

Windows (PowerShell):

```powershell
./scripts/quality/run-e2e-headed.ps1
```

Run a single spec (both scripts accept extra Playwright args):

```bash
scripts/quality/run-e2e-headed.sh tests/e2e/dashboard-filters.spec.cjs
```

```powershell
./scripts/quality/run-e2e-headed.ps1 -- tests/e2e/dashboard-filters.spec.cjs
```

Optional slow-motion override:

```bash
PW_SLOW_MO=300 scripts/quality/run-e2e-headed.sh
```

```powershell
./scripts/quality/run-e2e-headed.ps1 -SlowMoMs 300
```

## Concurrency Verification (Required for Scheduler Changes)

When modifying scheduler, preemption, or model lifecycle code, run concurrency-focused tests before merge:

```bash
pytest -q tests/inference/scheduler/test_scheduler.py tests/inference/scheduler/test_concurrency_coverage_edges.py tests/inference/scheduler/priority/*
```

Then run the complete suite in your normal CI/local workflow.

For local parity with CI complexity gates, run:

```bash
python3 -m radon cc -n B modules whisper_pro_asr.py
```

This command must produce no output. Any reported block means complexity rank `B` or worse and is a build failure condition.

## Troubleshooting

- **NPU preprocessing unavailable**: Some NPU versions have memory limits for static shapes. Set `ASR_PREPROCESS_DEVICE=CPU` to bypass NPU preprocessing; Whisper ASR itself uses CPU unless an Intel GPU is selected with `INTEL-WHISPER`.
- **Diarization not working**: Ensure `DIARIZATION_HF_TOKEN` is set and you have accepted the PyAnnote model license on Hugging Face.
- **Models consuming too much RAM when idle**: Set `MODEL_IDLE_TIMEOUT=300` to automatically unload models after 5 minutes of inactivity. A deferred cleanup timer starts after the last task completes and is cancelled when new tasks arrive, preventing unnecessary model reloads.
- **Out of memory or hangs during long movie transcription/vocal separation**: Ensure chunked processing is enabled. Check that `INTEL_ASR_CHUNK_DURATION` (default `300` seconds) and `UVR_CHUNK_DURATION` (default `600` seconds) are configured in your environment.
- **Optimization**: Check `docs/TUNING.md` for performance profiles.

The Docker Radon gate targets production Python and coverage tooling; test
orchestration remains enforced by the dedicated test, lint, and security stages.
