# Setup Guide

## Prerequisites

- Intel Core Ultra (Meteor Lake/Lunar Lake) with NPU, OR NVIDIA GPU (CUDA), OR AMD GPU (ROCm/DirectML), OR generic CPU
- Windows 11 (WSL2) or Linux (Ubuntu 22.04+)
- Intel NPU drivers installed (for NPU acceleration)
- Docker

## Installation

### Method 1: Docker Hub (Recommended)

```bash
docker run -d --name whisper-pro-asr -p 9000:9000 --device /dev/accel --device /dev/dri --user 65534:65534 --group-add 991 ventura8/whisper-pro-asr
```

### Method 2: Local Build

```bash
git clone https://github.com/ventura8/Whisper-Pro-ASR.git
cd Whisper-Pro-ASR
docker compose up -d --build
```

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

This controls the container timezone (`TZ`) without editing compose YAML. The default compose runtime keeps a non-root identity example in `docker-compose.yml`; Linux Intel hosts should also use `group_add: ["991"]` when their render/accel nodes require it.

Before starting, ensure local bind-mount directories exist:

```bash
mkdir -p data model_cache
```

If you cannot grant write access for the configured UID/GID, use a runtime identity that can write those paths.

For local quality runs, `Dockerfile.test` uses BuildKit cache mounts for `apt`, `pip`, `poetry`, `npm`, and Playwright browser downloads to speed repeated builds. Keep Docker BuildKit enabled.

**Note**: The system automatically detects and utilizes NVIDIA CUDA, Intel NPU, or Intel GPU. Manual device selection (`ASR_DEVICE`) is now optional.

**NVIDIA CUDA compose note**: Uncomment the NVIDIA reservation block in `docker-compose.yml` when deploying on CUDA hosts, then launch normally:

```bash
docker compose up -d --build --force-recreate
```

Set `NVIDIA_VISIBLE_DEVICES` in `.env` if you need to target specific GPUs (for example `NVIDIA_VISIBLE_DEVICES=0`).

**Intel Docker access note**: Intel GPU/NPU inference requires device nodes inside the container. The shipped `docker-compose.yml` now documents separate Linux and Windows/WSL2 Intel snippets:

- Linux Intel hosts: uncomment `group_add: ["991"]` plus `/dev/dri:/dev/dri` and `/dev/accel:/dev/accel`.
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

**Production persistence note**: The runtime writes history/telemetry to `/app/data` by default. Keep `./data:/app/data` mounted so tasks survive restarts.

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

## AMD GPU Acceleration (ROCm / DirectML)

### Supported Hardware Compatibility

AMD ROCm GPU acceleration inside Docker containers is supported for **discrete AMD Radeon and Instinct GPUs** with pre-compiled HIP kernel architectures:

| GPU Series | Models | Architecture | Status |
| --- | --- | --- | --- |
| **Radeon RX 7000 Series** | RX 7900 XTX, RX 7900 XT, RX 7900 GRE, RX 7800 XT, RX 7700 XT | `gfx1100`, `gfx1101` | ✅ **Full Hardware Acceleration** |
| **Radeon RX 6000 Series** | RX 6950 XT, RX 6900 XT, RX 6800 XT, RX 6800, RX 6700 XT | `gfx1030`, `gfx1031` | ✅ **Full Hardware Acceleration** |
| **Instinct Accelerators** | MI300X, MI300A, MI250X, MI250, MI210, MI100 | `gfx942`, `gfx90a` | ✅ **Full Hardware Acceleration** |
| **Ryzen iGPUs (APUs)** | Ryzen AI 9 HX 370 / Strix Point integrated graphics | `gfx1150` | ✅ **Supported on Linux** *(via `HSA_OVERRIDE_GFX_VERSION=11.0.0`)*<br>ℹ️ **CPU Fallback on Windows WSL2** |

---

### Setup on Windows 11 (WSL2 Docker Containers)

Docker containers on Windows WSL2 use AMD's **ROCDXG (`librocdxg`)** user-mode translation layer to accelerate UVR vocal separation over the DirectX bridge (`/dev/dxg`).

1. **Host Driver Requirement**: Install **AMD Software: Adrenalin Edition** (v26.2.2 or newer) on your Windows host.
2. **`docker-compose.yml` Configuration**:

   ```yaml
   services:
     whisper-pro-asr:
       image: whisper-pro-asr:latest
       ports:
         - "9000:9000"
       devices:
         - /dev/dxg:/dev/dxg # Windows 11 / WSL2 GPU bridge
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

1. **Host Driver Requirement**: Install `amdgpu-dkms` or standard AMD ROCm kernel drivers on the host Linux OS. Ensure the runtime user belongs to the `video` and `render` groups.
2. **`docker-compose.yml` Configuration**:

   ```yaml
   services:
     whisper-pro-asr:
       image: whisper-pro-asr:latest
       ports:
         - "9000:9000"
       devices:
         - /dev/kfd:/dev/kfd # AMD ROCm Kernel Fusion Driver
         - /dev/dri:/dev/dri # Direct Rendering Manager render nodes
       group_add:
         - video
         - render
   ```

3. **Ryzen iGPUs (`gfx1150` on Strix Point / Ryzen AI APUs)**:
   On Linux hosts, you can enable GPU acceleration for Ryzen iGPUs (`gfx1150`) by setting the ROCm architecture override flag in `environment`:

   ```yaml
       environment:
         - HSA_OVERRIDE_GFX_VERSION=11.0.0
   ```

4. **Equivalent `docker run` Command**:

   ```bash
   docker run -d \
     --name whisper-pro-asr \
     -p 9000:9000 \
     --device /dev/kfd \
     --device /dev/dri \
     --group-add video \
     --group-add render \
     whisper-pro-asr:latest
   ```

   *Optional for Ryzen APUs requiring GFX version override:*

   ```bash
   docker run -d \
     --name whisper-pro-asr \
     -p 9000:9000 \
     --device /dev/kfd \
     --device /dev/dri \
     --group-add video \
     --group-add render \
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

**AMD GPU note**: When an AMD GPU is detected (via `/dev/kfd`, `/dev/dri` DRM vendor `0x1002`, or DirectML/ROCm ONNX execution provider availability), the runtime loads `onnxruntime-rocm` from `/app/libs/amd` and registers an `amd:0` scheduler unit. UVR vocal isolation runs on the AMD GPU via ONNX ROCm/DirectML. Whisper ASR inference falls back to CPU because CTranslate2 does not have a ROCm backend.

For Linux AMD hosts, map `/dev/kfd` and `/dev/dri`. For Windows 11 / WSL2, map `/dev/dxg`. Set `MAX_AMD_UNITS=1` to enable the AMD unit. When both NVIDIA and AMD GPUs are present, Whisper ASR runs on NVIDIA CUDA while UVR preprocessing uses AMD.

**UVR OpenVINO device note**: For vocal separation, generic preprocess targets like `ASR_PREPROCESS_DEVICE=NPU` are resolved against OpenVINO runtime-reported device IDs. GPU slot selection is passed directly in OpenVINO `device_type` (for example `GPU.0`, `GPU.1`). NPU slot selection uses OpenVINO `load_config` with `DEVICE_ID` while keeping provider `device_type=NPU`, because the ORT OpenVINO provider in this runtime rejects dotted NPU `device_type` values such as `NPU.0`. When `ASR_PREPROCESS_DEVICE=AUTO`, the runtime selects the next available Intel accelerator in OpenVINO discovery order and falls back to CPU if no Intel accelerator is available.

**OpenVINO compatibility note**: The runtime image pins OpenVINO 2026.2.1 to match the public `onnxruntime-openvino` 1.24.x support matrix. Intel startup logs include ONNX Runtime/OpenVINO versions, provider paths, available devices, and Linux node visibility to distinguish device-mapping failures from provider compatibility failures.

**OpenVINO runtime env note**: The container now exports `INTEL_OPENVINO_DIR=/opt/intel/openvino` and extends `LD_LIBRARY_PATH` with OpenVINO runtime library paths so `onnxruntime-openvino` can load provider dependencies in non-interactive startup paths.

**Intel compose note**: `docker-compose.yml` now exposes `LIBVA_DRIVER_NAME`, `ONEAPI_DEVICE_SELECTOR`, `ZE_AFFINITY_MASK`, and `OCL_ICD_VENDORS` as optional pass-through variables so the banner can show the exact container runtime environment when OpenVINO reports `devices=['CPU']`.

**ONNX Runtime policy note**: The container now uses deterministic ONNX runtime paths under `/app/libs/*` (`/app/libs/cpu`, `/app/libs/nvidia`, `/app/libs/intel`) instead of relying on ambiguous site-packages resolution. This avoids accidental CPU-only runtime selection on Intel preprocessing targets.

**Numba cache note**: Runtime now sets `NUMBA_CACHE_DIR=/tmp/numba-cache` by default so libraries like `librosa` can write JIT caches, preventing `no locator available` cache failures during preprocessing.

**UVR accelerator fallback note**: When vocal separation explicitly targets Intel preprocessing (`ASR_PREPROCESS_DEVICE=NPU` or `GPU`) and OpenVINO cannot initialize the requested device, runtime retries other available Intel OpenVINO devices first, logs any ONNX Runtime CPU provider fallback, and then falls back to CPU preprocessing. `AUTO` may choose CPU immediately when OpenVINO reports no Intel accelerators.

**Dashboard hardware graph note**: The Hardware Acceleration chart now uses a hybrid distinction model for overlapping series with custom legend badges: explicit labels (`TYPE UNIT_ID - Name`) plus per-type line pattern and marker shape. Defaults are CUDA = solid/circle, Intel GPU = dashed/square, NPU = short-dashed/triangle. Marker spacing is also unit-specific so symbol cadence differs between series for easier visual separation.

Legacy telemetry fallback remains multi-unit aware: when `hardware_util` is missing, per-unit values are resolved by unit ID/index from legacy NVIDIA arrays and Intel GPU/NPU keyed/indexed telemetry fields.

First build exports model to INT8 (~5-10 min, ~4GB RAM).

## 3. Configuration & Device Selection

The service utilizes **Autonomous Hardware Sensing**. For `ASR_ENGINE=AUTO`, it resolves backend in the following order:

1. **NVIDIA CUDA**
2. **AMD GPU** (UVR on AMD ROCm/DirectML; ASR on CPU)
3. **Intel GPU (Arc/iGPU)**
4. **Intel NPU**
5. **CPU (Fallback)**

Engine mapping for `ASR_ENGINE=AUTO`:

- CUDA -> `FASTER-WHISPER`
- AMD -> `FASTER-WHISPER` (CPU mode) with UVR on AMD GPU
- Intel GPU -> `INTEL-WHISPER`
- Intel NPU -> `INTEL-WHISPER`
- CPU -> `FASTER-WHISPER`

If `ASR_ENGINE` is set explicitly to an unsupported value, startup fails fast with a clear validation error.

**Preprocessing Toggles:**

- `ENABLE_VOCAL_SEPARATION=true`: isolates vocals using UVR/MDX-NET (recommended for accuracy).

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

- **Model not loading on NPU**: Some NPU versions have memory limits for static shapes. If the model fails to load or the server crashes on startup, set `ASR_BEAM_SIZE=4` in `docker-compose.yml`.
- **Diarization not working**: Ensure `DIARIZATION_HF_TOKEN` is set and you have accepted the PyAnnote model license on Hugging Face.
- **Models consuming too much RAM when idle**: Set `MODEL_IDLE_TIMEOUT=300` to automatically unload models after 5 minutes of inactivity. A deferred cleanup timer starts after the last task completes and is cancelled when new tasks arrive, preventing unnecessary model reloads.
- **Out of memory or hangs during long movie transcription/vocal separation**: Ensure chunked processing is enabled. Check that `INTEL_ASR_CHUNK_DURATION` (default `300` seconds) and `UVR_CHUNK_DURATION` (default `600` seconds) are configured in your environment.
- **Optimization**: Check `docs/TUNING.md` for performance profiles.
