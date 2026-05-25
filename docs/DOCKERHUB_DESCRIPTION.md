![GitHub Release](https://img.shields.io/github/v/release/ventura8/Whisper-Pro-ASR)
![Docker Pulls](https://img.shields.io/docker/pulls/ventura8/whisper-pro-asr)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/ventura8/Whisper-Pro-ASR/ci.yml)
![GitHub License](https://img.shields.io/github/license/ventura8/Whisper-Pro-ASR)

# Whisper Pro ASR (Multilingual)

**Whisper Pro ASR** is a high-performance, production-ready AI transcription service. It is optimized for **Whisper Large V3** and designed for seamless integration with **Bazarr** and the *arr stack. 

It features native hardware acceleration for **Intel Core Ultra (NPU)**, **Intel iGPUS/Arc**, and **NVIDIA CUDA**, offloading heavy AI tasks from your CPU for industrial-grade speed.

---

## 🚀 Quick Start (Docker Compose)

Create a `docker-compose.yml`:

```yaml
services:
  whisper-pro-asr:
    image: ventura8/whisper-pro-asr:latest
    container_name: whisper-pro-asr
    ports:
      - "9000:9000"
    restart: unless-stopped
    environment:
      # --- [SSD WRITE PROTECTION] ---
      - WHISPER_TEMP_DIR=/tmp/whisper

    # --- [HARDWARE ACCELERATION] ---
    # The application performs automated detection of both Intel and NVIDIA hardware.
    
    # 1. Intel NPU / iGPU / Arc
    # devices:
    #   - /dev/dri:/dev/dri # Intel Integrated GPU
    #   - /dev/accel/accel0:/dev/accel/accel0 # Meteor/Lunar Lake NPU
    #   - /dev/dxg:/dev/dxg # Windows/WSL2 GPU mapping

    # 2. NVIDIA Silicon (CUDA)
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: 1
    #           capabilities: [gpu]

    tmpfs:
      - /tmp/whisper:size=2G
    volumes:
      # Persistent cache for AI models and pre-compiled hardware binaries
      - ./model_cache:/app/model_cache
      # Persistent storage for task history, telemetry, and system logs
      - ./state:/app/data
      # Recommended: Map your media volumes to enable instant (0-copy) local processing
      # The service will prioritize reading these files directly over network uploads.
      - /path/to/my/media:/media
      - /mnt/nas/tv:/tv
      - /mnt/nas/movies:/movies
```

Deploy with: `docker compose up -d`

> [!TIP]
> **Autonomous Hardware Detection**: The engine automatically identifies your hardware (NVIDIA GPU, Intel NPU, or Intel iGPU) and self-optimizes.

---

## 📝 v1.0.6 Changelog Summary
### Preemption Deadlock Prevention
This update resolves scheduling bugs in high-load setups and introduces exhaustive verification for heterogeneous accelerator pools.

- **Preemption Deadlock & Livelock Prevention**: Extended priority locking across the entire context lifetime of `early_task_registration` to serialize concurrent priority requests (like Language Detection), preventing double-preemption and deadlocking standard tasks.
- **Standard Task Yielding**: Standard tasks yield resource acquisition and loop-sleep instead of blocking on the model lock semaphore whenever priority tasks are present in the registry, preventing priority starvation.
- **Priority Preemption Bypass**: Running priority tasks ignore preemption requests, preventing them from pausing themselves if multiple priority tasks are queued.
- **Graceful Host CPU Fallback**: Automatically registers a default `"Host CPU"` slot when 0 hardware units are detected on startup, keeping the application fully functional on CPU-only hosts.
- **Exhaustive Concurrency Verification**: Added full unit test suites for 0, 1, 2, and 3 hardware unit resource configurations, including tests for yielding and preemption bypass.
- **Linter & PEP8 compliance**: All files maintain a perfect `10.00/10` pylint rating.
- **Strict Coverage & Testing**: Achieved >90% code coverage across all files (overall project coverage at 94.79%) with 312/312 tests passing.
- **Queued Priority & Preemption Observability**: Registered priority tasks in the registry immediately upon arrival, exposing them on the telemetry dashboard. Preempted standard tasks now transition to `"queued"` status with a `"Paused for Priority Task"` stage, ensuring they display in the dashboard queue.

---

## 📺 Bazarr Configuration

To use this service with **Bazarr**:

1.  **Provider**: Choose **Whisper** (or `whisper-asr-webservice`).
2.  **Endpoint**: `http://<WHISPER-PRO-ASR_DOCKER_IP/whisper-pro-asr>:9000`
3.  **Timeouts**: Should be set very high (54000) for long movies
4.  **Pass video filename to Whisper**: Should be enabled for volume mapping to work correctly
3.  **Volume Mapping (Highly Recommended)**:
    - Ensure your Bazarr and Whisper-Pro-ASR containers share the same media paths (e.g., both map `/tv` to the same actual folder).
    - When configured this way, Bazarr sends the *file path* to Whisper. Whisper Pro checks if it can read that path locally. If yes, it processes the file instantly without network overhead.
    - If paths don't match, Whisper Pro automatically falls back to handling the full file upload from Bazarr.

---

## 🌟 Key Features

- **16kHz WAV Standardization**: High-performance audio normalization layer for consistent cross-format results.
- **Global VAD & In-Memory Batch ID**: Optimized language identification using a single VAD pass and zero-I/O NumPy slicing.

### 🧩 Hardware Compatibility Matrix
| Pipeline Stage | CPU (Generic) | NVIDIA (CUDA) | Intel iGPU / Arc | Intel NPU |
| :--- | :---: | :---: | :---: | :---: |
| **Media Standardization** | ✅ | ✅ | ✅ | ✅ |
| **Vocal Isolation (UVR)** | ✅ | ✅ | ✅ (OpenVINO) | ✅ (OpenVINO) |
| **VAD Verification** | ✅ | ✅ | ✅ | ✅ |
| **Whisper ASR Inference** | ✅ | ✅ | ⚠️ (CPU Fallback) | ⚠️ (CPU Fallback) |

- **Re-entrant Hardware Orchestration**: Intelligent thread-local locking for nested AI sub-tasks.
- **Hybrid "Split" Architecture**: Efficiently distribute workloads across multiple accelerators (e.g., Intel NPU for isolation and NVIDIA for transcription).
- **Priority-First Engine**: Instant sub-second pre-emption for high-priority tasks (like language detection).
- **Bazarr Ready**: Direct compatibility with the full media automation stack via standard formats (SRT, VTT, JSON).
- **Industrial Telemetry**: Real-time speed multipliers, ETA calculation, and detailed hardware state reporting.
- **Interactive Documentation**: Full OpenAPI/Swagger interface available at `/docs`.

---

## ⚙️ Configuration

| Variable | Default | Description |
| :--- | :--- | :--- |
| **ASR_MODEL** | `Systran/faster-whisper-large-v3` | Model ID (HuggingFace) or local path |
| **ASR_DEVICE** | `AUTO` | Device: `AUTO`, `CUDA`, or `CPU` |
| **ASR_PREPROCESS_DEVICE** | `AUTO` | Device for Isolation: `AUTO`, `NPU`, `GPU`, `CUDA`, or `CPU` |
| **ENABLE_VOCAL_SEPARATION** | `true` | Pre-clean audio with UVR/MDX-NET engine |
| **OV_CACHE_DIR** | `./model_cache` | OpenVINO kernel cache directory (highly recommended) |
| **ASR_BEAM_SIZE** | `5` | Decoding beam width (Search depth) |
| **ASR_PARALLEL_LIMIT_ACCEL** | `1` | Max concurrent tasks on GPU/NPU |
| **DEBUG** | `false` | Enable verbose debug logging |

---

## 📦 Persistence
Mapping the following volumes is **strongly recommended**:

1.  **`/app/model_cache`**: Stores downloaded AI models and pre-compiled OpenVINO NPU/GPU blobs. Reduces startup time from minutes to milliseconds.
2.  **`/app/data`**: Stores the persistent state of the application, including task history, telemetry statistics, and system logs. Mapping this ensures your history survives container restarts and updates.

## 🐳 GPU/NPU Support

### NVIDIA GPU (CUDA)
- Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- Ensure you have current NVIDIA drivers on the host.

### Intel NPU/GPU
- Mapping `/dev/dri` and `/dev/accel` is recommended for native Linux access.
- For Windows/WSL2, ensure `/dev/dxg` is mapped.

---

**Maintained by**: [ventura8](https://github.com/ventura8)  
**Full Documentation**: [GitHub Repository](https://github.com/ventura8/Whisper-Pro-ASR)