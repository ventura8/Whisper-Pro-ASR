# Whisper Pro ASR

![Main Language](https://img.shields.io/github/languages/top/ventura8/Whisper-Pro-ASR)
![Coverage](assets/coverage.svg)
![Pylint](https://img.shields.io/badge/Pylint-10.0%2F10-brightgreen)

**Whisper Pro ASR** is a high-performance transcription microservice optimized for the **Whisper Large V3** model. It delivers enterprise-grade performance with native hardware acceleration for **Intel Core Ultra NPUs**, **Integrated GPUs**, and **NVIDIA CUDA** environments.

Engineered for seamless integration with **Bazarr** and the broader media automation stack, it offloads computationally intensive AI tasks from your primary system resources, providing industrial-strength transcription with rapid hardware context switching.

---

## ⚡ Quick Start

Deploy instantly using standard `docker-compose.yml`:

```yaml
services:
  whisper-pro-asr:
    image: ventura8/whisper-pro-asr:latest
    container_name: whisper-pro-asr
    ports:
      - "9000:9000"
    restart: unless-stopped

    # 1. Intel Silicon (NPU/GPU)
    # devices:
    #   - /dev/dri:/dev/dri # Intel iGPU / Arc
    #   - /dev/accel/accel0:/dev/accel/accel0 # Intel NPU
    #   - /dev/dxg:/dev/dxg # Windows/WSL2 GPU mapping

    # 2. NVIDIA Silicon (CUDA)
    # Note: Requires NVIDIA Container Toolkit on the HOST for driver passthrough.
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: 1
    #           capabilities: [ gpu ]
    
    environment:
      # --- [SSD WRITE PROTECTION] ---
      - WHISPER_TEMP_DIR=/tmp/whisper

    tmpfs:
      - /tmp/whisper:size=2G

    volumes:
      # Persistent cache for AI models and pre-compiled hardware binaries (NPU)
      - ./model_cache:/app/model_cache
      # Persistent storage for task history, telemetry, and system logs
      - ./state:/app/data
      # Recommended: Map your media volumes to enable instant (0-copy) local processing
      # The service will prioritize reading these files directly over network uploads.
      - /path/to/my/media:/media
      - /mnt/nas/tv:/tv
      - /mnt/nas/movies:/movies
```

1. Save the configuration.
2. Launch: `docker compose up -d`

> [!TIP]
> **Autonomous Hardware Resolution**: The engine automatically detects and adapts to your specific hardware (NVIDIA, Intel NPU, or Integrated GPU), optimizing the processing pipeline without requiring manual intervention.

## 🚀 Key Features

### Precision Architecture
- **Multi-Backend Support**: Specialized optimization profiles for **NVIDIA CUDA**, **Intel OpenVINO**, and **Generic CPU** runtimes.
- **Re-entrant Hardware Orchestration**: Utilizes a sophisticated thread-local locking system (`model_lock_ctx` in `scheduler.py`) that allows complex pipelines (UVR -> ASR) to share a single hardware claim without deadlocking.
- **FFmpeg 8.1.0 Integration**: Features optimized hardware-accelerated decoding. All media (MKV, AVI, MP4, etc.) is automatically standardized to **16kHz Mono WAV** using the `utils.py` core before entering the AI pipeline for maximum accuracy.

### Advanced Intelligence
- **Zero-Latency Pre-emption**: High-priority operations (such as language detection) instantly pause long-running transcription batches, ensuring immediate API responsiveness.
- **Consolidated Batch Montage**: Consolidates multiple sampling targets into a single high-density montage. This allows for a **single-pass UVR isolation** across multiple non-contiguous segments, eliminating repeated model loading overhead.
- **Global VAD & In-Memory Slicing**: Features a unified Voice Activity Detection scan across the entire montage. segments are then sliced as **NumPy arrays in memory**, eliminating temporary file I/O and reducing VAD overhead by up to 900%.
- **Deferred Persistence Engine**: Protects SSD longevity by buffering task history and telemetry in RAM, only syncing to physical storage after 10 tasks or 1 hour of activity.
- **Fail-Safe Dual-Path VAD**: Intelligent logic that verifies speech presence on both isolated and raw audio, selecting the optimal path automatically based on signal clarity.
- **Confusion-Matrix Tie Breaking**: Resolves linguistic ambiguities between similar pairs (e.g., NO/NN) with a weighted bias, eliminating common identification hallucinations.
- **Unified Session Orchestration**: Integrated task and queue tracking ensures that hardware resources are only reclaimed when the system is fully idle (zero active or waiting tasks).
- **Proactive Resource Reclamation**: Automatically offloads heavy models and clears hardware caches (CUDA/NPU) only when the queue is empty.
- **Weighted Multi-Segment Voting**: Aggregates probabilities from multiple zones with confidence-weighted averaging for industrial-strength accuracy.
- **Advanced Memory Hygiene**: Implements a "Nuclear Purge" strategy using `malloc_trim` and ctranslate2 cache clearing to ensure idle memory remains below 500MB even after heavy ASR sessions.
- **Centralized Storage Hygiene**: Features a thread-local tracking system that registers every transient asset (uploads, HQ prep files, isolated stems) created during a request. The system ensures a **100% cleanup rate** by purging all tracked files immediately upon request completion or failure.
- **On-Demand History Tiering**: Implements a dual-tier storage strategy. The dashboard and RAM are strictly capped at the last 20 tasks, while a durable history of up to 1000 tasks is maintained on the persistent volume.
- **Hardened Diagnostic Logging**: System logs (`whisper_pro.log`) are redirected to the persistent state volume with real-time flush-to-disk logic. Log downloads are optimized with zero-caching headers to ensure the latest diagnostic data is always available.

### Production Ready
- **OpenAI Standard API**: Drop-in compatible with the OpenAI whisper specification, allowing immediate integration with existing clients.
- **Interactive Documentation**: Full OpenAPI/Swagger interface available at `/docs` for testing and endpoint exploration.
- **Live SRT Streaming**: Features a real-time, auto-scrolling SubRip (SRT) display during processing, providing immediate visual feedback identical to the final output.
- **Persistent History Dashboard**: Maintains a durable log of all ASR and Language Detection tasks. Completed transcriptions are stored indefinitely and can be downloaded as `.srt` files directly from the dashboard.
- **Industrial Telemetry**: Real-time progress monitoring, including completion percentages (%), segment counts (`Seg 11 | 01:20 / 05:00`), active processing stages (e.g., UVR Preprocessing, Transcribing), and detailed hardware state reporting.
- **Granular Performance Auditing**: Every task provides a detailed breakdown of its execution phases, including exact time spent in **Queue**, **Vocal Isolation**, and **AI Inference**.
- **Material Design Dashboard**: A comprehensive monitoring interface at `/dashboard` (or the root `/` when accessed via browser) featuring live task progress bars, system resource visualization, real-time auto-scrolling logs, and a **Live Refresh** toggle for manual inspection.
- **Bazarr Optimized**: Purpose-built for high-volume subtitle automation with stable SRT, VTT, and verbose JSON output formats. Fully compatible with `whisper-asr-webservice` API.

---


### 🧩 Hardware Compatibility Matrix
| Pipeline Stage | CPU (Generic) | NVIDIA (CUDA) | Intel iGPU / Arc | Intel NPU |
| :--- | :---: | :---: | :---: | :---: |
| **Media Standardization** | ✅ | ✅ | ✅ | ✅ |
| **Vocal Isolation (UVR)** | ✅ | ✅ | ✅ (OpenVINO) | ✅ (OpenVINO) |
| **VAD Verification** | ✅ | ✅ | ✅ | ✅ |
| **Whisper ASR Inference** | ✅ | ✅ | ⚠️ (CPU Fallback) | ⚠️ (CPU Fallback) |

### System Architecture
The service utilizes a **Heterogeneous Model Pool** to orchestrate tasks across NVIDIA GPUs, Intel NPUs, and CPUs simultaneously. For a deep dive into the processing pipelines, resource locking, and hardware acceleration logic, see the [Technical Architecture](docs/ARCHITECTURE.md) documentation.

> [!TIP]
> View the [Concurrency & Resource Orchestration](docs/CONCURRENCY.md) guide for details on parallel preprocessing and pre-emption.

---

## Prerequisites
- **Silicon**: Any CPU or Intel GPU/NPU or NVIDIA Pascal+ GPU.
- **Environment**: Docker Engine 20.10+ / Docker Desktop.
- **NPU Requirements**: Latest Intel NPU driver package (NPU Plugin).

## Configuration Reference

The service is highly tunable via environment variables in `docker-compose.yml`.

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| **Runtime Control** | | |
| `ASR_DEVICE` | `AUTO` | Inference target: `AUTO`, `CUDA`, or `CPU`. |
| `ASR_PREPROCESS_DEVICE` | `AUTO` | Inference target: `AUTO`, `NPU`, `GPU`, or `CPU`. |
| `ASR_MODEL` | `Systran/faster-whisper-large-v3` | Model ID (HuggingFace) or local path. |
| `VOCAL_SEPARATION_MODEL` | `UVR-MDX-NET-Voc_FT` | Model ID (HuggingFace) or local path |
| `ASR_BATCH_SIZE` | `1` | Number of segments processed per pass. |
| `ASR_BEAM_SIZE` | `5` | Decoding beam width (Search depth). |
| `DEBUG` | `false` | Enables verbose stack traces and debug logging. |
| **Optimization** | | |
| `OV_PERFORMANCE_HINT` | `LATENCY` | OpenVINO scheduling hint (Latency/Throughput). |
| `OV_CACHE_DIR` | `./model_cache` | Persistent directory for compiled hardware blobs. |
| **Parallelism** | | |
| `ASR_THREADS` | `4` | CPU core allocation for inference (Auto-capped by hardware). |
| `ASR_PREPROCESS_THREADS` | `4` | CPU core allocation for UVR/ONNX (Auto-capped by hardware). |
| **SSD Protection** | | |
| `WHISPER_TEMP_DIR`| `/tmp/whisper`| Redirects transient I/O (uploads, WAVs, stems) to this path. |
| `WHISPER_TEMP_MIN_FREE_MB` | `512` | Fallback threshold to disk if RAM-disk is full. |
| **Preprocessing** | | |
| `ENABLE_VOCAL_SEPARATION`| `true` | Toggles UVR background removal engine for translate/transcribe. |
| `ENABLE_LD_PREPROCESSING`| `true` | Toggles UVR background removal engine for language detection. |
| `LD_VAD_THRESHOLD` | `0.3` | Aggressiveness of VAD during language identification (0.0 to 1.0). |
| `SMART_SAMPLING_SEARCH` | `true` | Enables localized entropy-based signal searching in sparse audio. |
| `MAX_CUDA_UNITS` | `1` | Max NVIDIA GPUs to utilize (supports `ALL`, `AUTO`). |
| `MAX_GPU_UNITS` | `1` | Max Intel GPUs to utilize (supports `ALL`, `AUTO`). |
| `MAX_NPU_UNITS` | `1` | Max Intel NPUs to utilize (supports `ALL`, `AUTO`). |
| `MAX_CPU_UNITS` | `1` | Max concurrent multi-core CPU tasks (VAD, FFmpeg, CPU-ASR). |
| `FFMPEG_HWACCEL` | `none` | FFmpeg hardware acceleration target (`cuda`, `vaapi`, `qsv`). |
| `FFMPEG_FILTER` | `dynaudnorm` | Normalization filter: `dynaudnorm` (Standard) or `loudnorm` (Broadcast). |

---

## 📜 Full `docker-compose.yml` Example
 
For an exhaustive deployment featuring all optimization toggles and hardware passthrough options:
 
```yaml
services:
  whisper-pro-asr:
    image: ventura8/whisper-pro-asr:latest
    container_name: whisper-pro-asr
    restart: unless-stopped
    ports:
      - "9000:9000"
 
    # --- [HARDWARE ACCELERATION] ---
    # The application performs automated detection of both Intel and NVIDIA hardware.
    # To enable hardware passthrough, uncomment the appropriate sections below.
 
    # 1. Intel Silicon (iGPU / NPU) - Used for Preprocessing
    # devices:
    #   - /dev/dri:/dev/dri # Integrated GPU
    #   - /dev/accel/accel0:/dev/accel/accel0 # Meteor/Lunar Lake NPU
    #   - /dev/dxg:/dev/dxg # Windows/WSL2 GPU mapping
 
    # 2. NVIDIA Silicon (CUDA)
    # Note: Requires NVIDIA Container Toolkit on the HOST for driver passthrough.
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: 1
    #           capabilities: [gpu]
 
    environment:
      - DEBUG=false
 
      # --- [ENGINE CONFIGURATION] ---
      # Hardware Target: AUTO (Automated detection), CUDA (NVIDIA), CPU
      - ASR_DEVICE=AUTO
      # Computation Precision: AUTO, int8, float16 (default: AUTO)
      - ASR_COMPUTE_TYPE=AUTO
      # Model Weight Source (Faster-Whisper ID or local path)
      - ASR_MODEL=Systran/faster-whisper-large-v3
 
      # --- [INFERENCE PARAMETERS] ---
      # Generation Search Breadth (Higher = more accurate, lower = faster)
      - ASR_BEAM_SIZE=5
      # Parallel segment batching (1 is recommended for single-GPU/NPU stability)
      - ASR_BATCH_SIZE=1
 
      # --- [PREPROCESSING (UVR / MDX-NET)] ---
      # Target Device: AUTO, CPU, CUDA (NVIDIA), GPU (Intel), NPU (Intel)
      - ASR_PREPROCESS_DEVICE=AUTO
      # Isolation Model Filename
      - VOCAL_SEPARATION_MODEL=UVR-MDX-NET-Inst_HQ_3.onnx
      # Vocal Separation Logic Toggles
      - ENABLE_VOCAL_SEPARATION=true
      - ENABLE_LD_PREPROCESSING=true
      - LD_VAD_THRESHOLD=0.3
      - LD_MIN_CONFIDENCE_THRESHOLD=0.8
      - SMART_SAMPLING_SEARCH=false
 
      # --- [RESOURCE ALLOCATION] ---
      # Core limit for Whisper ASR logic
      - ASR_THREADS=4
      # Core limit for Preprocessing (ONNX Runtime)
      - ASR_PREPROCESS_THREADS=4
      # Core limit for Media Normalization (0 = auto-detect system-wide)
      - FFMPEG_THREADS=4
      # Max number of physical accelerators to use (default: all)
      - ASR_MAX_ACCEL_UNITS=1

      # --- [SSD WRITE PROTECTION] ---
      - WHISPER_TEMP_DIR=/tmp/whisper

    tmpfs:
      - /tmp/whisper:size=2G

    volumes:
      # Persistent cache for AI models and pre-compiled hardware binaries (NPU)
      - ./model_cache:/app/model_cache
      # Recommended: Map your media volumes to enable instant (0-copy) local processing
      # The service will prioritize reading these files directly over network uploads.
      - /path/to/my/media:/media
      - /mnt/nas/tv:/tv
      - /mnt/nas/movies:/movies
```
 
---
 
## API Reference

Comprehensive Swagger documentation is hosted at **`/docs`**.

### 1. Identify Language
**POST** `/detect-language`  
Performs multi-zone analysis to identify source language metadata. Returns full language names (e.g., "English") for Bazarr compatibility.

### 2. Transcribe Media
**POST** `/asr`  
**POST** `/v1/audio/transcriptions`  
Main entry point for generating subtitles.
- **Formats**: `srt` (default), `vtt`, `txt`, `tsv`, `json` (with segments).
- **Optimization**: Prioritizes local file access if the path exists (via volume mapping), otherwise accepts file uploads.

### 3. Service Analytics & Dashboard
**GET** `/status`  
Health-check endpoint returning model metadata, hardware status, and versioning information.

**GET** `/dashboard` (or **GET** `/` via Browser)  
Interactive Material Design interface for real-time monitoring of task progress, hardware utilization, and application memory.

---

## 📺 Bazarr Configuration

To use this service with **Bazarr**:

1.  **Provider**: Choose **Whisper** (or `whisper-asr-webservice`).
2.  **Endpoint**: `http://<YOUR_DOCKER_IP>:9000`
3.  **Timeouts**: Should be set very high (54000) for long movies
4.  **Pass video filename to Whisper**: Should be enabled for volume mapping to work correctly
3.  **Volume Mapping (Highly Recommended)**:
    - Ensure your Bazarr and Whisper-Pro-ASR containers share the same media paths (e.g., both map `/tv` to the same actual folder).
    - When configured this way, Bazarr sends the *file path* to Whisper. Whisper Pro checks if it can read that path locally. If yes, it processes the file instantly without network overhead.
    - If paths don't match, Whisper Pro automatically falls back to handling the full file upload from Bazarr.

---

## Performance Notes
- **Golden Configuration**: We recommend **Large-V3** with **Batch=1** and **Beam=5** for the majority of CPU/GPU workloads.
- **VRAM/RAM Requirements**: Ensure at least **16GB of System RAM** when running both Vocal Isolation and Large-V3.

---

## 🛠 Project Structure
```text
/
├── whisper_pro_asr.py        # Master entry point
├── modules/                 # Service Logic
│   ├── api/                 # API Routes (ASR, Detection, System)
│   ├── inference/           # ML Engine (Model Manager, Scheduler, VAD, UVR)
│   ├── monitoring/          # Dashboard, Telemetry & Metrics
│   ├── config.py            # Global Settings
│   ├── logging_setup.py     # Task-specific Logging
│   └── utils.py             # System & Audio Utilities
├── tests/                   # Performance & Unit Test Suites
├── Dockerfile               # Packaging Definition
└── docker-compose.yml       # Orchestration Template
```
