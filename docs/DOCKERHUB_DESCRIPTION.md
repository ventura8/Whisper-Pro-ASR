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

## 📝 v1.0.4 Changelog Summary
### Strategic Sampling & Parallel Processing
This major update introduces high-concurrency signal processing and refined language identification logic to eliminate common hallucinations and reduce latency.

- **Parallel Preprocessing Pipeline**: Decouples extraction and vocal isolation from the inference cycle. Detection zones are now prepared concurrently, reducing identification wait-time by up to 60%.
- **Entropy-Adaptive Smart Search**: Automatically identifies speech regions in sparse audio (e.g., long musical intros) to ensure detection is based on valid signal energy.
- **Confusion-Matrix Tie Breaker**: Resolves linguistic ambiguities between similar pairs (e.g., Norwegian vs. Nynorsk) with a weighted bias, eliminating "nn" hallucinations.
- **Fail-Safe Dual-Path VAD**: Beyond silence detection, performs confidence-auditing on both isolated and raw audio to select the optimal processing path automatically.
- **Hardware-Aware Concurrency**: Integrated strict parallel limits (`ASR_PARALLEL_LIMIT_ACCEL`) to ensure specialized accelerators remain stable under concurrent loads.

## 📝 v1.0.3 Changelog Summary
### RAM Optimization & Storage Cleanup
This update focuses on significantly reducing the memory footprint of Whisper Pro ASR and finalizing the storage cleanup strategy for users migrating from version 1.0.0.

### 🧠 RAM Usage Optimization
- **Streamed Ingestion**: Completely refactored the file upload pipeline. Media files are now streamed directly from the network to disk, eliminating large RAM spikes previously caused by in-memory buffering.
- **Quantized Inference Defaults**: Standardized on `int8` quantization for CPU and NPU backends, halving the memory required for model weights while maintaining high transcription accuracy.
- **Proactive Memory Recovery**: Integrated explicit garbage collection and hardware cache clearing (CUDA/NPU) into the transcription lifecycle. Memory is now returned to the OS immediately after processing.
- **Dynamic UVR Offloading**: The heavy vocal isolation engine (UVR/MDX-NET) is now automatically offloaded from memory when idle, freeing up several hundred megabytes of RAM/VRAM for other system tasks.
- **ONNX Runtime Tuning**: Applied memory-reuse and pattern-matching optimizations to the preprocessing sessions to reduce peak allocation overhead.

### 🧹 Storage & SSD Endurance
- **Extra-Large File Optimization**: Implemented a dynamic persistent storage fallback (SSD/HDD) for uploads that exceed `tmpfs` capacity. The system now automatically migrates large media ingestion to disk, ensuring stability on low-RAM devices during massive file transfers.
- **Legacy Cleanup**: From Version 1.0.2 now automatically detects and purges orphaned 'preprocessing' directories and persistent temp artifacts in the persistent `./model_cache` volume.
- **Refined Tmpfs Usage**: Optimized the interaction between streamed uploads and `tmpfs` storage to ensure maximum SSD protection without risking system stability.


### 🎯 Dynamic Language Detection
- **Automated Chunk Scaling**: The identification engine now calculates an optimal sample size based on the total media duration.
- **Minimum 5-Minute Sample**: Ensures that even short clips have enough context for high-confidence identification.
- **Movie Optimization**: For a standard 4-hour movie, the system now extracts a 12-minute representative sample (5% of duration), significantly improving accuracy over 30-second snippets.
- **Single-Pass Efficiency**: Replaced the iterative 15-scan voting system with a single, high-fidelity inference pass. This reduces FFmpeg overhead and simplifies the processing pipeline.

### ⚡ Performance & Stability
- **Reduced Latency**: By eliminating multiple parallel extraction tasks and redundant inference cycles, `/detect-language` responses are now significantly faster.
- **Lower Memory Pressure**: The single-pass approach ensures more predictable VRAM/RAM utilization during the identification phase.
- **FFmpeg 8.1.0 Refinement**: Optimized the extraction command to use direct seek and duration flags for near-instantaneous chunk creation from any media container.

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

- **Industrial Acceleration**: Native **OpenVINO GenAI** and **Faster-Whisper** support for maximum hardware utilization.
- **Hardware Compatibility Matrix**:
    | Component | CPU | NVIDIA (CUDA) | Intel iGPU / Arc | Intel NPU |
    | :--- | :---: | :---: | :---: | :---: |
    | **Vocal Separation** | ✅ | ✅ | ✅ | ✅ |
    | **Whisper ASR** | ✅ | ✅ | ⚠️ (CPU Fallback) | ⚠️ (CPU Fallback) |
- **Parallel Preprocessing Pipeline**: Decouples extraction/isolation from inference, preparing all 5 zones simultaneously to reduce latency by up to 60%.
- **Entropy-Adaptive Smart Search**: Automatically identifies speech regions in sparse audio to ensure high-accuracy identification.
- **Fail-Safe Dual-Path VAD**: Beyond silence detection, performs confidence-auditing on both isolated and raw audio to select the optimal processing path.
- **Confusion-Matrix Resolution**: Resolves linguistic ambiguities between similar languages (e.g., NO/NN) with weighted tie-breaking logic.
- **Hybrid "Split" Architecture**: Efficiently distribute workloads across multiple accelerators (e.g., Intel NPU for isolation and NVIDIA for transcription).
- **Strategic Uniform Sampling**: Advanced identification logic using spatial audio zone voting for high-precision detection.
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

Mapping `./model_cache` to `/app/model_cache` is **strongly recommended**. This persistent volume stores:
1. **Downloaded Models**: Prevents re-downloading on container restarts.
2. **OpenVINO Kernels**: Stores pre-compiled NPU/GPU blobs, reducing subsequent startup times from minutes to milliseconds.

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