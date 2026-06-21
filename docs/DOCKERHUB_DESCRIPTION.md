![GitHub Release](https://img.shields.io/github/v/release/ventura8/Whisper-Pro-ASR)
![Docker Pulls](https://img.shields.io/docker/pulls/ventura8/whisper-pro-asr)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/ventura8/Whisper-Pro-ASR/ci.yml)
![GitHub License](https://img.shields.io/github/license/ventura8/Whisper-Pro-ASR)

# Whisper Pro ASR (Multilingual)

**Whisper Pro ASR** is a high-performance, production-ready AI transcription service with **speaker diarization**. It is optimized for **Whisper Large V3** and designed for seamless integration with **Bazarr** and the *arr stack. 

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
      # --- [SPEAKER DIARIZATION] ---
      # Required for speaker identification. Get a token at https://huggingface.co/settings/tokens
      # - HF_TOKEN=hf_your_token_here

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
      # Persistent cache for AI models, diarization models, and pre-compiled hardware binaries
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

- **🗣 Speaker Diarization**: Identify who said what using WhisperX alignment and PyAnnote speaker segmentation. Output formats (SRT, VTT, TXT) include speaker labels (e.g., `[SPEAKER_00]: Hello world`).
- **Intel ASR Chunking & Streaming**: Refactored OpenVINO engine transcription to split long media files dynamically into structured chunks guided by speech VAD timestamps, ensuring stability on very long movies.
- **O(1) Live Subtitle Updates**: Appends pre-formatted subtitle blocks incrementally to the live SRT stream during processing instead of doing full $O(N^2)$ stream reconstructions.
- **UVR Chunk Progress Tracking**: Computes and emits real-time preprocessing progress updates per UVR chunk to keep the dashboard progress bar fluid during vocal separation.
- **Graceful Temp-Storage Fallback**: Establishes a 2GB minimum free space threshold and 1.5x file-size headroom multiplier to fallback gracefully to persistent storage when tmpfs runs low on space, preventing ENOSPC crashes.
- **16kHz WAV Standardization**: High-performance audio normalization layer for consistent cross-format results.
- **Global VAD & In-Memory Batch ID**: Optimized language identification using a single VAD pass and zero-I/O NumPy slicing.
- **Customizable ASR Parameters**: Fine-tune transcription with `initial_prompt`, `vad_filter`, and `word_timestamps`.
- **Subtitle Layout Control**: Custom character-per-line wrapping (`max_line_width`) and max line limits (`max_line_count`) for SRT/VTT output.
- **Smart Model Lifecycle**: Configurable idle timeout (`MODEL_IDLE_TIMEOUT`) keeps models warm in memory for rapid response to bursty workloads. A deferred cleanup timer starts after the last task completes and is cancelled when new tasks arrive.
- **Service Analytics Dashboard**: Dedicated `/analytics` page with interactive charts showing cumulative and daily breakdown of task counts and durations.
- **Runtime Configuration**: Dynamic `/settings` endpoint allows model, device, and retention changes without container restart.
- **Telemetry Downsampling**: Dual-layer downsampling (server + client) caps chart data at 300 points for smooth dashboard rendering during extended operation.

### 🧩 Hardware Compatibility Matrix
| Pipeline Stage | CPU (Generic) | NVIDIA (CUDA) | Intel iGPU / Arc | Intel NPU |
| :--- | :---: | :---: | :---: | :---: |
| **Media Standardization** | ✅ | ✅ | ✅ | ✅ |
| **Vocal Isolation (UVR)** | ✅ | ✅ | ✅ (OpenVINO) | ✅ (OpenVINO) |
| **VAD Verification** | ✅ | ✅ | ✅ | ✅ |
| **Whisper ASR Inference** | ✅ | ✅ | ⚠️ (CPU Fallback) | ⚠️ (CPU Fallback) |
| **Speaker Diarization** | ✅ | ✅ | ✅ | ✅ |

- **Re-entrant Hardware Orchestration**: Intelligent thread-local locking for nested AI sub-tasks.
- **Hybrid "Split" Architecture**: Efficiently distribute workloads across multiple accelerators (e.g., Intel NPU for isolation and NVIDIA for transcription).
- **Priority-First Engine**: Instant sub-second pre-emption for high-priority tasks (like language detection).
- **Bazarr Ready**: Direct compatibility with the full media automation stack via standard formats (SRT, VTT, JSON).
- **Industrial Telemetry**: Real-time speed multipliers, ETA calculation, and detailed hardware state reporting.
- **Interactive Documentation**: Full OpenAPI/Swagger interface available at `/docs`.
- **Live SRT Streaming**: Real-time auto-scrolling subtitle display during processing for immediate visual feedback.

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
| **HF_TOKEN** | *(empty)* | Hugging Face token for speaker diarization (PyAnnote models) |
| **MODEL_IDLE_TIMEOUT** | `300` | Seconds to keep models loaded after last task (0 = immediate offload) |
| **INTEL_ASR_CHUNK_DURATION** | `300` | Chunk duration in seconds for Intel Whisper transcription |
| **INITIAL_PROMPT** | *(multilingual)* | Default context prompt for guiding transcription |
| **AGGRESSIVE_OFFLOAD** | `false` | Immediately unload models when idle (overridden by `MODEL_IDLE_TIMEOUT`) |
| **UVR_CHUNK_DURATION** | `600` | Chunk duration in seconds for UVR separation (0 to disable) |
| **WHISPER_TEMP_MIN_FREE_MB** | `2048` | Fallback threshold to disk if RAM-disk is full |
| **DEBUG** | `false` | Enable verbose debug logging |

---

## 📦 Persistence
Mapping the following volumes is **strongly recommended**:

1.  **`/app/model_cache`**: Stores downloaded AI models, WhisperX alignment models, PyAnnote diarization models, and pre-compiled OpenVINO NPU/GPU blobs. Reduces startup time from minutes to milliseconds.
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