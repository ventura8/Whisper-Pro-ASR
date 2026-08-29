![GitHub Release](https://img.shields.io/github/v/release/ventura8/Whisper-Pro-ASR)
![Docker Pulls](https://img.shields.io/docker/pulls/ventura8/whisper-pro-asr)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/ventura8/Whisper-Pro-ASR/ci.yml)
![GitHub License](https://img.shields.io/github/license/ventura8/Whisper-Pro-ASR)

# Whisper Pro ASR (Multilingual)

**Whisper Pro ASR** is a high-performance, production-ready AI transcription service with **speaker diarization**. It is optimized for **Whisper Large V3** and designed for seamless integration with **Bazarr** and the *arr stack.

It features native hardware acceleration for **Intel Core Ultra (NPU)**, **Intel iGPUS/Arc**, **NVIDIA CUDA**, and **native Linux AMD ROCm**, offloading heavy AI tasks from your CPU for industrial-grade speed.

Concurrency correctness is the top engineering priority: detect-language preemption is cooperative and unit-aware, critical priority waits remain indefinite under saturation (never failing immediately on simple scheduler timeout), and scheduler liveness is continuously validated by test gates.

---

## 📦 Which Image Should I Pick?

Match the image to the hardware you have. **No image ships model weights** -- they download
on first start into `./model_cache`.

| Your hardware | Use this image |
| :--- | :--- |
| No GPU | **`cpu`** |
| Intel iGPU / Arc / NPU | **`intel`** |
| NVIDIA GPU | **`nvidia`** |
| NVIDIA GPU + you need **speaker diarization** | **`full`** |
| NVIDIA GPU **and** an Intel iGPU in the same box | **`nvidia-intel`** |
| AMD Radeon (native Linux ROCm) | **`amd`** |

Two extra images exist only if you want to run the **openai-whisper** engine *on the GPU*.
They are large, and most people do not need them -- the default engines are already
GPU-accelerated in the images above.

| Special case | Use this image |
| :--- | :--- |
| openai-whisper on an Intel GPU | **`intel-xpu`** |
| openai-whisper on an AMD GPU | **`amd-rocm-torch`** |

### Capability Matrix

Sizes are uncompressed on-disk; Docker Hub reports a smaller compressed number.

| Image | Size | Transcription runs on | Vocal isolation (UVR) runs on | Engines available | Speaker diarization |
| :--- | ---: | :--- | :--- | :--- | :---: |
| **`cpu`** | 4.9 GB | CPU | CPU | Faster-Whisper, OpenAI-Whisper | — |
| **`intel`** | 5.2 GB | **Intel GPU** (OpenVINO); CPU fallback on NPU | **Intel GPU / NPU** (OpenVINO) | + Intel-Whisper | — |
| **`intel-xpu`** | 11.2 GB | **Intel GPU** (OpenVINO); CPU fallback on NPU | **Intel GPU / NPU** (OpenVINO) | + Intel-Whisper<br>OpenAI-Whisper also on **Intel GPU** **Requires Intel Arc (Alchemist) or newer** -- torch's XPU backend does not execute Whisper on older iGPUs (verified: UHD Graphics selects XPU but fails with a Level Zero error even for the `tiny` model). | — |
| **`nvidia`** | 17.5 GB | **NVIDIA GPU** (CUDA) | **NVIDIA GPU** (CUDA) | Faster-Whisper, OpenAI-Whisper | — |
| **`full`** | ~29.8 GB | **NVIDIA GPU** *and* **Intel GPU** (CUDA / OpenVINO); CPU fallback on NPU | either GPU, Intel NPU, or **AMD** (ROCm) | Faster-Whisper, Intel-Whisper, OpenAI-Whisper, WhisperX | ✅ |
| **`nvidia-whisperx`** | ~18.4 GB | **NVIDIA GPU** (CUDA) | **NVIDIA GPU** (CUDA) | + WhisperX | ✅ |
| **`nvidia-intel`** | 17.9 GB | **NVIDIA GPU** *and* **Intel GPU** at the same time | either GPU | Faster-Whisper, Intel-Whisper, OpenAI-Whisper | — |
| **`amd`** | 14.1 GB | CPU *(see note)* | **AMD GPU** (ROCm) | Faster-Whisper, OpenAI-Whisper | — |
| **`amd-rocm-torch`** | ~21.8 GB | CPU, except OpenAI-Whisper on **AMD GPU** | **AMD GPU** (ROCm) | Faster-Whisper, OpenAI-Whisper | — |

**Why AMD transcribes on the CPU:** the default engine is CTranslate2, which has no ROCm
backend at all. On AMD the GPU accelerates vocal isolation, and -- with `amd-rocm-torch` --
the openai-whisper engine. This is a limitation of the upstream engine, not of the image.

**Speaker diarization** needs WhisperX, which ships in **`full`** and in `nvidia-whisperx`.
Prefer `full` unless image size is the binding constraint -- it carries every vendor's ONNX
Runtime, so one tag runs on whatever hardware the host has.

## 🚀 Quick Start (Docker Compose)

Create a `docker-compose.yml`:

```yaml
services:
  whisper-pro-asr:
    # Choose the edition matching your hardware (see the table above):
    image: ventura8/whisper-pro-asr:latest
    container_name: whisper-pro-asr
    ports:
      - "9000:9000"
    restart: unless-stopped
    environment:
      # --- [SSD WRITE PROTECTION] ---
      - WHISPER_TEMP_DIR=/tmp/whisper
      # --- [SPEAKER DIARIZATION (requires a WhisperX image: full or nvidia-whisperx)] ---
      # Required for speaker identification. Get a token at https://huggingface.co/settings/tokens
      # - DIARIZATION_HF_TOKEN=hf_your_token_here

    # --- [HARDWARE ACCELERATION] ---
    # The application performs automated detection of both Intel and NVIDIA hardware.
    
    # 1. Intel NPU / iGPU / Arc
    # Linux Intel hosts — set HOST_INTEL_RENDER_GID explicitly (no silent default):
    #   echo "HOST_INTEL_RENDER_GID=$(stat -c '%g' /dev/dri/renderD128)" >> .env
    #   # or: echo "HOST_INTEL_RENDER_GID=$(stat -c '%g' /dev/dri/renderD* | head -n 1)" >> .env
    # group_add:
    #   - "${HOST_INTEL_RENDER_GID:?set HOST_INTEL_RENDER_GID from host render GID}"
    # devices:
    #   - /dev/dri:/dev/dri # Intel Integrated GPU / Arc (all render nodes)
    #   - /dev/accel:/dev/accel # Intel NPU (all accel nodes)
    # Windows 11 / WSL2 Intel hosts:
    # devices:
    #   - /dev/dxg:/dev/dxg # WSL GPU bridge
    #   - /dev/dri:/dev/dri # Optional if WSL exposes DRM render nodes
    #   - /dev/accel:/dev/accel # Optional if WSL exposes Intel NPU accel nodes
    # Optional WSL2 telemetry access (only if intel_gpu_top/npu_busy_time PMU/sysfs
    # access is blocked by container isolation): see the full docker-compose.yml
    # snippet and "Intel telemetry container-access note" in docs/SETUP.md.

    # 2. NVIDIA Silicon (CUDA)
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: 1
    #           capabilities: [gpu]

    # 3. AMD GPU (native Linux ROCm via ONNX Runtime)
    # Linux AMD hosts:
    # devices:
    #   - /dev/kfd:/dev/kfd # AMD KFD (ROCm GPU driver)
    #   - /dev/dri:/dev/dri # DRM render nodes
    # Windows 11 / WSL2 AMD hosts:
    # devices:
    #   - /dev/dxg:/dev/dxg # WSL GPU bridge (detection only in this Linux container; UVR falls back to CPU)
    # volumes:
    #   - /usr/lib/wsl:/usr/lib/wsl:ro # Optional: WSL2 host driver library mount

    tmpfs:
      - /tmp/whisper:size=2G,mode=1777
    volumes:
      # Persistent cache for AI models, diarization models, and pre-compiled hardware binaries
      - ./model_cache:/app/model_cache
      # Persistent storage for task history, telemetry, and system logs
      - ./data:/app/data
      # Recommended: Map your media volumes to enable instant (0-copy) local processing
      # The service will prioritize reading these files directly over network uploads.
      - /path/to/my/media:/media
      - /mnt/nas/tv:/tv
      - /mnt/nas/movies:/movies
```

Deploy with: `docker compose up -d`

> [!TIP]
> **Autonomous Hardware Detection**: The engine automatically identifies your hardware (NVIDIA GPU, AMD GPU, Intel NPU, or Intel iGPU) and self-optimizes.

---

## 📺 Bazarr Configuration

To use this service with **Bazarr**:

1. **Provider**: Choose **Whisper** (or `whisper-asr-webservice`).
2. **Endpoint**: `http://<IP_OR_HOSTNAME>:9000`
3. **Timeouts**: Should be set very high (54000) for long movies
4. **Pass video filename to Whisper**: Should be enabled for volume mapping to work correctly
5. **Volume Mapping (Highly Recommended)**:
    - Ensure your Bazarr and Whisper-Pro-ASR containers share the same media paths (e.g., both map `/tv` to the same actual folder).
    - When configured this way, Bazarr sends the *file path* to Whisper. Whisper Pro checks if it can read that path locally. If yes, it processes the file instantly without network overhead.
    - If paths don't match, Whisper Pro automatically falls back to handling the full file upload from Bazarr.

---

## 🌟 Key Features

- **🗣 Speaker Diarization**: Identify who said what using WhisperX alignment and PyAnnote speaker segmentation. Output formats (SRT, VTT, TXT) include speaker labels (e.g., `[SPEAKER_00]: Hello world`).
- **Intel ASR Chunking & Streaming**: Refactored OpenVINO engine transcription to split long media files dynamically into structured chunks guided by speech VAD timestamps, ensuring stability on very long movies.
- **O(1) Live Subtitle Updates**: Appends pre-formatted subtitle blocks incrementally to the live SRT stream during processing instead of doing full $O(N^2)$ stream reconstructions.
- **UVR Chunk Progress Tracking**: Computes and emits real-time preprocessing progress updates per UVR chunk to keep the dashboard progress bar fluid during vocal separation.
- **Graceful Temp-Storage Fallback**: Establishes a 2GB minimum free space threshold and 1.5x file-size headroom multiplier; both tmpfs and persistent fallback storage are validated so insufficient capacity fails early instead of causing an ENOSPC crash.
- **16kHz WAV Standardization**: High-performance audio normalization layer for consistent cross-format results.
- **Global VAD & In-Memory Batch ID**: Optimized language identification using a single VAD pass and zero-I/O NumPy slicing.
- **Customizable ASR Parameters**: Fine-tune transcription with `initial_prompt`, `vad_filter`, and `word_timestamps`.
- **Subtitle Layout Control**: Custom character-per-line wrapping (`max_line_width`) and max line limits (`max_line_count`) for SRT/VTT output.
- **Plex-Compatible AI Subtitle Tagging**: Subtitle files are named `<source>.<language>-ai.<format>` (e.g. `movie.en-ai.srt`). Plex maps the `-ai` suffix (ISO 3166-1 code for Anguilla) to display tracks as `<Language> (AI)` — for every language, for both transcription and translation, without falling back to `xx (Unknown)`.
- **Subtitle Word Highlighting**: `subtitle_highlight_words=true` highlights the active spoken word in SRT/VTT output with karaoke-style per-word timing.
- **Smart Model Lifecycle**: Configurable idle timeout (`MODEL_IDLE_TIMEOUT`) keeps models warm in memory for rapid response to bursty workloads. A deferred cleanup timer starts after the last task completes and is cancelled when new tasks arrive.
- **Service Analytics Dashboard**: Dedicated `/analytics` page with interactive charts showing cumulative and daily breakdown of task counts and durations, categorized by endpoint (/asr, /detect-language, /v1/audio/...).
- **Runtime Configuration**: Dynamic `/settings` endpoint allows model, device, and retention changes without container restart.
- **Telemetry Downsampling**: Dual-layer downsampling (server + client) caps chart data at 300 points for smooth dashboard rendering during extended operation.
- **Strict Lint Baseline**: CI/local parity enforces Ruff + Flake8 + Pylint on Python sources, with Flake8 configured at 140 chars and no ignore directives.
- **Nesting-Safe Hardware Orchestration**: Dedicated non-locking entry points let nested AI sub-tasks share a single hardware claim without re-locking.
- **Per-Task Accelerator Assignment**: Each incoming task is assigned a concrete hardware unit from `STATE.hw_pool` (FIFO), so different tasks can land on different accelerators (e.g., Intel NPU or NVIDIA GPU) and run independently.
- **Priority-First Engine**: Cooperative, unit-aware preemption lets high-priority tasks (like language detection) claim a saturated hardware unit as soon as the running task reaches its next yield checkpoint.
- **FIFO with Priority Yielding**: Requests are processed by `start_time` order (tie-break by registration order at acquisition time) within their own priority class while language detection still preempts ASR under saturation.
- **Stable Task Timeline**: Dashboard task cards (active + history) are displayed in deterministic `start_time` + `task_id` order for operations monitoring; this tie-break (by `task_id` string) is distinct from the scheduler's registration-order FIFO at acquisition time.
- **Bazarr Ready**: Direct compatibility with the full media automation stack via standard formats (SRT, VTT, JSON).
- **Industrial Telemetry**: Real-time speed multipliers, ETA calculation, and detailed hardware state reporting.
- **Intel GPU Saturation Reporting**: Intel iGPU/Arc dashboard charts report full busy-load saturation as `100%` instead of capping at `99%`.
- **Interactive Documentation**: Full OpenAPI/Swagger interface available at `/docs`.
- **Live SRT Streaming**: Real-time auto-scrolling subtitle display during processing for immediate visual feedback.

### 🧩 Hardware Compatibility Matrix

| Pipeline Stage | CPU (Generic) | NVIDIA (CUDA) | AMD (native Linux ROCm) | Intel iGPU / Arc | Intel NPU |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Media Standardization** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Vocal Isolation (UVR)** | ✅ | ✅ | ❔ (native Linux ROCm via `/dev/kfd`); WSL2 `/dev/dxg` detects AMD but falls back to CPU | ✅ (OpenVINO) | ✅ (OpenVINO) |
| **VAD Verification** | ✅ | ✅ | ❔ | ✅ | ✅ |
| **Whisper ASR Inference** | ✅ | ✅ | ⚠️ (CPU Fallback) | ✅ *with `ASR_ENGINE=INTEL-WHISPER`; otherwise CPU* | ⚠️ (CPU Fallback) |
| **Speaker Diarization** | ✅ | ✅ | ❔ | ✅ | ✅ |

✅ measured on real hardware &nbsp;·&nbsp; ⚠️ works, but on CPU &nbsp;·&nbsp; ❔ **implemented, never exercised on supported AMD silicon**

The AMD column is **not** a validation claim: the ROCm paths are implemented and the
kernels ship, but they have never run on supported AMD silicon. The only Radeon available
for testing was an integrated `gfx1036`, which upstream rocBLAS does not support at all.

---

## ⚙️ Configuration

| Variable | Default | Description |
| :--- | :--- | :--- |
| **ASR_MODEL** | `Systran/faster-whisper-large-v3` | Model ID (HuggingFace) or local path |
| **ASR_ENGINE** | `AUTO` | Selects ASR backend engine: `AUTO`, `FASTER-WHISPER`, `INTEL-WHISPER`, `OPENAI-WHISPER`, `WHISPERX`. `AUTO` resolves to `FASTER-WHISPER` on every host |
| **ASR_DEVICE** | `AUTO` | Device: `AUTO`, `CUDA`, or `CPU` |
| **ASR_PREPROCESS_DEVICE** | `AUTO` | Device for Isolation: `AUTO`, `NPU`, `GPU`, `CUDA`, or `CPU` |
| **ENABLE_VOCAL_SEPARATION** | `false` | Pre-clean audio with UVR/MDX-NET. Off by default: measured 76% slower for no gain on clean speech. Worth enabling for music-heavy audio |
| **OV_CACHE_DIR** | `./model_cache` | OpenVINO kernel cache directory (highly recommended) |
| **ASR_BEAM_SIZE** | `5` | Decoding beam width (Search depth) |
| **ASR_PARALLEL_LIMIT_ACCEL** | `1` | Max concurrent tasks on GPU/NPU |
| **DIARIZATION_HF_TOKEN** | *(empty)* | Hugging Face token for speaker diarization (PyAnnote models) |
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

1. **`/app/model_cache`**: Stores the AI models downloaded on first start, WhisperX alignment models, PyAnnote diarization models, and pre-compiled OpenVINO NPU/GPU blobs. **Mapping this is effectively required** -- without it every container recreation re-downloads several GB of models.
2. **`/app/data`**: Stores the persistent state of the application, including task history, telemetry statistics, and system logs. Mapping this ensures your history survives container restarts and updates.

## 🚦 First Start

Images ship **without model weights**; they are downloaded on first start into
`/app/model_cache` and reused afterwards.

- The container reports **healthy** immediately -- the download does not block startup.
- Requests submitted during the download **wait in the queue** (stage
  `Downloading Model (xx%)`) and run automatically once it finishes. They are not rejected.
- `GET /status` shows `engines.whisper.status: "downloading"` meanwhile.
- Budget roughly 3-4.5 GB on first start depending on the edition. Subsequent starts are
  immediate.

---

## 🐳 GPU/NPU Support

### NVIDIA GPU (CUDA)

- Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- Ensure you have current NVIDIA drivers on the host.

### AMD GPU (native Linux ROCm)

- **Linux**: Map `/dev/kfd` and `/dev/dri` into the container.
- **Windows 11 / WSL2**: Map `/dev/dxg` (WSL GPU bridge) only for AMD adapter detection in this Linux container.
- Set `MAX_AMD_UNITS=1` in environment to enable the AMD scheduler unit.
- UVR vocal isolation runs on the AMD GPU via `onnxruntime-rocm` only on native Linux ROCm hosts with `/dev/kfd`. On WSL2 `/dev/dxg`, UVR falls back to CPU. Whisper ASR falls back to CPU on AMD units since CTranslate2 does not have a ROCm backend.
- Published `amd` and `full` images support consumer Radeon RDNA2/RDNA3/RDNA4. Data-center and legacy
  AMD architectures use CPU fallback.

### Intel NPU/GPU

- Mapping `/dev/dri` and `/dev/accel` is recommended for native Linux access.
- For Windows/WSL2, ensure `/dev/dxg` is mapped only when you want AMD detection/telemetry in this Linux image; it does not enable GPU UVR.

---

**Maintained by**: [ventura8](https://github.com/ventura8)  
**Full Documentation**: [GitHub Repository](https://github.com/ventura8/Whisper-Pro-ASR)
