# Whisper Pro ASR

![Main Language](https://img.shields.io/github/languages/top/ventura8/Whisper-Pro-ASR)
![Coverage](assets/coverage.svg)
![Pylint](https://img.shields.io/badge/Pylint-10.00%2F10-brightgreen)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**Whisper Pro ASR** is a high-performance transcription microservice with **speaker diarization**, optimized for the **Whisper Large V3** model. It delivers enterprise-grade performance with native hardware acceleration for **Intel Core Ultra NPUs**, **Integrated GPUs**, and **NVIDIA CUDA** environments.

Engineered for seamless integration with **Bazarr** and the broader media automation stack, it offloads computationally intensive AI tasks from your primary system resources, providing industrial-strength transcription with speaker identification and rapid hardware context switching.

## Concurrency-First Priority

Concurrency correctness is the top project priority.

- Any change that can affect scheduling, locks, queues, events, or model lifecycle must preserve deadlock and livelock safety before feature throughput optimizations.
- Priority/preemption synchronization waits are intentionally unbounded (waiting indefinitely with periodic logging every 30 seconds to survive heavy load); requests must wait until hardware and preemption handoff are available instead of failing on scheduler timeouts.
- Concurrency-affecting changes require matching liveness regression tests and documentation updates in this repository.
- Guarantee model: practical high-confidence liveness with explicit assumptions and CI stress evidence, not absolute universal proof across all OS and third-party internals.

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

## Frontend Quality Gates

Dashboard UI quality is validated with ESLint, Stylelint, and Vitest coverage gates.

```bash
npm run lint:js
npm run lint:css
npm run test:js
npm run quality:frontend
```

Coverage policy for monitored dashboard JavaScript files (`modules/monitoring/templates/*.js`):
- Per-file minimum `90%` for `lines` and `statements`.
- CI fails when any monitored file drops below threshold.

CodeRabbit review guidance is stored in [.coderabbit.yaml](.coderabbit.yaml) and covers both dashboard JavaScript and Python modules.

## 🚀 Key Features

### 🗣 Speaker Diarization
- **WhisperX Integration**: Identify who said what with automatic speaker diarization powered by WhisperX alignment and PyAnnote speaker segmentation.
- **Speaker Labels**: Output formats (SRT, VTT, TXT, TSV) include speaker identification labels (e.g., `[SPEAKER_00]: Hello world`).
- **Configurable Speakers**: Control diarization with `min_speakers` and `max_speakers` parameters for optimal speaker count estimation.
- **Graceful Fallback**: If diarization fails or `HF_TOKEN` is not configured, the system seamlessly falls back to standard transcription.

### Precision Architecture
- **Multi-Backend Support**: Specialized optimization profiles for **NVIDIA CUDA**, **Intel OpenVINO**, and **Generic CPU** runtimes.
- **Re-entrant Hardware Orchestration**: Utilizes a sophisticated thread-local locking system (`model_lock_ctx` in `scheduler.py`) that allows complex pipelines (UVR → ASR → Diarization) to share a single hardware claim without deadlocking.
- **FFmpeg 8.1.0 Integration**: Features optimized hardware-accelerated decoding. All media (MKV, AVI, MP4, etc.) is automatically standardized to **16kHz Mono WAV** using the `utils.py` core before entering the AI pipeline for maximum accuracy.

### Advanced Intelligence
- **FIFO Fairness with Priority Yielding**: Tasks are processed in arrival order within the same priority tier. High-priority language detection still preempts ASR when needed, but detect-language requests are also processed FIFO among themselves.
- **Deterministic Dashboard Ordering**: Active and historical task cards are rendered in arrival order (`start_time`) so operators see the same sequence tasks entered the system.
- **Intel ASR Chunking & Streaming**: Refactored OpenVINO engine transcription to split long media files dynamically into structured chunks guided by speech VAD timestamps, ensuring stability on very long movies.
- **O(1) Live Subtitle Updates**: Appends pre-formatted subtitle blocks incrementally to the live SRT stream during processing instead of doing full $O(N^2)$ stream reconstructions.
- **UVR Chunk Progress Tracking**: Computes and emits real-time preprocessing progress updates per UVR chunk to keep the dashboard progress bar fluid during vocal separation.
- **Graceful Temp-Storage Fallback**: Establishes a 2GB minimum free space threshold and 1.5x file-size headroom multiplier to fallback gracefully to persistent storage when tmpfs runs low on space, preventing ENOSPC crashes.
- **Cooperative Pre-emption**: High-priority operations (such as language detection) pause long-running ASR at deterministic checkpoints, including pre-vocal-separation, HQ-prep FFmpeg progress boundaries, and pre-inference, ensuring responsive API behavior under saturation.
- **Consolidated Batch Montage**: Consolidates multiple sampling targets into a single high-density montage. This allows for a **single-pass UVR isolation** across multiple non-contiguous segments, eliminating repeated model loading overhead.
- **Global VAD & In-Memory Slicing**: Features a unified Voice Activity Detection scan across the entire montage. segments are then sliced as **NumPy arrays in memory**, eliminating temporary file I/O and reducing VAD overhead by up to 900%.
- **Customizable ASR Parameters**: Fine-tune transcription with `initial_prompt` (context guidance), `vad_filter` (silence suppression), and `word_timestamps` (word-level timing).
- **Subtitle Layout Control**: Custom character-per-line wrapping (`max_line_width`) and max line block limits (`max_line_count`) for SRT/VTT output.
- **Plex-Compatible AI Subtitle Tagging**: All subtitle output filenames use the `<source>.<language>-ai.<format>` naming convention (e.g. `movie.en-ai.srt`). The `-ai` suffix leverages the ISO 3166-1 country code for Anguilla (`AI`), which Plex's regional layout parser maps to display subtitles as `<Language> (AI)` — e.g. `English (AI)`, `Spanish (AI)`. Works for all languages and both transcription and translation tasks, preventing fall-through to `xx (Unknown)` in Plex.
- **Subtitle Word Highlighting**: `subtitle_highlight_words=true` renders the currently-spoken word in a highlight color within SRT/VTT blocks, automatically enabling word-level timestamps.
- **Configurable Subtitle Promo Card**: Prepends a promo subtitle block (e.g. `"Made with Whisper Pro ASR"`) to SRT and WebVTT outputs. Customizable display duration and text are fully configurable via Docker Compose.
- **Smart Model Lifecycle**: Configurable `MODEL_IDLE_TIMEOUT` keeps models warm in memory for rapid response to bursty workloads. A deferred cleanup timer starts only after the last task completes, and is automatically cancelled and rescheduled when new tasks arrive.
- **Deferred Persistence Engine**: Protects SSD longevity by buffering task history and telemetry in RAM, only syncing to physical storage after 10 tasks or 1 hour of activity.
- **Fail-Safe Dual-Path VAD**: Intelligent logic that verifies speech presence on both isolated and raw audio, selecting the optimal path automatically based on signal clarity.
- **Confusion-Matrix Tie Breaking**: Resolves linguistic ambiguities between similar pairs (e.g., NO/NN) with a weighted bias, eliminating common identification hallucinations.
- **Unified Session Orchestration**: Integrated task and queue tracking ensures that hardware resources are only reclaimed when the system is fully idle (zero active or waiting tasks).
- **Proactive Resource Reclamation**: Automatically offloads heavy models and clears hardware caches (CUDA/NPU) only when the queue is empty.
- **Weighted Multi-Segment Voting**: Aggregates probabilities from multiple zones with confidence-weighted averaging for industrial-strength accuracy.
- **Advanced Memory Hygiene**: Implements a "Nuclear Purge" strategy using `malloc_trim` and ctranslate2 cache clearing to ensure idle memory remains below 500MB even after heavy ASR sessions.
- **Telemetry Downsampling**: Dual-layer downsampling (server-side and client-side) caps telemetry chart data at 300 points, ensuring smooth dashboard rendering even after extended operation.
- **Centralized Storage Hygiene**: Features a thread-local tracking system that registers every transient asset (uploads, HQ prep files, isolated stems) created during a request. The system ensures a **100% cleanup rate** by purging all tracked files immediately upon request completion or failure.
- **On-Demand History Tiering**: Implements a dual-tier storage strategy. The dashboard and RAM are strictly capped at the last 20 tasks, while a durable history of up to 1000 tasks is maintained on the persistent volume.
- **Hardened Diagnostic Logging**: System logs (`whisper_pro.log`) are redirected to the persistent state volume with real-time flush-to-disk logic. Log downloads are served via atomic in-memory reads to prevent `RuntimeError: Response content longer than Content-Length` failures that occur when the log file is actively written during download. Zero-caching headers ensure the latest diagnostic data is always delivered.


### Production Ready
- **OpenAI Standard API**: Drop-in compatible with the OpenAI whisper specification, allowing immediate integration with existing clients.
- **Endpoint Taxonomy (Contract)**: `/asr` and `/v1/audio/...` are equivalent standard-priority ASR surfaces, while `/detect-language` (and alias `/detectlang`) is the high-priority language-identification surface.
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
| **Speaker Diarization** | ✅ | ✅ | ✅ | ✅ |

### System Architecture
The service utilizes a **Heterogeneous Model Pool** to orchestrate tasks across NVIDIA GPUs, Intel NPUs, and CPUs simultaneously, with integrated WhisperX diarization and configurable model lifecycle management. For a deep dive into the processing pipelines, resource locking, and hardware acceleration logic, see the [Technical Architecture](docs/ARCHITECTURE.md) documentation.

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
| `ASR_ENGINE` | `FASTER-WHISPER` | Selects ASR backend engine. Options: `AUTO`, `FASTER-WHISPER`, `INTEL-WHISPER`, `OPENAI-WHISPER`, `WHISPERX`. Invalid values fail startup. |
| `VOCAL_SEPARATION_MODEL` | `UVR-MDX-NET-Voc_FT` | Model ID (HuggingFace) or local path |
| `ASR_BATCH_SIZE` | `1` | Number of segments processed per pass. |
| `ASR_BEAM_SIZE` | `5` | Decoding beam width (Search depth). |
| `DEBUG` | `false` | Enables verbose stack traces and debug logging. |
| **Diarization** | | |
| `HF_TOKEN` | *(empty)* | Hugging Face token for speaker diarization (PyAnnote models). |
| **Transcription Tuning** | | |
| `INITIAL_PROMPT` | *(multilingual)* | Default context prompt to guide Whisper transcription. |
| `MODEL_IDLE_TIMEOUT` | `300` | Seconds to keep models loaded after last task (0 = immediate offload). |
| `INTEL_ASR_CHUNK_DURATION` | `300` | Chunk duration in seconds for Intel Whisper transcription. |
| `AGGRESSIVE_OFFLOAD` | `false` | Immediately unload models when idle (overridden by `MODEL_IDLE_TIMEOUT`). |
| **Subtitle Promo** | | |
| `SUBTITLE_PROMO_ENABLED` | `true` | Prepend a promo card "Made with Whisper Pro ASR" to SRT/VTT. |
| `SUBTITLE_PROMO_TEXT` | `Made with Whisper Pro ASR` | Text to display in the promo card. |
| `SUBTITLE_PROMO_DURATION` | `3.0` | Duration (in seconds) to display the promo card. |
| **Optimization** | | |
| `OV_PERFORMANCE_HINT` | `LATENCY` | OpenVINO scheduling hint (Latency/Throughput). |
| `OV_CACHE_DIR` | `./model_cache` | Persistent directory for compiled hardware blobs. |
| **Parallelism** | | |
| `ASR_THREADS` | `4` | CPU core allocation for inference (Auto-capped by hardware). |
| `ASR_PREPROCESS_THREADS` | `4` | CPU core allocation for UVR/ONNX (Auto-capped by hardware). |
| **SSD Protection** | | |
| `WHISPER_TEMP_DIR`| `/tmp/whisper`| Redirects transient I/O (uploads, WAVs, stems) to this path. |
| `WHISPER_TEMP_MIN_FREE_MB` | `2048` | Fallback threshold to disk if RAM-disk is full. |
| **Preprocessing** | | |
| `ENABLE_VOCAL_SEPARATION`| `true` | Toggles UVR background removal engine for translate/transcribe. |
| `UVR_CHUNK_DURATION` | `600` | Chunk duration in seconds for UVR separation (0 to disable). |
| `ENABLE_LD_PREPROCESSING`| `true` | Toggles UVR background removal engine for language detection. |
| `LD_VAD_THRESHOLD` | `0.3` | Aggressiveness of VAD during language identification (0.0 to 1.0). |
| `SMART_SAMPLING_SEARCH` | `true` | Enables localized entropy-based signal searching in sparse audio. |
| `MAX_CUDA_UNITS` | `1` | Max NVIDIA GPUs to utilize (supports `ALL`, `AUTO`). |
| `MAX_GPU_UNITS` | `1` | Max Intel GPUs to utilize (supports `ALL`, `AUTO`). |
| `MAX_NPU_UNITS` | `1` | Max Intel NPUs to utilize (supports `ALL`, `AUTO`). |
| `MAX_CPU_UNITS` | `1` | Max concurrent multi-core CPU tasks (VAD, FFmpeg, CPU-ASR). |
| `FFMPEG_HWACCEL` | `none` | FFmpeg hardware acceleration target (`cuda`, `vaapi`, `qsv`). |
| `FFMPEG_FILTER` | `dynaudnorm` | Normalization filter: `dynaudnorm` (Standard) or `loudnorm` (Broadcast). |

### ⚙️ ASR Backend Engines (ASR_ENGINE)

The service supports multiple ASR backend engines to run inference. You can configure this using the `ASR_ENGINE` environment variable. The following options are available:

- **`AUTO`**: Automatically resolves the engine by available hardware in this exact order: `CUDA` -> `Intel GPU` -> `Intel NPU` -> `CPU`.
  - `CUDA` -> `FASTER-WHISPER`
  - `Intel GPU` -> `INTEL-WHISPER`
  - `Intel NPU` -> `INTEL-WHISPER`
  - `CPU` -> `FASTER-WHISPER`
- **`FASTER-WHISPER`** (Default): Uses the CTranslate2 engine. This is the recommended choice for general CPU and NVIDIA CUDA environments, offering extremely fast processing and low memory footprint.
- **`INTEL-WHISPER`**: Uses the OpenVINO-based Intel Whisper engine (`IntelWhisperEngine`). Highly optimized for Intel NPUs and Integrated/Arc GPUs. If Intel GPU/NPU is unavailable, runtime falls back to `FASTER-WHISPER` (not OpenVINO CPU).
- **`OPENAI-WHISPER`**: Uses the reference OpenAI Whisper Python backend.
- **`WHISPERX`**: Uses the WhisperX backend, supporting batch inference.

When `ASR_ENGINE` is set explicitly, unsupported values are rejected at startup with a clear validation error.

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
 
      # AUTO resolution order: CUDA -> Intel GPU -> Intel NPU -> CPU
      - ASR_ENGINE=FASTER-WHISPER
      - INTEL_ASR_CHUNK_DURATION=300
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
Main entry point for generating subtitles with optional speaker diarization.
- **Formats**: `srt` (default), `vtt`, `txt`, `tsv`, `json` (with segments).
- **Diarization**: Add `diarize=true` to enable speaker identification (requires `HF_TOKEN`).
- **ASR Tuning**: `initial_prompt`, `vad_filter`, `word_timestamps` for fine-grained control.
- **Subtitle Layout**: `max_line_width` and `max_line_count` for custom subtitle formatting.
- **Word Highlighting**: `subtitle_highlight_words=true` highlights the active spoken word in SRT/VTT output.
- **Plex AI Tagging**: Subtitle files are named `<source>.<language>-ai.<ext>` so Plex displays them as `<Language> (AI)` for all languages.
- **Optimization**: Prioritizes local file access if the path exists (via volume mapping), otherwise accepts file uploads.

### 3. Service Analytics & Dashboard
**GET** `/status`  
Health-check endpoint returning model metadata, hardware status, and versioning information.

**GET** `/dashboard` (or **GET** `/` via Browser)  
Interactive Material Design interface for real-time monitoring of task progress, hardware utilization, and application memory.

**GET** `/analytics` (or **GET** `/analytics` via Browser)  
Cumulative and daily analytics dashboard with interactive charts, providing categorized breakdowns of task counts, durations, and usage patterns by endpoint (/asr, /detect-language, /v1/audio/...).

**GET/POST** `/settings`  
View or dynamically update service configuration (model, device, telemetry retention) at runtime without container restart.

---

## 📺 Bazarr Configuration

To use this service with **Bazarr**:

1.  **Provider**: Choose **Whisper** (or `whisper-asr-webservice`).
2.  **Endpoint**: `http://<YOUR_DOCKER_IP>:9000`
3.  **Timeouts**: Should be set very high (54000) for long movies
4.  **Pass video filename to Whisper**: Should be enabled for volume mapping to work correctly
3.  **Volume Mapping (Highly Recommended)**:
  - Ensure your Bazarr and Whisper-Pro-ASR containers share the same media paths (e.g., both map `/tv` to the same actual folder).
  - When configured this way, Bazarr sends the *file path* to Whisper. Whisper Pro checks if it can read that path locally. If yes, it uses the mapped file directly and skips upload materialization.
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
│   ├── bootstrap.py         # Hardware path patching & library redirection
│   ├── api/                 # API Routes
│   │   ├── routes_asr.py    # /asr, /v1/audio/transcriptions, /v1/audio/translations
│   │   ├── routes_detect.py # /detect-language
│   │   ├── routes_system.py # /dashboard, /status, /settings, /analytics, /history
│   │   └── routes_utils.py  # Shared request utilities & file handling
│   ├── inference/           # ML Engine
│   │   ├── model_manager.py # Model pool, transcription, diarization, idle monitoring
│   │   ├── scheduler.py     # Re-entrant locks & hardware orchestration
│   │   ├── language_detection.py  # Batch language identification pipeline
│   │   ├── preprocessing.py # UVR vocal separation
│   │   ├── vad.py           # Voice Activity Detection
│   │   └── intel_engine.py  # Intel NPU/GPU engine adapter
│   ├── monitoring/          # Dashboard, Telemetry & Metrics
│   │   ├── dashboard.py     # Dashboard entry point
│   │   ├── dashboard_ui.py  # Material Design dashboard renderer (loads from templates)
│   │   ├── analytics_ui.py  # Dynamic loader for analytics UI (loads from templates)
│   │   ├── templates/       # HTML, CSS, and JS dashboard/analytics templates
│   │   │   ├── dashboard.html
│   │   │   ├── dashboard.css
│   │   │   ├── dashboard_charts.js
│   │   │   ├── dashboard_main.js
│   │   │   ├── dashboard_state.js
│   │   │   ├── dashboard_utils.js
│   │   │   ├── analytics.html
│   │   │   ├── analytics.css
│   │   │   └── analytics.js
│   │   ├── telemetry.py     # Real-time telemetry collection
│   │   ├── telemetry_manager.py  # Persistent telemetry history
│   │   ├── history_manager.py    # Task history (dual-tier storage)
│   │   └── metrics_discovery.py  # Hardware metrics detection
│   ├── config.py            # Global Settings (HF_TOKEN, MODEL_IDLE_TIMEOUT, etc.)
│   ├── logging_setup.py     # Task-specific Logging
│   └── utils.py             # System & Audio Utilities (subtitle wrapping, speaker labels)
├── tests/                   # Performance & Unit Test Suites
│   ├── inference/           # Diarization, Language Detection, Concurrency tests
│   ├── integration/         # Route, Server, and Robustness tests
│   ├── monitoring/          # Dashboard, History, Telemetry tests
│   ├── performance/         # Coverage, RAM, SSD optimization tests
│   └── unit/                # Config, Logging, Utils tests
├── Dockerfile               # Packaging Definition
└── docker-compose.yml       # Orchestration Template
```
