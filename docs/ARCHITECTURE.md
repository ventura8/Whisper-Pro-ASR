# Technical Architecture
 
## 🧬 Module Ecosystem
 
| Component | Responsibility |
|:---|:---|
| `config.py` | Centralized environment management and application constants. |
| `logging_setup.py` | Orchestrates structured logging, hardware banners, and transformer filter stacks. |
| `model_manager.py` | Manages Faster-Whisper lifecycles, VAD integration, and priority task scheduling. |
| `preprocessing.py` | UVR/MDX-NET isolation engine with optimized ONNX/OpenVINO backend patching. |
| `routes.py` | Flask API layer implementing RESTful endpoints and OpenAI-compatible aliases. |
| `language_detection.py` | Multi-zone strategic sampling with confidence-weighted voting and dual-path VAD. |
| `utils.py` | Managed FFmpeg normalization and millisecond-accurate SRT generation. |
| `vad.py` | Silero Voice Activity Detection (VAD) for signal trimming and silence suppression. |
 
---
 
## 🏎 Processing Pipelines
 
### Transcription Flow (/asr)
1. **Ingress**: Media is received and analyzed for duration/metadata via `ffprobe`.
2. **Pre-Detection**: If the source language is unknown, the engine performs **Optimized Language ID**. It utilizes **Parallel Preprocessing** to prepare up to 5 strategic zones simultaneously, reducing latency.
3. **Standardization**: FFmpeg converts the source stream to **16kHz MONO WAV** with `-loudnorm` filtering, utilizing the project's global audio standard.
4. **UVR Isolation (Optional)**: If enabled, the signal is passed through the MDX-NET isolator to remove background noise/music.
5. **Signal Quality Audit**: Silero identifies active speech. If isolation is silent OR yields low confidence (<80%), the engine audits the raw signal and selects the path with higher confidence.
6. **Tie-Breaker Layer**: Before finalizing detection, the engine resolves ambiguities between similar languages (e.g., Norwegian vs. Nynorsk) using a **Confusion-Matrix Bias**.
7. **Inference**: The `model_manager` executes batched Faster-Whisper generation with real-time speed telemetry.
8. **Finalization**: Subtitles are generated and a millisecond-precision SRT or structured JSON is returned.
 
### Priority Detection Flow (/detect-language)
1. **Strategic Sampling**: The engine divides the media into up to 5 non-overlapping zones across the entire duration for maximum spatial representation.
2. **Unified Extraction**: Optimized FFmpeg commands extract strategic samples directly, bypassing full-file normalization.
3. **Dual-Path VAD**: Each zone is verified for speech on both isolated and raw audio paths to prevent false negatives in sparse media.
4. **Weighted Inference**: Probabilities from all zones are aggregated using confidence-weighted averaging to ensure a robust final identification.
 
---
 
## 🔒 Concurrency & Pre-emption
 
Whisper Pro implements a sophisticated **Priority Scheduling** system to manage shared hardware assets (NPU/GPU):
 
- **Task Serialization**: transcription requests are managed via a sequential queue to ensure stable memory pressure.
- **Priority Signal**: When a `/detect-language` request arrives, it sets a global `_PAUSE_REQUESTED` event.
- **Yielding Mechanism**:
    1. The active ASR loop checks for the pause event at the end of every segment.
    2. The ASR thread releases its model lock and hardware assets.
    3. The high-priority Detection task acquires the **Sequential Priority Lock** and utilizes the hardware.
    4. Upon completion, the Detection task releases assets and signals the `_RESUME_EVENT`.
    5. The paused ASR thread re-acquires its lock and restores the inference state.

### Unified Session Orchestration
To ensure stability across concurrent operations, the system tracks all active and queued tasks via global counters:
- **Synchronization**: Every request (ASR or Detection) increments `_ACTIVE_SESSIONS` on entry and decrements it on exit.
- **Queue Tracking**: Requests waiting for hardware locks (NPU/GPU) or priority pre-emption are tracked via `_QUEUED_SESSIONS`.
- **Proactive Reclamation**: Immediately after both active and queued counters hit zero, the system triggers a **Resource Offload** cycle. This ensures that heavy AI models remain in memory as long as there is pending work, eliminating initialization overhead for back-to-back requests.
 
---
 
## 🚀 Hardware Tuning & Optimizations
 
### OpenVINO (Intel Silicon)
- **Static Reshaping**: When targeting Intel NPU, the computational graph is reshaped into static dimensions to avoid runtime re-compilation.
- **Persistent Cache**: Compiled model blobs are cached in `OV_CACHE_DIR`, reducing startup time from minutes to seconds.
- **Performance Hints**: The engine dynamically toggles between `LATENCY` (single-request) and `THROUGHPUT` (batch) hints based on `ASR_BATCH_SIZE`.
 
### CUDA (NVIDIA Silicon)
- **CUDNN Tuning**: Leverages standard NVIDIA optimizations for high-speed half-precision (FP16/BF16) inference.
- **Batching**: Optimized for multi-segment parallel processing to saturate GPU compute cores.
 
### Runtime Patching
The service applies a series of recursive monkey-patches to **ONNX Runtime** and **audio-separator** to:
- Enforce strict thread limits (avoiding system-wide CPU starvation).
- Prioritize specific execution providers (OpenVINO vs CUDA) without environment pollution.
- Capture sub-segment progress telemetry for real-time Flask updates.
 
---
 
## 💾 SSD Write Optimization
 
To ensure longevity on SSD-backed hosts (common in Docker environments), Whisper Pro implements a **Transient RAM-Bypass** strategy:
 
- **Centralized Temp Store**: All high-frequency write operations (FFmpeg re-encoding, file uploads, UVR stems) are redirected to a single `WHISPER_TEMP_DIR`.
- **tmpfs Integration**: By mounting this directory as `tmpfs` (RAM-disk), the service eliminates physical SSD wear for transient files.
- **Dynamic Overflow Protection**: The engine monitors available space in the temp directory. If a file (e.g., a 4h+ movie) exceeds available RAM-disk space, the engine automatically falls back to the system's physical disk to ensure processing completes.
- **Transient Preprocessing**: Unlike model blobs which remain in persistent cache, intermediate UVR stems are stored in the temp directory and purged immediately after use.
 
---
 
## 🏛 Hardware Interface & Host Dependencies
 
To maintain high performance without compromising security, Whisper Pro separates the **AI Runtime** (inside the container) from the **Hardware Drivers** (on the Host):
 
- **Intel NPU/GPU**: Leverages the official `/dev/dri` and `/dev/accel` nodes. These are typically standard Linux device files and can be passed through via `devices:` mapping.
- **NVIDIA CUDA**: While the container includes CUDA 12.8 libraries, it requires the **NVIDIA Container Toolkit on the host. This toolkit handles the "injection" of host-specific `.so` driver files into the container namespace during startup, ensuring perfect alignment between the container's math operations and the host's physical GPU.
