# Technical Architecture
 
## 🧬 Module Ecosystem
 
| Component | Responsibility |
|:---|:---|
| `config.py` | Centralized environment management and application constants. |
| `logging_setup.py` | Orchestrates structured logging, hardware banners, and transformer filter stacks. |
| `model_manager.py` | Manages Faster-Whisper lifecycles, VAD integration, and priority task scheduling. |
| `preprocessing.py` | UVR/MDX-NET isolation engine with optimized ONNX/OpenVINO backend patching. |
| `routes.py` | Flask API layer implementing RESTful endpoints and OpenAI-compatible aliases. |
| `language_detection.py` | Multi-zone probabilistic identification with squared confidence weighting. |
| `utils.py` | Managed FFmpeg normalization and millisecond-accurate SRT generation. |
| `vad.py` | Silero Voice Activity Detection (VAD) for signal trimming and silence suppression. |
 
---
 
## 🏎 Processing Pipelines
 
### Transcription Flow (/asr)
1. **Ingress**: Media is received and analyzed for duration/metadata via `ffprobe`.
2. **Pre-Detection**: If the source language is unknown, the engine performs **Optimized Language ID** using source media chunks *before* full-file processing.
3. **Standardization**: FFmpeg converts the source stream to **16kHz MONO WAV** with `-loudnorm` filtering, utilizing the project's global audio standard.
4. **UVR Isolation (Optional)**: If enabled, the signal is passed through the MDX-NET isolator to remove background noise/music.
5. **VAD Alignment**: Silero identifies active speech regions to prevent hallucinations in silent periods.
6. **Inference**: The `model_manager` executes batched Faster-Whisper generation with real-time speed telemetry.
7. **Finalization**: Subtitles are generated and a millisecond-precision SRT or structured JSON is returned.
 
### Priority Detection Flow (/detect-language)
1. **Optimized Segmentation**: The engine uses FFmpeg to extract small **16kHz Mono chunks** directly from the source media (`-ss` / `-t`), bypassing the need for full-file normalization.
2. **Inference**: Parallel one-token scans are performed across these normalized chunks.
3. **Consensus**: Probabilities are aggregated using **Squared Confidence Weighting** to favors "peaks" over noise.
4. **ID**: The most probable language is selected and returned with a globally averaged confidence score.
 
---
 
## 🔒 Concurrency & Pre-emption
 
Whisper Pro implements a sophisticated **Priority Scheduling** system to manage shared hardware assets (NPU/GPU):
 
- **Task Serialization**: transcription requests are managed via a sequential queue to ensure stable memory pressure.
- **Priority Signal**: When a `/detect-language` request arrives, it sets a global `_PAUSE_REQUESTED` event.
- **Yielding Mechanism**:
    1. The active ASR loop checks for the pause event at the end of every chunk.
    2. The ASR thread releases its model lock and hardware assets.
    3. The high-priority Detection task acquires the **Sequential Priority Lock** and utilizes the hardware.
    4. Upon completion, the Detection task releases assets and signals the `_RESUME_EVENT`.
    5. The paused ASR thread re-acquires its lock and restores the inference state.
 
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
 
## 🏛 Hardware Interface & Host Dependencies
 
To maintain high performance without compromising security, Whisper Pro separates the **AI Runtime** (inside the container) from the **Hardware Drivers** (on the Host):
 
- **Intel NPU/GPU**: Leverages the official `/dev/dri` and `/dev/accel` nodes. These are typically standard Linux device files and can be passed through via `devices:` mapping.
- **NVIDIA CUDA**: While the container includes CUDA 12.8 libraries, it requires the **NVIDIA Container Toolkit on the host. This toolkit handles the "injection" of host-specific `.so` driver files into the container namespace during startup, ensuring perfect alignment between the container's math operations and the host's physical GPU.
