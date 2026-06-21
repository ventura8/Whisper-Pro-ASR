# Whisper Pro ASR

> **CRITICAL DEVELOPER NOTE**: Only build or run the Docker image locally if you have compatible hardware (Intel NPU/GPU or NVIDIA GPU) mapped to the container. Otherwise, rely on the CI/CD pipeline and mocks for verification.

## AI & Development Rules
- **File Size Constraint**: Never have a `.py` file larger than **500 lines**. If a file grows beyond this limit, refactor and modularize into smaller files within the `modules/` directory.
- **Logging Standard**: Use the project's central logger (`logging`) instead of `print()` statements for all modules and scripts.
- **Thread Compliance**: All multi-threaded components (FFmpeg, OpenVINO, ONNX Runtime) MUST strictly respect the thread limits set in `modules/config.py` (`ASR_THREADS`, `ASR_PREPROCESS_THREADS`, `FFMPEG_THREADS`). Language detection runs serially (single worker) to ensure thread limits are not exceeded.
- **Media Standardization**: All audio ingested MUST be converted to **16kHz, Mono, 16-bit PCM**. Always use `utils.STANDARD_AUDIO_FLAGS` and `utils.STANDARD_NORMALIZATION_FILTERS` for FFmpeg commands to ensure pipeline consistency.
- **Efficiency Optimization**: For non-ASR tasks (identification/status), ALWAYS favor segmented FFmpeg extraction (`-ss` / `-t`) from the source media over full-file normalization. Avoid using `soundfile` (`sf.read`) directly on non-WAV video containers as it causes expensive full-file probes.
- **Resource Cleanup & Stability**: All temporary files and system resources (file descriptors, locks) MUST be managed using `try...finally` blocks to ensure absolute cleanup even on catastrophic failures. Use the project's unified `decrement_active_session()` helper to ensure synchronized resource reclamation and VRAM offloading.


## Documentation Index

| Document | Description |
|----------|-------------|
| [docs/SETUP.md](docs/SETUP.md) | Installation & configuration (includes diarization setup) |
| [docs/API.md](docs/API.md) | API endpoint reference (diarization, ASR params, subtitle layout) |
| [docs/TUNING.md](docs/TUNING.md) | Performance optimization guide (idle timeout, initial prompt) |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Technical module documentation (pipelines, caching, lifecycle) |

## Key Features
- **Industrial Hardware Acceleration**: Native support for **Intel NPU**, **Intel iGPU**, and **NVIDIA CUDA**.
- **Speaker Diarization**: Identify who said what using WhisperX alignment and PyAnnote speaker segmentation. Requires `HF_TOKEN` for PyAnnote model access.
- **Hardware Compatibility Matrix**:
    | Engine | CPU | NVIDIA (CUDA) | Intel GPU | Intel NPU |
    | :--- | :---: | :---: | :---: | :---: |
    | **Vocal Separation** | ✅ | ✅ | ✅ | ✅ |
    | **Whisper ASR** | ✅ | ✅ | ❌ (CPU Fallback) | ❌ (CPU Fallback) |
    | **Speaker Diarization** | ✅ | ✅ | ✅ | ✅ |
- **Hybrid Device Support**: Simultaneously utilize multiple accelerators (e.g., Intel for isolation and NVIDIA for transcription).
- **Advanced Preprocessing Stack**: Integrated **UVR/MDX-NET** (Vocal Isolation) with hardware acceleration and dedicated thread pooling (`ASR_PREPROCESS_THREADS`).
- **OpenAI Compatible**: Native support for `/v1/audio/transcriptions` and `/v1/audio/translations` OpenAI API format.
- **Swagger Documentation**: Interactive API testing available at `/docs`.
- **Customizable ASR Parameters**: `initial_prompt`, `vad_filter`, and `word_timestamps` can be set per-request or via environment variables.
- **Subtitle Layout Control**: `max_line_width` and `max_line_count` for SRT/VTT formatting.
- **Smart Model Lifecycle**: Configurable `MODEL_IDLE_TIMEOUT` keeps models warm in memory for rapid response, with background idle monitoring.
- **Probabilistic Language ID**: Robust identification using **Strategic Uniform Sampling** and **Weighted Voting**. The system automatically samples up to 5 non-overlapping zones across the media and aggregates high-confidence evidence to eliminate false positives.

- **Sequential Priority Queue**: Multiple high-priority requests (Language Detection) are automatically queued and processed one-by-one, ensuring hardware stability.
- **Deadlock-Free Yielding**: Ongoing transcription threads release hardware model locks during priority wait periods, allowing metadata tasks to pre-empt decoding instantly.
- **Robust Multi-Stage Cleanup**: Automated intermediate file management ensures that all UVR outputs, converted WAVs, and temporary files are strictly purged.
- **Unified Precision Logs**: Real-time logging with stable ETA estimation and professional `HH:MM:SS` formatting for all stages.
- **Unified Session Management**: Integrated task tracking (`_ACTIVE_SESSIONS`) ensures that hardware resources (VRAM/RAM) are only reclaimed when the entire system is idle, preventing race conditions during concurrent requests.
- **Proactive Resource Reclamation**: Automated model offloading and hardware cache clearing (CUDA/NPU) triggered immediately after task completion.
- **Direct Path Support**: Use the `local_path` parameter to avoid HTTP upload overhead for multi-gigabit files.

## Model Selection
The service is highly flexible and supports any **Faster-Whisper** compatible model from Hugging Face. 

- **Runtime (Recommended)**: Change the `ASR_MODEL` environment variable in your `docker-compose.yml` (e.g., `Systran/faster-whisper-medium`) to switch models instantly. The new model will be downloaded to your persistent cache on startup.
- **Offline/Baked-in**: Modify the `WHISPER_MODEL` build argument in the `Dockerfile` and rebuild if you need a specific model permanently baked into the image without internet access.

- **Multilingual**: `large-v3-turbo`, `large-v3`, `medium`, `small`, `tiny`.
- **English-Only**: Add `.en` suffix for better performance (e.g., `medium.en`).
- **Distil-Whisper**: Use `distil-whisper/distil-large-v3` for maximum speed (**English-only for all tasks**).

## Hardware Support Details
The service utilizes different backends for transcription and isolation:

Powered by **Faster-Whisper**. Supports **CPU** (Intel MKL/OpenMP) and **NVIDIA CUDA**. 
*Note: Intel OpenVINO ASR is currently disabled to ensure maximum transcription quality (Beam Search support). Hardware auto-detection will default to CPU/CUDA for all decoding tasks.*

### Vocal Separation (Preprocessing)
Powered by **ONNX Runtime (patched for OpenVINO)**. Supports **CPU**, **NVIDIA CUDA**, **Intel GPU**, and **Intel NPU**. 
*This allows for a "Split Architecture" where you can offload vocal cleaning to an Intel iGPU/NPU while keeping your NVIDIA card focused on transcription.*

## Quick Start
```bash
docker run -d \
  --name whisper-pro-asr \
  -p 9000:9000 \
  --device /dev/accel/accel0 \
  --device /dev/dri \
  -v ./model_cache:/app/model_cache \
  ventura8/whisper-pro-asr:latest
```

> [!TIP]
> **Autonomous Detection**: The service automatically identifies and utilizes NVIDIA GPUs, Intel NPUs, or Intel iGPUs. Manual device configuration is optional.

> [!IMPORTANT]
> **NPU Compilation Cache**: Always map the `./model_cache` volume. This stores the binary blobs unique to your specific NPU hardware/driver version, reducing startup time from minutes to seconds on subsequent runs.

## Zero-Wait Detection & Priority Yielding
The service implements a **Full-Pipeline Priority Yielding** system with sequentialized request management. When a `/detect-language` request arrives:
1. **Pre-emption Detection**: Ongoing ASR tasks (including both heavy UVR/MDX-NET preprocessing and Whisper decoding stages) detect the pause request at the next available yield point.
2. **Hardware Yielding**: The processing thread releases its claim on the hardware model lock (`_MODEL_LOCK`) and enters a wait state. This prevents deadlocks and ensures the high-priority task has immediate access to acceleration.
3. **Sequential Queueing**: If multiple `/detect-language` calls arrive simultaneously, they are serialized using a re-entrant sequential lock. This manages hardware load while maintaining the ASR pause.
4. **Resumption**: Once the entire priority queue is empty, the `_RESUME_EVENT` is triggered. The ASR thread re-acquires the model lock and resumes exactly where it left off.

## CI/CD and Verification
The project features an optimized CI/CD pipeline that consolidates linting and testing for maximum efficiency.

### Local Verification
You can run the full test and lint suite locally using Docker to mirror the CI environment:
```bash
# Build the test image (cached layers from production are used automatically)
docker build -t whisper-npu-test -f Dockerfile.test .

# Run the suite (Pylint + Pytest + Coverage)
docker run --rm whisper-npu-test
```

## Release Notes v1.1.3
- **FEAT**: Refactored the live transcription progress pipeline to append pre-formatted SubRip (SRT) blocks incrementally (reducing live update complexity to $O(1)$) instead of rebuilding the entire stream on every segment, preventing performance bottlenecks and memory bloat on large media files.
- **FEAT**: Refactored OpenVINO engine transcription (`IntelWhisperEngine`) to split long media files dynamically into structured chunks guided by speech VAD timestamps, and auto-detecting/locking the language on the first chunk to ensure stability on very long movies.
- **FEAT**: Patched the UVR vocal separation process dynamically on the scheduler to compute and emit real-time chunk progress status to prevent visual hangs during vocal separation.
- **STAB**: Enhanced RAM-disk / SSD protection by raising `WHISPER_TEMP_MIN_FREE_MB` to `2048` MB and implementing a 1.5x headroom factor fallback based on estimated WAV sizes.
- **FEAT**: Deconstructed the monolithic `dashboard.html` into a cleaner layout under a dedicated `templates/` directory containing separate CSS and JS components.
- **TEST**: Passed all unit, integration, and coverage tests with a perfect **10.00/10** score on all files under `pylint`.

## Release Notes v1.1.2
- **FIX**: Replaced fixed-interval `ModelIdleMonitor` polling thread with a deferred `threading.Timer` that starts after the last task completes. New tasks cancel and reschedule the timer, keeping models warm during bursty workloads.
- **FIX**: Resolved dashboard hardware utilization chart showing `0%` on initial page load (only displaying correct values after F5 refresh).
- **FIX**: Dashboard chart no longer incorrectly reports GPU/NPU utilization for Whisper ASR when it runs on CPU.
- **FIX**: Extended metrics discovery to correctly account for `/detect-language` tasks alongside `/asr` tasks in utilization calculations.
- **FIX**: Removed all `# pylint: disable` suppressions from production code. All lint compliance achieved through genuine code improvements (specific exception types, reduced locals, clean module access).
- **STAB**: Added `_POOL_LOCK` to serialize model loading and unloading, preventing race conditions during concurrent engine state transitions.
- **TEST**: Verified all unit and integration tests passing with a clean **10.00/10** pylint score on all files.

## Release Notes v1.1.1
- **FIX**: Resolved all pylint complexity warnings (`too-many-locals` and `too-many-positional-arguments` in `diarization.py`) by converting `run_diarization` parameters to keyword-only and extracting sub-helpers.
- **FIX**: Eliminated static and runtime cyclic imports between `model_manager`, `concurrency`, and `language_detection_core` using runtime `sys.modules` lookup.
- **FEAT**: Standardized duration metrics formatting across analytics pages to a unified, zero-padded `dd:hh:mm:ss` display.
- **TEST**: Passed all 345 unit/integration tests with a final **95.24%** coverage and a clean **10.00/10** score on all files under `pylint`.

## Release Notes v1.1.0
- **FEAT**: Integrated automatic Speaker Diarization using the WhisperX alignment, PyAnnote diarization, and speaker assignment pipeline, including per-device pools caching.
- **FEAT**: Added custom subtitle formatting controls (`max_line_width`, `max_line_count`) across SRT and VTT formatters.
- **FEAT**: Added a background daemon idle model timeout monitor (`MODEL_IDLE_TIMEOUT`) to automatically offload warm models after inactivity.
- **FEAT**: Extended API support with standard OpenAI-compatible and query parameters (`initial_prompt`, `vad_filter`, `word_timestamps`, `diarize`, `min_speakers`, `max_speakers`, `hf_token`, `max_line_width`, `max_line_count`).
- **TEST**: Verified coverage to 94.77% (exceeding 90% target) with 323/323 unit and integration tests passing.

## Release Notes v1.0.6
- **FIX**: Resolved a critical deadlock/livelock bug in the preemption scheduler where concurrent priority requests (like Language Detection) could race and permanently lock standard transcription tasks in a paused state.
- **STAB**: Implemented graceful fallback to Host CPU execution if the hardware unit registry is empty on startup, preventing worker threads from blocking indefinitely.
- **TEST**: Added exhaustive concurrency test suite covering 0, 1, 2, and 3 hardware unit configurations to prevent scheduling regressions.
- **STAB**: Addressed code hygiene and python linters: resolved unused arguments, inconsistent return statements, and local imports, achieving a perfect 10/10 pylint score and keeping coverage above 90% for all files.
- **OBS**: Registered queued priority tasks immediately upon arrival, exposing them on the telemetry dashboard with "Waiting for Priority Slot" status during resource contention.

## Release Notes v1.0.5
- **FEAT**: Implemented request-local file tracking registry for 100% reclamation of temporary files (uploaded media, standardized WAVs, isolated stems, and HQ prep files) on error/success.
- **STAB**: Hardened persistent diagnostic logging to survive container updates and app restarts.
- **STAB**: Added real-time log flushing and zero-caching headers for the log download endpoint.

## Release Notes v1.0.4
- **FIX**: Resolved "nn" (Nynorsk) language hallucination on silent or non-speech audio segments.
- **FEAT**: Implemented **High-Performance Batch Montage**. Consolidated all sampling targets into a single high-density montage with **30s Grid Padding**, enabling single-pass UVR isolation and reducing identification latency by up to 80%.
- **FEAT**: Added **Entropy-Adaptive Smart Search**. The engine now recovers from silent zones by scanning strides for peak signal energy.
- **FEAT**: Implemented **Confusion-Matrix Tie-Breaker**. Resolves ambiguities between similar language pairs (e.g., NO/NN) with weighted bias resolution.
- **FEAT**: Added **Fail-Safe Dual-Path VAD**. Automatically verifies speech and signal confidence on both isolated and raw audio segments. If confidence is <80% on isolated audio, it audits the raw signal and selects the one with higher confidence.
- **FEAT**: Implemented **Strategic Uniform Sampling** for language detection. The engine now samples up to 5 non-overlapping zones across the entire media to ensure representative identification.
- **FEAT**: Enhanced **Dynamic Chunk Sizing**. Samples now scale linearly (5m to 20m) to provide deep context for both short clips and feature films.
- **FEAT**: Implemented **Weighted Multi-Segment Voting**. Aggregates probabilities with confidence-weighted averaging to eliminate Nynorsk (nn) hallucinations.
- **STAB**: Integrated hardware-aware parallel limits (`ASR_PARALLEL_LIMIT_ACCEL`) to prevent VRAM/NPU congestion during concurrent tasks.
- **STAB**: Implemented **Unified Session Orchestration** and proactive resource reclamation. AI models are now automatically offloaded from VRAM/RAM immediately after task completion, ensuring long-term system stability.

## Release Notes v1.0.1
- **FIX**: Automatically clean up legacy storage leaks from version 1.0.0 in the preprocessing cache at startup.
- **FIX**: Resolved an issue where temporary vocal and instrumental stems were not properly cleaned up, causing `model_cache/preprocessing` to grow indefinitely.
- **FIX**: Warmup inference stems are now properly captured and deleted on startup.
- **FIX**: Temporary input files during in-memory segment processing are now cleaned up via `finally` blocks, preventing leaks on separator exceptions.
- **FIX**: Closed leaked file descriptors from `tempfile.mkstemp` in segmented isolation and added error-path cleanup for partial output files.
- **FIX**: Resolved relative path reconciliation issues for audio stems in Windows and Docker environments.
- **STAB**: Hardened FFmpeg error parsing for corrupted or zero-byte input files.
- **FFmpeg 8.1.0**: Upgraded to official static builds with optimized multithreading paths.

## Release Notes v1.0.0
- **Improved Language Detection**: Enhanced accuracy via **Squared Confidence Weighting** and full probability distribution analysis.
- **OpenAI/Bazarr Compatibility**: Added `/v1/audio/transcriptions` and `/v1/audio/translations` aliases with support for `response_format` and `file` parameters.
- **Swagger UI**: Integrated interactive documentation at `/docs`.
- **Race Condition Fix**: Implemented thread synchronization in Vocal Separation to prevent `NoneType` errors during concurrent processing.
- **Stability Fix**: Resolved audio truncation issues in FFmpeg preprocessing by switching to 16-bit PCM and optimizing audio filters.
- **Hallucination Filtering Fix**: Fixed a bug where filtered hallucinations would still appear in the final consolidated text field.
- **Code Quality**: Refactored core modules to achieve 100% Pylint compliance and verified >90% code coverage across the entire project.
- **High-Fidelity Pipeline**: Transitioned to **UVR/MDX-NET** for superior vocal isolation and natural signal mix-back (15%) for maximum Whisper confidence.
- **Precision Logging**: Unified professional `HH:MM:SS` formatting and dynamic step numbering across all modules.
- **Smart Detection**: Enhanced with iterative speech search and unified progress bars for language ID segments.
- **Responsive Pre-empting**: Multi-stage yielding mechanism covers both preprocessing and decoding stages for zero-wait detection.
- **Startup Warmup**: Eager model initialization and hardware verification.

