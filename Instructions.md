# Whisper Pro ASR

> **CRITICAL DEVELOPER NOTE**: Only build or run the Docker image locally if you have compatible hardware (Intel NPU/GPU or NVIDIA GPU) mapped to the container. Otherwise, rely on the CI/CD pipeline and mocks for verification.

## AI & Development Rules
- **File Size Constraint**: Never have a `.py` file larger than **500 lines**. If a file grows beyond this limit, refactor and modularize into smaller files within the `modules/` directory.
- **Logging Standard**: Use the project's central logger (`logging`) instead of `print()` statements for all modules and scripts.
- **Thread Compliance**: All multi-threaded components (FFmpeg, OpenVINO, ONNX Runtime) MUST strictly respect the thread limits set in `modules/config.py` (`ASR_THREADS`, `ASR_PREPROCESS_THREADS`, `FFMPEG_THREADS`). Language detection runs serially (single worker) to ensure thread limits are not exceeded.
- **Media Standardization**: All audio ingested MUST be converted to **16kHz, Mono, 16-bit PCM**. Always use `utils.STANDARD_AUDIO_FLAGS` and `utils.STANDARD_NORMALIZATION_FILTERS` for FFmpeg commands to ensure pipeline consistency.
- **Efficiency Optimization**: For non-ASR tasks (identification/status), ALWAYS favor chunked FFmpeg extraction (`-ss` / `-t`) from the source media over full-file normalization. Avoid using `soundfile` (`sf.read`) directly on non-WAV video containers as it causes expensive full-file probes.


## Documentation Index

| Document | Description |
|----------|-------------|
| [docs/SETUP.md](docs/SETUP.md) | Installation & configuration |
| [docs/API.md](docs/API.md) | API endpoint reference |
| [docs/TUNING.md](docs/TUNING.md) | Performance optimization guide |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Technical module documentation |

## Key Features
- **Industrial Hardware Acceleration**: Native support for **Intel NPU**, **Intel iGPU**, and **NVIDIA CUDA**.
- **Hardware Compatibility Matrix**:
    | Engine | CPU | NVIDIA (CUDA) | Intel GPU | Intel NPU |
    | :--- | :---: | :---: | :---: | :---: |
    | **Vocal Separation** | ✅ | ✅ | ✅ | ✅ |
    | **Whisper ASR** | ✅ | ✅ | ❌ (CPU Fallback) | ❌ (CPU Fallback) |
- **Hybrid Device Support**: Simultaneously utilize multiple accelerators (e.g., Intel for isolation and NVIDIA for transcription).
- **Advanced Preprocessing Stack**: Integrated **UVR/MDX-NET** (Vocal Isolation) with hardware acceleration and dedicated thread pooling (`ASR_PREPROCESS_THREADS`).
- **OpenAI Compatible**: Native support for `/v1/audio/transcriptions` and `/v1/audio/translations` OpenAI API format.
- **Swagger Documentation**: Interactive API testing available at `/docs`.
- **Smart Voting Detection**: Robust language ID using probability aggregation with **Squared Confidence Weighting**. This prioritizes high-confidence segments to eliminate false positives in multilingual or noisy content. Density scales with file length (up to 25 samples).

- **Sequential Priority Queue**: Multiple high-priority requests (Language Detection) are automatically queued and processed one-by-one, ensuring hardware stability.
- **Deadlock-Free Yielding**: Ongoing transcription threads release hardware model locks during priority wait periods, allowing metadata tasks to pre-empt decoding instantly.
- **Robust Multi-Stage Cleanup**: Automated intermediate file management ensures that all UVR outputs, converted WAVs, and temporary files are strictly purged.
- **Unified Precision Logs**: Real-time logging with stable ETA estimation and professional `HH:MM:SS` formatting for all stages.
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
*Note: The test suite enforces 90%+ code coverage for all critical modules.*

## Release Notes v1.0.0
- **FFmpeg 8.0.1 (Huffman)**: Upgraded to official static builds with optimized multithreading paths.
- **Improved Language Detection**: Enhanced accuracy via **Squared Confidence Weighting** and full probability distribution analysis.
- **OpenAI/Bazarr Compatibility**: Added `/v1/audio/transcriptions` and `/v1/audio/translations` aliases with support for `response_format` and `file` parameters.
- **Swagger UI**: Integrated interactive documentation at `/docs`.
- **Race Condition Fix**: Implemented thread synchronization in Vocal Separation to prevent `NoneType` errors during concurrent processing.
- **Stability Fix**: Resolved audio truncation issues in FFmpeg preprocessing by switching to 16-bit PCM and optimizing audio filters.
- **Hallucination Filtering Fix**: Fixed a bug where filtered hallucinations would still appear in the final consolidated text field.
- **Code Quality**: Refactored core modules to achieve 100% Pylint compliance and verified >90% code coverage across the entire project.
- **High-Fidelity Pipeline**: Transitioned to **UVR/MDX-NET** for superior vocal isolation and natural signal mix-back (15%) for maximum Whisper confidence.
- **Precision Logging**: Unified professional `HH:MM:SS` formatting and dynamic step numbering across all modules.
- **Smart Detection**: Enhanced with iterative speech search and unified progress bars for language ID chunks.
- **Responsive Pre-empting**: Multi-stage yielding mechanism covers both preprocessing and decoding stages for zero-wait detection.
- **Startup Warmup**: Eager model initialization and hardware verification.

