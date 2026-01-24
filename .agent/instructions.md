# AI Instructions
 
## Global Rules
- **Strict File Size Limit**: Any `.py` file MUST NOT exceed **500 lines**.
- **Logging Only**: Use `logger` (logging module) for all output. No `print()` calls allowed.
- **Hardware Agnostic**: Ensure code works across NPU, GPU (Intel/NVIDIA), and CPU. Use `modules.config` for device selection.
 
## Resource Management
- **Thread Compliance**: All multi-threaded components (FFmpeg, OpenVINO, ONNX Runtime) MUST respect the thread limits set in `modules.config` (`ASR_THREADS`, `PREPROCESS_THREADS`, `FFMPEG_THREADS`).
- **FFmpeg Parallelization**: Prefer parallelizing across segments (as seen in `language_detection.py`) rather than relying on high thread counts for single-file filters, as some FFmpeg filters are inherently single-threaded.
- **Library Patching**: If a third-party library (e.g., `audio-separator`) does not expose thread configuration, use monkey-patching on the underlying engine (e.g., `onnxruntime`) to enforce limits.
 
## Media Standardization
- **Uniform Specification**: All audio ingested MUST be converted to **16kHz, Mono, 16-bit PCM**.
- **Shared Constants**: Always use `utils.STANDARD_AUDIO_FLAGS` and `utils.STANDARD_NORMALIZATION_FILTERS` for FFmpeg commands to ensure pipeline consistency.
 
## Performance & Efficiency
- **Chunked Processing**: For identification or status tasks, ALWAYS favor chunked FFmpeg extraction (`-ss` / `-t`) from the source media over full-file normalization.
- **I/O Optimization**: Avoid using `soundfile` (`sf.read`) directly on non-WAV video containers as it causes expensive full-file probes. Use FFmpeg to extract small chunks to temporary WAV files first if signal analysis (RMS) is required.
- **Priority Yielding**: Maintain "Full-Pipeline Priority Yielding" logic—high-priority tasks (LD) must pause batch operations (ASR).
- **Silent Status**: Ensure hardware acceleration warnings (ONNX, OpenVINO) are suppressed in favor of custom authoritative status logs.
 
## Code Style
- Follow PEP 8 and use type hints.
- Maintain high test coverage (min 90%).
