# Release v1.1.3 - Incremental Subtitle Updates, Intel Engine ASR Chunking, UVR Progress Patching & UI Modularization

This release introduces significant optimizations for processing very long media files, including O(1) incremental subtitle updates, chunked transcription for the Intel OpenVINO engine, real-time chunk-based UVR progress patching, robust temporary storage fallback logic, and a modular refactoring of the dashboard UI.

## 🚀 Key Improvements & Bug Fixes

### ⚡ O(1) Incremental Subtitle Updates
- **Performance Optimization**: Refactored the live transcription progress pipeline in `model_manager.py` and `routes_asr.py` to append pre-formatted SubRip (SRT) blocks incrementally (reducing live update complexity to $O(1)$) using the new `utils.format_single_srt_block()` helper. This eliminates the CPU-heavy $O(N^2)$ rebuilding of the entire SRT stream from all accumulated segments on every new segment, preventing performance bottlenecks and memory bloat on large media files.

### 🧠 Intel Engine ASR Chunking & Streaming
- **Robustness for Long Movies**: Refactored `IntelWhisperEngine` in `modules/inference/intel_engine.py` to transcribe audio in consecutive chunks determined by `INTEL_ASR_CHUNK_DURATION` (default 300 seconds) instead of a single-pass inference which was prone to execution failures on long files.
- **Dynamic VAD and Auto-Detection**: Uses Voice Activity Detection (VAD) timestamps to dynamically find optimal chunk split points (`find_split_points()`). The engine auto-detects and locks the source language on the first chunk to ensure linguistic consistency. Quiet/silent chunks are skipped automatically, and speech-present chunks are masked with zeros outside speech regions to preserve temporal alignment.

### 📊 UVR Vocal Separation Chunk Progress
- **Real-Time Progress Tracking**: Patched the UVR vocal separation process dynamically on the scheduler to compute and emit real-time chunk progress status (e.g. `Vocal Separation (Chunk X/Y | A/B)`) according to `UVR_CHUNK_DURATION` (default 600 seconds). This eliminates long visual hangs during preprocessing on large files and keeps the dashboard progress bar fluid.

### 🗄 Graceful Storage Fallback & Protection
- **Enhanced SSD/RAM-Disk Protection**: Increased `WHISPER_TEMP_MIN_FREE_MB` default threshold to `2048` MB.
- **Graceful Storage Headroom**: Applied a 1.5x headroom multiplier to the estimated WAV size check. The system now performs an early fallback to persistent (SSD) storage if the RAM-disk lacks sufficient breathing room, preventing `ENOSPC` (no space left on device) crashes during concurrent processing of long files.

### 🎨 Modular Dashboard UI Architecture
- **Maintainability Refactoring**: Deconstructed the massive monolithic `dashboard.html` into a cleaner layout under a dedicated `templates/` directory containing separate `dashboard.html`, `dashboard.css`, and modular JavaScript state and utility files (`dashboard_state.js`, `dashboard_utils.js`, `dashboard_charts.js`, and `dashboard_main.js`).
- **Telemetry Enhancements**: Real-time task progress tracks and calculates estimated transcription speed using early audio duration extraction.

## 🧪 Full Verification & Validation
- **100% Pass Rate**: Passed all unit and integration tests successfully, including the newly added OpenVINO engine chunking coverage tests.
- **Pylint**: Maintained a perfect **10.00/10** score on all repository code files with zero suppression directives.
- **Coverage**: Maintained strong project test coverage, exceeding the 90% build-gate threshold.

---
*For deployment and configuration instructions, refer to the [README.md](../../README.md) or [ARCHITECTURE.md](../ARCHITECTURE.md).*
