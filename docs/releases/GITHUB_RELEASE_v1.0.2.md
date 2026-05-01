# Release v1.0.2 — RAM Optimization & Storage Cleanup

This update focuses on significantly reducing the memory footprint of Whisper Pro ASR and finalizing the storage cleanup strategy for users migrating from version 1.0.0.

## 🚀 Key Improvements

### 🧠 RAM Usage Optimization (Phase 1 & 2)
- **Streamed Ingestion**: Completely refactored the file upload pipeline. Media files are now streamed directly from the network to disk, eliminating large RAM spikes previously caused by in-memory buffering.
- **Quantized Inference Defaults**: Standardized on `int8` quantization for CPU and NPU backends, halving the memory required for model weights while maintaining high transcription accuracy.
- **Proactive Memory Recovery**: Integrated explicit garbage collection and hardware cache clearing (CUDA/NPU) into the transcription lifecycle. Memory is now returned to the OS immediately after processing.
- **Dynamic UVR Offloading**: The heavy vocal isolation engine (UVR/MDX-NET) is now automatically offloaded from memory when idle, freeing up several hundred megabytes of RAM/VRAM for other system tasks.
- **ONNX Runtime Tuning**: Applied memory-reuse and pattern-matching optimizations to the preprocessing sessions to reduce peak allocation overhead.

### 🧹 Storage & SSD Endurance
- **Extra-Large File Optimization**: Implemented a dynamic persistent storage fallback (SSD/HDD) for uploads that exceed `tmpfs` capacity. The system now automatically migrates large media ingestion to disk, ensuring stability on low-RAM devices during massive file transfers.
- **Legacy Cleanup**: Version 1.0.2 now automatically detects and purges orphaned 'preprocessing' directories and persistent temp artifacts in the persistent `./model_cache` volume.
- **Refined Tmpfs Usage**: Optimized the interaction between streamed uploads and `tmpfs` storage to ensure maximum SSD protection without risking system stability.

## 🛠️ Internal Changes
- **Version**: Bumped to `1.0.2`.
- **Backend**: Enhanced `PreprocessingManager` with model lifecycle controls.
- **Routes**: Optimized `_handle_upload` with zero-copy stream-to-disk logic.

## 📦 Deployment
Update your `docker-compose.yml` to use the `1.0.2` tag (or `latest`) and restart the service:
```bash
docker compose pull
docker compose up -d
```

---
*Optimized for high-concurrency and long-duration media processing.*
