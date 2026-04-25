# 🚀 Whisper Pro ASR v1.0.1 - Stability & Reliability Update

We are proud to announce the **v1.0.1** release of **Whisper Pro ASR**. This update focuses on long-term system stability, resource management, and reliability. 

This version addresses critical storage and memory leaks identified in the preprocessing pipeline, ensuring the service can run indefinitely in high-load production environments without manual cache maintenance.

## 🌟 v1.0.1 Highlights

### 🧹 Industrial-Grade Cache Management
Fixed a critical issue where intermediate stems from the UVR/MDX-NET vocal isolation process were being left in the `model_cache/preprocessing` directory. The system now guarantees absolute cleanup of all temporary files, even when processing fails or is interrupted. Additionally, version 1.0.1 now **automatically purges any legacy orphaned files** left over from version 1.0.0 during the initial startup sequence.

### 🛡️ Resource Leak Prevention
Implemented comprehensive `finally` block coverage across the entire inference lifecycle. This ensures that:
- **File Descriptors**: All temporary file handles are closed immediately.
- **Temporary Files**: `segment_in_*.wav` and other transient assets are deleted even if an exception occurs during isolation.
- **Warmup Artifacts**: Model warmup stems are now properly captured and purged at startup.

### 📈 Enhanced Test Coverage
The test suite has been expanded to cover over 90% of all code paths, including edge-case error handling for FFmpeg, soundfile, and hardware-specific fallback scenarios.

---

## 🛠️ Deployment

The recommended way to deploy **Whisper Pro ASR** is via `docker-compose.yml`. This allows you to easily manage hardware mapping and persistent caches.

### **Quick Install (Default)**
```bash
git clone https://github.com/ventura8/Whisper-Pro-ASR.git
cd Whisper-Pro-ASR
docker compose up -d
```

### **Full docker-compose.yml Example**
```yaml
services:
  whisper-asr:
    image: ventura8/whisper-pro-asr:latest
    container_name: whisper-asr
    restart: unless-stopped
    ports:
      - "9000:9000"
    volumes:
      - ./model_cache:/app/model_cache
    
    # --- HARDWARE MAPPING ---
    # 1. Intel GPU/NPU: Uncomment to enable
    # devices:
    #   - /dev/dri:/dev/dri # Intel iGPU / Arc
    #   - /dev/accel/accel0:/dev/accel/accel0 # Intel NPU
    #   - /dev/dxg:/dev/dxg # Windows/WSL2 GPU mapping
    
    # 2. NVIDIA GPU: Uncomment to enable
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: 1
    #           capabilities: [gpu]
```

## 📝 Changelog Summary
- **FIX**: Automatically clean up legacy storage leaks from version 1.0.0 in the preprocessing cache at startup.
- **FIX**: Resolved an issue where temporary vocal and instrumental stems were not properly cleaned up, causing `model_cache/preprocessing` to grow indefinitely.
- **FIX**: Warmup inference stems are now properly captured and deleted on startup.
- **FIX**: Temporary input files during in-memory segment processing are now cleaned up via `finally` blocks, preventing leaks on separator exceptions.
- **FIX**: Closed leaked file descriptors from `tempfile.mkstemp` in segmented isolation and added error-path cleanup for partial output files.
- **FIX**: Resolved relative path reconciliation issues for audio stems in Windows and Docker environments.
- **STAB**: Hardened FFmpeg error parsing for corrupted or zero-byte input files.

---
*For a full list of environment variables and advanced tuning guides, please refer to the [README.md](https://github.com/ventura8/Whisper-Pro-ASR/blob/main/README.md).*
