# Setup Guide

## Prerequisites
- Intel Core Ultra (Meteor Lake/Lunar Lake) with NPU
- Windows 11 (WSL2) or Linux (Ubuntu 22.04+)
- Intel NPU drivers installed
- Docker

## Installation
### Method 1: Docker Hub (Recommended)
```bash
docker run -d --name whisper-pro-asr -p 9000:9000 --device /dev/accel/accel0 --device /dev/dri ventura8/whisper-pro-asr
```

### Method 2: Local Build
```bash
git clone https://github.com/ventura8/Whisper-Pro-ASR.git
cd Whisper-Pro-ASR
docker compose up -d --build
```

**Note**: The system automatically detects and utilizes NVIDIA CUDA, Intel NPU, or Intel GPU. Manual device selection (`ASR_DEVICE`) is now optional.

First build exports model to INT8 (~5-10 min, ~4GB RAM).

## 3. Configuration & Device Selection
The service utilizes **Autonomous Hardware Sensing**. It will prioritize accelerators in the following order:
1. **NVIDIA CUDA**
2. **Intel NPU**
3. **Intel GPU (Arc/iGPU)**
4. **CPU (Fallback)**

**Preprocessing Toggles:**
-   `ENABLE_VOCAL_SEPARATION=true`: isolates vocals using UVR/MDX-NET (recommended for accuracy).

## Volume Mapping
Edit `docker-compose.yml`:
```yaml
volumes:
  - ./model_cache:/app/model_cache     # NPU compilation blobs (Critical for fast reload)
  - /mnt/nas/movies:/movies            # Your media (mapped to same path as in Bazarr)
  - /mnt/nas/tv:/tv
```

## Verify
```bash
docker compose logs -f
# Look for: "Model loaded successfully!"
```

## Troubleshooting
- **Model not loading on NPU**: Some NPU versions have memory limits for static shapes. If the model fails to load or the server crashes on startup, set `ASR_BEAM_SIZE=4` in `docker-compose.yml`.
- **Optimization**: Check `docs/TUNING.md` for performance profiles.
