# Setup Guide

## Prerequisites
- Intel Core Ultra (Meteor Lake/Lunar Lake) with NPU, OR NVIDIA GPU (CUDA), OR generic CPU
- Windows 11 (WSL2) or Linux (Ubuntu 22.04+)
- Intel NPU drivers installed (for NPU acceleration)
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

## Speaker Diarization Setup
To enable speaker diarization (identifying who said what), you need a **Hugging Face token** with access to PyAnnote speaker segmentation models:

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Accept the license terms for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
3. Generate an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Set the `HF_TOKEN` environment variable in your `docker-compose.yml`:

```yaml
environment:
  - HF_TOKEN=hf_your_token_here
```

> [!IMPORTANT]
> **Without `HF_TOKEN`**, diarization requests will fall back to standard transcription (without speaker labels). The token is only required if you use `diarize=true` in API calls.

## Volume Mapping
Edit `docker-compose.yml`:
```yaml
volumes:
  - ./model_cache:/app/model_cache     # NPU compilation blobs + diarization models (Critical for fast reload)
  - ./state:/app/data                  # Task history, telemetry, and system logs
  - /mnt/nas/movies:/movies            # Your media (mapped to same path as in Bazarr)
  - /mnt/nas/tv:/tv
```

> [!TIP]
> The `model_cache` volume now also stores cached WhisperX alignment and PyAnnote diarization models. Mapping this volume avoids re-downloading these models on container restarts.

## SSD Protection
If running on an SSD, consider adding a `tmpfs` mount to minimize write wear. See `docs/TUNING.md` for details.

## Verify
```bash
docker compose logs -f
# Look for: "Model loaded successfully!"
```

## Troubleshooting
- **Model not loading on NPU**: Some NPU versions have memory limits for static shapes. If the model fails to load or the server crashes on startup, set `ASR_BEAM_SIZE=4` in `docker-compose.yml`.
- **Diarization not working**: Ensure `HF_TOKEN` is set and you have accepted the PyAnnote model license on Hugging Face.
- **Models consuming too much RAM when idle**: Set `MODEL_IDLE_TIMEOUT=300` to automatically unload models after 5 minutes of inactivity instead of keeping them resident.
- **Optimization**: Check `docs/TUNING.md` for performance profiles.
