# Performance Tuning

## Default "Golden Standard"
| Setting | Value | Reason |
|---------|-------|--------|
| Quantization | INT8 | Best accuracy |
| Batch Size | 1 | Safe for all systems |
| Beam Size | 5 | Highest accuracy (Max 4 if NPU load fails) |

## Profiles

| Goal | Batch | Beam | OpenVINO Mode | Notes |
|------|-------|------|---------------|-------|
| **Quality** | 1-2 | 5 | `LATENCY` | Default, ~0.65x realtime |
| **Speed** | 4-8 | 1 | `THROUGHPUT` | Fastest, lower accuracy |
| **Low RAM** | 1 | 1 | `LATENCY` | For <16GB systems |

> [!NOTE]
> The OpenVINO mode and stream count are now automatically optimized based on your `ASR_BATCH_SIZE`. You can manually override them using `OV_PERFORMANCE_HINT` and `OV_NUM_STREAMS`.

## Changing Quantization
Edit `Dockerfile`:
```dockerfile
# INT8 (default)
--weight-format int8

# INT4 (faster, less accurate)
--weight-format int4
```
Then rebuild: `docker compose up -d --build`

## Long Movies (4h+)
- **Intel ASR Chunking & Streaming**: For OpenVINO engine transcription, set `INTEL_ASR_CHUNK_DURATION` (default `300` seconds) to chunk audio processing. This prevents execution hangs and out-of-memory errors on massive files while maintaining continuous progress metrics.
- **UVR Preprocessing Chunking**: Set `UVR_CHUNK_DURATION` (default `600` seconds / 10 minutes) to segment vocal separation. This caps peak RAM utilization and enables periodic chunk-level progress updates on the dashboard.
- **Bazarr timeout**: Set to `36000` (10 hours) for high reliability.
- **RAM**: 32GB recommended for language detection on extremely large libraries.

## Troubleshooting
| Issue | Fix |
|-------|-----|
| NPU hangs | Reduce `ASR_BATCH_SIZE` to 1 or `ASR_BEAM_SIZE` to 4 |
| Model load fails | Reduce `ASR_BEAM_SIZE` to 4 |
| Build fails | Check disk space/RAM (~17GB needed) |
| Slow first run | Normal - NPU compilation takes 2-5 min |

## 🛠 Hardware Acceleration (FFmpeg)

By default, media standardization runs on the CPU to ensure maximum compatibility. You can offload this to your GPU to reduce CPU load:

| Variable | Value | Hardware |
|:---|:---|:---|
| `FFMPEG_HWACCEL` | `cuda` | NVIDIA GPUs |
| `FFMPEG_HWACCEL` | `qsv` | Intel GPUs (Recommended) |
| `FFMPEG_HWACCEL` | `vaapi` | AMD / Generic Linux |

## 🧩 Granular Resource Orchestration

As of v1.0.4, you can control exactly how many hardware units the service utilizes:

- **`MAX_CUDA_UNITS`**: Caps NVIDIA GPUs utilized.
- **`MAX_GPU_UNITS` / `MAX_NPU_UNITS`**: Caps Intel Silicon units.
- **`MAX_CPU_UNITS`**: Caps concurrent multi-threaded CPU tasks (VAD, FFmpeg). Set to `AUTO` to let the system decide based on your core count.

## 🧠 Model Lifecycle Management

The service provides two strategies for managing model memory when the system is idle:

### Aggressive Offload (Default)
```yaml
environment:
  - AGGRESSIVE_OFFLOAD=true
```
Models are immediately unloaded from memory when all active sessions complete. This is ideal for shared-resource environments where RAM must be reclaimed as fast as possible.

### Idle Timeout
```yaml
environment:
  - MODEL_IDLE_TIMEOUT=300
```
When set to a positive value (in seconds), models remain warm in memory after the last session completes. A deferred `threading.Timer` is started after the last task finishes and only purges models after the timeout elapses. If new tasks arrive during the waiting period, the timer is automatically cancelled and rescheduled, keeping models warm for bursty workloads.

> [!TIP]
> Set `MODEL_IDLE_TIMEOUT=300` (5 minutes) for a good balance between memory efficiency and response latency. The deferred timer has zero CPU overhead while waiting (compared to the previous polling approach).

When `MODEL_IDLE_TIMEOUT > 0`, it takes precedence over `AGGRESSIVE_OFFLOAD`.

## 🗣 Transcription Tuning

### Initial Prompt
Use `INITIAL_PROMPT` to provide context that guides the transcription model:
```yaml
environment:
  - INITIAL_PROMPT=This video contains speech in English with technical terminology.
```
This can also be overridden per-request using the `initial_prompt` query parameter.

### VAD Filter
The `vad_filter` parameter (default: `true`) enables Voice Activity Detection to suppress silence and reduce hallucinations. You can disable it per-request with `vad_filter=false` if you need timestamps for silent segments.

### Word Timestamps
Enable `word_timestamps=true` in API calls to get word-level timing information in JSON output. This is useful for precise subtitle alignment and karaoke-style displays.

## SSD Protection (RAM-disk)

For high-volume transcription, it is highly recommended to use a `tmpfs` mount to protect your SSD from write wear.

### Configuration
In your `docker-compose.yml`, add:
```yaml
environment:
  - WHISPER_TEMP_DIR=/tmp/whisper
tmpfs:
  - /tmp/whisper:size=2G
```

### Sizing Guidance
- **Default (2GB)**: Sufficient for 95% of use cases (including ≤4h movies).
- **Large (4GB+)**: Recommended if you frequently process 4h+ movies or 4K videos with large upload sizes.
- **Dynamic Fallback**: If the free space in the RAM-disk drops below `WHISPER_TEMP_MIN_FREE_MB` (default `2048` MB), or if the estimated audio size exceeds the tmpfs capacity factoring in a 1.5× headroom multiplier, the service automatically falls back to persistent storage (`PERSISTENT_TEMP_DIR` / SSD) to prevent ENOSPC crashes.
