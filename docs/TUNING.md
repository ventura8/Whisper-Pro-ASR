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
- Processing: 6h for 4h movie
- Bazarr timeout: Set to `36000` (10 hours)
- RAM: 32GB recommended for language detection

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
When set to a positive value (in seconds), models remain warm in memory after the last session completes. A background daemon thread monitors inactivity and only purges models after the timeout elapses. This is ideal for environments with bursty workloads where you want fast response times for subsequent requests within the timeout window.

> [!TIP]
> Set `MODEL_IDLE_TIMEOUT=300` (5 minutes) for a good balance between memory efficiency and response latency. The idle monitor thread has negligible CPU overhead (polling every 5 seconds).

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
- **Default (2GB)**: Sufficient for 95% of use cases (including 2h movies).
- **Large (4GB+)**: Recommended if you frequently process 4h+ movies or 4K videos with large upload sizes.
- **Dynamic Fallback**: If the RAM-disk fills up, Whisper Pro will automatically fallback to physical disk to prevent crashes.
