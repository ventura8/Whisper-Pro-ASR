# Performance Tuning

## Concurrency Safety First

Before throughput tuning, keep liveness and lock safety stable:

- Do not increase parallelism at the expense of queue fairness and scheduler liveness.
- Keep priority preemption behavior deterministic while allowing parallel execution across available hardware units.
- Validate tuning changes with concurrency stress tests, not only single-request benchmarks.

### Liveness-Related Runtime Guards

The scheduler uses queued waiting and cooperative yielding under contention. Tune throughput carefully and always validate with matching stress-test coverage so queued work eventually progresses.

## Default "Golden Standard"

| Setting | Value | Reason |
| --------- | ------- | -------- |
| Quantization | INT8 | Best accuracy |
| Batch Size | 1 | Safe for all systems |
| Beam Size | 5 | Highest accuracy (Max 4 if NPU load fails) |

## Profiles

| Goal | Batch | Beam | OpenVINO Mode | Notes |
| ------ | ------- | ------ | --------------- | ------- |
| **Quality** | 1-2 | 5 | `LATENCY` | Default, ~0.65x realtime |
| **Speed** | 4-8 | 1 | `THROUGHPUT` | Fastest, lower accuracy |
| **Low RAM** | 1 | 1 | `LATENCY` | For <16GB systems |

> [!NOTE]
> The OpenVINO mode and stream count are now automatically optimized based on your `ASR_BATCH_SIZE`. You can manually override them using `OV_PERFORMANCE_HINT` and `OV_NUM_STREAMS`.
>
> UVR preprocessing uses a dedicated OpenVINO configuration for stability under mixed GPU/NPU detect-language bursts:
>
> - OpenVINO UVR sessions are pinned to `num_streams=1`.
> - OpenVINO UVR cache is partitioned per accelerator family (`.../uvr/gpu`, `.../uvr/npu`) to avoid first-batch cross-device cache contention.

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

- **Intel ASR Chunking & Streaming**: For Intel Whisper runtime workloads, set `INTEL_ASR_CHUNK_DURATION` (default `300` seconds) to chunk audio processing. This keeps long jobs bounded and preserves continuous progress metrics. `INTEL-WHISPER` accelerates ASR on Intel GPUs; Intel NPUs are for vocal isolation, so ASR falls back to CPU when no Intel GPU is available.
- **UVR Preprocessing Chunking**: Set `UVR_CHUNK_DURATION` (default `600` seconds / 10 minutes) to segment vocal separation. This caps peak RAM utilization and enables periodic chunk-level progress updates on the dashboard.
- **Bazarr timeout**: Set to `36000` (10 hours) for high reliability.
- **RAM**: 32GB recommended for language detection on extremely large libraries.

## Troubleshooting

| Issue | Fix |
| ------- | ----- |
| NPU preprocessing fails or hangs | Use `ASR_PREPROCESS_DEVICE=CPU`, or reduce `ASR_BATCH_SIZE` to 1 / `ASR_BEAM_SIZE` to 4. The NPU is not an ASR execution target. |
| Model load fails | Reduce `ASR_BEAM_SIZE` to 4 |
| Build fails | Check disk space/RAM (~17GB needed) |
| Slow first run | Normal — OpenVINO compilation for Intel GPU/NPU preprocessing can take 2–5 min |

## 🛠 Hardware Acceleration (FFmpeg)

By default, media standardization runs on the CPU to ensure maximum compatibility. You can offload this to your GPU to reduce CPU load:

| Variable | Value | Hardware |
| :--- | :--- | :--- |
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

Models are immediately unloaded from memory when all active sessions complete and no tasks remain waiting in the queue. This is ideal for shared-resource environments where RAM must be reclaimed as fast as possible without interrupting work that is already queued.
Reclaim logs report process RSS explicitly (`RAM(RSS)`) and, on NVIDIA hosts with `nvidia-smi` available, include CUDA VRAM before/after plus delta.

### Idle Timeout

```yaml
environment:
  - MODEL_IDLE_TIMEOUT=300
```

When set to a positive value (in seconds), models remain warm in memory after the last session completes. A deferred `threading.Timer` is started after the last task finishes and only purges models after the timeout elapses. If new tasks arrive during the waiting period, the timer is automatically cancelled and rescheduled, keeping models warm for bursty workloads.

> [!TIP]
> Set `MODEL_IDLE_TIMEOUT=300` (5 minutes) for a good balance between memory efficiency and response latency. The deferred timer has zero CPU overhead while waiting (compared to the previous polling approach).

When `MODEL_IDLE_TIMEOUT > 0`, it takes precedence over `AGGRESSIVE_OFFLOAD`.

## ⚙️ ASR Backend Engines (ASR_ENGINE)

The service supports multiple ASR backend engines to run inference. You can configure this using the `ASR_ENGINE` environment variable. The following options are available:

- **`AUTO`** (Default): The configured default value of `ASR_ENGINE`. Always resolves to `FASTER-WHISPER`, for reproducible decoding across hardware.
  Hardware selection still chooses the task's unit; CUDA accelerates Faster-Whisper, while
  Intel/AMD units remain available for preprocessing. An explicit `ASR_DEVICE` constrains
  that unit selection, and an explicit engine selects an engine-specific backend.
- **`FASTER-WHISPER`**: Uses the CTranslate2 engine, and is what `AUTO` resolves to. This is the recommended choice for general CPU and NVIDIA CUDA environments, offering extremely fast processing and low memory footprint.
- **`INTEL-WHISPER`**: Uses the Intel Whisper engine (`IntelWhisperEngine`) on Intel Integrated/Arc GPUs. Intel NPUs accelerate vocal isolation only; ASR falls back to CPU when no Intel GPU is available.
- **`OPENAI-WHISPER`**: Uses the reference OpenAI Whisper Python backend.
- **`WHISPERX`**: Uses the WhisperX backend, supporting batch inference.

Invalid explicit `ASR_ENGINE` values fail startup with a validation error listing supported values.

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
  - /tmp/whisper:size=2G,mode=1777
```

`mode=1777` is required for restart-safe uploads: `docker compose restart`
remounts tmpfs, so permissions from the image layer are not retained.

### Sizing Guidance

- **Default (2GB)**: Sufficient for 95% of use cases (including ≤4h movies).
- **Large (4GB+)**: Recommended if you frequently process 4h+ movies or 4K videos with large upload sizes.
- **Dynamic Fallback**: If the free space in the RAM-disk drops below `WHISPER_TEMP_MIN_FREE_MB` (default `2048` MB), or if the estimated audio size exceeds the tmpfs capacity factoring in a 1.5× headroom multiplier, the service automatically falls back to persistent storage (`PERSISTENT_TEMP_DIR` / SSD) to prevent ENOSPC crashes.
