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
