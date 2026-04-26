# API Reference

## Endpoints

### `GET /`
Health check. Returns JSON with service identity.

### `GET /status`
Returns hardware pool status, active sessions, and version.

### `POST /detect-language`
Detect audio language with **High Priority**. Utilizes a specialized **16kHz Stereo Batch Montage** for maximum hardware efficiency across Intel iGPU/NPU and NVIDIA backends. Supports all media containers (MKV, AVI, MP4, etc.) without full-file standardization.

**Parameters**: `audio_file` (upload) OR `local_path` (server path)

**Success Response (JSON)**:
| Field | Type | Description |
|:---|:---|:---|
| `detected_language` | string | ISO 639-1 code (e.g. `en`) |
| `language_name` | string | Full English name (e.g. `english`) |
| `confidence` | float | 0.0 to 1.0 confidence score |
| `voting_details` | dict | Weights for all detected candidates |

**Observability**: Task status is visible on the dashboard with a `translate` icon and `/detect-language` label.

```bash
curl -X POST -F "audio_file=@movie.mp4" http://localhost:9000/detect-language
```

### `POST /asr`
Transcribe audio to SRT/JSON.

**Parameters**:
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `audio_file` | file | - | Binary upload |
| `local_path` | string | - | Server file path (faster) |
| `task` | string | `transcribe` | `transcribe` or `translate` |
| `language` | string | auto | Source language (`en`, `es`, etc.) |
| `output` | string | `srt` | `srt` or `json` |
| `batch_size` | int | config | Override batch size |

**Observability**: 
- **Live Stream**: Monitor progress via real-time SRT stream on the dashboard.
- **Persistence**: Completed transcriptions are saved to persistent history for download.
- **Performance Auditing**: JSON responses include a `performance` object with `queue_sec`, `isolation_sec`, and `inference_sec`.
- **Iconography**: Identified on the dashboard with a `record_voice_over` icon and `/asr` label.

**Error Codes**:
*   `400`: Malformed request or media format standardization failed.
*   `503`: Inference engine unavailable or warming up.

```bash
# Local file (fast)
curl -X POST "http://localhost:9000/asr?local_path=/movies/avatar.mkv&language=en"

# Upload
curl -X POST -F "audio_file=@video.mp4" http://localhost:9000/asr
```

## Bazarr Integration
1. Settings → Providers → Whisper
2. Endpoint: `http://<IP>:9000`
3. Read Timeout: `36000` (for long movies)

## Subtitle Edit Integration
1. Video → Audio-to-text (Whisper)
2. Provider: `OpenAI / Custom`
3. URL: `http://<IP>:9000/v1/audio/transcriptions`
