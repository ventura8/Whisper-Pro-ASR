# API Reference

## Endpoints

### `GET /`
Health check. Returns `"Whisper ASR API is running"`.

### `GET /status`
Returns model status and version.

### `POST /detect-language`
Detect audio language with **High Priority**.

> [!NOTE]
> This endpoint uses the **Priority-Based Queueing** system. If a transcription is currently running, this request will safely pause the transcription, perform detection in ~1-2 seconds, and then resume the transcription automatically.

**Parameters**: `audio_file` (upload) OR `local_path` (server path)

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
