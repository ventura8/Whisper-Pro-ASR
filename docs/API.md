# API Reference

## Endpoints

## Endpoint Treatment Policy

Request handling is normalized into two execution classes:

- **Standard ASR class**: `/asr`, `/v1/audio/transcriptions`, `/v1/audio/translations`
  - Same scheduler priority (standard)
  - Same preprocessing/inference pipeline
  - `/v1/audio/translations` forces `task=translate`
- **Priority Language-ID class**: `/detect-language`, `/detectlang`
  - Same scheduler priority (high)
  - Same language-detection pipeline

Operationally, `/v1/audio/...` is treated as an OpenAI-compatible surface over the same ASR core behavior used by `/asr`.

### `GET /`
Health check and dashboard entry point. Returns JSON with service identity when called programmatically, or renders the HTML monitoring dashboard when accessed via browser (`Accept: text/html`).

### `GET /status`
Returns hardware pool status, active sessions, telemetry history, and version.

**Ordering Guarantee**: Returned task lists are sorted by task start/arrival time (`start_time`, then `task_id`) so dashboard and API consumers observe a stable FIFO chronology.

### `POST /detect-language`
Detect audio language with **High Priority**. All media inputs are automatically standardized to **16kHz Mono/Stereo WAV** before entering the voting consensus, ensuring maximum compatibility across Intel iGPU/NPU and NVIDIA backends.

`/detectlang` is a compatibility alias with identical behavior.

**Concurrency Semantics**:
- Priority requests can run in parallel across multiple accelerators when units are available or borrowed.
- Under hardware saturation, detect-language can preempt standard ASR work at cooperative checkpoints.
- Priority synchronization waits are unbounded by design; requests remain queued until preemption handoff and hardware become available.

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

# Alias (same behavior)
curl -X POST -F "audio_file=@movie.mp4" http://localhost:9000/detectlang
```

### `POST /asr`
Transcribe audio to SRT/VTT/JSON with optional speaker diarization. All incoming media is automatically standardized to **16kHz Mono WAV** via an optimized FFmpeg pipeline before inference.

**Concurrency Semantics**:
- ASR tasks follow FIFO order within standard-priority tier.
- ASR can be temporarily paused/resumed to honor queued high-priority detect-language requests.
- Cooperative preemption checkpoints exist before vocal separation, during HQ-prep FFmpeg progress, and before inference starts.
- Hardware acquisition waits until a unit is available; scheduler timeouts do not terminate queued work.

**Input Precedence**:
- If `local_path` is readable inside the service container (volume mapping works), the request uses that path directly and skips upload materialization.
- If `local_path` is not readable/mapped, the service falls back to the uploaded payload (`audio_file`/`file`) when present.

**Parameters**:
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `audio_file` | file | - | Binary upload |
| `local_path` | string | - | Server file path (faster) |
| `task` | string | `transcribe` | `transcribe` or `translate` |
| `language` | string | auto | Source language (`en`, `es`, etc.) |
| `output` | string | `srt` | `srt`, `vtt`, `txt`, `tsv`, or `json` |
| `initial_prompt` | string | config | Context prompt to guide transcription |
| `vad_filter` | bool | `true` | Enable Voice Activity Detection filtering |
| `word_timestamps` | bool | `false` | Include word-level timestamps |
| `subtitle_highlight_words` | bool | `false` | Highlight active word in SRT/VTT (enables `word_timestamps` automatically) |
| `diarize` | bool | `false` | Enable speaker diarization (WhisperX) |
| `min_speakers` | int | - | Minimum expected speakers (diarization) |
| `max_speakers` | int | - | Maximum expected speakers (diarization) |
| `hf_token` | string | config | Hugging Face token (required for diarization) |
| `max_line_width` | int | - | Max characters per subtitle line |
| `max_line_count` | int | - | Max lines per subtitle block |

**Speaker Diarization**:
When `diarize=true`, the service runs the WhisperX post-processing pipeline:
1. **Alignment**: Aligns transcription segments to audio using `whisperx.align`.
2. **Diarization**: Identifies speakers using `whisperx.diarization.DiarizationPipeline` (requires `HF_TOKEN`).
3. **Speaker Assignment**: Maps speaker IDs to segments via `whisperx.assign_word_speakers`.

Output formats (SRT, VTT, TXT, TSV) will include speaker labels (e.g., `[SPEAKER_00]: Hello world`).

**Subtitle Customization**:
Use `max_line_width` and `max_line_count` to control subtitle layout for SRT and VTT formats:
- `max_line_width=42` wraps text at 42 characters per line.
- `max_line_count=2` limits each subtitle block to 2 lines maximum.

**Subtitle Filename & Plex AI Tagging**:
All subtitle responses are returned with a structured filename of the form `<source>.<language>-ai.<format>` (e.g. `movie.en-ai.srt`, `episode.es-ai.vtt`). The `-ai` regional suffix uses the ISO 3166-1 country code for **Anguilla** (`AI`), which Plex maps via its regional layout parser to display as **`<Language> (AI)`** — e.g. `English (AI)`, `Spanish (AI)`. This correctly labels both transcriptions and translations in all target languages without falling back to `xx (Unknown)`.

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

# With speaker diarization
curl -X POST -F "audio_file=@meeting.mp3" "http://localhost:9000/asr?diarize=true&min_speakers=2&max_speakers=5"

# With custom subtitles
curl -X POST -F "audio_file=@video.mp4" "http://localhost:9000/asr?output=vtt&max_line_width=42&max_line_count=2"

# With ASR tuning
curl -X POST -F "audio_file=@video.mp4" "http://localhost:9000/asr?initial_prompt=Medical%20terminology&vad_filter=true&word_timestamps=true"
```

### `POST /v1/audio/translations`
OpenAI-compatible translation endpoint. Behaves identically to `/asr` with `task=translate`. Accepts the same parameters.

### `POST /v1/audio/transcriptions`
OpenAI-compatible transcription endpoint. Behaves identically to `/asr` with `task=transcribe` defaults and accepts equivalent parameters.

---

## System Management Endpoints

### `GET /analytics`
Returns cumulative and daily breakdown of task counts and durations separated by category (`asr`, `detectlang`, and `audio`). When accessed via browser (`Accept: text/html`), renders a dedicated analytics dashboard with interactive charts.

```bash
# JSON response
curl http://localhost:9000/analytics

# HTML dashboard (open in browser)
http://localhost:9000/analytics
```

#### JSON Response Schema
The response contains structured cumulative and daily statistics. Each breakdown segments tasks and processing durations (seconds) into three categories corresponding to endpoint scopes:
- `asr`: Transcription tasks triggered via `/asr`.
- `detectlang`: Language ID tasks triggered via `/detect-language`.
- `audio`: OpenAI-compatible transcription or translation calls under `/v1/audio/...`.

**Example Payload**:
```json
{
  "cumulative": {
    "all_time": 150.0,
    "today": 150.0,
    "this_month": 150.0,
    "this_year": 150.0,
    "count_all_time": 3,
    "count_today": 3,
    "asr": {
      "count": 1,
      "duration": 60.0
    },
    "detectlang": {
      "count": 1,
      "duration": 40.0
    },
    "audio": {
      "count": 1,
      "duration": 50.0
    }
  },
  "daily": {
    "2026-06-21": {
      "count": 3,
      "duration": 150.0,
      "asr": {
        "count": 1,
        "duration": 60.0
      },
      "detectlang": {
        "count": 1,
        "duration": 40.0
      },
      "audio": {
        "count": 1,
        "duration": 50.0
      }
    }
  }
}
```


### `GET/POST /settings`
View or update service settings at runtime.

**GET**: Returns current configuration values (`ASR_MODEL`, `ASR_DEVICE`, `TELEMETRY_RETENTION_HOURS`).

**POST** (JSON body):
| Field | Type | Description |
|:---|:---|:---|
| `ASR_MODEL` | string | Update the active model (triggers reload) |
| `ASR_DEVICE` | string | Update inference target |
| `telemetry_retention_hours` | int | Telemetry history retention |
| `log_retention_days` | int | Log file retention period |

```bash
# View settings
curl http://localhost:9000/settings

# Update model
curl -X POST -H "Content-Type: application/json" -d '{"ASR_MODEL": "Systran/faster-whisper-large-v3"}' http://localhost:9000/settings
```

### `GET /history`
Retrieves the full list of completed and active tasks from persistent storage.

```bash
curl http://localhost:9000/history
```

### `POST /system/history/clear`
Purges all task records from the history manager.

```bash
curl -X POST http://localhost:9000/system/history/clear
```

### `POST /system/cleanup`
Manually triggers removal of old temporary audio files and transient assets.

```bash
curl -X POST http://localhost:9000/system/cleanup
```

### `GET /logs/download`
Downloads the system log file (`whisper_pro.log`) with forced flush-to-disk and zero-caching headers for guaranteed freshness. The log is read atomically into memory before the response is sent, preventing `RuntimeError: Response content longer than Content-Length` failures that could occur when the log file is written to during an active streaming download.

```bash
curl -O http://localhost:9000/logs/download
```

### `GET /help`
API discovery endpoint. Returns a list of all available endpoints and a link to the Swagger documentation.

```bash
curl http://localhost:9000/help
```

---

## Bazarr Integration
1. Settings → Providers → Whisper
2. Endpoint: `http://<IP>:9000`
3. Read Timeout: `36000` (for long movies)

## Subtitle Edit Integration
1. Video → Audio-to-text (Whisper)
2. Provider: `OpenAI / Custom`
3. URL: `http://<IP>:9000/v1/audio/transcriptions`
