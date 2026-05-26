# Release v1.1.0 - Speaker Diarization, Custom Subtitle Layouts & Smart Lifecycle Management

This release adds major new features to the Whisper Pro ASR service, introducing high-fidelity speaker diarization, advanced transcription tuning parameters, customizable subtitle wrapping/layouts, and background idle resource reclamation.

## 🚀 Key Features & Enhancements

### 🗣️ Speaker Diarization (WhisperX & PyAnnote)
- **High-Fidelity Speaker Tracking**: Identify who said what using WhisperX alignment and PyAnnote speaker segmentation. Output formats (SRT, VTT, TXT, TSV) prepend speaker tags automatically (e.g. `[SPEAKER_00]: Hello world`).
- **Parallel Hardware Alignment & Caching Pools**: Models for diarization and alignment are cached in dedicated Pools (`_ALIGN_POOL` and `_DIARIZE_POOL`) per hardware accelerator slot, ensuring fast context-switching and optimal throughput.
- **Graceful Token Fallbacks**: Diarization is fully optional and requires a Hugging Face Hub token (`HF_TOKEN`). If the token is missing or diarization fails, the system automatically falls back to standard transcription.

### ⚙️ Customizable ASR Parameters
- **OpenAI-Compatible & Query Options**: Exposes `initial_prompt`, `vad_filter`, and `word_timestamps` to the Flask ASR endpoints (`/asr`, `/v1/audio/transcriptions`, `/v1/audio/translations`).
- **Context Guidance**: Custom prompts can guide Whisper in transcribing challenging jargon, acronyms, or specific multilingual contents.

### 📝 Subtitle Wrapping & Constraints
- **Layout Control**: Enforce precise formatting boundaries on subtitles with `max_line_width` and `max_line_count`.
- **Word Wrapping**: Automatically wraps lines using local formatting algorithms in `utils.py` and limits the output segments to the target line counts (e.g. max 2 lines at 42 chars per line).

### 🧠 Smart Model Lifecycle Management
- **Inactivity Reclamation**: Introduces `MODEL_IDLE_TIMEOUT` configuration (in seconds). A background daemon monitor thread watches the engine sessions and unloads the warm models from RAM/VRAM once the inactivity limit is exceeded, avoiding aggressive/instant offloading between consecutive API requests.

### 🧹 Code Quality, Testing, & PEP8 Compliance
- Authored comprehensive unit and integration tests covering the new routes, model forwarding, timeout monitors, formatters, and diarization.
- Maintained a strict **>90% test coverage** for all files (overall project coverage at **94.77%** with 323/323 unit and integration tests passing successfully).
- Fully validated to preserve a perfect **10.00/10** score on all repository code files under `pylint`.

---
*For deployment and configuration instructions, refer to the [README.md](README.md) or [ARCHITECTURE.md](docs/ARCHITECTURE.md).*
