# Document Update Skill

This skill automates the process of synchronizing all project documentation with the current state of the codebase.

## Objective
Review the whole current commit for documentation drift, then update `README.md`, all files in `docs/`, and all Mermaid diagrams in `.md` files to reflect recent architectural changes including Speaker Diarization (WhisperX), ASR parameter customization, Model Idle Timeout, and Subtitle Layout Customization.

Also enforce the Concurrency-First policy: concurrency correctness must be reflected consistently across user docs and agent governance docs.
Also enforce endpoint taxonomy consistency: `/asr` and `/v1/audio/...` are one standard ASR class; `/detect-language` and `/detectlang` are one priority language-ID class.

## Required Concurrency-First Checklist
- `README.md`: includes project-level concurrency priority statement.
- `Instructions.md`: includes mandatory concurrency checklist.
- `docs/CONCURRENCY.md`: canonical lock order and bounded-wait policy.
- `docs/ARCHITECTURE.md`: concurrency safety/liveness boundaries.
- `docs/API.md`: endpoint concurrency semantics.
- `docs/TUNING.md`: liveness-safe tuning guidance.
- `docs/SETUP.md`: concurrency verification commands.
- `docs/DOCKERHUB_DESCRIPTION.md`: concise reliability wording.
- `.agent/instructions.md` and relevant `.agent/skills/*.md`: concurrency-first governance alignment.
- `docs/API.md`, `README.md`, `docs/ARCHITECTURE.md`, `docs/CONCURRENCY.md`, `Instructions.md`: endpoint taxonomy and alias policy alignment.

## Procedure

### 0. Review the Full Current Commit
Before editing docs, inspect the full current commit diff so documentation and release notes are checked against every changed file, not just the obvious feature surface.

### 1. Update Mermaid Diagrams
Locate and update Mermaid diagrams in all project `.md` files to match the current architecture, including the diarization post-processing pipeline and idle timeout monitor thread.

### 2. Update `README.md`
*   Highlight the new **Speaker Diarization** feature (WhisperX alignment, diarization, speaker assignment).
*   Document new ASR parameters: `initial_prompt`, `vad_filter`, `word_timestamps`, `diarize`, `min_speakers`, `max_speakers`, `hf_token`.
*   Document subtitle customization: `max_line_width`, `max_line_count`.
*   Update Configuration Reference with `HF_TOKEN`, `MODEL_IDLE_TIMEOUT`, and `INITIAL_PROMPT` environment variables.
*   Update the **Hardware Orchestration** section to mention re-entrant locking.
*   Update the **Language Detection** section to describe the new global VAD + batch inference flow.

### 3. Update `docs/ARCHITECTURE.md`
*   Deep dive into the `model_lock_ctx` re-entrancy implementation.
*   Describe the `run_batch_language_detection` optimization.
*   Document the WhisperX diarization pipeline stages and caching pools (`_ALIGN_POOL`, `_DIARIZE_POOL`).
*   Document the `_monitor_idleness()` background thread and `MODEL_IDLE_TIMEOUT` lifecycle.

### 4. Update `docs/CONCURRENCY.md`
*   Explain how priority requests and re-entrant locks prevent deadlocks in high-load scenarios.
*   Document Model Idle Timeout as an alternative to `AGGRESSIVE_OFFLOAD`.

### 5. Update `docs/API.md`
*   Ensure the `/detect-language` and `/asr` endpoint documentation matches current status codes (400 for media errors) and the unified 16kHz WAV pipeline.
*   Document new `/asr` parameters: `diarize`, `min_speakers`, `max_speakers`, `hf_token`, `initial_prompt`, `vad_filter`, `word_timestamps`, `max_line_width`, `max_line_count`.
*   Document output formats include speaker labels when diarization is enabled.

### 6. Update `docs/DOCKERHUB_DESCRIPTION.md`
*   Synchronize with README updates (Speaker Diarization, new config vars, new features).

### 7. Update `docs/SETUP.md`
*   Document `HF_TOKEN` requirement for speaker diarization.
*   Add volume mapping guidance for persistent diarization model cache.

### 8. Update `docs/TUNING.md`
*   Document `MODEL_IDLE_TIMEOUT` as an alternative to `AGGRESSIVE_OFFLOAD` for memory management.
*   Document `INITIAL_PROMPT` for guiding transcription context.

### 9. Hardware Compatibility Matrix
Ensure a **Hardware Compatibility Matrix** is present and updated in `README.md`, `docs/ARCHITECTURE.md`, and `docs/DOCKERHUB_DESCRIPTION.md`. The matrix must accurately reflect current backend support for Vocal Separation, ASR Inference, and Speaker Diarization across CPU, NVIDIA, and Intel architectures.
