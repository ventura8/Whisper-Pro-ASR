# Concurrency & Resource Orchestration Skill

This skill documents how to test, debug, and verify the thread-local re-entrant lock pools and priority request preemption pipelines under high task loads.

## Objective
Verify that standard transcription requests (`/asr`) safely yield resources to high-priority language detection tasks (`/detect-language`) without introducing deadlocks or leaking thread-local file storage.

Concurrency correctness is Priority 1. Any optimization that weakens liveness guarantees is considered a regression.

### Mandate: Preserve Helpful Comments
Code clarity around synchronization, lock ordering, and state transitions is essential for maintainability. **Never delete comments that explain logic**, even when optimizing for line count. Comments about why pauses are requested, why waits remain queued, or how generation tokens enforce atomicity must be preserved.

---

## Architectural Mechanisms

### 1. Re-entrant Locking (`model_lock_ctx`)
Ensures that a request context can obtain the model lock once and share it seamlessly across:
1.  **Vocal Separation (UVR)**
2.  **Language Detection (Whisper)**
3.  **ASR Inference (Whisper)**
4.  **Speaker Diarization (PyAnnote/WhisperX)**

This prevents a standard task from being preempted mid-pipeline, avoiding state corruption.

### 2. Preemption & Resumption
- When a priority task arrives, standard tasks yield model access and sleep.
- Paused tasks transition their status to `"queued"` with a `"Paused for Priority Task"` stage.
- Priority tasks can execute in parallel across multiple available/borrowed hardware units.
- ASR preprocessing must expose cooperative checkpoints during HQ-prep FFmpeg progress so long preparation windows do not block detect-language preemption.
- On completion, the targeted unit's `unit_sync[unit_id]["resume_event"]` is fired to resume the paused pipeline exactly where it yielded.
- Execution gates must remain unit-scoped; shared scheduler events are compatibility mirrors only.

### 3. File Hygiene
Ensures thread context registers and deletes all temporary audio WAV stems (standardized WAVs, UVR isolated stems) under a `finally` clause, guaranteeing a 100% deletion rate.

---

## Verification & Debugging Procedure

### 0. Mandatory Safety Checks
- Confirm lock-order compliance for modified paths.
- Confirm all new waits follow current policy and do not introduce timeout-based request failures in priority/preemption paths.
- Confirm a regression test exists for each changed liveness pathway.

### 1. Simulate Concurrency Races
Run integration tests checking priority request preemption behavior:
```bash
python3 -m pytest tests/inference/priority/test_priority_concurrency.py \
                  tests/inference/priority/test_priority_fifo_ordering.py \
                  tests/inference/priority/test_priority_stage_preemption.py
```

### 2. Assert Priority Parallelism and Unit Resume
- Verify that concurrent priority requests can register and run across different available/borrowed units.
- Assert that paused tasks resume processing when their targeted unit is resumed.

### 3. Verify Temp Storage Cleanup
- Verify that mock tasks registering files via `tracked_files` successfully clear all listed paths upon execution exit.
- Check test results in `tests/inference/priority/test_priority_concurrency.py` targeting file cleanup routines.
