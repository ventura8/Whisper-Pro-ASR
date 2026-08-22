# Concurrency & Resource Orchestration Skill

This skill documents how to test, debug, and verify the nesting-safe hardware-claim pools (a single global permit semaphore gating concurrent claims, with concrete unit assignment tracked separately via `STATE.hw_pool`, plus non-locking "_direct" sub-stage entry points) and priority request preemption pipelines under high task loads.

## Objective

Verify that standard transcription requests (`/asr`) safely yield resources to high-priority language detection tasks (`/detect-language`) without introducing deadlocks or leaking thread-local file storage.

Concurrency correctness is Priority 1. Any optimization that weakens liveness guarantees is considered a regression.

### Mandate: Preserve Helpful Comments

Code clarity around synchronization, lock ordering, and state transitions is essential for maintainability. **Never delete comments that explain logic**, even when optimizing for line count. Comments about why pauses are requested, why waits remain queued, or how generation tokens enforce atomicity must be preserved.

---

## Architectural Mechanisms

### 1. Nesting-Safe Locking (`model_lock_ctx`)

The model lock (`STATE.model_lock`, a plain `threading.Semaphore`) is not reentrant. Instead, a request context obtains it once at the top level, and internal sub-stages avoid re-acquiring it by calling dedicated non-locking "_direct" entry points (e.g. `run_vocal_isolation_direct`, `run_batch_language_detection_direct`, `run_language_detection_core`) that skip the lock and reuse the already-claimed unit, seamlessly sharing it across:

1. **Vocal Separation (UVR)**
2. **Language Detection (Whisper)**
3. **ASR Inference (Whisper)**
4. **Speaker Diarization (PyAnnote/WhisperX)**

This prevents a standard task from being preempted mid-pipeline, avoiding state corruption.

**Test structural verification**: The "_direct" sibling invariant is confirmed by
`test_nested_subtasks_route_through_direct_variants_not_model_lock_ctx` using an
AST-based check (`ast.walk`) rather than a fragile substring scan. The AST check
correctly handles aliased imports and ignores string literals/comments.

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
- Confirm all new waits follow current policy (indefinite under saturation) and do not introduce timeout-based request failures in priority/preemption paths.
- **Daemon threads**: All worker threads in integration/concurrency tests MUST be created with `daemon=True`. This ensures a test failure that leaves a thread blocked does not hang the entire test runner. Every daemon worker thread must still be followed by a bounded `thread.join(timeout=...)` and an `assert not thread.is_alive()` — `daemon=True` only stops a stuck thread from hanging the *process* at interpreter exit, it does not detect or fail the test when a thread hangs mid-run, so the join+assert pair is what actually catches that regression.
- **Worker error capture**: Use the `_capture_worker_exc(errors)` context manager pattern (defined in `tests/inference/scheduler/priority/_preemption_test_helpers.py`) instead of bare `except Exception` in worker thread bodies. The bare catch lives only inside the named context manager, keeping worker bodies clean.
- **Deadline-based polling**: Replace `time.sleep(N)` synchronization waits with `_poll_until(lambda: condition, timeout=T)`. This eliminates race conditions on slow CI runners without requiring large fixed sleeps.
- Confirm a regression test exists for each changed liveness pathway.

### 1. Simulate Concurrency Races & Full End-to-End Suite

Run the comprehensive end-to-end concurrency & lifecycle test suite:

```bash
docker build -f Dockerfile.test --target test -t whisper-pro-asr-test .
docker run --rm -v "$(pwd):/app" -w /app whisper-pro-asr-test python3 -m pytest tests/integration/concurrency/ -v
```

### 2. Verify UI Correctness During Concurrency (Playwright E2E)

Run the Playwright E2E UI concurrency & preemption suite:

```bash
docker build -f Dockerfile.test --target test -t whisper-pro-asr-test .
docker run --rm -v "$(pwd):/app" -w /app whisper-pro-asr-test npm run test:e2e
```

### 3. Assert Priority Parallelism, Unit Resume, and Temp File Hygiene

- Verify that concurrent priority requests register and execute across available/borrowed hardware units.
- Assert that paused tasks resume processing without data loss when their targeted unit resumes.
- Verify 100% temporary audio WAV file cleanup via `utils.get_tracked_files()` assertions across all concurrency tiers.
