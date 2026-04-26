# Document Update Skill

This skill automates the process of synchronizing all project documentation with the current state of the codebase.

## Objective
Update `README.md`, all files in `docs/`, and all Mermaid diagrams in `.md` files to reflect recent architectural optimizations (Standardized Audio, Efficient Batch LD, Hardware Re-entrancy).

## Procedure

### 1. Update Mermaid Diagrams
Locate and update Mermaid diagrams in all project `.md` files to match the new architecture.

### 2. Update `README.md`
*   Highlight the new **Standardized Audio Pipeline**.
*   Update the **Hardware Orchestration** section to mention re-entrant locking.
*   Update the **Language Detection** section to describe the new global VAD + batch inference flow.

### 3. Update `docs/ARCHITECTURE.md`
*   Deep dive into the `model_lock_ctx` re-entrancy implementation.
*   Describe the `run_batch_language_detection` optimization.

### 4. Update `docs/CONCURRENCY.md`
*   Explain how priority requests and re-entrant locks prevent deadlocks in high-load scenarios.

### 5. Update `docs/API.md`
*   Ensure the `/detect-language` and `/asr` endpoint documentation matches current status codes (400 for media errors) and the unified 16kHz WAV pipeline.

### 6. Update `docs/DOCKERHUB_DESCRIPTION.md`
*   Synchronize with README updates.

### 8. Hardware Compatibility Matrix
Ensure a **Hardware Compatibility Matrix** is present and updated in `README.md`, `docs/ARCHITECTURE.md`, and `docs/DOCKERHUB_DESCRIPTION.md`. The matrix must accurately reflect current backend support for Vocal Separation and ASR Inference across CPU, NVIDIA, and Intel architectures.
