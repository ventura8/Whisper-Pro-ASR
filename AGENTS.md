# AI Instructions

## Global Rules

- **Mandatory Pre-Task Agent Review**: Before starting any implementation, read `.agent/instructions.md`, `.agent/skills/SKILLS_CATALOG.md`, and every directly relevant skill/workflow file. Do not begin coding until this review is complete.
- **Mandatory Markdown Update (Every Task, No Exceptions)**: Every time you work on this project, you MUST update all relevant markdown files in the same task. Never ship code-only changes. This applies to every kind of change, including CI/CD, Docker, and other infrastructure/tooling work, not just application code. Before closing work, sync every impacted file among `README.md`, `docs/*.md`, and `.agent/*.md` so they describe the current implementation. A task is incomplete until relevant docs are updated.
- **Concurrency Correctness Priority #1**: For scheduler/resource/lifecycle work, deadlock/livelock prevention and bounded progress are mandatory and take precedence over throughput optimizations.
- **Preserve Helpful Comments**: Never delete comments that explain logic, especially around synchronization, error handling, or non-obvious code paths. Code clarity and maintainability take precedence over aggressive line count reduction.
- **Strict File Size Limit**: Any `.py` file MUST NOT exceed **500 lines**.
- **Logging Only**: Use `logger` (logging module) for all output. No `print()` calls allowed.
- **Hardware Agnostic**: Ensure code works across NPU, GPU (Intel/NVIDIA), and CPU. Use `modules.core.config` for device selection.
- **Agent Asset Maintenance (Mandatory)**: Whenever code, architecture, CI flow, testing strategy, release process, or operational behavior changes, update all affected agent assets in `.agent/` during the same task. This includes `instructions.md`, relevant files in `.agent/skills/`, and `.agent/workflows/` so agent guidance stays accurate.
- **Frontend Gate Maintenance (Mandatory)**: Whenever dashboard HTML/JS/CSS changes, run and keep aligned the frontend quality gates (`npm run quality:frontend`) and update related documentation/skills if commands or thresholds change.
- **Frontend Security Gate (Mandatory)**: CI and local build parity scripts must fail on any npm audit vulnerability using `npm audit --audit-level=low` after `npm ci`.
- **Build Script Bootstrap (Mandatory)**: `scripts/ci/build-and-test.sh` and `scripts/ci/build-and-test.ps1` are allowed to bootstrap missing `npm`/`docker` dependencies on Linux via `apt-get`, but must fail clearly when automatic installation is unavailable.
- **Playwright + MCP Tooling (Mandatory)**: For frontend and dashboard verification, use Playwright CLI commands (`npx playwright ...` or npm scripts wrapping Playwright CLI) and MCP browser tooling for DOM/state inspection and deterministic troubleshooting. Do not rely on ad-hoc browser/manual-only verification paths when these automated tools are available.
- **Docker-Only Quality Execution**: All linting, formatting, schema validation, type checking, dependency auditing, security scanning, and testing MUST be executed exclusively within the Docker test image pipeline (`Dockerfile.test` target `test`). Host-side executions of these quality gates are forbidden; the local/CI wrappers (`build-and-test.sh` and `build-and-test.ps1`) must only build and run the Docker test image and consume its results.
- **PR Review-Thread Closure (Mandatory)**: After fixing (or determining no change is needed for) a PR review comment/thread, reply to that specific thread on GitHub explaining the outcome (fixed / already resolved / not applicable / deferred, with a brief reason), then mark it resolved. Do this for every thread addressed in the pass, not just a single summary comment on the PR. Only reply/resolve once the underlying fix is actually pushed and visible in the PR diff — do not resolve a thread against a local, unpushed commit.

## Task & Status Display Priority (Mandatory)

Correct task status display on the dashboard is a critical user-facing contract.
Any change to scheduling, preemption, or monitoring MUST preserve correctness and requires synchronized agent asset and test updates.

### Mandatory Checklist

- **Status Enum Stability**: The 7 task statuses (`initializing`, `queued`, `active`, `post-processing`, `completed`, `failed`, `unknown`) are immutable. Do not add, remove, or rename without corresponding updates to `.agent/skills/monitoring/task_status_display_specification_skill.md`, frontend tests, and backend telemetry tests. Note: `unknown` is an **internal-only fallback** status; it must be normalized to a concrete canonical label (e.g., `initializing`) before any API payload serialization or dashboard rendering — it must never appear as a visible display value.
- **Stage Transitions**: When adding new pipeline stages or modifying preemption behavior, update `.agent/skills/monitoring/task_status_display_specification_skill.md` with the new stage semantics. Verify dashboard rendering reflects the change and tests validate the transition.
- **No Placeholder Display Values (Hard Rule)**: Dashboard-visible status/stage output must never expose placeholder-like values (`unknown`, `none`, `null`, `undefined`, `(0/0)`, `resuming`, or any sentinel placeholder text). If internal state is undefined, normalize it to a concrete canonical runtime label before API payload/rendering.
- **Preemption Visibility**: Paused tasks MUST display as `status='queued'` + stage containing 'Paused for Priority Task'. This distinction allows dashboard to show "Paused for priority" vs "Waiting for hardware" hints correctly.
- **Order Determinism**: Task rendering order MUST be deterministic (active-first, then arrival/start_time based) independent of map iteration or async fetch order. Validate via `test_task_ordering_*` regression tests in `tests/monitoring/`.
- **Frontend Alignment**: After any status/stage change, run `npm run test:js` and verify dashboard feature script coverage remains ≥90% lines/statements for critical status-rendering modules (`dashboard/features/runtime.js`, `dashboard/features/speed_status.js`; branches/functions pragmatic per `quality/frontend_quality_gates_skill.md`).

### Change Impact Triggers

- **Any scheduler status update**: Re-validate display combinations in `monitoring/task_status_display_specification_skill.md`; run backend monitoring tests.
- **Any preemption logic change**: Verify "Paused for Priority Task" stage is set correctly during pause/resume; run priority preemption tests.
- **Any preprocessing/yield checkpoint change**: Preserve cooperative preemption hooks inside HQ-prep FFmpeg progress and keep docs/tests synchronized for those checkpoints.
- **Any priority scheduling change**: Preserve same-priority FIFO at acquisition time, but do not reintroduce global whole-task serialization for detect-language across multiple hardware units.
- **Any task ordering change**: Add regression test to `tests/monitoring/` to verify deterministic ordering across concurrent task arrivals.
- **Any frontend status rendering change**: Run `npm run test:js` with coverage; update test matrix if new UI states added.

## Resource Management

- **Priority Wait Semantics**: Critical priority/preemption synchronization paths require explicit wait steps for hardware/preemption handoff instead of failing immediately on simple scheduler timeouts; waits remain indefinite under saturation.
- **Thread Compliance**: All multi-threaded components (FFmpeg, OpenVINO, ONNX Runtime) MUST respect the thread limits set in `modules.core.config` (`ASR_THREADS`, `PREPROCESS_THREADS`, `FFMPEG_THREADS`).
- **FFmpeg Parallelization**: Prefer parallelizing across segments (as seen in `language_detection.py`) rather than relying on high thread counts for single-file filters, as some FFmpeg filters are inherently single-threaded.
- **Library Patching**: If a third-party library (e.g., `audio-separator`) does not expose thread configuration, use monkey-patching on the underlying engine (e.g., `onnxruntime`) to enforce limits.
- **Stability & Cleanup**: ALL temporary files, file descriptors, and memory-intensive assets MUST be managed using `try...finally` blocks.
- **Path Resolution**: Always use absolute paths (resolved against `config.CACHE_DIR`) for temporary file management to ensure consistency across different OS environments (Windows/Docker).

## Media Standardization

- **Uniform Specification**: All audio ingested MUST be converted to **16kHz, Mono, 16-bit PCM**.
- **Shared Constants**: Always use `utils.STANDARD_AUDIO_FLAGS` and `utils.STANDARD_NORMALIZATION_FILTERS` for FFmpeg commands to ensure pipeline consistency.

## Performance & Efficiency

- **Chunked Processing**: For identification or status tasks, ALWAYS favor chunked FFmpeg extraction (`-ss` / `-t`) from the source media over full-file normalization.
- **I/O Optimization**: Avoid using `soundfile` (`sf.read`) directly on non-WAV video containers as it causes expensive full-file probes. Use FFmpeg to extract small chunks to temporary WAV files first if signal analysis (RMS) is required.
- **Priority Yielding**: Maintain "Full-Pipeline Priority Yielding" logic—high-priority tasks (LD) must pause batch operations (ASR).
- **Silent Status**: Ensure hardware acceleration warnings (ONNX, OpenVINO) are suppressed in favor of custom authoritative status logs.
- **Model Lifecycle**: Respect `MODEL_IDLE_TIMEOUT` and `AGGRESSIVE_OFFLOAD` settings in `modules/core/config.py`. If `MODEL_IDLE_TIMEOUT > 0`, models are purged by a deferred `threading.Timer` (started after the last active session ends) after the configured idle period instead of immediate unloading; the timer is automatically cancelled and rescheduled if new work arrives before it fires.

## Speaker Diarization

- **WhisperX Pipeline**: Speaker diarization uses `whisperx` (alignment → diarization → speaker assignment) inside an isolated worker subprocess (`whisperx_worker.py` via `multiprocessing` `spawn`; `whisper_pro_asr.py` and package inits are spawn-safe so the child does not import the main torch stack before isolation). The parent caches opaque handles in `ALIGN_POOL` and `DIARIZE_POOL` per hardware unit; RPC access is serialized on one shared worker (`WHISPERX_WORKER_CALL_TIMEOUT_SEC` defaults to `0` / no deadline).
- **DIARIZATION_HF_TOKEN Requirement**: Diarization requires a valid Hugging Face token (`DIARIZATION_HF_TOKEN` environment variable) for access to PyAnnote speaker segmentation models.
- **Graceful Fallback**: If diarization fails or `DIARIZATION_HF_TOKEN` is missing, the system must fall back to standard transcription without speaker labels.

## ASR Parameters

- **Forwarding**: `initial_prompt`, `vad_filter`, and `word_timestamps` parameters from the API must be forwarded to `model.transcribe()` in `model_manager.run_transcription()`.
- **Subtitle Layout**: `max_line_width` and `max_line_count` parameters control text wrapping in SRT/VTT formatters via `_wrap_text()` in `modules/utils.py`.

## Endpoint Taxonomy Contract

- `/asr`, `/v1/audio/transcriptions`, and `/v1/audio/translations` are one standard-priority ASR execution class.
- `/detect-language` and `/detectlang` are one high-priority language-detection execution class.
- `/v1/audio/...` calls are protocol compatibility surfaces and must not be treated as a separate scheduler priority class.

## Security & Access Control

- **CORS Allowlist Requirement**: Never enable wildcard CORS (`*`) by default. CORS origins must be configured explicitly via `CORS_ORIGINS` or explicitly opted into via `CORS_ALLOW_ALL=true`.
- **Model Supply Chain Guarding**: Dynamic model loading via `/system/settings` must be validated against `is_valid_model_name()` to prevent arbitrary external downloads and path traversals (`..`).
- **Administrative Endpoint Protection**: State-changing administrative endpoints (`/system/settings`, `/system/history/clear`, `/system/telemetry/clear`, `/system/cleanup`, `/logs/download`) must require `ADMIN_API_KEY` authentication when configured. If no key is configured, Origin/Referer validation is used only as anti-CSRF protection (not as the primary authorization mechanism).
- **Default Localhost Publish**: `docker-compose.yml` must publish `127.0.0.1:9000:9000` by default so unauthenticated management data is not reachable on the LAN. All-interface publish (`9000:9000`) requires `API_KEY`. Startup must warn when both keys are unset.
- **Audit Logging**: All administrative mutations and purges must be recorded with client IP and User-Agent using `security.audit_log_admin_action()`.

## Code Style

- Follow PEP 8 and use type hints.
- Maintain high test coverage (min 90%).
- Enforce a maximum cyclomatic complexity score of A (Radon rank A, complexity <= 5) for all Python functions, methods, and blocks.
- **No Inline Suppressions or Disables (Hard Rule)**: Do not use inline lint suppressions, excludes, ignores, or disables anywhere in code or tests (such as linter disable comments, type-ignore annotations, or warning bypasses). All code and tests must be written cleanly to pass quality gates without inline suppressions.
