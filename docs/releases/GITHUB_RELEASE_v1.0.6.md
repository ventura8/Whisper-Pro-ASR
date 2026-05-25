# Release v1.0.6 - Preemption Deadlock Prevention & Concurrency Verification

This update resolves a critical scheduler deadlock/livelock bug in high-load preemption scenarios and establishes exhaustive verification for heterogeneous accelerator pools.

## 🚀 Key Enhancements

### 🛡️ Preemption Deadlock & Livelock Prevention
Fixed a critical bug in the preemption orchestrator where standard transcription tasks could get permanently stuck in a `Paused for Priority Task` state.
- **Sequential Priority Locking**: Extended the scope of `STATE.priority_sequential_lock` to cover the entire context lifecycle inside `early_task_registration`. This guarantees high-priority tasks (e.g., Language Detection) are strictly serialized when requesting hardware resources, preventing concurrent preemption races.
- **Cooperative Yield Recovery**: Ensured standard transcription tasks can safely resume and reclaim hardware slots once sequential priority requests are cleared, eliminating live-lock conditions on single-accelerator hosts (such as NPU-only setups).
- **Early Priority Registration Observability**: Registered priority tasks in the scheduler registry immediately upon arrival (with `"queued"` status and `"Waiting for Priority Slot"` stage) prior to acquiring the sequential preemption lock. This ensures queued priority requests are fully visible on the telemetry dashboard during high resource contention instead of waiting silently in the background.
- **Standard Task Yielding & Non-Blocking Loops**: Prevented standard tasks from starving queued priority tasks by having them yield the lock context (looping and sleeping instead of blocking on the model semaphore) as long as priority tasks exist in the task registry. Standard tasks now use a 0.5-second timeout on resource acquisition to check for newly arriving priority tasks.
- **Priority Preemption Bypass**: Fixed a preemption re-entrancy bug where a running priority task would incorrectly pause itself if another priority task was queued. Priority tasks now ignore preemption checks.

### 📊 Observability & Reporting Enhancements
- **Preemption Visibility**: Dynamically update the status of preempted/paused standard tasks to `"queued"` and set their stage to `"Paused for Priority Task"` during preemption. This ensures paused tasks display properly in the dashboard's queued list with appropriate hardware waiting notifications.
- **Precise Queue Duration**: Upgraded `log_completed_task` to compute the real queue duration for history items using precise engine metrics, with fallbacks to task resource-allocation delta or total elapsed time (for tasks aborted in the queue).
- **FFmpeg Standardization Metrics**: Log the final conversion speed (e.g. `25.4x`) upon completion of FFmpeg audio standardization.

### ⚙️ Hardened Startup Validation
- **Graceful Host CPU Fallback**: Hardened startup routines in the scheduler. If the resolved `HARDWARE_UNITS` collection is empty at startup, the system now automatically registers a default `"Host CPU"` execution slot. This prevents worker threads from blocking indefinitely on zero-value resource semaphores while keeping the application fully functional on CPU-only hosts.

### 🧪 Exhaustive Concurrency Tests
Authored a complete concurrency test suite under `tests/inference/test_priority_concurrency.py` validating task scheduling across multiple hardware slot configurations:
- **0 Units Configuration**: Asserts that the system successfully registers and executes on a fallback Host CPU slot when no accelerators are detected.
- **1 Unit Configuration (Intel NPU)**: Asserts sequential execution of competing priority tasks and proper resumption of paused standard tasks.
- **2 & 3 Units Configurations (Intel NPU, Intel GPU, NVIDIA GPU)**: Asserts correct resource pooling, load distribution, and preemption triggers on heterogeneous setups.
- **Yielding & Preemption Bypass Tests**: Added tests asserting standard task yielding to queued priority requests and priority task bypass of preemption signals.

### 🧹 Code Quality, Testing, & PEP8 Compliance
- Cleaned up formatting styles and linting warnings across all modules (fixed `inconsistent-return-statements`, `unused-argument`, and `import-outside-toplevel` warnings).
- Achieved a perfect **10.00/10** score on all repository code files under `pylint --rcfile=.pylintrc`.
- Maintained a strict **>90% test coverage** for all files (overall project coverage at **94.79%** with 312/312 unit and integration tests passing successfully).

---
*For deployment instructions, refer to the [README.md](README.md) or [ARCHITECTURE.md](docs/ARCHITECTURE.md).*
