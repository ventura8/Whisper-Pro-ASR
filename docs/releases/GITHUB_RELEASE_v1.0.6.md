# Release v1.0.6 - Preemption Deadlock Prevention & Concurrency Verification

This update resolves a critical scheduler deadlock/livelock bug in high-load preemption scenarios and establishes exhaustive verification for heterogeneous accelerator pools.

## 🚀 Key Enhancements

### 🛡️ Preemption Deadlock & Livelock Prevention
Fixed a critical bug in the preemption orchestrator where standard transcription tasks could get permanently stuck in a `Paused for Priority Task` state.
- **Sequential Priority Locking**: Extended the scope of `STATE.priority_sequential_lock` to cover the entire context lifecycle inside `early_task_registration`. This guarantees high-priority tasks (e.g., Language Detection) are strictly serialized when requesting hardware resources, preventing concurrent preemption races.
- **Cooperative Yield Recovery**: Ensured standard transcription tasks can safely resume and reclaim hardware slots once sequential priority requests are cleared, eliminating live-lock conditions on single-accelerator hosts (such as NPU-only setups).

### ⚙️ Hardened Startup Validation
- **Graceful Host CPU Fallback**: Hardened startup routines in the scheduler. If the resolved `HARDWARE_UNITS` collection is empty at startup, the system now automatically registers a default `"Host CPU"` execution slot. This prevents worker threads from blocking indefinitely on zero-value resource semaphores while keeping the application fully functional on CPU-only hosts.

### 🧪 Exhaustive Concurrency Tests
Authored a complete concurrency test suite under `tests/inference/test_priority_concurrency.py` validating task scheduling across multiple hardware slot configurations:
- **0 Units Configuration**: Asserts that the system successfully registers and executes on a fallback Host CPU slot when no accelerators are detected.
- **1 Unit Configuration (Intel NPU)**: Asserts sequential execution of competing priority tasks and proper resumption of paused standard tasks.
- **2 & 3 Units Configurations (Intel NPU, Intel GPU, NVIDIA GPU)**: Asserts correct resource pooling, load distribution, and preemption triggers on heterogeneous setups.

### 🧹 Code Quality & PEP8 Compliance
- Cleaned up formatting styles and linting warnings across all modified scripts.
- Achieved a perfect **10.00/10** score on all repository code files under `pylint --rcfile=.pylintrc`.

---
*For deployment instructions, refer to the [README.md](README.md) or [ARCHITECTURE.md](docs/ARCHITECTURE.md).*
